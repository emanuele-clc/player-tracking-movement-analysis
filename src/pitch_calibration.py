"""
Stage 2: homography from broadcast pixel coordinates to real-world pitch
coordinates (meters), turning tracked bounding-box foot-points into positions
on an actual football pitch.

Two ways to supply the point correspondences a homography needs:

1. Manual (works today, no extra model required): you pick >=4 pitch
   keypoints that are visible in a representative frame of the clip (penalty
   box corners, centre circle, halfway line...) and record their pixel (x, y)
   alongside which of the 32 standard keypoints (see pitch_config.py) they
   are. `--make-template` writes a starter JSON file to fill in by hand
   (e.g. by opening one frame in any image viewer and reading off pixel
   coordinates).

2. Automatic (once a pitch-keypoint detection model is available — e.g.
   Roboflow's football-field-detection model, see data/README.md): per-frame
   keypoint detections replace the manual file, giving a homography that
   adapts to camera movement/zoom instead of one static matrix per clip.

Important honest limitation of mode 1: broadcast cameras pan/zoom/cut
between angles, so a single static homography fitted from one frame is only
valid for the (sub-)segment of the clip where the camera framing hasn't
materially changed. For a fixed wide-angle camera or a short single-shot
clip this is fine; for a full match with camera cuts, mode 2 (per-frame
calibration) is what makes the pitch coordinates reliable throughout.

The homography math (ViewTransformer) is a small, standard OpenCV
find/apply-homography wrapper, adapted from Roboflow's open-source `sports`
project (MIT-licensed):
https://github.com/roboflow/sports/blob/main/sports/common/view.py

Usage:
    python src/pitch_calibration.py --clip-id smoketest --make-template
    # ... fill in data/processed/correspondences/smoketest.json by hand ...
    python src/pitch_calibration.py --clip-id smoketest \
        --correspondences data/processed/correspondences/smoketest.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class ViewTransformer:
    """Homography wrapper: fit from N>=4 (pixel <-> pitch-plane) point pairs,
    then project any pixel point into pitch coordinates."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape:
            raise ValueError("source and target must have the same shape")
        if source.shape[1] != 2:
            raise ValueError("points must be 2D (x, y)")
        if source.shape[0] < 4:
            raise ValueError(f"homography needs >=4 point pairs, got {source.shape[0]}")

        self.m, mask = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))
        if self.m is None:
            raise ValueError("cv2.findHomography failed to find a solution for these points")
        self.inliers = int(mask.sum()) if mask is not None else source.shape[0]
        self.n_points = source.shape[0]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(reshaped, self.m)
        return out.reshape(-1, 2).astype(np.float32)


def make_template(clip_id: str, config: SoccerPitchConfiguration) -> Path:
    """Write a starter correspondences JSON listing all 32 standard pitch
    keypoints with empty pixel coordinates, to be filled in by hand for
    whichever ones are actually visible in a representative frame."""
    out_dir = PROCESSED_DIR / "correspondences"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.json"

    template = {
        "clip_id": clip_id,
        "frame": 0,
        "instructions": (
            "Fill in pixel_x/pixel_y for at least 4 keypoints you can see in "
            "this clip's reference frame; leave the rest null. vertex_id "
            "matches SoccerPitchConfiguration.vertices (1-indexed)."
        ),
        "points": [
            {"vertex_id": i + 1, "pixel_x": None, "pixel_y": None}
            for i in range(len(config.vertices))
        ],
    }
    out_path.write_text(json.dumps(template, indent=2))
    return out_path


def load_correspondences(path: Path, config: SoccerPitchConfiguration):
    data = json.loads(path.read_text())
    pixel_pts, pitch_pts = [], []
    for p in data["points"]:
        if p.get("pixel_x") is None or p.get("pixel_y") is None:
            continue
        vertex_idx = p["vertex_id"] - 1
        pixel_pts.append([p["pixel_x"], p["pixel_y"]])
        pitch_pts.append(config.vertices[vertex_idx])

    if len(pixel_pts) < 4:
        raise SystemExit(
            f"Only {len(pixel_pts)} filled-in correspondences in {path} — need >=4. "
            "Open the clip's reference frame and fill in more pixel_x/pixel_y values."
        )

    # pitch coordinates stored in cm in the config -> convert to meters here
    pitch_pts_m = np.array(pitch_pts, dtype=np.float32) / 100.0
    return np.array(pixel_pts, dtype=np.float32), pitch_pts_m


def run(clip_id: str, correspondences_path: Path):
    config = SoccerPitchConfiguration()
    pixel_pts, pitch_pts_m = load_correspondences(correspondences_path, config)
    transformer = ViewTransformer(source=pixel_pts, target=pitch_pts_m)
    print(f"Fitted homography from {transformer.n_points} point pairs.")

    tracklets_path = PROCESSED_DIR / "tracklets" / f"{clip_id}.parquet"
    if not tracklets_path.exists():
        raise SystemExit(f"No tracklets found at {tracklets_path} — run detect_and_track.py first.")

    df = pd.read_parquet(tracklets_path)
    feet = df[["foot_x", "foot_y"]].to_numpy(dtype=np.float32)
    pitch_xy = transformer.transform_points(feet)
    df["x_m"] = pitch_xy[:, 0]
    df["y_m"] = pitch_xy[:, 1]

    # Sanity bounds: a real pitch is ~105m x 68m (this config: 120m x 70m
    # touchline-to-touchline incl. run-off convention above) - flag rows that
    # land wildly outside plausible bounds, which usually means either a bad
    # correspondence or the camera framing drifted from the reference frame.
    margin = 15.0  # meters of slack for detection/homography noise
    plausible = (
        (df["x_m"] >= -margin) & (df["x_m"] <= config.pitch_length_m + margin) &
        (df["y_m"] >= -margin) & (df["y_m"] <= config.pitch_width_m + margin)
    )
    n_bad = (~plausible).sum()
    if n_bad:
        print(f"Warning: {n_bad}/{len(df)} rows ({n_bad / len(df):.1%}) project outside "
              f"plausible pitch bounds — check correspondences or expect camera drift.")

    out_dir = PROCESSED_DIR / "pitch_coords"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows with pitch coordinates -> {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--make-template", action="store_true",
                     help="Write a starter correspondences JSON for this clip and exit")
    ap.add_argument("--correspondences", type=Path, default=None,
                     help="Path to a filled-in correspondences JSON (default: "
                          "data/processed/correspondences/<clip-id>.json)")
    args = ap.parse_args()

    config = SoccerPitchConfiguration()

    if args.make_template:
        path = make_template(args.clip_id, config)
        print(f"Wrote template -> {path}\nFill in pixel_x/pixel_y for >=4 visible keypoints, then re-run without --make-template.")
        return

    correspondences_path = args.correspondences or (
        PROCESSED_DIR / "correspondences" / f"{args.clip_id}.json"
    )
    if not correspondences_path.exists():
        raise SystemExit(
            f"No correspondences file at {correspondences_path}. "
            f"Run with --make-template first, fill it in, then re-run."
        )

    run(args.clip_id, correspondences_path)


if __name__ == "__main__":
    main()
