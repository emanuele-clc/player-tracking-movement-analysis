"""
Stage 4: team/role assignment via jersey-color clustering.

Approach:
- For each detection classified as a person (player/goalkeeper/referee), crop
  a torso-region patch from the source frame (middle 45% of box height,
  center 60% of box width - avoids grass/background bleeding into the color
  estimate, and avoids shorts/socks which are noisier across a squad).
- Reduce each patch to a single dominant color in HSV space (median of patch
  pixels - a robust one-cluster "dominant color" that's much cheaper than
  running k-means per patch across thousands of detections).
- Average each track_id's dominant color across several sampled frames (more
  stable than any single frame - cuts down on lighting/motion-blur noise).
- Cluster track-level average colors (k-means, k=2 by default) to split
  detections into two teams. A person whose color sits far from either
  cluster centroid (distance > --outlier-factor times the cluster's own
  spread) is flagged as a likely referee/neutral instead of forced into a
  team - referees are usually a small minority in a clip, so this is a
  simple outlier rule rather than a proper 3-cluster fit, which tends to be
  unstable with so few examples.

This only requires the source video + the tracklets parquet from
detect_and_track.py - no separate model.

Usage:
    python src/team_classification.py --video data/raw/clip.mp4 --clip-id clip01
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def torso_patch(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = int(x1 + 0.20 * bw)
    px2 = int(x1 + 0.80 * bw)
    py1 = int(y1 + 0.25 * bh)
    py2 = int(y1 + 0.70 * bh)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)
    if px2 <= px1 or py2 <= py1:
        return None
    return frame[py1:py2, px1:px2]


def dominant_color_hsv(patch):
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    return np.median(hsv, axis=0)


def extract_dominant_colors(video_path, df, sample_per_track):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    person_df = df[df["class"].isin(["person", "player", "goalkeeper", "referee"])].copy()

    sampled_rows = (
        person_df.groupby("track_id", group_keys=False)
        .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(sample_per_track, len(g))).astype(int)])
    )
    wanted_frames = set(sampled_rows["frame"].unique().tolist())
    rows_by_frame = {f: g for f, g in sampled_rows.groupby("frame")}

    colors_by_frame = {}
    frame_idx = 0
    max_wanted = max(wanted_frames) if wanted_frames else -1
    while frame_idx <= max_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted_frames:
            for _, row in rows_by_frame[frame_idx].iterrows():
                patch = torso_patch(frame, row.x1, row.y1, row.x2, row.y2)
                color = dominant_color_hsv(patch)
                if color is not None:
                    colors_by_frame.setdefault(row.track_id, []).append(color)
        frame_idx += 1

    cap.release()

    records = []
    for track_id, colors in colors_by_frame.items():
        colors = np.array(colors)
        records.append({
            "track_id": track_id,
            "n_samples": len(colors),
            "h": float(np.median(colors[:, 0])),
            "s": float(np.median(colors[:, 1])),
            "v": float(np.median(colors[:, 2])),
        })
    return pd.DataFrame(records)


def assign_teams(color_df, k, outlier_factor):
    features = color_df[["h", "s", "v"]].to_numpy()
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(features_norm)

    dist_to_centroid = np.linalg.norm(features_norm - km.cluster_centers_[labels], axis=1)
    cluster_spread = np.array([
        dist_to_centroid[labels == c].std() if (labels == c).sum() > 1 else dist_to_centroid[labels == c].mean()
        for c in range(k)
    ])
    is_outlier = dist_to_centroid > outlier_factor * cluster_spread[labels]

    color_df = color_df.copy()
    color_df["cluster"] = labels
    color_df["dist_to_centroid"] = dist_to_centroid
    color_df["team"] = [
        "referee_or_neutral" if outlier else f"team_{c}"
        for outlier, c in zip(is_outlier, labels)
    ]
    return color_df


def run(video_path, clip_id, k, sample_per_track, outlier_factor):
    tracklets_path = PROCESSED_DIR / "tracklets" / f"{clip_id}.parquet"
    if not tracklets_path.exists():
        raise SystemExit(f"No tracklets at {tracklets_path} - run detect_and_track.py first.")

    df = pd.read_parquet(tracklets_path)
    color_df = extract_dominant_colors(video_path, df, sample_per_track)
    if len(color_df) < k:
        raise SystemExit(f"Only {len(color_df)} tracks with usable color samples, need >= {k} for {k}-means.")

    result = assign_teams(color_df, k=k, outlier_factor=outlier_factor)

    out_dir = PROCESSED_DIR / "teams"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.parquet"
    result.to_parquet(out_path, index=False)

    print(result.sort_values("team").to_string(index=False))
    print(f"\nWrote team assignment for {len(result)} tracks -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--k", type=int, default=2, help="Number of team clusters (default 2)")
    ap.add_argument("--sample-per-track", type=int, default=8,
                     help="Max frames sampled per track to estimate jersey color (default 8)")
    ap.add_argument("--outlier-factor", type=float, default=2.5,
                     help="Std-dev multiple beyond which a track is flagged referee/neutral instead of a team")
    args = ap.parse_args()

    run(args.video, args.clip_id, args.k, args.sample_per_track, args.outlier_factor)


if __name__ == "__main__":
    main()
