"""
End-to-end orchestrator: runs every pipeline stage on ONE arbitrary uploaded
video, calling each stage's already-validated code directly (not via
subprocess) so a UI (see app.py) can report progress step by step and render
a full report at the end - "upload a video, get a report", the way this
project would actually be pitched to a club.

Automatic pitch calibration (see auto_calibrate.py) is attempted on several
sampled frames and the best-confidence result is kept. Below a confidence
threshold, the run falls back to pixel-space analysis rather than silently
producing bad real-world distances: tracking, team clustering, and heatmap
*shape* are still genuine and correct in that mode, but anything that needs
real metres (speeds in m/s, the space-creation score in m^2) is skipped and
clearly flagged in the returned warnings, never faked.

Usage (library):
    from pipeline import run_pipeline
    summary = run_pipeline("data/raw/my_clip.mp4", "my_clip", max_frames=150)

Usage (CLI, mainly for testing):
    python src/pipeline.py --video data/raw/clip.mp4 --clip-id clip01 --max-frames 150
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import detect_and_track
import team_classification
import build_tracking_dataset
import cluster_movement
import generate_heatmaps
import space_creation
from auto_calibrate import auto_calibrate
from pitch_calibration import ViewTransformer
from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CALIBRATION_CONFIDENCE_THRESHOLD = 0.5


def _sample_frames(video_path, n_samples=6):
    """Grab a handful of frames spread through the clip to try calibration
    on - camera motion, players occluding lines, or motion blur mean not
    every frame shows the pitch markings equally well, so trying several
    and keeping the best result is far more robust than trying just one."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 1:
        cap.release()
        return []
    idxs = sorted(set(int(total * f) for f in np.linspace(0.1, 0.85, n_samples)))
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


def best_auto_calibration(video_path, config=None, n_samples=6, tol=0.15):
    """Try automatic calibration on several sampled frames, keep the
    highest-confidence result. Always returns a dict with at least
    status/confidence/notes, even if every frame failed to calibrate."""
    config = config or SoccerPitchConfiguration()
    best = None
    best_frame_idx = None
    for idx, frame in _sample_frames(video_path, n_samples):
        result = auto_calibrate(frame, config=config, tol=tol)
        if best is None or result["confidence"] > best["confidence"]:
            best = result
            best_frame_idx = idx
    if best is None:
        best = {"status": "failed", "confidence": 0.0,
                "notes": "Could not read any frames from the video for calibration."}
    best["frame_idx"] = best_frame_idx
    return best


def _write_pitch_coords(clip_id, calib, config, frame_size=None):
    """Project tracked foot-points into pitch coordinates if calibration was
    confident enough; otherwise fall back to a SCHEMATIC pixel-space layout
    (still valid for tracking playback and team clustering shape - just not
    real metres, distances, or speeds). The fallback is rescaled from raw
    pixel coordinates into the same 0-length_m x 0-width_m box the metric
    mode uses (by the source frame's width/height), purely so both modes can
    share one dashboard canvas - it is NOT a real calibration, just a fit-to-
    frame relayout, and is always labeled as such downstream (clip_meta.mode
    == "pixel"). Returns the mode used: "metric" or "pixel"."""
    tracklets_path = PROCESSED_DIR / "tracklets" / f"{clip_id}.parquet"
    df = pd.read_parquet(tracklets_path)
    feet = df[["foot_x", "foot_y"]].to_numpy(dtype=np.float32)

    mode = "metric" if calib.get("confidence", 0.0) >= CALIBRATION_CONFIDENCE_THRESHOLD else "pixel"
    if mode == "metric":
        pixel_pts = np.array(calib["pixel_pts"], dtype=np.float32)
        pitch_pts = np.array(calib["pitch_pts_m"], dtype=np.float32)
        transformer = ViewTransformer(source=pixel_pts, target=pitch_pts)
        xy = transformer.transform_points(feet)
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
    else:
        frame_w, frame_h = frame_size or (feet[:, 0].max() or 1.0, feet[:, 1].max() or 1.0)
        df["x_m"] = feet[:, 0] / float(frame_w) * config.pitch_length_m
        df["y_m"] = feet[:, 1] / float(frame_h) * config.pitch_width_m

    out_dir = PROCESSED_DIR / "pitch_coords"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{clip_id}.parquet", index=False)

    _record_calibration_mode(clip_id, mode, calib)
    return mode


def _record_calibration_mode(clip_id, mode, calib):
    """Persist this clip's calibration mode/confidence/notes so the dashboard
    can honestly label whether positions are real pitch metres or a
    schematic pixel-space fallback - read by generate_dashboard_data.py."""
    import json
    meta_path = PROCESSED_DIR / "calibration_mode.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[clip_id] = {
        "mode": mode,
        "confidence": calib.get("confidence", 0.0),
        "notes": calib.get("notes", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def run_pipeline(video_path, clip_id, weights="yolov8n.pt", max_frames=None,
                  conf=0.25, annotate=True, progress_cb=None):
    """Runs every stage end-to-end for one uploaded video.

    progress_cb(fraction: float, message: str), if given, is called before
    each stage - wire it to a UI progress bar (see app.py).

    Returns a summary dict the report UI can render directly. Raises only
    for hard failures (unreadable video, zero detections at all); expected
    degraded cases (calibration not confident, too few tracks for role
    clustering, too little teammate co-visibility for the space-creation
    score) are caught and recorded in `warnings` instead, never hidden and
    never crashing the run.
    """
    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    warnings = []
    config = SoccerPitchConfiguration()
    video_path = Path(video_path)

    report(0.05, "Detecting players & the ball, then tracking them across frames...")
    tracklets_df = detect_and_track.run(video_path, clip_id, weights, conf, annotate=annotate, max_frames=max_frames)
    if tracklets_df.empty:
        raise RuntimeError(
            "No detections at all in this video. Check it's a readable video file with visible "
            "people in frame, and that the weights file loaded correctly."
        )

    report(0.35, "Looking for a goal box or penalty box to calibrate real-world distances...")
    calib = best_auto_calibration(video_path, config=config)
    cap = cv2.VideoCapture(str(video_path))
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) or None, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or None)
    cap.release()
    calib_mode = _write_pitch_coords(clip_id, calib, config, frame_size=frame_size)
    calib["mode"] = calib_mode
    if calib_mode == "pixel":
        warnings.append(
            "Automatic pitch calibration wasn't confident enough on this video "
            f"({calib.get('notes', '')}) - falling back to pixel-space analysis. Tracking, team "
            "clustering, and heatmap shape below are still real and correct; distances, speeds, "
            "and the space-creation score (which need real metres) are skipped rather than faked."
        )

    report(0.5, "Classifying players into teams by jersey colour...")
    try:
        team_classification.run(video_path, clip_id, k=2, sample_per_track=8, outlier_factor=2.5)
    except SystemExit as e:
        warnings.append(f"Team classification skipped: {e}")

    report(0.65, "Assembling the tracking dataset...")
    build_tracking_dataset.run(clip_id)
    if calib_mode == "pixel":
        # Speed computed from schematic (not real) coordinates would look like
        # real m/s but isn't - null it out rather than show a fabricated number.
        tds_path = PROCESSED_DIR / "tracking_dataset.parquet"
        tds = pd.read_parquet(tds_path)
        tds.loc[tds["clip_id"] == clip_id, "speed_mps"] = np.nan
        tds.to_parquet(tds_path, index=False)

    report(0.78, "Clustering player roles by movement pattern...")
    try:
        cluster_movement.run(clip_id, k=3, min_frames=3)
    except SystemExit as e:
        warnings.append(f"Role clustering skipped: {e}")

    report(0.88, "Building heatmaps...")
    if calib_mode == "metric":
        try:
            generate_heatmaps.run(clip_id, None, 34, 22)
        except SystemExit as e:
            warnings.append(f"Heatmaps skipped: {e}")
    else:
        warnings.append("Heatmaps are skipped in this report (pixel-space fallback mode) - see the calibration note above.")

    report(0.95, "Computing the off-ball space-creation score...")
    if calib_mode == "metric":
        try:
            space_creation.run(clip_id, window_s=2.0, grid_res=1.0, min_valid_frames=3)
        except SystemExit as e:
            warnings.append(f"Space-creation score skipped: {e}")
    else:
        warnings.append("Space-creation score needs real metres, so it's skipped in pixel-space fallback mode.")

    report(1.0, "Done.")

    return {
        "clip_id": clip_id,
        "n_detections": int(len(tracklets_df)),
        "n_tracks": int(tracklets_df["track_id"].nunique()),
        "n_frames": int(tracklets_df["frame"].nunique()),
        "calibration": calib,
        "warnings": warnings,
        "annotated_video": str(PROCESSED_DIR / "annotated" / f"{clip_id}.mp4") if annotate else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    def cli_progress(pct, msg):
        print(f"[{pct*100:5.1f}%] {msg}")

    summary = run_pipeline(
        args.video, args.clip_id, weights=args.weights, max_frames=args.max_frames,
        conf=args.conf, annotate=not args.no_annotate, progress_cb=cli_progress,
    )
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
