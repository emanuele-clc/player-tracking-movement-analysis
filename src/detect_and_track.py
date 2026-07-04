"""
Stage 1: player/ball detection + multi-object tracking on raw broadcast clips.

Plan:
- YOLOv8 (ultralytics) fine-tuned or zero-shot on a football-detection checkpoint
  to detect players, goalkeepers, referees, and the ball per frame.
- ByteTrack (via the `supervision` package) to link per-frame detections into
  consistent per-player tracklets across a clip, surviving short occlusions.
- Output: one row per (frame, track_id) with bounding box + class, written to
  data/processed/tracklets/<clip_id>.parquet

Not yet implemented — see README.md pipeline table for status.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tracklets"


def main():
    raise NotImplementedError(
        "Detection + tracking pipeline not yet built. "
        "Planned: ultralytics YOLOv8 + supervision ByteTrack per clip in data/raw, "
        "writing per-frame track boxes to data/processed/tracklets/."
    )


if __name__ == "__main__":
    main()
