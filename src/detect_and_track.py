"""
Stage 1: player/ball detection + multi-object tracking on a broadcast clip.

Approach:
- YOLOv8 (ultralytics) for per-frame detection. Model-agnostic on purpose:
  it reads whatever classes the loaded weights actually expose (via
  `model.names`) rather than hardcoding ids, so the same script runs against
  two very different weight sets without code changes:
    1. stock COCO-pretrained yolov8n.pt (default) - only "person" and
       "sports ball" are kept, remapped to "person"/"ball". Works
       immediately on any clip, no setup required.
    2. a football-fine-tuned checkpoint with classes {ball, goalkeeper,
       player, referee} (e.g. Roboflow's football-players-detection /
       football-ball-detection models - see data/README.md for how to get
       one) - all four classes are kept as-is. Pass it via --weights.
  Detection quality/tracking stability is materially better with (2): see
  README.md "Status" section for the measured COCO-baseline numbers this
  replaces.
- ByteTrack (via the `supervision` package) links per-frame detections into
  persistent per-object tracklets across the clip, surviving brief occlusion
  and missed detections.
- Output: one row per (frame, tracker_id): timestamp, class, bbox, and the
  bbox's bottom-center "foot point" in pixel coordinates - the point
  pitch_calibration.py projects into real-world pitch coordinates.

Usage:
    python src/detect_and_track.py --video data/raw/clip.mp4 --clip-id clip01
    python src/detect_and_track.py --video data/raw/clip.mp4 --clip-id clip01 --annotate
    python src/detect_and_track.py --video data/raw/clip.mp4 --clip-id clip01 \
        --weights models/football-players-detection.pt
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

FOOTBALL_LABELS = {"ball", "goalkeeper", "player", "referee"}
COCO_KEEP = {"person": "person", "sports ball": "ball"}


def resolve_classes(model_names):
    """Pick which of the loaded model's classes to keep and how to label them,
    based on what the weights actually expose - not a hardcoded assumption.

    - Football-fine-tuned weights (ball/goalkeeper/player/referee): keep all four.
    - Generic COCO weights: keep only person + sports ball, renamed to
      person/ball so downstream code doesn't care which model produced them.
    - Anything else: keep every class the model has (best effort).
    """
    name_to_id = {str(v).lower(): k for k, v in model_names.items()}

    if FOOTBALL_LABELS.issubset(name_to_id.keys()):
        ids = [name_to_id[label] for label in sorted(FOOTBALL_LABELS)]
        names = {i: model_names[i] for i in ids}
        return ids, names

    ids, names = [], {}
    for coco_name, out_name in COCO_KEEP.items():
        if coco_name in name_to_id:
            i = name_to_id[coco_name]
            ids.append(i)
            names[i] = out_name

    if ids:
        return ids, names

    return list(model_names.keys()), dict(model_names)


def foot_point(xyxy):
    """Bottom-center of each bounding box - the pixel point that corresponds
    to where a player is standing on the pitch (used for homography, not the
    box center, since the box center floats above the ground for a person)."""
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2, y2], axis=1)


def _make_browser_playable(mp4_path: Path):
    """cv2.VideoWriter's 'mp4v' fourcc writes MPEG-4 Part 2 video, which most
    browsers (Chrome/Edge included) refuse to play natively in an HTML5
    <video> tag - the file is valid and plays fine in VLC, but silently shows
    blank/nothing in st.video() or a plain <video> element, which is exactly
    the "the video doesn't show" symptom this fixes. If ffmpeg is on PATH,
    re-encode in place to H.264/yuv420p, the one combination every browser
    supports. If ffmpeg isn't installed, leave the original file as-is
    (still downloadable/playable outside the browser) rather than fail the
    whole run over a missing optional tool."""
    if not mp4_path.exists() or shutil.which("ffmpeg") is None:
        return
    tmp_path = mp4_path.with_suffix(".h264.mp4")
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4_path),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(tmp_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(mp4_path)
        else:
            print(f"Note: ffmpeg re-encode for browser playback failed ({proc.returncode}); "
                  f"keeping the original file, which may not play in a browser.")
            tmp_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Note: ffmpeg re-encode skipped ({e}); keeping the original file, "
              f"which may not play in a browser.")


def run(video_path, clip_id, weights, conf, annotate, max_frames):
    model = YOLO(weights)
    class_ids, class_name = resolve_classes(model.names)
    print(f"Using weights '{weights}' -> tracking classes: {list(class_name.values())}")

    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_writer = None
    if annotate:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = PROCESSED_DIR / "annotated" / f"{clip_id}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        result = model.predict(
            frame,
            classes=class_ids,
            conf=conf,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        if len(detections) > 0:
            feet = foot_point(detections.xyxy)
            for i in range(len(detections)):
                rows.append(
                    {
                        "clip_id": clip_id,
                        "frame": frame_idx,
                        "timestamp_s": frame_idx / fps,
                        "track_id": int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
                        "class": class_name.get(int(detections.class_id[i]), str(detections.class_id[i])),
                        "conf": float(detections.confidence[i]),
                        "x1": float(detections.xyxy[i, 0]),
                        "y1": float(detections.xyxy[i, 1]),
                        "x2": float(detections.xyxy[i, 2]),
                        "y2": float(detections.xyxy[i, 3]),
                        "foot_x": float(feet[i, 0]),
                        "foot_y": float(feet[i, 1]),
                    }
                )

        if annotate and out_writer is not None:
            labels = [
                f"#{tid} {class_name.get(int(cid), cid)}"
                for tid, cid in zip(
                    detections.tracker_id if detections.tracker_id is not None else [-1] * len(detections),
                    detections.class_id,
                )
            ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            out_writer.write(annotated)

        frame_idx += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()
        _make_browser_playable(PROCESSED_DIR / "annotated" / f"{clip_id}.mp4")

    df = pd.DataFrame(rows)
    out_dir = PROCESSED_DIR / "tracklets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} detection-frames from {frame_idx} frames -> {out_path}")
    if annotate:
        print(f"Annotated preview -> {PROCESSED_DIR / 'annotated' / f'{clip_id}.mp4'}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path, help="Path to a broadcast clip")
    ap.add_argument("--clip-id", required=True, help="Identifier used for output filenames")
    ap.add_argument("--weights", default="yolov8n.pt", help="Ultralytics weights (default: stock COCO yolov8n)")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--annotate", action="store_true", help="Also write an annotated preview video")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (useful for quick tests)")
    args = ap.parse_args()

    run(args.video, args.clip_id, args.weights, args.conf, args.annotate, args.max_frames)


if __name__ == "__main__":
    main()
