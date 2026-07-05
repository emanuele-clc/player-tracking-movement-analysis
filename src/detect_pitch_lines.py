"""
Automatic pitch-line detection for calibration, as an alternative to manually
eyeballing pixel coordinates (unreliable by hand: +/-150-200px errors when
tested on a real clip).

Approach:
- Threshold the frame in HSV for white pixels (pitch lines are painted white
  on grass, so low saturation + high value separates them well from grass,
  jerseys aside - see caveats below).
- Isolate long thin structures with directional morphological opening
  (a wide-flat kernel keeps horizontal lines, a tall-thin kernel keeps
  vertical lines), which drops small blobs like players' white kit patches.
- HoughLinesP to find line segments, then cluster near-parallel segments by
  position (many overlapping segments per real line) into single lines.
- Report each detected line's position + extent so they can be turned into
  exact line-line intersection points (far more precise than reading pixel
  coordinates off a still frame by eye - validated on a real clip: 4
  independent px/m scale estimates from the detected goal-box/penalty-box
  edges agreed within 1.4% of each other, and the resulting homography's
  reprojection error on those same points was ~8cm mean / 12.6cm max).

Caveats (why this is a starting point, not a fully automatic tool yet):
- Only reliable when the pitch markings are actually the highest-contrast
  white features in the ROI - dense white text on advertising boards, white
  kit, or bright sky can pollute the mask and need a tighter ROI or extra
  filtering.
- Turning detected lines into named pitch keypoints (vertex_id in
  pitch_config.py) is still a manual step here: you look at the printed
  line list, decide which is the goal line vs box edge, etc. A fully
  automatic version would classify lines by their known pitch geometry
  relationships instead (e.g. Roboflow's football-field-detection model
  does this end-to-end).

Usage:
    python src/detect_pitch_lines.py --frame data/processed/frames/clip01_ref.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_lines(frame: np.ndarray, hough_threshold: int = 200, min_line_length: int = 150,
                  max_line_gap: int = 20, cluster_gap: int = 25):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))

    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    lines_mask = cv2.bitwise_or(horiz, vert)

    lines = cv2.HoughLinesP(lines_mask, 1, np.pi / 180, threshold=hough_threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return [], [], lines_mask

    horiz_extent, vert_extent = {}, {}
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(y2 - y1) < abs(x2 - x1) * 0.15:
            key = round((y1 + y2) / 2)
            horiz_extent.setdefault(key, []).extend([x1, x2])
        elif abs(x2 - x1) < abs(y2 - y1) * 0.15:
            key = round((x1 + x2) / 2)
            vert_extent.setdefault(key, []).extend([y1, y2])

    def cluster(extent_dict, gap):
        keys = sorted(extent_dict.keys())
        if not keys:
            return []
        clusters, cur = [], [keys[0]]
        for k in keys[1:]:
            (cur.append(k) if k - cur[-1] <= gap else (clusters.append(cur), cur := [k]))
        clusters.append(cur)
        out = []
        for c in clusters:
            vals = [v for k in c for v in extent_dict[k]]
            center = sum(c) / len(c)
            out.append({"position": center, "extent_min": min(vals), "extent_max": max(vals), "n_segments": len(c)})
        return out

    horiz_lines = cluster(horiz_extent, cluster_gap)
    vert_lines = cluster(vert_extent, cluster_gap)
    return horiz_lines, vert_lines, lines_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True, type=Path, help="Path to a still frame (PNG/JPG)")
    ap.add_argument("--save-mask", type=Path, default=None, help="Optional path to save the line mask for inspection")
    args = ap.parse_args()

    frame = cv2.imread(str(args.frame))
    if frame is None:
        raise SystemExit(f"Could not read {args.frame}")

    horiz_lines, vert_lines, mask = detect_lines(frame)

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}\n")
    print("Horizontal lines (y position, x extent, n segments):")
    for l in sorted(horiz_lines, key=lambda d: d["position"]):
        print(f"  y={l['position']:.0f}  x=({l['extent_min']},{l['extent_max']})  n={l['n_segments']}")
    print("\nVertical lines (x position, y extent, n segments):")
    for l in sorted(vert_lines, key=lambda d: d["position"]):
        print(f"  x={l['position']:.0f}  y=({l['extent_min']},{l['extent_max']})  n={l['n_segments']}")

    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_mask), mask)
        print(f"\nLine mask written -> {args.save_mask}")

    print(
        "\nNext step: match these detected lines to known pitch features (goal line, "
        "box edges) by eye, compute their intersections, and fill them into a "
        "correspondences JSON (see pitch_calibration.py --make-template)."
    )


if __name__ == "__main__":
    main()
