"""
Fully automatic pitch calibration: detects a goal-box / penalty-box rectangle
in a reference frame using detect_pitch_lines.detect_lines(), matches it
against known FIFA box dimensions, and derives a homography with no manual
correspondence-picking step.

This turns the already-validated *manual* method used for the drone_box clip
(see data/processed/correspondences/drone_box.json - built by a person
looking at detected lines and deciding which was which) into an algorithm:
instead of a human matching detected lines to pitch features, this module
searches combinations of detected lines for a rectangle whose measured
width/depth ratio matches a known box (goal box, or 6-yard/18-yard penalty
box) to within tolerance. When both boxes are visible and nested (sharing
the goal line, as in drone_box), it cross-checks 4 independent px/m scale
estimates against each other - exactly the same cross-validation already
documented for drone_box, just computed automatically instead of by eye.
Agreement between those estimates becomes a numeric confidence score.

Honest limitations (by design, not hidden):
- Requires a camera framing close to fronto-parallel/elevated, so pitch
  lines render close to axis-aligned in the image (detect_lines()'s
  requirement). Oblique wide broadcast shots of a full pitch will not
  calibrate this way.
- Needs a goal-box or penalty-box edge actually visible in frame.
- When only one box is detected (not the nested pair), there is no way to
  tell which end of the pitch is in view, so it is always anchored at the
  "own" (pitch x=0) end. That's an arbitrary but harmless choice for
  distances/speeds/heatmap shapes; only the absolute left/right placement on
  a full-pitch backdrop could be mirrored end-to-end versus reality.
- Below a confidence threshold, calibration is reported as failed rather
  than silently producing bad meters - callers should fall back to
  pixel-space analysis in that case (see pipeline.py).

Usage:
    python src/auto_calibrate.py --frame data/processed/frames/drone_box_ref.png
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from detect_pitch_lines import detect_lines
from pitch_config import SoccerPitchConfiguration

# (pitch depth, pitch width) in meters for each candidate box type.
BOX_TYPES = {
    "goal_box": {"depth_m": 5.5, "width_m": 18.32},
    "penalty_box": {"depth_m": 16.5, "width_m": 40.32},
}


def _best_box_match(depth_px, width_px, tol):
    """Given a candidate rectangle's two pixel spans, find the best-matching
    known box type and the relative ratio error for that match."""
    if depth_px <= 1 or width_px <= 1:
        return None
    ratio = depth_px / width_px
    best = None
    for box_name, dims in BOX_TYPES.items():
        known_ratio = dims["depth_m"] / dims["width_m"]
        err = abs(ratio - known_ratio) / known_ratio
        if err <= tol and (best is None or err < best[1]):
            best = (box_name, err)
    return best


def _rect_candidates(horiz_lines, vert_lines, min_gap_px=40):
    """All (h_lo, h_hi, v_lo, v_hi) line quadruples whose spans overlap
    plausibly (each vertical line's y-extent should cover the horizontal
    pair, and vice versa) - i.e. a real drawn rectangle, not two unrelated
    lines that just happen to exist somewhere in the frame."""
    h_sorted = sorted(horiz_lines, key=lambda l: l["position"])
    v_sorted = sorted(vert_lines, key=lambda l: l["position"])
    out = []
    for h1, h2 in itertools.combinations(h_sorted, 2):
        if abs(h2["position"] - h1["position"]) < min_gap_px:
            continue
        lo_h, hi_h = sorted([h1, h2], key=lambda l: l["position"])
        for v1, v2 in itertools.combinations(v_sorted, 2):
            if abs(v2["position"] - v1["position"]) < min_gap_px:
                continue
            lo_v, hi_v = sorted([v1, v2], key=lambda l: l["position"])
            y_lo, y_hi = lo_h["position"], hi_h["position"]
            x_lo, x_hi = lo_v["position"], hi_v["position"]
            v_ok = all(
                v["extent_min"] <= y_lo + min_gap_px * 0.5 and v["extent_max"] >= y_hi - min_gap_px * 0.5
                for v in (lo_v, hi_v)
            )
            h_ok = all(
                h["extent_min"] <= x_lo + min_gap_px * 0.5 and h["extent_max"] >= x_hi - min_gap_px * 0.5
                for h in (lo_h, hi_h)
            )
            if v_ok and h_ok:
                out.append((lo_h, hi_h, lo_v, hi_v))
    return out


def _single_box_search(horiz_lines, vert_lines, tol=0.15):
    """Best single-rectangle match across both possible axis orientations
    (pitch-depth-along-image-rows, or pitch-depth-along-image-columns)."""
    best = None
    for lo_h, hi_h, lo_v, hi_v in _rect_candidates(horiz_lines, vert_lines):
        h_span = hi_h["position"] - lo_h["position"]
        v_span = hi_v["position"] - lo_v["position"]

        for depth_px, width_px, orientation in (
            (h_span, v_span, "depth_is_rows"),
            (v_span, h_span, "depth_is_cols"),
        ):
            match = _best_box_match(depth_px, width_px, tol)
            if match is None:
                continue
            box_name, err = match
            candidate = {
                "box_name": box_name,
                "ratio_error": err,
                "orientation": orientation,
                "lo_h": lo_h, "hi_h": hi_h, "lo_v": lo_v, "hi_v": hi_v,
                "depth_px": depth_px, "width_px": width_px,
            }
            if best is None or err < best["ratio_error"]:
                best = candidate
    return best


def _corners_from_rect(lo_h, hi_h, lo_v, hi_v, box_name, orientation, config):
    """Pixel <-> pitch (meters) correspondences for the 4 corners of one
    detected box, anchored at the pitch-x=0 ("own") end."""
    dims = BOX_TYPES[box_name]
    depth_m, width_m = dims["depth_m"], dims["width_m"]
    W = config.pitch_width_m
    y0, y1 = (W - width_m) / 2, (W + width_m) / 2

    x_lo, x_hi = lo_v["position"], hi_v["position"]
    y_lo, y_hi = lo_h["position"], hi_h["position"]

    pixel_pts, pitch_pts = [], []
    if orientation == "depth_is_rows":
        for (px, py, mx, my) in (
            (x_lo, y_lo, 0.0, y0), (x_hi, y_lo, 0.0, y1),
            (x_lo, y_hi, depth_m, y0), (x_hi, y_hi, depth_m, y1),
        ):
            pixel_pts.append([px, py])
            pitch_pts.append([mx, my])
    else:
        for (px, py, mx, my) in (
            (x_lo, y_lo, 0.0, y0), (x_lo, y_hi, 0.0, y1),
            (x_hi, y_lo, depth_m, y0), (x_hi, y_hi, depth_m, y1),
        ):
            pixel_pts.append([px, py])
            pitch_pts.append([mx, my])
    return pixel_pts, pitch_pts


def _nested_box_search_oriented(h_lines, v_lines, tol):
    """Core nested-box search for ONE fixed orientation (h_lines carry pitch
    depth, v_lines carry pitch width). No recursion, no orientation-swapping
    here - the public _nested_box_search() calls this twice with roles
    swapped and picks the better result."""
    h_sorted = sorted(h_lines, key=lambda l: l["position"])
    v_sorted = sorted(v_lines, key=lambda l: l["position"])
    if len(h_sorted) < 3 or len(v_sorted) < 4:
        return None

    best = None
    for h0, h1, h2 in itertools.permutations(h_sorted, 3):
        d1 = h1["position"] - h0["position"]
        d2 = h2["position"] - h0["position"]
        if d1 <= 0 or d2 <= d1:
            continue

        for v_pb_lo, v_gb_lo, v_gb_hi, v_pb_hi in itertools.permutations(v_sorted, 4):
            if not (v_pb_lo["position"] < v_gb_lo["position"] < v_gb_hi["position"] < v_pb_hi["position"]):
                continue
            gb_width_px = v_gb_hi["position"] - v_gb_lo["position"]
            pb_width_px = v_pb_hi["position"] - v_pb_lo["position"]
            centre_gb = (v_gb_lo["position"] + v_gb_hi["position"]) / 2
            centre_pb = (v_pb_lo["position"] + v_pb_hi["position"]) / 2
            if pb_width_px <= 0 or gb_width_px <= 0:
                continue
            if abs(centre_gb - centre_pb) / pb_width_px > 0.08:
                continue

            gb = BOX_TYPES["goal_box"]
            pb = BOX_TYPES["penalty_box"]
            scale_gb_w = gb["width_m"] / gb_width_px
            scale_gb_d = gb["depth_m"] / d1
            scale_pb_w = pb["width_m"] / pb_width_px
            scale_pb_d = pb["depth_m"] / d2
            scales = [scale_gb_w, scale_gb_d, scale_pb_w, scale_pb_d]
            mean_scale = sum(scales) / len(scales)
            max_disagreement = max(abs(s - mean_scale) / mean_scale for s in scales)
            if max_disagreement > tol:
                continue

            candidate = {
                "h0": h0, "h1": h1, "h2": h2,
                "v_gb_lo": v_gb_lo, "v_gb_hi": v_gb_hi,
                "v_pb_lo": v_pb_lo, "v_pb_hi": v_pb_hi,
                "max_disagreement": max_disagreement,
                "scales": scales,
            }
            if best is None or max_disagreement < best["max_disagreement"]:
                best = candidate
    return best


def _nested_box_search(horiz_lines, vert_lines, tol=0.15):
    """Best case: goal box nested inside penalty box, sharing the goal
    line. Gives 4 independent px/m scale estimates to cross-check against
    each other, same validation style already used for drone_box. Tries
    both orientations (pitch depth along image rows, or along image
    columns) and returns whichever fits better."""
    rows_result = _nested_box_search_oriented(horiz_lines, vert_lines, tol)
    if rows_result is not None:
        rows_result = dict(rows_result)
        rows_result["orientation"] = "depth_is_rows"

    cols_raw = _nested_box_search_oriented(vert_lines, horiz_lines, tol)
    cols_result = None
    if cols_raw is not None:
        cols_result = dict(cols_raw)
        cols_result["orientation"] = "depth_is_cols"
        cols_result["v0"], cols_result["v1"], cols_result["v2"] = (
            cols_result.pop("h0"), cols_result.pop("h1"), cols_result.pop("h2")
        )
        cols_result["h_gb_lo"], cols_result["h_gb_hi"] = cols_result.pop("v_gb_lo"), cols_result.pop("v_gb_hi")
        cols_result["h_pb_lo"], cols_result["h_pb_hi"] = cols_result.pop("v_pb_lo"), cols_result.pop("v_pb_hi")

    candidates = [c for c in (rows_result, cols_result) if c is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["max_disagreement"])


def _corners_from_nested(candidate, config):
    W = config.pitch_width_m
    gb, pb = BOX_TYPES["goal_box"], BOX_TYPES["penalty_box"]
    gy0, gy1 = (W - gb["width_m"]) / 2, (W + gb["width_m"]) / 2
    py0, py1 = (W - pb["width_m"]) / 2, (W + pb["width_m"]) / 2

    pixel_pts, pitch_pts = [], []
    if candidate["orientation"] == "depth_is_rows":
        h0, h1, h2 = candidate["h0"], candidate["h1"], candidate["h2"]
        vgl, vgh = candidate["v_gb_lo"], candidate["v_gb_hi"]
        vpl, vph = candidate["v_pb_lo"], candidate["v_pb_hi"]
        cols = [(vgl["position"], gy0), (vgh["position"], gy1), (vpl["position"], py0), (vph["position"], py1)]
        for px, y_m in cols:
            pixel_pts.append([px, h0["position"]]); pitch_pts.append([0.0, y_m])
        for px, y_m in ((vgl["position"], gy0), (vgh["position"], gy1)):
            pixel_pts.append([px, h1["position"]]); pitch_pts.append([gb["depth_m"], y_m])
        for px, y_m in ((vpl["position"], py0), (vph["position"], py1)):
            pixel_pts.append([px, h2["position"]]); pitch_pts.append([pb["depth_m"], y_m])
    else:
        v0, v1, v2 = candidate["v0"], candidate["v1"], candidate["v2"]
        hgl, hgh = candidate["h_gb_lo"], candidate["h_gb_hi"]
        hpl, hph = candidate["h_pb_lo"], candidate["h_pb_hi"]
        for py, y_m in ((hgl["position"], gy0), (hgh["position"], gy1), (hpl["position"], py0), (hph["position"], py1)):
            pixel_pts.append([v0["position"], py]); pitch_pts.append([0.0, y_m])
        for py, y_m in ((hgl["position"], gy0), (hgh["position"], gy1)):
            pixel_pts.append([v1["position"], py]); pitch_pts.append([gb["depth_m"], y_m])
        for py, y_m in ((hpl["position"], py0), (hph["position"], py1)):
            pixel_pts.append([v2["position"], py]); pitch_pts.append([pb["depth_m"], y_m])
    return pixel_pts, pitch_pts


def auto_calibrate(frame: np.ndarray, config: SoccerPitchConfiguration = None, tol: float = 0.15):
    """Try nested (goal box + penalty box) detection first (highest
    confidence, 4 cross-checked scale estimates); fall back to a single-box
    match (2 estimates, lower confidence ceiling); report failure otherwise.

    Returns a dict: status ("nested" | "single" | "failed"), confidence
    (0-1), pixel_pts / pitch_pts_m (correspondences, if any), and a
    human-readable `notes` string.
    """
    config = config or SoccerPitchConfiguration()
    horiz_lines, vert_lines, _mask = detect_lines(frame)

    if len(horiz_lines) < 2 or len(vert_lines) < 2:
        return {"status": "failed", "confidence": 0.0,
                "notes": f"Only {len(horiz_lines)} horizontal / {len(vert_lines)} vertical lines detected - not enough to attempt a match."}

    nested = _nested_box_search(horiz_lines, vert_lines, tol=tol)
    if nested is not None:
        pixel_pts, pitch_pts = _corners_from_nested(nested, config)
        confidence = max(0.0, 1.0 - nested["max_disagreement"] / tol) * 0.5 + 0.5
        return {
            "status": "nested",
            "confidence": round(confidence, 3),
            "pixel_pts": pixel_pts,
            "pitch_pts_m": pitch_pts,
            "max_scale_disagreement_pct": round(nested["max_disagreement"] * 100, 2),
            "scales_px_per_m": [round(1 / s, 2) for s in nested["scales"]],
            "notes": (f"Nested goal-box + penalty-box match found. 4 independent px/m scale "
                      f"estimates agree within {nested['max_disagreement']*100:.1f}%."),
        }

    single = _single_box_search(horiz_lines, vert_lines, tol=tol)
    if single is not None:
        pixel_pts, pitch_pts = _corners_from_rect(
            single["lo_h"], single["hi_h"], single["lo_v"], single["hi_v"],
            single["box_name"], single["orientation"], config,
        )
        confidence = max(0.0, 1.0 - single["ratio_error"] / tol) * 0.35 + 0.15
        return {
            "status": "single",
            "confidence": round(confidence, 3),
            "pixel_pts": pixel_pts,
            "pitch_pts_m": pitch_pts,
            "box_name": single["box_name"],
            "ratio_error_pct": round(single["ratio_error"] * 100, 2),
            "notes": (f"Single {single['box_name'].replace('_', ' ')} match found (ratio error "
                      f"{single['ratio_error']*100:.1f}%). Only one box visible, so which end of "
                      f"the pitch this is cannot be confirmed - distances/speeds/heatmap shapes "
                      f"are still valid, absolute pitch position is a best guess."),
        }

    return {"status": "failed", "confidence": 0.0,
            "notes": "No goal-box or penalty-box rectangle matched known FIFA dimensions in this frame."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    frame = cv2.imread(str(args.frame))
    if frame is None:
        raise SystemExit(f"Could not read {args.frame}")

    result = auto_calibrate(frame, tol=args.tol)
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']}")
    print(result["notes"])
    if "pixel_pts" in result:
        print(f"\n{len(result['pixel_pts'])} correspondences:")
        for px, pm in zip(result["pixel_pts"], result["pitch_pts_m"]):
            print(f"  pixel {px} -> pitch(m) {[round(v, 2) for v in pm]}")


if __name__ == "__main__":
    main()
