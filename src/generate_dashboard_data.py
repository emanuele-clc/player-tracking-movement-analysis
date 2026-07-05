"""
Stage 8 (data prep + build): export every analysis output (tracking
positions, pitch geometry, heatmap grids, role clusters, space-creation
scores) into docs/assets/dashboard_data.json, then render the final
docs/index.html from docs/_index_template.html by injecting that JSON in
place of the /*__DASHBOARD_DATA_JSON__*/ marker - the same self-contained,
zero-backend dashboard approach used in expected-goals-xg-model, but with
the HTML template kept separate from the data so re-running the pipeline
never requires hand-editing the dashboard markup.

The data-building logic is also exposed as build_dashboard_data(), so
src/pipeline.py / app.py can reuse it to render a one-off per-video report
for a single uploaded clip without touching the public docs/ site.

Usage:
    python src/generate_dashboard_data.py
    python src/generate_dashboard_data.py --clip-id my_upload
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
TEMPLATE_PATH = DOCS_DIR / "_index_template.html"
PLACEHOLDER = "/*__DASHBOARD_DATA_JSON__*/"


def load_parquet_safe(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def build_dashboard_data(clip_filter=None):
    """Assemble the same JSON payload the dashboard consumes, optionally
    restricted to a subset of clip_ids (clip_filter). Returns a plain dict -
    callers decide whether/where to write it."""
    config = SoccerPitchConfiguration()

    tracking = load_parquet_safe(PROCESSED_DIR / "tracking_dataset.parquet")
    roles = load_parquet_safe(PROCESSED_DIR / "role_clusters.parquet")
    scores = load_parquet_safe(PROCESSED_DIR / "space_creation_scores.parquet")

    heatmap_grids_path = PROCESSED_DIR / "heatmap_grids.json"
    heatmap_grids = json.loads(heatmap_grids_path.read_text()) if heatmap_grids_path.exists() else {}

    calib_mode_path = PROCESSED_DIR / "calibration_mode.json"
    clip_meta = json.loads(calib_mode_path.read_text()) if calib_mode_path.exists() else {}

    if clip_filter is not None:
        clip_filter = set(clip_filter)
        if not tracking.empty:
            tracking = tracking[tracking["clip_id"].isin(clip_filter)]
        if not roles.empty:
            roles = roles[roles["clip_id"].isin(clip_filter)]
        if not scores.empty:
            scores = scores[scores["clip_id"].isin(clip_filter)]
        heatmap_grids = {k: v for k, v in heatmap_grids.items() if k in clip_filter}
        clip_meta = {k: v for k, v in clip_meta.items() if k in clip_filter}

    clips = sorted(tracking["clip_id"].unique().tolist()) if not tracking.empty else []
    # Any clip without an explicit entry (e.g. older runs from before this field
    # existed) is assumed metric - that was the only mode this project produced
    # before the automatic-calibration/honest-fallback feature was added.
    for clip_id in clips:
        clip_meta.setdefault(clip_id, {"mode": "metric", "confidence": None, "notes": ""})

    frames_by_clip = {}
    for clip_id in clips:
        clip_df = tracking[tracking["clip_id"] == clip_id]
        frames = {}
        for frame, group in clip_df.groupby("frame"):
            frames[int(frame)] = [
                {
                    "track_id": int(r["track_id"]),
                    "class": r["class"],
                    "team": r["team"],
                    "x_m": round(float(r["x_m"]), 2),
                    "y_m": round(float(r["y_m"]), 2),
                    "speed_mps": None if pd.isna(r["speed_mps"]) else round(float(r["speed_mps"]), 2),
                }
                for _, r in group.iterrows()
            ]
        frames_by_clip[clip_id] = {
            "n_frames": int(clip_df["frame"].nunique()),
            "n_tracks": int(clip_df["track_id"].nunique()),
            "frames": frames,
        }

    return {
        "pitch": {
            "length_m": config.pitch_length_m,
            "width_m": config.pitch_width_m,
            "vertices_m": [[v[0] / 100, v[1] / 100] for v in config.vertices],
            "edges": config.edges,
        },
        "clips": clips,
        "clip_meta": clip_meta,
        "tracking_by_clip": frames_by_clip,
        "heatmap_grids": heatmap_grids,
        "role_clusters": roles.to_dict(orient="records") if not roles.empty else [],
        "space_creation_scores": scores.to_dict(orient="records") if not scores.empty else [],
    }


def render_html(data: dict, template_path: Path = TEMPLATE_PATH) -> str:
    """Inject a data dict into the dashboard HTML template, returning the
    rendered HTML as a string (caller decides where to write it)."""
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"Placeholder {PLACEHOLDER!r} not found in {template_path}")
    return template.replace(PLACEHOLDER, json.dumps(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", default=None, help="Restrict to one clip (default: every clip in the dataset)")
    args = ap.parse_args()

    clip_filter = [args.clip_id] if args.clip_id else None
    data = build_dashboard_data(clip_filter=clip_filter)
    data_json = json.dumps(data)

    out_path = DOCS_DIR / "assets" / "dashboard_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(data_json)
    print(f"Wrote dashboard data ({len(data_json)} bytes) -> {out_path}")
    print(f"Clips: {data['clips']}")

    if TEMPLATE_PATH.exists():
        html = render_html(data)
        html_out_path = DOCS_DIR / "index.html"
        html_out_path.write_text(html, encoding="utf-8")
        print(f"Rendered dashboard ({len(html)} bytes) -> {html_out_path}")
    else:
        print(f"No template at {TEMPLATE_PATH} - skipped rendering docs/index.html")


if __name__ == "__main__":
    main()
