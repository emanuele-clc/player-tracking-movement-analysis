"""
Stage 8 (data prep + build): export every analysis output (tracking
positions, pitch geometry, heatmap grids, role clusters, space-creation
scores) into docs/assets/dashboard_data.json, then render the final
docs/index.html from docs/_index_template.html by injecting that JSON in
place of the /*__DASHBOARD_DATA_JSON__*/ marker - the same self-contained,
zero-backend dashboard approach used in expected-goals-xg-model, but with
the HTML template kept separate from the data so re-running the pipeline
never requires hand-editing the dashboard markup.

Usage:
    python src/generate_dashboard_data.py
"""
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


def main():
    config = SoccerPitchConfiguration()

    tracking = load_parquet_safe(PROCESSED_DIR / "tracking_dataset.parquet")
    roles = load_parquet_safe(PROCESSED_DIR / "role_clusters.parquet")
    scores = load_parquet_safe(PROCESSED_DIR / "space_creation_scores.parquet")

    heatmap_grids_path = PROCESSED_DIR / "heatmap_grids.json"
    heatmap_grids = json.loads(heatmap_grids_path.read_text()) if heatmap_grids_path.exists() else {}

    clips = sorted(tracking["clip_id"].unique().tolist()) if not tracking.empty else []

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

    data = {
        "pitch": {
            "length_m": config.pitch_length_m,
            "width_m": config.pitch_width_m,
            "vertices_m": [[v[0] / 100, v[1] / 100] for v in config.vertices],
            "edges": config.edges,
        },
        "clips": clips,
        "tracking_by_clip": frames_by_clip,
        "heatmap_grids": heatmap_grids,
        "role_clusters": roles.to_dict(orient="records") if not roles.empty else [],
        "space_creation_scores": scores.to_dict(orient="records") if not scores.empty else [],
    }

    data_json = json.dumps(data)

    out_path = DOCS_DIR / "assets" / "dashboard_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(data_json)
    print(f"Wrote dashboard data ({len(data_json)} bytes) -> {out_path}")
    print(f"Clips: {clips}")

    if TEMPLATE_PATH.exists():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        if PLACEHOLDER not in template:
            raise SystemExit(f"Placeholder {PLACEHOLDER!r} not found in {TEMPLATE_PATH}")
        html = template.replace(PLACEHOLDER, data_json)
        html_out_path = DOCS_DIR / "index.html"
        html_out_path.write_text(html, encoding="utf-8")
        print(f"Rendered dashboard ({len(html)} bytes) -> {html_out_path}")
    else:
        print(f"No template at {TEMPLATE_PATH} - skipped rendering docs/index.html")


if __name__ == "__main__":
    main()
