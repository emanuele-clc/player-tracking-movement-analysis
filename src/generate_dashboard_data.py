"""
Stage 8 (data prep): export every analysis output (tracking positions,
pitch geometry, heatmap grids, role clusters, space-creation scores) into a
single JSON file docs/index.html loads - the same self-contained,
zero-backend dashboard approach used in expected-goals-xg-model.

Usage:
    python src/generate_dashboard_data.py
"""
import json
from pathlib import Path

import pandas as pd

from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


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

    out_path = DOCS_DIR / "assets" / "dashboard_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    print(f"Wrote dashboard data ({len(json.dumps(data))} bytes) -> {out_path}")
    print(f"Clips: {clips}")


if __name__ == "__main__":
    main()
