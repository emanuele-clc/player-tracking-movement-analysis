"""
Stage 6b: per-player and per-team positional heatmaps from the tracking
dataset, rendered as static plots (plots/) and as pitch-normalized grid data
for the dashboard (data/processed/heatmap_grids.json) - no plotting library
needed client-side, just a grid of counts.

Usage:
    python src/generate_heatmaps.py --clip-id clip01
    python src/generate_heatmaps.py --clip-id clip01 --track-id 6
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def draw_pitch(ax, config):
    l, w = config.pitch_length_m, config.pitch_width_m
    ax.set_xlim(-2, l + 2)
    ax.set_ylim(-2, w + 2)
    ax.set_facecolor("#1e5c2e")
    ax.add_patch(plt.Rectangle((0, 0), l, w, fill=False, color="white", linewidth=1.5))
    for (i1, i2) in config.edges:
        p1 = np.array(config.vertices[i1 - 1]) / 100.0
        p2 = np.array(config.vertices[i2 - 1]) / 100.0
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="white", linewidth=1.2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def plot_heatmap(x, y, config, title, out_path, bins=30):
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax, config)
    if len(x) > 0:
        ax.hexbin(x, y, gridsize=bins, cmap="inferno", alpha=0.75,
                   extent=[0, config.pitch_length_m, 0, config.pitch_width_m], mincnt=1)
    ax.set_title(title, color="white")
    fig.patch.set_facecolor("#1e5c2e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def grid_counts(x, y, config, grid_w=34, grid_h=22):
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=[grid_w, grid_h],
        range=[[0, config.pitch_length_m], [0, config.pitch_width_m]],
    )
    return hist.tolist()


def run(clip_id, track_id, grid_w, grid_h):
    dataset_path = PROCESSED_DIR / "tracking_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(f"No tracking dataset at {dataset_path} - run build_tracking_dataset.py first.")

    df = pd.read_parquet(dataset_path)
    if clip_id:
        df = df[df["clip_id"] == clip_id]
    person_df = df[df["class"].isin(["person", "player", "goalkeeper"])].copy()
    if person_df.empty:
        raise SystemExit("No person/player rows found for the requested clip_id.")

    config = SoccerPitchConfiguration()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    grids = {}

    label = clip_id or "all_clips"
    plot_heatmap(person_df["x_m"], person_df["y_m"], config,
                 f"All players - {label}", PLOTS_DIR / f"heatmap_all_{label}.png")
    grids["all"] = grid_counts(person_df["x_m"].to_numpy(), person_df["y_m"].to_numpy(), config, grid_w, grid_h)
    print(f"Wrote plots/heatmap_all_{label}.png ({len(person_df)} points)")

    for team, group in person_df.groupby("team"):
        safe_team = str(team).replace("/", "_")
        plot_heatmap(group["x_m"], group["y_m"], config,
                     f"{team} - {label}", PLOTS_DIR / f"heatmap_{safe_team}_{label}.png")
        grids[f"team:{team}"] = grid_counts(group["x_m"].to_numpy(), group["y_m"].to_numpy(), config, grid_w, grid_h)
        print(f"Wrote plots/heatmap_{safe_team}_{label}.png ({len(group)} points)")

    if track_id is not None:
        tdf = person_df[person_df["track_id"] == track_id]
        if tdf.empty:
            print(f"Warning: no rows for track_id={track_id} in this selection.")
        else:
            plot_heatmap(tdf["x_m"], tdf["y_m"], config,
                         f"Track {track_id} - {label}", PLOTS_DIR / f"heatmap_track{track_id}_{label}.png")
            grids[f"track:{track_id}"] = grid_counts(tdf["x_m"].to_numpy(), tdf["y_m"].to_numpy(), config, grid_w, grid_h)
            print(f"Wrote plots/heatmap_track{track_id}_{label}.png ({len(tdf)} points)")

    grid_path = PROCESSED_DIR / "heatmap_grids.json"
    existing = json.loads(grid_path.read_text()) if grid_path.exists() else {}
    existing[label] = {"grid_w": grid_w, "grid_h": grid_h,
                        "pitch_length_m": config.pitch_length_m, "pitch_width_m": config.pitch_width_m,
                        "grids": grids}
    grid_path.write_text(json.dumps(existing))
    print(f"Wrote grid data for '{label}' -> {grid_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--grid-w", type=int, default=34)
    ap.add_argument("--grid-h", type=int, default=22)
    args = ap.parse_args()
    run(args.clip_id, args.track_id, args.grid_w, args.grid_h)


if __name__ == "__main__":
    main()
