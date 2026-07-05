"""
Stage 7: original contribution - off-ball space creation score.

The idea: standard heatmaps/touch maps only show where a player *was*. This
metric instead asks what a player's movement *did* for their teammates -
specifically, whether moving into a new spot expanded the total space their
team controls, using each frame's Voronoi-style pitch tessellation (every
point on the pitch "belongs" to whichever tracked player is nearest to it).

Method:
1. Per frame, discretize the pitch into a grid and assign each cell to the
   nearest tracked player (a standard, fast approximation to a true Voronoi
   diagram - exact Voronoi needs bounded-polygon clipping at the pitch edges,
   which the grid approach sidesteps entirely while converging to the same
   answer as the grid gets finer).
2. Sum a player's own team's controlled cells, excluding the player's own
   cell, into "teammate space" for that frame.
3. For each player, correlate their own frame-to-frame displacement with the
   change in their teammates' space over the following window (default
   2 seconds) - a player whose movement reliably precedes their teammates'
   controlled area growing is creating space; a player who is a pure
   passenger isn't.

Honest limitation, found while testing this on the project's current real
clip (drone_box): the metric needs several teammates simultaneously visible
for enough frames to mean anything, and a single tight box-focused camera
angle mostly doesn't have that (in drone_box: only 13 of 114 frames have
both teams present at all, and only 8-16 frames have >=2 players from the
*same* team visible together). The grid-Voronoi math itself is verified
correct on a synthetic symmetric test (see test_space_creation.py-style
checks in this module's __main__ block), but a real, meaningful ranking of
players needs a full match's worth of tracking (SoccerNet), not a few
seconds of one camera angle.

Usage:
    python src/space_creation.py --clip-id clip01 --window-s 2.0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_config import SoccerPitchConfiguration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def voronoi_control_grid(positions: np.ndarray, grid_res: float, pitch_length: float, pitch_width: float):
    """positions: (N, 2) array of (x_m, y_m) for the N players visible in a
    frame. Returns an (N,) array of controlled area in m^2 per player, via
    nearest-player grid assignment."""
    n = positions.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([pitch_length * pitch_width])

    xs = np.arange(grid_res / 2, pitch_length, grid_res)
    ys = np.arange(grid_res / 2, pitch_width, grid_res)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (G, 2)

    # (G, N) distance matrix - fine for grid_res ~1-2m on a single pitch;
    # for many frames at once, call this per-frame rather than batching
    # everything into one huge matrix.
    d2 = ((grid_pts[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    cell_area = grid_res * grid_res
    areas = np.bincount(nearest, minlength=n).astype(float) * cell_area
    return areas


def per_frame_team_space(df: pd.DataFrame, config: SoccerPitchConfiguration, grid_res: float):
    """For every (clip_id, frame), compute each player's controlled area and
    their team's total controlled area excluding themselves ("teammate
    space"). Returns a long dataframe: clip_id, frame, track_id, team,
    own_area_m2, teammate_space_m2."""
    rows = []
    for (clip_id, frame), group in df.groupby(["clip_id", "frame"]):
        positions = group[["x_m", "y_m"]].to_numpy()
        areas = voronoi_control_grid(positions, grid_res, config.pitch_length_m, config.pitch_width_m)
        group = group.reset_index(drop=True)
        group["own_area_m2"] = areas
        for team, team_group in group.groupby("team"):
            teammate_total = team_group["own_area_m2"].sum()
            for _, row in team_group.iterrows():
                rows.append({
                    "clip_id": clip_id,
                    "frame": frame,
                    "track_id": row["track_id"],
                    "team": team,
                    "own_area_m2": row["own_area_m2"],
                    "teammate_space_m2": teammate_total - row["own_area_m2"],
                })
    return pd.DataFrame(rows)


def compute_space_creation_scores(space_df: pd.DataFrame, tracking_df: pd.DataFrame, window_s: float):
    """For each track, correlate its own displacement in a frame with the
    change in ITS TEAMMATES' total space over the following `window_s`
    seconds. Returns one row per track: n_valid_frames, score (mean
    teammate-space delta per meter of the player's own displacement)."""
    ts = tracking_df.set_index(["clip_id", "track_id", "frame"])["timestamp_s"]

    merged = space_df.merge(
        tracking_df[["clip_id", "track_id", "frame", "x_m", "y_m", "timestamp_s"]],
        on=["clip_id", "track_id", "frame"], how="left",
    )
    merged = merged.sort_values(["clip_id", "track_id", "frame"])
    merged["dx"] = merged.groupby(["clip_id", "track_id"])["x_m"].diff()
    merged["dy"] = merged.groupby(["clip_id", "track_id"])["y_m"].diff()
    merged["own_displacement_m"] = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2)

    results = []
    for (clip_id, track_id), g in merged.groupby(["clip_id", "track_id"]):
        g = g.sort_values("frame")
        valid_pairs = 0
        deltas_per_m = []
        for _, row in g.iterrows():
            if pd.isna(row["own_displacement_m"]) or row["own_displacement_m"] <= 0:
                continue
            t0 = row["timestamp_s"]
            future = g[(g["timestamp_s"] > t0) & (g["timestamp_s"] <= t0 + window_s)]
            if future.empty:
                continue
            space_delta = future["teammate_space_m2"].iloc[-1] - row["teammate_space_m2"]
            deltas_per_m.append(space_delta / row["own_displacement_m"])
            valid_pairs += 1
        if valid_pairs > 0:
            results.append({
                "clip_id": clip_id,
                "track_id": track_id,
                "n_valid_frames": valid_pairs,
                "space_creation_score": float(np.mean(deltas_per_m)),
            })
    return pd.DataFrame(results)


def run(clip_id, window_s, grid_res, min_valid_frames):
    dataset_path = PROCESSED_DIR / "tracking_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(f"No tracking dataset at {dataset_path} - run build_tracking_dataset.py first.")

    df = pd.read_parquet(dataset_path)
    if clip_id:
        df = df[df["clip_id"] == clip_id]
    df = df[df["class"].isin(["person", "player", "goalkeeper"])].copy()
    if df.empty:
        raise SystemExit("No person/player rows found for the requested clip_id.")

    config = SoccerPitchConfiguration()
    space_df = per_frame_team_space(df, config, grid_res)
    scores = compute_space_creation_scores(space_df, df, window_s)
    scores = scores[scores["n_valid_frames"] >= min_valid_frames].sort_values(
        "space_creation_score", ascending=False
    )

    out_path = PROCESSED_DIR / "space_creation_scores.parquet"
    scores.to_parquet(out_path, index=False)

    if scores.empty:
        print(f"No tracks had >= {min_valid_frames} valid frames for this clip - "
              f"expected on short/sparse clips, needs a longer clip with more simultaneous teammates.")
    else:
        print(scores.to_string(index=False))
    print(f"\nWrote {len(scores)} scored tracks -> {out_path}")
    return scores


def _self_test():
    """Sanity check the grid-Voronoi math on a symmetric synthetic case
    where the answer is known exactly: 4 players placed at the 4 quadrant
    centers of a rectangular pitch should each control ~1/4 of the area."""
    config = SoccerPitchConfiguration()
    l, w = config.pitch_length_m, config.pitch_width_m
    positions = np.array([
        [l * 0.25, w * 0.25], [l * 0.75, w * 0.25],
        [l * 0.25, w * 0.75], [l * 0.75, w * 0.75],
    ])
    areas = voronoi_control_grid(positions, grid_res=0.5, pitch_length=l, pitch_width=w)
    expected = l * w / 4
    max_err_pct = np.max(np.abs(areas - expected) / expected) * 100
    print(f"Self-test: 4 symmetric players, expected {expected:.1f} m^2 each, "
          f"got {np.round(areas, 1).tolist()} (max error {max_err_pct:.2f}%)")
    assert max_err_pct < 2, "grid-Voronoi areas deviate more than 2% from the symmetric expectation"
    print("Self-test passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--window-s", type=float, default=2.0,
                     help="Seconds after a player's movement to measure teammate space change over (default 2.0)")
    ap.add_argument("--grid-res", type=float, default=1.0, help="Grid cell size in meters (default 1.0)")
    ap.add_argument("--min-valid-frames", type=int, default=3,
                     help="Minimum scored frames for a track to be included in the ranking (default 3)")
    ap.add_argument("--self-test", action="store_true", help="Run the synthetic correctness check and exit")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    run(args.clip_id, args.window_s, args.grid_res, args.min_valid_frames)


if __name__ == "__main__":
    main()
