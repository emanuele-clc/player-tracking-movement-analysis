"""
Stage 5: assemble the final structured tracking dataset for a clip, joining
the outputs of the three earlier stages:

- data/processed/pitch_coords/<clip_id>.parquet  (detect_and_track.py output
  + pitch_calibration.py's added x_m/y_m real-world coordinates)
- data/processed/teams/<clip_id>.parquet         (team_classification.py's
  per-track team assignment)

...into one row-per-(frame, track_id) dataset with a team label and a
derived per-frame speed, which is what every downstream analysis stage
(clustering, heatmaps, the space-creation metric) is built on.

Speed is computed per track from consecutive real-world positions (x_m,
y_m) divided by the real elapsed time between frames (timestamp_s), so it's
correct even across frames where a track was briefly not detected (no
assumption of constant frame spacing baked in).

Usage:
    python src/build_tracking_dataset.py --clip-id clip01
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def compute_speed(df):
    df = df.sort_values(["track_id", "frame"]).copy()
    dx = df.groupby("track_id")["x_m"].diff()
    dy = df.groupby("track_id")["y_m"].diff()
    dt = df.groupby("track_id")["timestamp_s"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dist / dt, np.nan)
    df["speed_mps"] = speed
    return df


def run(clip_id):
    pitch_coords_path = PROCESSED_DIR / "pitch_coords" / f"{clip_id}.parquet"
    if not pitch_coords_path.exists():
        raise SystemExit(f"No pitch coordinates at {pitch_coords_path} - run pitch_calibration.py first.")

    df = pd.read_parquet(pitch_coords_path)

    teams_path = PROCESSED_DIR / "teams" / f"{clip_id}.parquet"
    if teams_path.exists():
        teams = pd.read_parquet(teams_path)[["track_id", "team"]]
        df = df.merge(teams, on="track_id", how="left")
        n_missing = df["team"].isna().sum()
        df["team"] = df["team"].fillna("unknown")
        if n_missing:
            print(f"Note: {n_missing} rows have no team assignment (short/noisy tracks "
                  f"that team_classification.py couldn't sample a jersey color for) - labeled 'unknown'.")
    else:
        print(f"No team assignment found at {teams_path} - run team_classification.py first if you want teams. "
              f"Continuing without a team column.")
        df["team"] = "unknown"

    df = compute_speed(df)

    out_path = PROCESSED_DIR / "tracking_dataset.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing = existing[existing["clip_id"] != clip_id]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(out_path, index=False)

    n_tracks = df[df["clip_id"] == clip_id]["track_id"].nunique()
    print(f"Wrote {len(df)} total rows ({len(df[df['clip_id']==clip_id])} for clip '{clip_id}', "
          f"{n_tracks} tracks) -> {out_path}")

    speed_summary = df[df["clip_id"] == clip_id].groupby("class")["speed_mps"].agg(["count", "mean", "max"])
    print("\nSpeed sanity check by class (m/s):")
    print(speed_summary.to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", required=True)
    args = ap.parse_args()
    run(args.clip_id)


if __name__ == "__main__":
    main()
