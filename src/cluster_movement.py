"""
Stage 6a: positional role clustering on the assembled tracking dataset.

Approach:
- For each track (a tracked person across a clip), build a feature vector
  from their real-world positions: mean x/y, std x/y (how much ground they
  cover - a winger roams differently than a center-back), and mean speed.
- Cluster these per-track feature vectors (k-means) into role archetypes -
  independent of whatever nominal position a lineup sheet would give them,
  purely from where they actually stood and moved during the clip.

Honest scope limitation: role clustering is only as meaningful as the
positional variety in the data feeding it. A few seconds of one penalty-box
camera angle (this project's current real test clip) mostly shows players
already congregated in one part of the pitch for one phase of play - not
enough to separate "center-back" from "winger" in any real footballing
sense. This becomes a genuinely meaningful analysis once run on a full match
(SoccerNet, once access comes through) where players actually occupy
distinct zones over 90 minutes. The code is real and tested now; the
football insight it should eventually produce needs more/longer data.

Usage:
    python src/cluster_movement.py --clip-id clip01 --k 3
    python src/cluster_movement.py --k 3   # uses every clip in tracking_dataset.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_track_features(df):
    person_df = df[df["class"].isin(["person", "player", "goalkeeper"])].copy()
    feats = person_df.groupby(["clip_id", "track_id"]).agg(
        team=("team", "first"),
        n_frames=("frame", "count"),
        mean_x=("x_m", "mean"),
        mean_y=("y_m", "mean"),
        std_x=("x_m", "std"),
        std_y=("y_m", "std"),
        mean_speed=("speed_mps", "mean"),
    ).reset_index()
    feats[["std_x", "std_y", "mean_speed"]] = feats[["std_x", "std_y", "mean_speed"]].fillna(0.0)
    return feats


def cluster_roles(feats, k):
    feature_cols = ["mean_x", "mean_y", "std_x", "std_y", "mean_speed"]
    X = feats[feature_cols].to_numpy()
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X_norm)

    feats = feats.copy()
    feats["role_cluster"] = labels
    return feats


def run(clip_id, k, min_frames):
    dataset_path = PROCESSED_DIR / "tracking_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(f"No tracking dataset at {dataset_path} - run build_tracking_dataset.py first.")

    df = pd.read_parquet(dataset_path)
    if clip_id:
        df = df[df["clip_id"] == clip_id]
        if df.empty:
            raise SystemExit(f"No rows for clip_id={clip_id!r} in {dataset_path}")

    feats = build_track_features(df)
    feats = feats[feats["n_frames"] >= min_frames].reset_index(drop=True)
    if len(feats) < k:
        raise SystemExit(
            f"Only {len(feats)} tracks have >= {min_frames} frames, need >= {k} for {k}-means. "
            f"Lower --min-frames or --k, or use a longer clip."
        )

    result = cluster_roles(feats, k)

    out_path = PROCESSED_DIR / "role_clusters.parquet"
    result.to_parquet(out_path, index=False)

    print(result.sort_values(["role_cluster", "clip_id"]).to_string(index=False))
    print(f"\nWrote {len(result)} track role assignments -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", default=None, help="Restrict to one clip (default: all clips in the dataset)")
    ap.add_argument("--k", type=int, default=3, help="Number of role clusters (default 3)")
    ap.add_argument("--min-frames", type=int, default=5,
                     help="Drop tracks shorter than this many frames before clustering (default 5)")
    args = ap.parse_args()
    run(args.clip_id, args.k, args.min_frames)


if __name__ == "__main__":
    main()
