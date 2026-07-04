"""
Stage 3: assemble the final structured tracking dataset from per-clip
tracklet + pitch-coordinate outputs.

Plan:
- Join tracklets (data/processed/tracklets/) with pitch coordinates
  (data/processed/pitch_coords/) per clip.
- Assign team labels via jersey-color k-means clustering on detection crops
  (team A / team B / referee / goalkeeper).
- Derive per-frame speed and smoothed trajectories.
- Output: data/processed/tracking_dataset.parquet, one row per
  (clip_id, frame, track_id): timestamp, team, x_m, y_m, speed_mps

This is the dataset all downstream analysis (clustering, heatmaps, the
space-creation metric) is built on.

Not yet implemented — see README.md pipeline table for status.
"""


def main():
    raise NotImplementedError(
        "Tracking dataset assembly not yet built. "
        "Planned: join tracklets + pitch coordinates, assign teams via "
        "jersey-color clustering, derive speed/trajectories."
    )


if __name__ == "__main__":
    main()
