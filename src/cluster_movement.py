"""
Stage 4: movement analysis on the assembled tracking dataset.

Plan:
- Positional role clustering: per-player average position + positional
  dispersion (std of x/y, time spent in pitch thirds) as a feature vector,
  clustered (k-means / GMM) into role archetypes independent of the nominal
  position label the broadcast/lineup gives them.
- Trajectory clustering: DTW or resampled-trajectory clustering on individual
  attacking sequences to find recurring off-ball movement patterns (e.g.
  overlapping runs, half-space occupation).
- Original contribution (see README): Voronoi-tessellation-based off-ball
  space creation score per player per possession phase.

Not yet implemented — see README.md pipeline table for status.
"""


def main():
    raise NotImplementedError(
        "Movement clustering not yet built. "
        "Planned: role clustering (k-means/GMM) + trajectory clustering + "
        "Voronoi-based space-creation metric on data/processed/tracking_dataset.parquet."
    )


if __name__ == "__main__":
    main()
