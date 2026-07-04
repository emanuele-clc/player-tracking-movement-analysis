"""
Stage 5: per-player and per-team positional heatmaps from the tracking
dataset, rendered both as static plots (plots/) and as data for the
interactive dashboard (data/processed/heatmap_grids.json).

Plan:
- 2D histogram / KDE of (x_m, y_m) per player, per team, and per game phase
  (in-possession / out-of-possession / transition, inferred from ball
  proximity and team-in-possession state).
- Export as pitch-normalized grid data the dashboard can render as a canvas
  heatmap without needing a plotting library client-side.

Not yet implemented — see README.md pipeline table for status.
"""


def main():
    raise NotImplementedError(
        "Heatmap generation not yet built. "
        "Planned: KDE/2D-histogram of tracked positions per player/team/phase, "
        "exported as pitch-grid JSON for plots/ and the dashboard."
    )


if __name__ == "__main__":
    main()
