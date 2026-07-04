# Player Tracking & Movement Analysis

Computer-vision pipeline that turns **real broadcast football video** into player tracking data, then mines that data for movement patterns that matter: role clustering, off-ball space creation, pressing structure, and player-specific heatmaps.

This is a companion piece to [expected-goals-xg-model](https://github.com/emanuele-clc/expected-goals-xg-model) (shot-quality modeling from event data). Where that project asked "how good was this shot?", this one asks "what were the other 21 players doing while it happened?" — using video, not pre-packaged tracking feeds, as the source.

**Status: in progress.** This README documents the finished project's target shape and the build plan. Sections marked `[done]` / `[in progress]` / `[planned]` reflect actual state.

## Why video, not a tracking feed

Most public "tracking data" football projects start from a CSV of already-extracted (x, y) positions (e.g. Metrica Sports' sample data). That's a data-analysis exercise. This project starts one step earlier — from raw match footage — and builds the extraction layer itself:

`broadcast video → player detection → multi-object tracking → pitch calibration (homography) → real-world coordinates → analysis`

That's the difference between "I analyzed a tracking dataset" and "I built the system that produces one."

## Data source

**[SoccerNet](https://www.soccer-net.org/)** — the standard open dataset in football computer vision research (used in CVPR workshop challenges). It provides real broadcast footage from professional matches together with camera calibration data, which is what makes turning raw video into pitch coordinates possible without building a homography model from zero.

- Access: free, gated behind a short request form (NDA for the video assets) — see `data/README.md` for the request link and what to do once the password arrives.
- What's used: a small set of full match clips (broadcast angle) plus the calibration/field-registration data for those same clips.
- Raw video is **not** committed to this repo (see `.gitignore`) — it's large and its redistribution isn't part of the SoccerNet license. Only derived data (detections, tracks, coordinates, aggregates, plots, dashboard JSON) is checked in. `data/README.md` has the exact steps to re-download and regenerate everything from scratch.

## Pipeline

| Stage | What it does | Status |
|---|---|---|
| 1. Data access | Request SoccerNet credentials, download target clips + calibration | `[planned]` |
| 2. Detection & tracking | YOLOv8 player/ball detection per frame, multi-object tracking (ByteTrack) to get consistent per-player tracklets across a clip | `[planned]` |
| 3. Team/role assignment | Cluster shirt-color pixels (k-means on jersey crops) to split detections into team A / team B / referee / goalkeeper | `[planned]` |
| 4. Pitch calibration | Homography from broadcast pixel coordinates to real-world pitch coordinates (105x68m), using SoccerNet's field calibration data | `[planned]` |
| 5. Tracking dataset | Aggregate per-frame, per-player (x, y) into a structured dataset (parquet/csv): player_id, team, frame, timestamp, x, y, speed | `[planned]` |
| 6. Movement analysis | Positional role clustering (k-means/GMM on average-position + dispersion vectors), trajectory clustering, per-player and per-team heatmaps | `[planned]` |
| 7. Original contribution | See below | `[planned]` |
| 8. Dashboard | Single-file interactive HTML (`docs/index.html`), same self-contained style as the xG project | `[planned]` |

## Planned original contribution

Not committed to the exact framing yet, but the target is a metric in the spirit of the xG project's Scouting Radar — something a recruitment/analytics team would actually want, not just a chart:

**Off-ball space creation score** — using tracked positions of all 22 players (not just the ball-carrier), compute Voronoi tessellation of the pitch at each frame and measure how a player's movement expands teammates' controlled space over the following 1-3 seconds. Rewards intelligent off-ball runs that standard heatmaps and touch maps can't see, since those only show where a player *was*, not what space their movement *created*.

Fallback/parallel idea if Voronoi proves too noisy on broadcast-derived (single-camera, partially-occluded) tracking: pressing-trigger detection via clustering of simultaneous team-wide velocity vectors, to flag coordinated pressing moments a scout could review on video.

## Why broadcast video (and its honest limitations)

Broadcast tracking (vs. professional 25-camera optical tracking systems like Second Spectrum) means: only players visible in-frame are tracked at any moment, off-screen players are unavailable, and calibration accuracy depends on camera cuts/zooms. This is called out explicitly in the write-up rather than hidden — same honesty standard as the xG project's dataset-imbalance and model-comparison sections. The interesting engineering problem is making a real analysis pipeline work *despite* those constraints, which is closer to what a real analytics team building tools on broadcast footage (rather than a $ per-match optical tracking contract) actually has to deal with.

## Repository structure

```
data/
  raw/        downloaded video + calibration (gitignored, not committed)
  processed/  detections, tracks, pitch-coordinate datasets, aggregates (committed)
  README.md   exact SoccerNet access + download steps
src/          detection, tracking, calibration, clustering, heatmap, dashboard-data scripts
models/       saved detection/clustering model artifacts
plots/        evaluation and analysis charts
docs/         index.html — the live dashboard (GitHub Pages) + assets/
notebooks/    exploratory analysis
```

## Reproduction

See `data/README.md` for data access, then (once implemented):

```
pip install -r requirements.txt
python src/download_data.py
python src/detect_and_track.py
python src/pitch_calibration.py
python src/build_tracking_dataset.py
python src/cluster_movement.py
python src/generate_heatmaps.py
python src/generate_dashboard_data.py
```

## License

Code: MIT. Match footage and derived positional data remain subject to the SoccerNet data license — this repo redistributes only aggregate/derived outputs, not raw video.
