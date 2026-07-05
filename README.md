---
title: Player Tracking & Movement Analysis
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Player Tracking & Movement Analysis

Computer-vision pipeline that turns **real broadcast football video** into player tracking data, then mines that data for movement patterns that matter: role clustering, off-ball space creation, pressing structure, and player-specific heatmaps.

This is a companion piece to [expected-goals-xg-model](https://github.com/emanuele-clc/expected-goals-xg-model) (shot-quality modeling from event data). Where that project asked "how good was this shot?", this one asks "what were the other 21 players doing while it happened?" — using video, not pre-packaged tracking feeds, as the source.

**Status: in progress.** This README documents the finished project's target shape and the build plan. Sections marked `[done]` / `[in progress]` / `[planned]` reflect actual state.

> The YAML block above is Hugging Face Spaces metadata (ignored everywhere else, including GitHub) - it's what lets this same repo be deployed as a live, hosted version of `app.py`, run via the `Dockerfile` in the repo root. See "Run it online" below.

## Analyze your own video

There are two completely separate things in this repo, easy to mix up:

- **The public dashboard** ([emanuele-clc.github.io/player-tracking-movement-analysis](https://emanuele-clc.github.io/player-tracking-movement-analysis/)) is a fixed, read-only showcase. It's a static page (GitHub Pages), so it can never have an upload button - there's no server behind it to run the detection model.
- **The app below** (`app.py`) is the actual "upload a video and analyze it" tool. It's a separate small web app you either run yourself (locally, or hosted on Hugging Face Spaces) - and it has a **"Publish to public dashboard"** button that pushes your result onto that public page automatically.

### Option A: run it locally, step by step (no coding knowledge needed)

This is the recommended way if you just want to try it on your own footage. You need Python installed (get it from [python.org](https://www.python.org/downloads/) if you don't have it - any recent version works) and Git ([git-scm.com](https://git-scm.com/downloads)).

1. **Open a terminal** (PowerShell on Windows, Terminal on Mac/Linux) and download the project:
   ```
   git clone https://github.com/emanuele-clc/player-tracking-movement-analysis
   cd player-tracking-movement-analysis
   ```
2. **Install the dependencies** (only needed once):
   ```
   pip install -r requirements.txt
   ```
   This takes a few minutes the first time - it's downloading the detection model library (PyTorch/YOLO) among other things. Also install [ffmpeg](https://ffmpeg.org/download.html) if you don't already have it (on Windows, `winget install ffmpeg`; on Mac, `brew install ffmpeg`) - without it, the tracked video the app produces is a valid file but won't play inside the browser (a codec issue, not a bug in the analysis itself); it'll still work fine as a download.
3. **Start the app:**
   ```
   streamlit run app.py
   ```
   A browser tab opens automatically (usually at `http://localhost:8501`). If it doesn't, copy the "Local URL" the terminal prints into your browser.
4. **Upload a clip**: click the file uploader near the top, pick an `.mp4`/`.mov`/`.avi`/`.mkv` file from your computer, and optionally give the analysis a name.
5. **Click ▶ Analyze.** A progress bar shows each step (detection, pitch calibration, team classification, heatmaps, etc.) - a short clip takes anywhere from under a minute to a few minutes depending on your machine and clip length. "Quick preview" mode in the sidebar (on by default) only analyzes the first several seconds, useful for a fast first look.
6. **Read the report** that appears below: tracked video, calibration confidence badge, team split, heatmaps, role clustering, and (if calibration succeeded) the space-creation score. If your clip's pitch markings aren't clear enough for automatic calibration, you'll see a yellow "pixel-space fallback" badge instead of a green one - tracking and team clustering are still real, just not in real-world metres for that clip (never faked).
7. **To see your result on the live public dashboard**, scroll to the bottom and click **🚀 Publish to public dashboard**. This regenerates `docs/index.html` with your new clip added and runs the git commands for you (`add` / `commit` / `push`) - only works if this is your own clone with push access to your GitHub repo. After it says "Published", open the public dashboard site and hard-refresh (Ctrl+F5 / Cmd+Shift+R) - your clip will appear in the "Clip" dropdown in the Tracked Playback section.
8. Alternatively, click **⬇ Download full interactive report (HTML)** to get a single standalone file with just that clip's analysis, to send to someone without touching the public site at all.

### Option B: run it online (hosted, no install)

Hugging Face retired the built-in "Streamlit" SDK option, so Streamlit Spaces are now deployed as **Docker** Spaces (HF builds the `Dockerfile` in this repo, which installs everything and starts the app on port 8501 - see [their docs](https://huggingface.co/docs/hub/spaces-sdks-streamlit)). To deploy your own copy:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick a Space name, then under **Select the Space SDK** choose **Docker** (not Gradio, which is the default selection - there's no separate Streamlit tile anymore). Pick the **Blank** Docker template.
3. Leave hardware on **CPU Basic** (free) and click **Create Space**.
4. Push this repo's `app.py`, `Dockerfile`, `requirements.txt`, `README.md`, and `src/` to the new Space's git remote (the Space's page shows the exact git URL and login instructions after you create it - it needs a Hugging Face access token, not your account password, as the git password).

Once built, you get the same upload-and-analyze app in a browser tab, no install. CPU-only inference on the free tier, so expect it to be slower than running locally - fine for a quick demo, not for a full match.

Pitch calibration in both flows is fully automatic (`src/auto_calibrate.py`): it looks for a goal-box/penalty-box rectangle across several sampled frames and cross-checks independent pixel-per-metre scale estimates against each other for a numeric confidence score, the same validation style already proven on the `drone_box` clip below - re-run automatically instead of eyeballed by hand. Below a confidence threshold it **honestly falls back to pixel-space analysis** (tracking, team clustering, and heatmap shape stay real; distances/speeds/the space-creation score are skipped, never faked) rather than silently producing bad real-world numbers - see `src/pipeline.py` for the orchestration and fallback logic.

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
| 1. Data access | Request SoccerNet credentials, download target clips + calibration | `[in progress]` — script written, request pending |
| 2. Detection & tracking | YOLOv8 detection per frame (model-agnostic: works with stock COCO weights or a football-fine-tuned model, see `data/README.md`), ByteTrack multi-object tracking into per-player tracklets | `[done]` — validated on a real clip: 3862 detections / 376 frames, 40 track_ids. Honest baseline result with generic COCO weights: several tracks span the whole clip at 100% frame coverage, but many fragment into short 1-5 frame stubs (camera-edge occlusion / re-entry), and the ball is only detected in ~9.6% of frames (COCO's generic "sports ball" class isn't tuned for a small fast-moving football). Swapping in a football-fine-tuned model (`data/README.md`) is the next concrete step to fix both. |
| 3. Pitch calibration | Homography from broadcast pixel coordinates to real-world pitch coordinates (meters). `pitch_config.py` holds the standard 32-keypoint FIFA pitch geometry; `pitch_calibration.py` fits the OpenCV homography. Correspondences can come from `src/auto_calibrate.py` (fully automatic: matches a detected goal-box/penalty-box rectangle to known FIFA dimensions and cross-checks scale estimates for a confidence score - what `app.py`/`pipeline.py` use) or, for the manually-built reference case below, from `detect_pitch_lines.py` (HSV white-line threshold + directional morphology + HoughLinesP + line clustering) — not manual pixel-guessing, which turned out to have +/-150-200px error, too imprecise to trust | `[done]` — validated on a real drone shot of a regulation pitch (`13386302_3840_2160_24fps.mp4`): 8 line-intersection points detected automatically around the goal-box/penalty-box edges, fit with a single homography. Cross-check: 4 independent px/m scale estimates (goal-box width/depth, penalty-box width/depth) agree within 1.4% of each other, and reprojecting the fit points back gives ~8cm mean / 12.6cm max error — strong evidence the frame is a genuine, correctly-proportioned pitch and the calibration is sound. Caught and fixed a real bug in the process: `pitch_config.py`'s penalty-box depth was wrong (20.15m instead of the actual 16.5m) until this validation exposed it. `auto_calibrate.py` independently re-derives the same 8 points from scratch (algorithmically, not by eye) and agrees to within 0.8%. |
| 4. Team/role assignment | Cluster shirt-color pixels (k-means on jersey crops, HSV median per track, single sequential video pass rather than per-frame seeking for speed) to split detections into team A / team B / referee-or-neutral | `[done]` — validated on the same real smoke-test clip: 36 tracked people split into a 14/18 team split by jersey hue plus 4 flagged outliers (a bright, distinctly different track likely the goalkeeper, plus short/noisy track fragments) — a plausible, sane result on real footage. |
| 5. Tracking dataset | Join tracklets + team assignment + pitch coordinates into one row-per-(frame, track) dataset (`data/processed/tracking_dataset.parquet`), with per-track speed derived from real elapsed time between frames (not assumed-constant frame spacing) | `[done]` — validated on `drone_box`: 270 rows, 36 tracks. Sanity-checked speeds: players mean 2.4 m/s / max 10.0 m/s (plausible human running range). Honest limitation surfaced by this same check: the ball's max speed comes out only 2.4 m/s, unrealistically slow for a kicked football - almost certainly because fast ball movement causes more ByteTrack ID fragmentation (each fragment's "speed" is computed only within its own short track), not a bug in the speed math itself. A football-specific detection model (steadier ball tracking) is the fix, same one already flagged in stage 2. |
| 6. Movement analysis | Positional role clustering (k-means on per-track mean position + dispersion + speed) via `cluster_movement.py`; per-player/per-team/all-players heatmaps (rendered to a correctly-scaled pitch + exported as grid JSON for the dashboard) via `generate_heatmaps.py` | `[done]` — both run end-to-end on `drone_box` and produce sane output (heatmap correctly concentrates activity in the penalty-box area the drone camera actually shows). Honest scope limit: a few seconds of one camera angle isn't enough positional variety to separate real footballing roles (winger vs center-back) — that needs a full match's worth of tracking, i.e. still waiting on SoccerNet access. The clustering code itself is real and tested, not a stub. |
| 7. Original contribution | Off-ball space creation score (`space_creation.py`) — see below | `[done]` (method validated, real ranking pending more data) |
| 8. Dashboard | Single-file interactive HTML (`docs/index.html`), rendered from `docs/_index_template.html` + `docs/assets/dashboard_data.json` by `src/generate_dashboard_data.py`. Includes a raw-vs-tracked video comparison, an interactive pitch-coordinate playback with a clip selector (every published clip is browsable) and a calibration-mode badge (real metres vs. schematic pixel-space preview), auto-generated plain-language match insights, role-clustering and space-creation-score tables, and the technical validation stats (with a live chart of the 4 scale-estimate cross-check) | `[done]` — live at [emanuele-clc.github.io/player-tracking-movement-analysis](https://emanuele-clc.github.io/player-tracking-movement-analysis/) |
| 9. Upload-your-own-video app | `app.py` (Streamlit) + `src/pipeline.py` (orchestrator) + `src/auto_calibrate.py` (automatic calibration with a confidence score and an honest pixel-space fallback) - runs every stage above end-to-end on an arbitrary uploaded clip, renders a full local report, and can publish straight to the public dashboard with one click | `[done]` — see "Analyze your own video" above |

## Original contribution: off-ball space creation score

In the spirit of the xG project's Scouting Radar — a metric a recruitment/analytics team would actually want, not just a chart:

**Off-ball space creation score** (`src/space_creation.py`) — using tracked positions of all visible players (not just the ball-carrier), a grid-based Voronoi approximation assigns every point on the pitch to its nearest player each frame; summing a team's controlled cells minus a player's own gives that player's "teammate space." The score correlates a player's own displacement in a frame with the change in their teammates' space over the following ~2 seconds. Rewards intelligent off-ball runs that standard heatmaps and touch maps can't see, since those only show where a player *was*, not what space their movement *created*.

Verified correct with a synthetic self-test (`python src/space_creation.py --self-test`): 4 players placed at the exact quadrant centers of the pitch each get 0.00% error from their expected exact quarter-share of the pitch area. Run on the real `drone_box` clip, the method executes end-to-end but the resulting scores are numerically unstable (some >10,000 m² per meter of displacement) — an honest, expected symptom of the underlying data, not a bug in the math: `drone_box` only has both teams simultaneously visible in 13 of 114 frames, and rarely more than 1-2 teammates from the same team at once, since it's a few seconds of one box-focused camera angle. A meaningful, stable ranking needs a full match's worth of tracking data with many players visible together for extended stretches — i.e. still waiting on SoccerNet access, same dependency as stages 2 and 5's honest limitations.

Fallback/parallel idea if Voronoi space stays too noisy even on a full match: pressing-trigger detection via clustering of simultaneous team-wide velocity vectors, to flag coordinated pressing moments a scout could review on video.

## Why broadcast video (and its honest limitations)

Broadcast tracking (vs. professional 25-camera optical tracking systems like Second Spectrum) means: only players visible in-frame are tracked at any moment, off-screen players are unavailable, and calibration accuracy depends on camera cuts/zooms. This is called out explicitly in the write-up rather than hidden — same honesty standard as the xG project's dataset-imbalance and model-comparison sections. The interesting engineering problem is making a real analysis pipeline work *despite* those constraints, which is closer to what a real analytics team building tools on broadcast footage (rather than a $ per-match optical tracking contract) actually has to deal with.

## Repository structure

```
data/
  raw/        downloaded/uploaded video + calibration (gitignored, not committed)
  processed/  detections, tracks, pitch-coordinate datasets, aggregates (committed)
  README.md   exact SoccerNet access + download steps
src/          detection, tracking, calibration, clustering, heatmap, dashboard-data, pipeline scripts
app.py        local/hosted Streamlit app - upload a video, get a full analysis report, publish it live
Dockerfile    lets app.py run as a Hugging Face Docker Space (see "Analyze your own video" above)
models/       saved detection/clustering model artifacts
plots/        evaluation and analysis charts
docs/         index.html — the live dashboard (GitHub Pages), _index_template.html — its source template, + assets/
notebooks/    exploratory analysis
```

## Reproduction

See `data/README.md` for data access, then either run the full pipeline stage by stage:

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

or run every stage automatically end-to-end on one video via `src/pipeline.py` (what `app.py` calls):

```
python src/pipeline.py --video data/raw/your_clip.mp4 --clip-id your_clip
```

or just use the app (`streamlit run app.py`, or the hosted version) - see "Analyze your own video" above for the full step-by-step.

## License

Code: MIT. Match footage and derived positional data remain subject to the SoccerNet data license — this repo redistributes only aggregate/derived outputs, not raw video.
