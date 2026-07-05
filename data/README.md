# Data access

This project needs two separate things: **match video** (SoccerNet, below) and **football-specific detection weights** (Roboflow, below) — generic COCO-pretrained YOLOv8 only knows "person" and "sports ball", which works but tracks poorly and barely sees the ball (measured: 9.6% of frames on a first smoke test). A model fine-tuned on actual football footage separates player/goalkeeper/referee/ball properly and is a large, free upgrade.

## Match video — SoccerNet

Real broadcast match video + camera calibration data from **SoccerNet** (https://www.soccer-net.org/), the open dataset used in the SoccerNet CVPR workshop challenges (tracking, calibration, action spotting).

### 1. Request access

1. Go to https://www.soccer-net.org/data and follow the "get the password" / NDA request form (a Google Form). This is free and typically returns a password by email within minutes to a day.
2. Install the downloader:
   ```
   pip install SoccerNet
   ```
3. Download target assets (example — tracking + calibration subsets):
   ```python
   from SoccerNet.Downloader import SoccerNetDownloader

   dl = SoccerNetDownloader(LocalDirectory="data/raw")
   dl.password = "<password from the form>"
   dl.downloadDataTask(task="tracking", split=["test"])       # broadcast video + tracklet annotations
   dl.downloadDataTask(task="calibration", split=["test"])    # camera calibration / field registration
   ```

### 2. What lands where

- `data/raw/` — downloaded video clips + calibration JSON (gitignored — not committed, large + license-restricted for redistribution)
- `data/processed/` — everything this project derives from the raw video: detections, tracks, pitch-coordinate datasets, clustering results, heatmap aggregates, dashboard JSON (committed — this is the actual portfolio artifact)

### 3. Licensing note

SoccerNet data is free for research use under its own terms (no commercial redistribution of the raw video). This repo never commits raw video or full-resolution frames — only derived numerical/positional data and rendered chart images, consistent with that license.

### Note on the current smoke-test clip

`data/raw/13386254_3840_2160_24fps.mp4` (a free stock clip used to validate the detection/tracking/team-clustering code end-to-end) is casual five-a-side football on a small enclosed artificial pitch — no regulation penalty box, centre circle, or standard pitch markings. It's fine for what it's been used for (proving the code runs correctly on real video), but it is **not** valid input for `pitch_calibration.py`: there are no standard keypoints in frame to calibrate a homography against, and using it would produce meaningless pitch coordinates. Real calibration needs either a SoccerNet clip (above) or another source that clearly shows a regulation pitch's markings.

## Football-specific detection weights — Roboflow

[Roboflow Universe](https://universe.roboflow.com/) hosts free, pre-trained YOLOv8 models (originally built from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) Kaggle dataset by the [roboflow/sports](https://github.com/roboflow/sports) project) that detect **ball / goalkeeper / player / referee** as four separate classes, instead of the generic "person"/"sports ball" a stock COCO model gives you.

1. Sign up for a free account at https://roboflow.com and grab an API key from your workspace settings (Settings → API Keys).
2. Install the SDK: `pip install roboflow` (already in requirements.txt).
3. Download the player-detection model (repeat for `football-ball-detection-rejhg` and, later, `football-field-detection-f07vi` for pitch-keypoint auto-calibration):
   ```python
   from roboflow import Roboflow

   rf = Roboflow(api_key="<your API key>")
   project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
   model = project.version(1).model
   model_path = project.version(1).download("yolov8", location="models/").location
   ```
4. Point `detect_and_track.py` at the downloaded weights:
   ```
   python src/detect_and_track.py --video data/raw/clip.mp4 --clip-id clip01 --weights models/football-players-detection.pt
   ```
   `detect_and_track.py` reads whichever classes the weights expose (see `resolve_classes()`), so this "just works" without touching the script.

### Licensing note

Ultralytics (the YOLOv8 library itself) is [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) — using it as a dependency (as this project does) is standard practice, but be aware of that license if you ever redistribute a modified copy of ultralytics itself, as opposed to just depending on the published package. The pitch-geometry/homography code in this repo (`src/pitch_config.py`, `src/pitch_calibration.py`) is an original implementation crediting the widely-used keypoint convention from roboflow/sports (MIT-licensed).
