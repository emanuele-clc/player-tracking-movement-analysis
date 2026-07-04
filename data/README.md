# Data access — SoccerNet

This project uses real broadcast match video + camera calibration data from **SoccerNet** (https://www.soccer-net.org/), the open dataset used in the SoccerNet CVPR workshop challenges (tracking, calibration, action spotting).

## 1. Request access

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

## 2. What lands where

- `data/raw/` — downloaded video clips + calibration JSON (gitignored — not committed, large + license-restricted for redistribution)
- `data/processed/` — everything this project derives from the raw video: detections, tracks, pitch-coordinate datasets, clustering results, heatmap aggregates, dashboard JSON (committed — this is the actual portfolio artifact)

## 3. Regenerating processed data from scratch

```
python src/download_data.py        # wraps the steps above given a password in $SOCCERNET_PASSWORD
python src/detect_and_track.py
python src/pitch_calibration.py
python src/build_tracking_dataset.py
```

## Licensing note

SoccerNet data is free for research use under its own terms (no commercial redistribution of the raw video). This repo never commits raw video or full-resolution frames — only derived numerical/positional data and rendered chart images, consistent with that license.
