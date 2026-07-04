"""
Download SoccerNet broadcast video + calibration data used as the raw input
for this project's detection/tracking/calibration pipeline.

Requires a SoccerNet password (see data/README.md for how to request one),
passed via the SOCCERNET_PASSWORD environment variable.

Usage:
    SOCCERNET_PASSWORD=xxxx python src/download_data.py
"""
import os
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise SystemExit(
            "Set SOCCERNET_PASSWORD (see data/README.md for how to request access)."
        )

    from SoccerNet.Downloader import SoccerNetDownloader

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dl = SoccerNetDownloader(LocalDirectory=str(RAW_DIR))
    dl.password = password

    # Broadcast video + tracklet annotations
    dl.downloadDataTask(task="tracking", split=["test"])
    # Camera calibration / field registration for the same clips
    dl.downloadDataTask(task="calibration", split=["test"])

    print(f"Done. Raw data in {RAW_DIR}")


if __name__ == "__main__":
    main()
