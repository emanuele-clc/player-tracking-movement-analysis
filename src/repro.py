import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from detect_pitch_lines import detect_lines
from pitch_config import SoccerPitchConfiguration

print("IMPORTS DONE")

def main():
    print("MAIN START")
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, type=Path)
    args = ap.parse_args()
    print("PARSED:", args.frame)
    frame = cv2.imread(str(args.frame))
    print("FRAME SHAPE:", None if frame is None else frame.shape)

if __name__ == "__main__":
    main()
    print("MAIN DONE")
