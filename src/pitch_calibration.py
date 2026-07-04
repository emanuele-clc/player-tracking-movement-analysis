"""
Stage 2: homography from broadcast pixel coordinates to real-world pitch
coordinates (105m x 68m), using SoccerNet's field calibration annotations
(detected pitch line intersections / keypoints per frame or per camera shot).

Plan:
- Load SoccerNet calibration JSON for a clip (camera parameters or line
  correspondences).
- Fit/apply a homography matrix per frame (or per stable camera segment)
  mapping detection-box foot-points (bottom-center of each bounding box) to
  pitch-plane (x, y) in meters.
- Output: data/processed/pitch_coords/<clip_id>.parquet with
  (frame, track_id, team, x_m, y_m)

Not yet implemented — see README.md pipeline table for status.
"""


def main():
    raise NotImplementedError(
        "Pitch calibration not yet built. "
        "Planned: apply SoccerNet camera calibration to project tracked "
        "bounding-box foot-points into real-world (x, y) pitch coordinates."
    )


if __name__ == "__main__":
    main()
