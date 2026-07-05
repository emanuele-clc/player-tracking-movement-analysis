"""
Standard football pitch geometry (FIFA regulation dimensions, in cm) and the
32 line-intersection keypoints used as homography anchors.

The pitch measurements themselves are just facts (FIFA's regulation pitch
dimensions), not anyone's creative work - but the specific 32-keypoint
numbering/edge-graph convention used here follows the widely-adopted scheme
from Roboflow's open-source "sports" project (MIT-licensed):
https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py
Reusing a shared, well-known keypoint convention (rather than inventing a new
one) matters here specifically because it's what public pitch-keypoint
detection models (e.g. Roboflow's football-field-detection) are trained to
output - using the same numbering means detector output plugs directly into
`pitch_calibration.py` without a translation layer.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    width: int = 6800       # touchline-to-touchline, cm (68m - standard analytics pitch)
    length: int = 10500     # goal-line-to-goal-line, cm (105m - standard analytics pitch)
    penalty_box_width: int = 4032    # 40.32m
    penalty_box_length: int = 1650   # 16.5m
    goal_box_width: int = 1832       # 18.32m
    goal_box_length: int = 550       # 5.5m
    centre_circle_radius: int = 915  # 9.15m
    penalty_spot_distance: int = 1100  # 11m

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """32 line-intersection keypoints, in pitch-plane cm, origin at one
        corner flag. Index order matches labels below (1-indexed in labels,
        0-indexed here)."""
        w, l = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        r = self.centre_circle_radius
        spot = self.penalty_spot_distance
        return [
            (0, 0),                                   # 1  own corner
            (0, (w - pbw) / 2),                        # 2  own penalty box corner (top)
            (0, (w - gbw) / 2),                         # 3  own goal box corner (top)
            (0, (w + gbw) / 2),                         # 4  own goal box corner (bottom)
            (0, (w + pbw) / 2),                         # 5  own penalty box corner (bottom)
            (0, w),                                     # 6  own corner (far side)
            (gbl, (w - gbw) / 2),                       # 7  own goal box (top, pulled in)
            (gbl, (w + gbw) / 2),                       # 8  own goal box (bottom, pulled in)
            (spot, w / 2),                              # 9  own penalty spot
            (pbl, (w - pbw) / 2),                       # 10 own penalty box (top, pulled in)
            (pbl, (w - gbw) / 2),                       # 11
            (pbl, (w + gbw) / 2),                       # 12
            (pbl, (w + pbw) / 2),                       # 13 own penalty box (bottom, pulled in)
            (l / 2, 0),                                 # 14 halfway line (touchline, top)
            (l / 2, w / 2 - r),                         # 15 centre circle x halfway (top)
            (l / 2, w / 2 + r),                         # 16 centre circle x halfway (bottom)
            (l / 2, w),                                 # 17 halfway line (touchline, bottom)
            (l - pbl, (w - pbw) / 2),                   # 18 far penalty box (pulled in, top)
            (l - pbl, (w - gbw) / 2),                    # 19
            (l - pbl, (w + gbw) / 2),                    # 20
            (l - pbl, (w + pbw) / 2),                    # 21 far penalty box (pulled in, bottom)
            (l - spot, w / 2),                          # 22 far penalty spot
            (l - gbl, (w - gbw) / 2),                    # 23 far goal box (pulled in, top)
            (l - gbl, (w + gbw) / 2),                    # 24 far goal box (pulled in, bottom)
            (l, 0),                                     # 25 far corner
            (l, (w - pbw) / 2),                          # 26 far penalty box corner (top)
            (l, (w - gbw) / 2),                          # 27 far goal box corner (top)
            (l, (w + gbw) / 2),                          # 28 far goal box corner (bottom)
            (l, (w + pbw) / 2),                          # 29 far penalty box corner (bottom)
            (l, w),                                      # 30 far corner (far side)
            (l / 2 - r, w / 2),                          # 31 centre circle, left point
            (l / 2 + r, w / 2),                          # 32 centre circle, right point
        ]

    labels: List[str] = field(default_factory=lambda: [str(i) for i in range(1, 33)])

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17),
        (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    ])

    @property
    def pitch_length_m(self) -> float:
        return self.length / 100

    @property
    def pitch_width_m(self) -> float:
        return self.width / 100
