"""Compute simulation-only goal visibility and image coordinates from ground truth."""
from __future__ import annotations

import numpy as np

import goal_expert as gx

KINDS = ["door", "window", "prop", "mailbox", "garage_door", "vehicle"]
D = 4 + len(KINDS)


def kind_onehot(kind: str) -> np.ndarray:
    v = np.zeros(len(KINDS), np.float32)
    v[KINDS.index(kind)] = 1.0
    return v


def extent_px(eye, q, pts) -> float:
    uv = [gx.project(eye, q, p) for p in pts]
    uv = np.asarray([x for x in uv if x is not None])
    return float(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))) if len(uv) else 0.0


def oracle_vec(yard: "gx.GoalYard", pts, kind: str, x) -> np.ndarray:
    """x: drone state [p(3), v(3), q_xyzw(4)] in NED (the decision state)."""
    pos, q = np.asarray(x[0:3], float), np.asarray(x[6:10], float)
    vis, px = gx.visible(yard, pos, q, pts)
    out = np.zeros(D, np.float32)
    if vis:
        out[0] = 1.0
        out[1] = (px[0] - gx.CAM["w"] / 2) / (gx.CAM["w"] / 2)
        out[2] = (px[1] - gx.CAM["h"] / 2) / (gx.CAM["h"] / 2)
        out[3] = extent_px(pos, q, pts) / gx.CAM["w"]
    out[4:] = kind_onehot(kind)
    return out
