#!/usr/bin/env python3
"""
syn_exact_collision.py: exact occupancy scoring of a flight on a generated yard.

The deploy harness scores collisions with a forward-cone metric-depth test (an approximation,
terra/collision_metric.py). Generated yards carry exact labels, so a flown trajectory can be scored
against synth_yards/yards/<yard>/recon/truth.npz: occ3d (nx, ny, nz) bool, lo (3,) NED, cell (m);
the centre of cell (i, j, k) is lo + ([i, j, k] + 0.5) * cell. The score is the minimum 3-D distance
from every flown position to the nearest occupied cell centre (cKDTree, the same rule the syn expert
uses for its own clearance), and collided = min distance < BODY_R (0.30 m).

Inputs: one or more deploy archives written by sv_deploy_canonical.py and kept by SAVE_TRAJ=1 as
<results>/traj/<ckpt>_<start>_deploy.pt (dict; the flown states are pt["policy"]["Xro"] (N, 10) with
positions in columns 0:3; "expert_sim" and "expert_recorded" hold the expert panels), or an expert
tXUd.npy (15, N) whose positions are rows 1 to 3. Reports, per trajectory: min_clearance_m, its
index and time, collided, closest approach to door_ned, final distance to door_ned, terminal speed,
and frames. door_ned comes from terra/syn_registry/<yard>.json unless --door_ned is given.

    python3 syn_exact_collision.py --yard syn_002 --pt results/traj/commNet_ep030_s000_deploy.pt ...
    python3 syn_exact_collision.py --yard syn_002 --txud outputs/syn_002/visual_expert/s000/tXUd.npy
    options: --which policy|expert_sim|expert_recorded  --body_r 0.30  --tsv out.tsv  --json

Runs inside figs-sv (needs torch for the .pt files; numpy and scipy suffice for a tXUd.npy). Paths
are taken as given, so pass container paths when running in the container.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

SY_ROOT = os.environ.get("SY_ROOT", FLAMBE_YARDS)
REG_DIR = os.environ.get("SYN_REG_DIR", os.path.join(FLAMBE_TERRA, "syn_registry"))
TSV_COLS = ["source", "which", "frames", "duration_s", "min_clearance_m", "t_min_clearance_s",
            "collided", "n_frames_in_contact", "closest_m", "final_m", "terminal_speed_m_s"]


class ExactYard:
    def __init__(self, yard: str, root: str = SY_ROOT):
        path = os.path.join(root, "yards", yard, "recon", "truth.npz")
        assert os.path.isfile(path), f"missing {path}"
        t = np.load(path)
        self.lo = np.asarray(t["lo"], float)
        self.cell = float(t["cell"])
        self.occ3d = np.asarray(t["occ3d"]).astype(bool)
        idx = np.argwhere(self.occ3d)
        self.pts = self.lo + (idx + 0.5) * self.cell
        self.tree = cKDTree(self.pts)
        self.source = str(t["source"]) if "source" in t.files else "?"

    def clearance(self, P: np.ndarray) -> np.ndarray:
        d, _ = self.tree.query(np.asarray(P, float))
        return d


def load_door(yard: str, override=None):
    if override:
        v = [float(x) for x in override.split(",")]
        assert len(v) == 3, "door_ned must be x,y,z"
        return np.array(v)
    with open(os.path.join(REG_DIR, f"{yard}.json")) as fh:
        return np.array(json.load(fh)["door_ned"], float)


def positions_from_pt(path: str, which: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    panel = d[which]
    Xro = np.asarray(panel["Xro"], float)
    Tro = np.asarray(panel["Tro"], float).ravel()
    if Xro.shape[0] != len(Tro) and Xro.shape[1] == len(Tro):
        Xro = Xro.T
    n = min(len(Tro), Xro.shape[0])
    return Tro[:n], Xro[:n, 0:3], Xro[:n, 3:6], d.get("door_ned")


def positions_from_txud(path: str):
    t = np.load(path)
    assert t.ndim == 2 and t.shape[0] == 15, f"tXUd must be (15,N), got {t.shape}"
    return t[0, :], t[1:4, :].T, t[4:7, :].T, None


def score(yard: ExactYard, door: np.ndarray, Tro, P, V, body_r: float) -> dict:
    clr = yard.clearance(P)
    k = int(np.argmin(clr))
    d_door = np.linalg.norm(P - door, axis=1)
    speed = float(np.linalg.norm(V[-1])) if V is not None and len(V) else float("nan")
    return {
        "frames": int(len(P)),
        "duration_s": round(float(Tro[-1] - Tro[0]), 3),
        "min_clearance_m": round(float(clr[k]), 4),
        "t_min_clearance_s": round(float(Tro[k]), 3),
        "collided": bool(clr[k] < body_r),
        "n_frames_in_contact": int(np.sum(clr < body_r)),
        "closest_m": round(float(d_door.min()), 4),
        "final_m": round(float(d_door[-1]), 4),
        "terminal_speed_m_s": round(speed, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yard", required=True)
    ap.add_argument("--pt", nargs="*", default=[], help="deploy .pt archives")
    ap.add_argument("--txud", nargs="*", default=[], help="expert tXUd.npy files")
    ap.add_argument("--which", default="policy", choices=["policy", "expert_sim", "expert_recorded"])
    ap.add_argument("--body_r", type=float, default=0.30)
    ap.add_argument("--door_ned", default=None, help="override x,y,z (default: registry door_ned)")
    ap.add_argument("--tsv", default=None, help="append one row per trajectory to this TSV")
    ap.add_argument("--json", action="store_true", help="print one JSON object per trajectory")
    args = ap.parse_args()
    if not args.pt and not args.txud:
        ap.error("give --pt and/or --txud")

    yard = ExactYard(args.yard)
    door = load_door(args.yard, args.door_ned)
    print(f"[exact] yard={args.yard} occupied cells={len(yard.pts)} cell={yard.cell} m "
          f"source={yard.source} door_ned={np.round(door, 3).tolist()} body_r={args.body_r}", flush=True)

    rows = []
    for p in args.pt:
        Tro, P, V, door_pt = positions_from_pt(p, args.which)
        if door_pt is not None and np.linalg.norm(np.asarray(door_pt, float) - door) > 1e-3:
            print(f"  [warn] {os.path.basename(p)}: archive door_ned {np.round(door_pt, 3).tolist()} "
                  f"differs from the scoring door", flush=True)
        r = {"source": os.path.basename(p), "which": args.which}
        r.update(score(yard, door, Tro, P, V, args.body_r))
        rows.append(r)
    for p in args.txud:
        Tro, P, V, _ = positions_from_txud(p)
        root_prefix = os.path.join(FLAMBE_ROOT, "")
        label = p[len(root_prefix):] if p.startswith(root_prefix) else p
        r = {"source": label, "which": "txud"}
        r.update(score(yard, door, Tro, P, V, args.body_r))
        rows.append(r)

    for r in rows:
        if args.json:
            print(json.dumps(r), flush=True)
        else:
            print(f"  {r['source']} [{r['which']}]: frames={r['frames']} dur={r['duration_s']}s "
                  f"min_clearance={r['min_clearance_m']} m at t={r['t_min_clearance_s']}s "
                  f"collided={r['collided']} (contact frames {r['n_frames_in_contact']}) "
                  f"closest={r['closest_m']} m final={r['final_m']} m "
                  f"term_speed={r['terminal_speed_m_s']} m/s", flush=True)
    if args.tsv:
        new = not os.path.isfile(args.tsv)
        with open(args.tsv, "a") as fh:
            if new:
                fh.write("\t".join(TSV_COLS) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in TSV_COLS) + "\n")
        print(f"[exact] appended {len(rows)} row(s) to {args.tsv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
