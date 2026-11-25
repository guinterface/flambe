"""Generate goal location, depth, and semantic labels aligned with rendered flight frames."""
import json, os, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

import goal_expert as gx

SV = Path(os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts")); WS = Path(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace", "outputs"))
YD = Path(os.path.join(FLAMBE_YARDS, "yards"))
NG, CELL = 16, 2.0
RH, RW, DMAX, STEP = 9, 16, 20.0, 0.2
SEM2C = {0: 6, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 17: 1, 6: 2, 7: 2, 15: 2, 16: 2, 18: 2, 8: 3, 9: 4, 10: 5,
         11: 6, 12: 6, 13: 6, 14: 6, 19: 6}
LUT = np.array([SEM2C.get(i, 6) for i in range(256)], np.uint8)
C = gx.CAM
# The policy image is Resize(256, 256) then CenterCrop(224) of the 640 x 360 frame (pilot.py), so it shows the central
# 224/256 of the frame in both axes; the rays cover exactly that region.
_FR = 224.0 / 256.0
US = C["w"] * (0.5 - _FR / 2 + (np.arange(RW) + 0.5) * _FR / RW); VS = C["h"] * (0.5 - _FR / 2 + (np.arange(RH) + 0.5) * _FR / RH)
UU, VV = np.meshgrid(US, VS)                                              # (RH, RW)
DIR_B = np.stack([np.ones_like(UU), (UU - C["cx"]) / C["fx"], (VV - C["cy"]) / C["fy"]], -1).reshape(-1, 3)   # body x fwd, y right, z down
SS = np.arange(STEP, DMAX + 1e-6, STEP)


class Truth:
    def __init__(self, yard, goal):
        t = np.load(YD / yard / "recon" / "truth.npz")
        self.occ = t["occ3d"].astype(bool); self.lo = t["lo"].astype(float); self.cell = float(t["cell"])
        self.h = t["height_top"].astype(float); self.sem = t["sem_top"]; ex = t["extent_blender"]
        self.hx0, self.hy0 = float(ex[0]), float(ex[2]); self.zg = -float(t["z_ground"])   # NED z of the ground
        gy = gx.GoalYard(YD / yard, goal)
        self.face, self.n = np.asarray(gy.face, float), np.asarray(gy.n, float)
        self.t = np.array([-self.n[1], self.n[0], 0.0])
        Pd = np.asarray(goal["points_ned"], float)                               # door corner points (goals.json)
        lat = (Pd - self.face) @ self.t
        self.lat = (lat.min() - 0.1, lat.max() + 0.1); self.zr = (Pd[:, 2].min() - 0.1, Pd[:, 2].max() + 0.1)

    def range_ray(self, pos, d_world):
        """Range (m) along one world-frame unit ray to the first occupied point (ground, occupancy or height map), or
        None within DMAX. Used by arm G3 as an ideal depth sensor (perception.py)."""
        P = np.asarray(pos)[None, :] + np.asarray(d_world)[None, :] * SS[:, None]
        hit = P[:, 2] >= self.zg
        I = np.floor((P - self.lo) / self.cell).astype(int)
        inb = np.all((I >= 0) & (I < np.array(self.occ.shape)), axis=-1); Ic = np.where(inb[:, None], I, 0)
        hit |= inb & self.occ[Ic[:, 0], Ic[:, 1], Ic[:, 2]]
        hx = ((P[:, 0] - self.hx0) / self.cell).astype(int); hy = ((-P[:, 1] - self.hy0) / self.cell).astype(int)
        inh = (hx >= 0) & (hx < self.h.shape[0]) & (hy >= 0) & (hy < self.h.shape[1])
        hit |= (~inb) & inh & (self.h[np.where(inh, hx, 0), np.where(inh, hy, 0)] > -P[:, 2] + 0.05)
        return float(SS[hit.argmax()]) if hit.any() else None

    def cast(self, pos, q):
        R = Rotation.from_quat(q).as_matrix(); D = DIR_B @ R.T; D /= np.linalg.norm(D, axis=1, keepdims=True)
        P = pos[None, None, :] + D[:, None, :] * SS[None, :, None]           # (rays, samples, 3)
        hit = P[..., 2] >= self.zg                                            # ground
        I = np.floor((P - self.lo) / self.cell).astype(int)
        inb = np.all((I >= 0) & (I < np.array(self.occ.shape)), axis=-1)
        Ic = np.where(inb[..., None], I, 0)
        hit |= inb & self.occ[Ic[..., 0], Ic[..., 1], Ic[..., 2]]
        hx = ((P[..., 0] - self.hx0) / self.cell).astype(int); hy = ((-P[..., 1] - self.hy0) / self.cell).astype(int)
        inh = (hx >= 0) & (hx < self.h.shape[0]) & (hy >= 0) & (hy < self.h.shape[1])
        hxc, hyc = np.where(inh, hx, 0), np.where(inh, hy, 0)
        hit |= (~inb) & inh & (self.h[hxc, hyc] > -P[..., 2] + 0.05)
        first = np.where(hit.any(1), hit.argmax(1), -1)
        depth = np.where(first >= 0, SS[np.maximum(first, 0)], DMAX).astype(np.float32)
        r = np.arange(len(first)); Ph = P[r, np.maximum(first, 0)]
        sx = np.clip(((Ph[:, 0] - self.hx0) / self.cell).astype(int), 0, self.sem.shape[0] - 1)
        sy = np.clip(((-Ph[:, 1] - self.hy0) / self.cell).astype(int), 0, self.sem.shape[1] - 1)
        cls = LUT[self.sem[sx, sy]]
        cls = np.where(Ph[:, 2] >= self.zg - 0.02, np.where(cls == 1, 1, cls), cls)
        dv = Ph - self.face                                                    # door box around the goal face
        lt = dv @ self.t
        door = (np.abs(dv @ self.n) < 0.35) & (lt > self.lat[0]) & (lt < self.lat[1]) & (Ph[:, 2] > self.zr[0]) & (Ph[:, 2] < self.zr[1])
        cls = np.where(door, 3, cls); cls = np.where(first < 0, 0, cls).astype(np.uint8)
        return depth.reshape(RH, RW), cls.reshape(RH, RW)


def one(job):
    cohort, course, v, what = job
    yard = v["yard"]; flight = v["flight"]
    src = v.get("src", "goal_data")
    fd = WS / yard / src / flight
    L = np.load(fd / "labels.npz"); n = len(L["goal_body"])
    out = {}
    if "r2h1" in what:
        gb = L["goal_body"]; ix = np.floor((gb[:, 0] + NG * CELL / 2) / CELL).astype(int); iy = np.floor((gb[:, 1] + NG * CELL / 2) / CELL).astype(int)
        ok = (ix >= 0) & (ix < NG) & (iy >= 0) & (iy < NG)
        out["r2h1"] = dict(cell=np.where(ok, ix * NG + iy, -1).astype(np.int64), gz=gb[:, 2].astype(np.float32),
                           phase=L["phase"].astype(np.int64), vis=L["visible"].astype(np.int64))
    if "r3" in what:
        X = np.load(fd / "tXUd.npy")[1:11]                                       # rows: pos 3, vel 3, quat 4 (xyzw)
        goals = {g["id"]: g for g in json.load(open(YD / yard / "goals.json"))["goals"]}
        tr = Truth(yard, goals["door"])                                         # the door is class 3 in every view
        dep = np.zeros((n, RH, RW), np.float16); seg = np.zeros((n, RH, RW), np.uint8)
        for t in range(n):
            x = X[:, min(t + 1, X.shape[1] - 1)]
            d, c = tr.cast(x[0:3], x[6:10]); dep[t] = d; seg[t] = c
        out["r3"] = dict(depth=dep, seg=seg)
    for k, a in out.items():
        os.makedirs(SV / cohort / f"s10_{k}", exist_ok=True)
        np.savez_compressed(SV / cohort / f"s10_{k}" / f"{course}.npz", **a)
    return course, n


if __name__ == "__main__":
    cohort, man, yre, kre = sys.argv[1:5]; what = (sys.argv[5] if len(sys.argv) > 5 else "r2h1,r3").split(",")
    M = json.load(open(man))["courses"]
    jobs = [(cohort, c, v, what) for c, v in sorted(M.items()) if re.fullmatch(yre, v["yard"]) and re.fullmatch(kre, v["kind"])]
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for c, n in p.imap_unordered(one, jobs):
            print(f"[s10lab] {c} {n}", flush=True)
    print(f"S10_LABELS_DONE {len(jobs)} courses {what}", flush=True)
