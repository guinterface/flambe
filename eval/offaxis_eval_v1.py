"""Measure how often the goal falls outside the camera view during evaluation flights."""
import json, os, sys, glob, math
import numpy as np
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

CARL = os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "configs", "frame", "carl.json"); HALF = 35.0; VMIN = 0.1
TC2B = np.array(json.load(open(CARL))["camera_to_body_transform"], float)
def quat_R(q):                                                                  # (n,4) x, y, z, w -> (n,3,3), as contacts_gate_v1.quat_R
    q = q / np.linalg.norm(q, axis=-1, keepdims=True); x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
def off_axis(X):                                                                # off-axis angle (deg) and horizontal speed per state
    T = np.tile(np.eye(4), (len(X), 1, 1)); T[:, :3, :3] = quat_R(X[:, 6:10]); T[:, :3, 3] = X[:, :3]; Tc = T @ TC2B
    a = -Tc[:, :2, 2]; a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12); v = X[:, 3:5]
    return np.degrees(np.abs(np.arctan2(a[:, 0] * v[:, 1] - a[:, 1] * v[:, 0], (a * v).sum(1)))), np.linalg.norm(v, axis=1)
def group(f):
    o = f.get("outcome", ""); return "success" if f.get("success") else ("contact" if f.get("t_contact") is not None or "contact" in o else ("timeout" if o == "timeout" else "other"))
def acc(): return {"flights": 0, "steps_cut": 0, "off_cut": 0, "steps_all": 0, "off_all": 0}
def add(A, ang, s):
    m = s >= VMIN; A["flights"] += 1; A["steps_cut"] += int(m.sum()); A["off_cut"] += int((ang[m] > HALF).sum()); A["steps_all"] += len(ang); A["off_all"] += int((ang > HALF).sum())
def fin(A): return dict(A, share_cut=round(A["off_cut"] / A["steps_cut"], 4) if A["steps_cut"] else None, share_all=round(A["off_all"] / A["steps_all"], 4) if A["steps_all"] else None)
def ids_of(d): return {(json.load(open(e))["yard"], f["id"]) for e in glob.glob(f"{d}/eval_*.json") for f in json.load(open(e))["flights"]}
def run(d, keep=None):
    R = {"dir": d, "by_group": {}, "by_yard": {}, "missing_tracks": [], "n_flights_in_eval": 0}
    for e in sorted(glob.glob(f"{d}/eval_*.json")):
        E = json.load(open(e)); y = E.get("yard") or os.path.basename(e)[5:-5]
        for f in E["flights"]:
            if keep is not None and (y, f["id"]) not in keep: continue
            R["n_flights_in_eval"] += 1; p = f"{d}/tracks_{y}/{f['id']}.npz"
            if not os.path.exists(p): R["missing_tracks"].append(f"{y}/{f['id']}"); continue
            z = np.load(p); X = z["X"].astype(np.float64); T = z["T"] if "T" in z.files else np.arange(len(X)) * 0.05
            if f.get("t_contact") is not None: X = X[:max(1, min(int(np.searchsorted(T, f["t_contact"])), len(X)))]
            ang, s = off_axis(X); g = group(f)
            for key, tab in ((g, R["by_group"]), ("all", R["by_group"])): tab.setdefault(key, acc()); add(tab[key], ang, s)
            R["by_yard"].setdefault(y, {}).setdefault(g, acc()); add(R["by_yard"][y][g], ang, s)
    R["by_group"] = {k: fin(v) for k, v in R["by_group"].items()}; R["by_yard"] = {y: {k: fin(v) for k, v in t.items()} for y, t in R["by_yard"].items()}
    R["n_missing_tracks"] = len(R["missing_tracks"]); return R
if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; out, dirs = a[0], a[1:]; same = "--same-starts" in sys.argv
    keep = set.intersection(*[ids_of(d) for d in dirs]) if same and len(dirs) > 1 else None
    res = {"script": "offaxis_eval_v1.py", "half_angle_deg": HALF, "vmin": VMIN, "same_starts": bool(keep), "n_common_starts": len(keep) if keep else None, "runs": [run(d, keep) for d in dirs]}
    assert not os.path.exists(out), f"{out} exists"; json.dump(res, open(out, "w"), indent=1)
    for r in res["runs"]: print(r["dir"].split("/gate2/")[-1], {g: (v["flights"], v["share_cut"]) for g, v in r["by_group"].items()}, "missing tracks", r["n_missing_tracks"])
