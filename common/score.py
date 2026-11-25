"""Score simulated delivery flights by goal proximity, speed, hold time, and clearance."""
import argparse, csv, glob, json, os, sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

R_GOAL, V_STOP, HOLD_S, BODY_R, BUDGET_X = 0.5, 0.2, 1.0, 0.30, 1.5

DATA = os.environ.get("FE_DATA", os.path.join(FLAMBE_ROOT, "research", "02_experiments", "syn_closedloop", "data"))
SY = os.environ.get("SY_ROOT", FLAMBE_YARDS)
TERRA = os.environ.get("FE_TERRA", FLAMBE_TERRA)
MAN = os.environ.get("GMAN", f"{TERRA}/SousVide_official/cohorts/syn_L8p/goal_manifest.json")


def score_track(t, pos, vel, goal, clearance, budget_s, r_goal=R_GOAL, v_stop=V_STOP, hold_s=HOLD_S, body_r=BODY_R):
    """t (N,) s; pos, vel (N, 3); goal (3,); clearance (N,) m to the nearest occupied cell; budget_s: time allowed from t[0].
    Returns dict: success, outcome (success|contact|overshoot|timeout), t_success, t_contact, closest_m, min_clearance_m."""
    t = np.asarray(t, float); pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    clr = np.asarray(clearance, float); goal = np.asarray(goal, float)
    assert len(t) == len(pos) == len(vel) == len(clr) and len(t) >= 1, "track arrays must have the same length"
    assert np.all(np.diff(t) > 0), "time must increase"
    d = np.linalg.norm(pos - goal, axis=1); spd = np.linalg.norm(vel, axis=1)
    ok = (d < r_goal) & (spd < v_stop)
    t_end = t[0] + budget_s
    t_success = None
    i = 0
    while i < len(t):                                  # first frame at which a continuous ok-run has lasted hold_s
        if ok[i]:
            j = i
            while j + 1 < len(t) and ok[j + 1] and t[j + 1] - t[i] < hold_s - 1e-9:
                j += 1
            if j + 1 < len(t) and ok[j + 1]:           # run reaches hold_s at frame j + 1
                t_success = t[j + 1]; break
            i = j + 1
        else:
            i += 1
    if t_success is not None and t_success > t_end + 1e-9:
        t_success = None
    hit = np.nonzero(clr < body_r)[0]
    t_contact = float(t[hit[0]]) if len(hit) else None
    win = t <= (t_success if t_success is not None else t_end) + 1e-9
    if t_contact is not None and t_contact <= (t_success if t_success is not None else t_end) + 1e-9:
        outcome = "contact"
    elif t_success is not None:
        outcome = "success"
    elif np.any(d[win] < r_goal) and d[win][-1] >= r_goal:
        outcome = "overshoot"
    else:
        outcome = "timeout"
    return {"success": outcome == "success", "outcome": outcome, "t_success": None if t_success is None else float(t_success),
            "t_contact": t_contact, "closest_m": float(d[win].min()) if win.any() else float(d.min()),
            "min_clearance_m": float(clr.min()), "budget_s": float(budget_s), "flown_s": float(t[-1] - t[0])}


def goal_point(yard, goal_id):
    G = {g["id"]: g for g in json.load(open(f"{SY}/yards/{yard}/goals.json"))["goals"]}
    return np.asarray(G[goal_id]["land_ned"], float)


_YARDS = {}


def clearance_fn(yard):
    if yard not in _YARDS:
        from scipy.spatial import cKDTree
        tr = np.load(f"{SY}/yards/{yard}/recon/truth.npz")
        idx = np.argwhere(np.asarray(tr["occ3d"]).astype(bool))
        _YARDS[yard] = cKDTree(np.asarray(tr["lo"], float) + (idx + 0.5) * float(tr["cell"]))
    return lambda P: _YARDS[yard].query(np.asarray(P, float))[0]


def score_deploy(path, yard, goal_id, budget_x=BUDGET_X):
    import torch
    p = torch.load(path, weights_only=False, map_location="cpu")
    goal = goal_point(yard, goal_id)
    assert np.allclose(goal, np.asarray(p["door_ned"], float), atol=1e-3), f"{path}: archive door_ned differs from goals.json"
    X = np.asarray(p["policy"]["Xro"], float); T = np.asarray(p["policy"]["Tro"], float)
    TE = np.asarray(p["expert_recorded"]["Tro"], float)
    r = score_track(T, X[:, 0:3], X[:, 3:6], goal, clearance_fn(yard)(X[:, 0:3]), budget_x * (TE[-1] - TE[0]))
    r["expert_s"] = float(TE[-1] - TE[0]); return r


def old_rule(arm, yard, ckpt):
    """goal_score.py's rule from the stored tsvs: closest < 0.5 m and no exact contact at any time."""
    d = os.path.join(DATA, arm, yard); ex, out = {}, {}
    pe = os.path.join(d, "exact_results.tsv")
    if os.path.isfile(pe):
        for r in csv.DictReader(open(pe), delimiter="\t"):
            s = r["source"]
            if s.startswith(ckpt + "_") and s.endswith("_deploy.pt"):
                ex[s[len(ckpt) + 1:-len("_deploy.pt")]] = r["collided"] in ("True", "1")
    for r in csv.DictReader(open(os.path.join(d, "results.tsv")), delimiter="\t"):
        if r["ckpt"] == ckpt:
            out[r["start"]] = r["reached_0p5"] == "1" and ex.get(r["start"]) is False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+"); ap.add_argument("--yards", nargs="+", default=["syn_002"])
    ap.add_argument("--ckpt", default="commNet_ep015"); ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--json", default=None); ap.add_argument("--budget_x", type=float, default=BUDGET_X)
    a = ap.parse_args()
    man = {(c["yard"], c["flight"]): c for c in json.load(open(MAN))["courses"].values()}
    allres = {}
    for arm in a.arms:
        rows = {}
        for y in a.yards:
            old = old_rule(arm, y, a.ckpt)
            for f in sorted(glob.glob(os.path.join(DATA, arm, y, "traj", f"{a.ckpt}_*_deploy.pt"))):
                fl = os.path.basename(f)[len(a.ckpt) + 1:-len("_deploy.pt")]
                c = man.get((y, fl))
                if c is None or (a.kinds and c["kind"] not in a.kinds):
                    continue
                r = score_deploy(f, y, c["goal"], a.budget_x); r["old_success"] = old.get(fl); rows[f"{y}/{fl}"] = r
        allres[arm] = rows
        n = len(rows); s = sum(r["success"] for r in rows.values()); so = sum(bool(r["old_success"]) for r in rows.values())
        sp = {k: sum(r["outcome"] == k for r in rows.values()) for k in ("contact", "overshoot", "timeout")}
        both = sum(r["success"] and bool(r["old_success"]) for r in rows.values())
        print(f"{arm} ({a.ckpt}, budget {a.budget_x}x): {n} flights | new rule success {s}/{n} ({100 * s / max(n, 1):.1f}%) "
              f"contact {sp['contact']} overshoot {sp['overshoot']} timeout {sp['timeout']} | old rule {so}/{n} "
              f"({100 * so / max(n, 1):.1f}%) | success under both {both}", flush=True)
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
