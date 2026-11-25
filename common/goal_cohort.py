"""Convert goal-expert flights into training courses and write per-frame labels."""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

for _p in (os.path.join(FLAMBE_TERRA, "SousVide_official", "src"), os.path.join(FLAMBE_TERRA, "SousVide_official", "FiGS", "src"), FLAMBE_TERRA,
           os.path.join(FLAMBE_TERRA, "goalspec")):
    if _p not in sys.path:
        sys.path[:0] = [_p]
sys.path[:0] = LOCAL_MODULE_PATHS

import visual_expert_syn as ve          # noqa: E402
import goal_expert as gx                # noqa: E402
import g4_oracle as g4                  # noqa: E402

SV = Path(os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts"))
WS = Path(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace", "outputs"))
SY = Path(os.path.join(FLAMBE_YARDS, "yards"))
R0, R1 = 0.2, 1.0          # descent ramp (syn_vel_labels.py)
DH, DZ = 0.2, 0.10         # stop rule (syn_stop_labels.py)


def flights(yard: str, src: str, only: str = ""):
    """(flight name, summary, labels) for every copyable flight of the yard."""
    out = []
    # Optional list of flight names to leave out (re-flown flights that fail the forward-camera rule)
    excl = set(open(os.environ["GOAL_COHORT_EXCLUDE"]).read().split()) if os.environ.get("GOAL_COHORT_EXCLUDE") else set()
    for d in sorted((WS / yard / src).iterdir()):
        if not d.is_dir() or not d.name.startswith(only) or not (d / "summary.json").exists() or d.name in excl:
            continue
        s = json.load(open(d / "summary.json"))
        if not s["success"] or s["blind_commit"] or s["collided"] or s.get("no_route"):
            continue
        L = np.load(d / "labels.npz")
        if "guid" not in L.files:
            print(f"  [skip] {d.name}: no recorded guidance (flown before 11-03 evening)")
            continue
        out.append((d.name, s, L))
    return out


def course_of(yard, flight):
    return f"{yard}_{flight}_canon"


def u_replay(X, guid):
    """recompute the body-rate command from the recorded guidance, as GeometricExpert._u (prev_oz rate limit included)"""
    class _E:
        prev_oz = 0.0
    e = _E(); U = np.zeros((len(X), 4))
    for k in range(len(X)):
        U[k], _ = ve.GeometricExpert._u(e, X[k, 0:3], X[k, 3:6], X[k, 6:10], np.asarray(guid[k, 0:2]), float(guid[k, 2]), float(guid[k, 3]))
    return U


def vel_labels(X, L, n):
    G = np.asarray(L["guid"], float)[:n]; com = np.asarray(L["committed"])[:n]
    hov = np.asarray(L["hover"], float)[:n]; land = np.asarray(L["land"], float)[:n]
    a = np.zeros((n, 4), np.float32)
    for k in range(n):
        x = X[k]; yaw = ve.yaw_of(x[6:10]); c, s = math.cos(-yaw), math.sin(-yaw); v = G[k, 0:2]; z_set = G[k, 2]
        if com[k]:
            dh = float(np.linalg.norm(x[0:2] - hov[k, :2]))
            if dh < R1:
                z_set = land[k, 2] + (hov[k, 2] - land[k, 2]) * min(1.0, max(0.0, (dh - R0) / (R1 - R0)))
        a[k] = [c * v[0] - s * v[1], s * v[0] + c * v[1], z_set - x[2], ve.wrap(G[k, 3] - yaw)]
    return a


def stop_labels(X, L, n):
    com = np.asarray(L["committed"])[:n]; hov = np.asarray(L["hover"], float)[:n]; land = np.asarray(L["land"], float)[:n]
    dh = np.linalg.norm(X[:, 0:2] - hov[:, :2], axis=1)
    return ((dh < DH) & (X[:, 2] >= land[:, 2] - DZ) & (com > 0)).astype(np.uint8)


def check_training_yard(yard):
    import priors
    if priors.held_out(yard):
        print(f"[FATAL] {yard} is a held-out yard (priors.held_out); goal flights there are never training data"); sys.exit(2)


def cmd_convert(a):
    import sv_canonical_to_rollout as conv
    check_training_yard(a.yard)
    n_new = 0
    for name, s, L in flights(a.yard, a.src, a.only):
        course = course_of(a.yard, name)
        base = Path(conv.COHORTS) / a.cohort / "rollout_data" / course
        if (base / "trajectories" / "001.pt").exists() and (base / "images" / "001.pt").exists():
            continue
        if not (WS / a.yard / a.src / name / "images").is_dir():
            print(f"  [skip] {name}: images already removed"); continue
        conv.convert_start(name, use_svu=True, cohort=a.cohort, keep_hover_frames=40, scene=a.yard,
                           ve_dir=str(WS / a.yard / a.src), door_ned=None)
        n_new += 1; print(f"  [ok] {course}", flush=True)
    print(f"[goal-cohort] convert {a.yard}: {n_new} new courses")


def cmd_labels(a):
    check_training_yard(a.yard)
    co = SV / a.cohort
    for sub in ("vel_labels_ramp", "stop_labels", "g4_oracle"):
        (co / sub).mkdir(exist_ok=True)
    mpath = co / "goal_manifest.json"
    man = json.load(open(mpath)) if mpath.exists() else {"courses": {}}
    goals = {g["id"]: g for g in json.load(open(SY / a.yard / "goals.json"))["goals"]}
    r2s, agree, npos = [], [], []
    for name, s, L in flights(a.yard, a.src, a.only):
        course = course_of(a.yard, name)
        T = np.load(WS / a.yard / a.src / name / "tXUd.npy"); N = T.shape[1]; n = N - 1
        assert len(L["guid"]) >= n, (name, len(L["guid"]), N)
        X = T[1:11, 1:].T; U = T[11:15, :-1].T                   # decision k: state col k+1, command col k
        Ur = u_replay(X, np.asarray(L["guid"], float)[:n])
        r2 = 1 - ((Ur - U) ** 2).sum(0) / (((U - U.mean(0)) ** 2).sum(0) + 1e-12); r2s.append(r2)
        av = vel_labels(X, L, n); sv = stop_labels(X, L, n)
        g = goals[s["goal"]]; gy = gx.GoalYard(SY / a.yard, g); pts = [np.asarray(p, float) for p in g["points_ned"]]
        O = np.stack([g4.oracle_vec(gy, pts, g["kind"], X[k]) for k in range(n)])
        agree.append(float(np.mean(O[:, 0] == np.asarray(L["visible"])[:n])))
        np.savez(co / "vel_labels_ramp" / f"{course}.npz", a=av, replay_r2=r2)
        np.savez(co / "stop_labels" / f"{course}.npz", s=sv)
        np.savez(co / "g4_oracle" / f"{course}.npz", o=O)
        npos.append(int(sv.sum()))
        man["courses"][course] = {"yard": a.yard, "flight": name, "start": s["start"], "rollout": s.get("rollout", 0),
                                  "goal": s["goal"], "kind": s["kind"], "instruction": s["instruction"], "frames": int(N),
                                  "land_ned": [float(v) for v in np.asarray(L["land"])[-1]],
                                  "hover_ned": [float(v) for v in np.asarray(L["hover"])[-1]], "src": a.src,
                                  # Every face's land point, for rescoring a landing at another face of the same goal
                                  "faces_land_ned": [f["land_ned"] for f in g.get("faces", [g])]}
        print(f"  {course}: n {n}  R2 {np.round(r2, 3).tolist()}  oracle=expert visibility {agree[-1]:.3f}  "
              f"visible {O[:, 0].mean():.2f}  stop frames {int(sv.sum())}", flush=True)
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(man, open(mpath, "w"), indent=1)
    if r2s:
        R = np.array(r2s)
        print(f"[goal-cohort] labels {a.yard}: {len(R)} courses; R2 min {np.round(R.min(0), 3).tolist()} median "
              f"{np.round(np.median(R, 0), 3).tolist()}; oracle/expert visibility agreement min {min(agree):.3f}; "
              f"courses with no stop frame {sum(p == 0 for p in npos)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "labels"]); ap.add_argument("cohort"); ap.add_argument("yard")
    ap.add_argument("--src", default="goal_data"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    {"convert": cmd_convert, "labels": cmd_labels}[a.cmd](a)
