"""Relabel student flight states with expert guidance and generate action chunks."""
from __future__ import annotations

import argparse, copy, glob, json, math, os, re, sys, time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

for _p in (os.path.join(FLAMBE_TERRA, "FiGS-Standalone"), os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "src"), os.path.join(FLAMBE_TERRA, "goalspec", "v4"), os.path.join(FLAMBE_TERRA, "goalspec"), FLAMBE_TERRA):
    if _p not in sys.path:
        sys.path[:0] = [_p]
sys.path[:0] = LOCAL_MODULE_PATHS
import figs.simulator                   # noqa: E402  the installed FiGS-Standalone, as run_episode; must precede goal_cohort,
import figs.utilities.trajectory_helper # noqa: E402  which puts SousVide_official/FiGS/src (another figs) on sys.path

import visual_expert_syn as ve          # noqa: E402
import goal_expert as gx                # noqa: E402
import goal_expert_v4 as gv             # noqa: E402
import grid3d as g3                     # noqa: E402
import goal_cohort as gc                # noqa: E402
import g4_oracle as g4                  # noqa: E402

ROOT = Path(os.environ.get("DAGGER_ROOT", os.path.join(FLAMBE_YARDS_FALLBACK, "dagger_v4")))
CO = Path(os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts", "syn_v4"))
SY = Path(os.path.join(FLAMBE_YARDS, "yards"))
TV = Path(os.path.join(FLAMBE_YARDS_FALLBACK, "train_v4", "flights"))
K = int(os.environ.get("CHUNK_K", "20"))
REPLAN_M, PULL_R, PHASE_BACK_M = 1.0, 0.45, 1.0
FAR, MIN_FRAMES = 1.5, 40


# ---------------------------------------------------------------- dynamics (no gsplat)
class Dyn:
    """the FiGS-Standalone integrator of run_episode (Simulator.load_rollout('baseline') + load_frame('carl')), without the splat"""

    def __init__(self):
        from figs.simulator import Simulator
        s = Simulator.__new__(Simulator)
        s.configs_path = Path(sys.modules["figs.simulator"].__file__).parent.parent.parent / "configs"
        s.conFiG = {"rollout": {}, "drone": {}, "perception": {}}; s.solver = None
        cwd = os.getcwd(); os.makedirs("/tmp/w_dyn", exist_ok=True); os.chdir("/tmp/w_dyn")
        s.load_rollout("baseline"); s.load_frame("carl"); os.chdir(cwd)
        r = s.conFiG["rollout"]; self.sim = s
        self.n_sub = int(r["frequency"] / ve.HZ_CTL); self.n_delay = int(r["delay"] * r["frequency"])
        self.mu = np.array(r["model_noise"]["mean"]) / r["frequency"]
        assert np.allclose(r["model_noise"]["std"], 0) and np.allclose(self.mu, 0), "lookahead assumes the noiseless baseline rollout"

    def step(self, x, u_prev, u):
        from figs.utilities.trajectory_helper import obedient_quaternion as oq
        xp = x.copy()
        for i in range(self.n_sub):
            uin = u_prev if i < self.n_delay else u
            x = self.sim.solver.simulate(x=x, u=uin); x[6:10] = oq(x[6:10], xp[6:10]); xp = x.copy()
        return x


# ---------------------------------------------------------------- expert replay
def snapshot(ex):
    s = copy.copy(ex); s.log = []; s.counts = dict(ex.counts); return s


def replan(ex, pos, cache):
    g = ex.yard.goal_rec; fi = ex.plan[2]
    for r in (g3.R_FREE, g3.R_MIN):
        k = (g["id"], fi, gv.SOFT, r)
        if k not in cache:
            cache[k] = ex.gr.field(ex.pose["hover"], g3.ceil_for(ex.pose), r, gv.SOFT)
        f = cache[k]
        R = f(np.asarray(pos, float)) if f is not None else None
        if R is not None and len(R) >= 2:
            ex.route = [np.asarray(p, float) for p in R]; ex.vv = ex._vertex_speeds(); ex.seg = 0; ex._t_on_seg = 0.0
            return True
    return False


def maybe_replan(ex, x, cache):
    """the only departure from the expert's code: off-route states get a route from where the student is"""
    if ex.pose is None:
        return False
    pos = np.asarray(x[0:3], float)
    if ex.phase in ("approach", "descent"):
        ref = ex.pose["hover"] if ex.phase == "approach" else ex.pose["land"]
        if np.linalg.norm(ref[:2] - pos[:2]) <= PHASE_BACK_M:
            return False
        if replan(ex, pos, cache):
            ex.phase = "cruise"; ex.hold = 0; return True
        return False
    if ex.phase != "cruise":
        return False
    P = ex.route; best = None
    for i in range(len(P) - 1):                  # distance to the whole remaining route (not only seg..seg+3)
        a, b = P[i], P[i + 1]; ab = b - a; L = float(np.linalg.norm(ab)) + 1e-9
        t = float(np.clip(np.dot(pos - a, ab) / L, 0.0, L)); c = a + ab / L * t; d = float(np.linalg.norm(c - pos))
        if best is None or d < best[0]:
            best = (d, c)
    if best is None:
        return False
    if best[0] > REPLAN_M or not ex.gr.seg_clear(pos, best[1], PULL_R):
        return replan(ex, pos, cache)
    return False


def vel_label(x, lab):
    """goal_cohort.vel_labels for one decision (committed = 1 for the v4 expert)"""
    L = {"guid": np.asarray([lab["guid"]]), "committed": np.asarray([lab["committed"]]),
         "hover": np.asarray([lab["hover"]]), "land": np.asarray([lab["land"]])}
    return gc.vel_labels(np.asarray([x]), L, 1)[0]


def lookahead(ex, x0, u_prev, dyn, cache):
    """K-step velocity-label chunk of the expert flying from x0 with its current internal state (a copy)"""
    e = snapshot(ex); x = np.asarray(x0, float).copy(); C = np.zeros((K, 4), np.float32); j = 0
    for j in range(K):
        maybe_replan(e, x, cache)
        u = e.decide(0.0, x.copy()); C[j] = vel_label(x, e.log[-1])
        if e.landed:
            C[j + 1:] = C[j]; break
        x = dyn.step(x, u_prev, u); u_prev = u
    return C


def relabel_states(ex, X, Uprev, dyn, cache, chunks=True):
    """replay along states X (n, 10); returns per-decision labels, chunks, replan flags, landed index"""
    ex.reset(); n = len(X); A = np.zeros((n, 4), np.float32); C = np.zeros((n, K, 4), np.float32) if chunks else None
    logs, rp, landed_at, U = [], np.zeros(n, np.uint8), None, np.zeros((n, 4))
    for k in range(n):
        rp[k] = maybe_replan(ex, X[k], cache)
        if chunks:
            C[k] = lookahead(ex, X[k], Uprev[k], dyn, cache)
        U[k] = ex.decide(k / ve.HZ_CTL, X[k].copy()); lab = ex.log[-1]; logs.append(lab); A[k] = vel_label(X[k], lab)
        if chunks:
            assert np.allclose(C[k, 0], A[k], atol=1e-5), (k, C[k, 0], A[k])
        if ex.landed and landed_at is None:
            landed_at = k; break
    m = len(logs)
    return A[:m], (C[:m] if chunks else None), rp[:m], landed_at, logs, U[:m]


def make_expert(yard, goal_id, p0, cache, gr):
    goals = {g["id"]: g for g in json.load(open(SY / yard / "goals.json"))["goals"]}
    g = goals[goal_id]; gy = gx.GoalYard(SY / yard, g)
    ex = gv.DirectGoalExpert(gy, gr, np.asarray(p0, float), cache)
    return ex, gy, g


# ---------------------------------------------------------------- commands
def cmd_check(a):
    """replay on the expert's own train_v4 flights: labels vs the cohort's vel_labels_ramp, lookahead chunk vs a[k:k+K]"""
    gr = g3.Grid(SY / a.yard); cache = {}; dyn = Dyn()
    man = json.load(open(CO / "goal_manifest.json"))["courses"]
    names = sorted(c for c, v in man.items() if v["yard"] == a.yard and v["kind"] == "door" and v.get("src") == "train_v4"
                   and (not a.only or v["flight"].startswith(a.only)))[: a.limit or None]
    for course in names:
        v = man[course]; T = np.load(TV / a.yard / v["flight"] / "tXUd.npy"); N = T.shape[1]; n = N - 1
        X = T[1:11, 1:].T; Ucmd = T[11:15].T
        ref = np.load(CO / "vel_labels_ramp" / f"{course}.npz")["a"]
        ex, gy, g = make_expert(a.yard, v["goal"], T[1:4, 0], cache, gr)
        t0 = time.time()
        Uprev = np.vstack([Ucmd[0:1], Ucmd[:n - 1]])
        A, C, rp, la, logs, U = relabel_states(ex, X, Uprev, dyn, cache, chunks=True)
        m = len(A); r2 = 1 - ((A - ref[:m]) ** 2).sum(0) / (((ref[:m] - ref[:m].mean(0)) ** 2).sum(0) + 1e-12)
        ap = np.concatenate([ref, np.repeat(ref[-1:], K - 1, 0)]); Cref = np.stack([ap[t:t + K] for t in range(m)])
        r2c = 1 - ((C - Cref) ** 2).sum((0, 1)) / (((Cref - Cref.mean((0, 1))) ** 2).sum((0, 1)) + 1e-12)
        ej = np.abs(C - Cref).mean((0, 2))
        print(f"{course}: n {n} replayed {m} replans {int(rp.sum())} landed_at {la}  R2 a vs cohort {np.round(r2, 4).tolist()}  "
              f"R2 lookahead chunk vs a[k:k+K] {np.round(r2c, 4).tolist()}  |err| step0 {ej[0]:.4f} step{K - 1} {ej[-1]:.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)


def cmd_relabel(a):
    gr = g3.Grid(SY / a.yard); cache = {}; dyn = Dyn()
    man = json.load(open(CO / "goal_manifest.json"))["courses"]
    fd = Path(a.flight_dir); outy = ROOT / "flights" / a.yard; outy.mkdir(parents=True, exist_ok=True)
    tps = sorted(glob.glob(str(fd / "traj" / "commNet_ep*_deploy.pt")))
    tps = [t for t in tps if re.search(r"_(s\d+__door)_deploy\.pt$", t) and (not a.only or f"_{a.only}" in t)][: a.limit or None]
    import torch
    for tp in tps:
        sid = re.search(r"_(s\d+__door)_deploy\.pt$", tp).group(1); name = f"{sid}_{a.tag}"; out = outy / name
        if (out / "labels.npz").exists():
            print("exists", out); continue
        d = torch.load(tp, map_location="cpu", weights_only=False)
        course0 = d["course"]; v = man[course0]
        X = np.asarray(d["policy"]["Xro"], float); Tt = np.asarray(d["policy"]["Tro"], float); Us = np.asarray(d["policy"]["Uro"], float)
        n = len(Us)                                       # decisions (= video frames)
        t0 = time.time()
        ex, gy, g = make_expert(a.yard, v["goal"], X[0, 0:3], cache, gr)
        land = np.asarray(ex.pose["land"], float)
        dland = float(np.linalg.norm(land - np.asarray(d["door_ned"], float)))
        clr = gr.clearance(X[:n, 0:3]); dist = np.linalg.norm(X[:n, 0:3] - land, axis=1)
        cut = n; why = "end"
        hit = np.where(clr < ve.BODY_R)[0]
        if len(hit) and hit[0] < cut: cut, why = int(hit[0]), "contact"
        far = np.where(dist > FAR * dist[0])[0]
        if len(far) and far[0] < cut: cut, why = int(far[0]), "far"
        Uprev = np.vstack([Us[0:1], Us[:n - 1]])          # command applied before decision k (the student's)
        A, C, rp, la, logs, U = relabel_states(ex, X[:cut], Uprev[:cut], dyn, cache, chunks=True)
        if la is not None and la + 1 < cut: cut, why = la + 1, "expert_landed"
        A, C, rp = A[:cut], C[:cut], rp[:cut]
        if cut < MIN_FRAMES:
            print(f"short {a.yard} {name}: {cut} frames ({why}), skipped", flush=True); continue
        out.mkdir(parents=True, exist_ok=True)
        Xs = np.vstack([X[0:1], X[0:cut - 1]])            # column k+1 = decision-k state
        tXUd = np.vstack([Tt[:cut][None], Xs.T, Us[:cut].T])
        np.save(out / "tXUd.npy", tXUd)
        L = logs[:cut]
        np.savez_compressed(out / "labels.npz", **{k: np.asarray([l[k] for l in L]) for k in L[0]}, a=A, c=C, replan=rp,
                            u_expert=U[:cut])
        meta = {"dagger_tag": a.tag, "policy_traj": tp, "src_course": course0, "goal": v["goal"], "frames": int(cut),
                "flight_frames": int(n), "cut_reason": why, "min_clearance_m": float(clr[:cut].min()),
                "replans": int(rp.sum()), "land_vs_deploy_target_m": dland, "start_dist_m": float(dist[0]),
                "expert": "goal_expert_v4.DirectGoalExpert replayed along the student's states (dagger_v4.py)",
                "relabel_s": round(time.time() - t0, 1)}
        json.dump(meta, open(out / "dagger.json", "w"), indent=1)
        print(f"{a.yard} {name}: {cut}/{n} frames ({why}), replans {int(rp.sum())}, land vs deploy target {dland:.3f} m, "
              f"{time.time() - t0:.0f}s", flush=True)
    print("DAGGER_RELABEL_DONE", a.yard, a.tag)


def cmd_render(a):
    """images at the student's decision states with run_episode's camera (GPU container)"""
    import cv2
    from figs.simulator import Simulator
    from figs.utilities.trajectory_helper import xv_to_T
    sim = Simulator(ve.find_checkpoint(a.yard), "baseline", "carl")
    T_c2b = np.array(sim.conFiG["drone"]["T_c2b"]); camera = sim.gsplat.generate_output_camera(sim.conFiG["drone"]["camera"])
    for out in sorted((ROOT / "flights" / a.yard).glob(f"*_{a.tag}")):
        if not (out / "labels.npz").exists() or (out / "images.done").exists():
            continue
        T = np.load(out / "tXUd.npy"); n = T.shape[1]
        X = np.hstack([T[1:11, 1:], np.full((10, 1), np.nan)]).T     # decision k state = column k+1; last one from the deploy file
        import torch
        meta = json.load(open(out / "dagger.json")); d = torch.load(meta["policy_traj"], map_location="cpu", weights_only=False)
        X[n - 1] = np.asarray(d["policy"]["Xro"], float)[n - 1]
        (out / "images").mkdir(exist_ok=True); t0 = time.time()
        for k in range(n):
            rgb = sim.gsplat.render_rgb(camera, xv_to_T(X[k]) @ T_c2b)["rgb"]
            cv2.imwrite(str(out / "images" / f"frame_{k:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if a.compare_mp4:
            mp4 = Path(meta["policy_traj"]).parent.parent / "videos" / (Path(meta["policy_traj"]).name.replace("_deploy.pt", "_policy_rgb.mp4"))
            cap = cv2.VideoCapture(str(mp4)); ps = {-1: [], 0: [], 1: []}; vid = []
            while len(vid) < min(n, 200):
                ok, f = cap.read()
                if not ok: break
                vid.append(f.astype(np.float32))
            for k in range(5, len(vid) - 5, 10):
                for s in ps:
                    im = cv2.imread(str(out / "images" / f"frame_{k + s:05d}.png")).astype(np.float32)
                    ps[s].append(10 * np.log10(255 ** 2 / (np.mean((im - vid[k]) ** 2) + 1e-9)))
            print(f"  {out.name}: PSNR(render frame k+s vs deploy mp4 frame k): " + ", ".join(f"s={s} {np.mean(v):.2f} dB" for s, v in ps.items()))
        (out / "images.done").touch()
        print(f"render {a.yard} {out.name}: {n} frames {time.time() - t0:.0f}s", flush=True)
    print("DAGGER_RENDER_DONE", a.yard, a.tag)


def cmd_convert(a):
    """in its own process (dagger_v4_convert.py): convert_start needs SousVide's figs, this module imports FiGS-Standalone's"""
    import subprocess
    subprocess.run([sys.executable, "-u", os.path.join(FLAMBE_TERRA, "goalspec", "v4", "dagger_v4_convert.py"), a.yard, a.tag, str(ROOT)], check=True)


def cmd_labels(a):
    """write the cohort label files for converted DAgger courses + dagger_manifest.json"""
    for sub in ("vel_labels_ramp", "stop_labels", "g4_oracle", "dagger_chunk"):
        (CO / sub).mkdir(exist_ok=True)
    mp = CO / "dagger_manifest.json"; man = json.load(open(mp)) if mp.exists() else {"courses": {}}
    goals = {g["id"]: g for g in json.load(open(SY / a.yard / "goals.json"))["goals"]}
    for out in sorted((ROOT / "flights" / a.yard).glob(f"*_{a.tag}")):
        if not (out / "labels.npz").exists():
            continue
        course = f"{a.yard}_{out.name}_canon"
        if not (CO / "rollout_data" / course / "trajectories" / "001.pt").exists():
            print("not converted", course); continue
        T = np.load(out / "tXUd.npy"); N = T.shape[1]; n = N - 1; L = np.load(out / "labels.npz")
        X = T[1:11, 1:].T
        A = L["a"][:n]; C = L["c"][:n]; assert len(A) == n and C.shape == (n, K, 4), (course, A.shape, C.shape, n)
        meta = json.load(open(out / "dagger.json")); g = goals[meta["goal"]]; gy = gx.GoalYard(SY / a.yard, g)
        pts = [np.asarray(p, float) for p in g["points_ned"]]
        S = gc.stop_labels(X, L, n); O = np.stack([g4.oracle_vec(gy, pts, g["kind"], X[k]) for k in range(n)])
        np.savez(CO / "vel_labels_ramp" / f"{course}.npz", a=A)
        np.savez(CO / "dagger_chunk" / f"{course}.npz", c=C)
        np.savez(CO / "stop_labels" / f"{course}.npz", s=S)
        np.savez(CO / "g4_oracle" / f"{course}.npz", o=O)
        man["courses"][course] = {"yard": a.yard, "flight": out.name, "tag": a.tag, "frames": int(N), "kind": "door",
                                  "goal": meta["goal"], "cut_reason": meta["cut_reason"], "replans": meta["replans"],
                                  "src_course": meta["src_course"]}
        print(f"  {course}: n {n} stop frames {int(S.sum())} replans {meta['replans']} cut {meta['cut_reason']}", flush=True)
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S"); json.dump(man, open(mp, "w"), indent=1)
    print("DAGGER_LABELS_DONE", a.yard, a.tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "relabel", "render", "convert", "labels"]); ap.add_argument("yard")
    ap.add_argument("tag", nargs="?", default="d1"); ap.add_argument("flight_dir", nargs="?", default="")
    ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare_mp4", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "relabel": cmd_relabel, "render": cmd_render, "convert": cmd_convert, "labels": cmd_labels}[a.cmd](a)
