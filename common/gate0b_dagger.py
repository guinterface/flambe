"""Generate expert action chunks from states visited by the state-only student."""
import json, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
sys.path[:0] = LOCAL_MODULE_PATHS
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import gate0b_student as st

D = sys.argv[1]; L = st.land()
_CACHE = {}


def relabel_one(args):
    seed, tag, m = args
    import obs_relabel_z as oz, goal_expert_v4 as gv, gate0b_expert as ge
    z = np.load(f"{D}/tracks_student_s{seed}{tag}/{m['id']}.npz"); X, clr = z["X"], z["clr"][:len(z["X"])]
    F, Y = [], []; n_clr = n_plan = 0; errs = {}; n_fb = [0]
    for k in range(len(X)):
        if clr[k] < 0.30:
            n_clr += 1; continue
        x = X[k].copy()
        try:
            r = ge.fly_one(dict(m, pos=x[:3].tolist(), _build_only=True, _cache=_CACHE, _relabel=True))
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        if FALLBACK and (isinstance(r, dict) or r[0].pose is None) and 0.30 <= clr[k] < 0.85:
            try:
                r = fallback_plan(m, x, float(clr[k]))
            except Exception as e:
                r = {"error": f"fallback: {type(e).__name__}: {e}"}
            if not isinstance(r, dict): n_fb[0] += 1
        if isinstance(r, dict) or r[0].pose is None:
            n_plan += 1; key = (r.get("error", "no route") if isinstance(r, dict) else "no route")[:70]; errs[key] = errs.get(key, 0) + 1
            continue
        ex, cache = r[0], r[2]
        XR, GR, PH = [], [], []; u_prev = ge.U0.copy(); xr = x.copy()
        for j in range(st.K):
            oz.dv.maybe_replan(ex, xr, cache); u = ex.decide(j / 20, xr.copy())
            XR.append(xr.copy()); GR.append(np.asarray(ex._guid, float)); PH.append(gv.PHASE_ID.get(ex.phase, 6))
            xr = oz.DYN.step(xr, u_prev, u); u_prev = u
        XR, GR, PH = np.array(XR), np.array(GR), np.array(PH)
        g = np.c_[GR[:, 0], GR[:, 1], GR[:, 2] - XR[:, 2], st.wrap(GR[:, 3] - st.yaw_of(XR[:, 6:10])), (PH == 2).astype(float)]
        F.append(st.feats(x[None], L)[0]); Y.append(g.reshape(-1).astype(np.float32))
    return {"id": m["id"], "F": np.array(F, np.float32).reshape(-1, 10), "Y": np.array(Y, np.float32).reshape(-1, st.K * 5),
            "visited": int(len(X)), "dropped_clearance": n_clr, "dropped_no_plan": n_plan, "errors": errs, "fallback_used": n_fb[0]}


FALLBACK = os.environ.get("RELABEL_FALLBACK", "0") == "1"


def fallback_plan(m, x, c0):
    """Relabel fallback when RELABEL_FALLBACK=1: for a visited state with no plan and
    clearance 0.30-0.85 m, search a 0.1 m grid over the exact geometry with radius min(0.60, clearance - 0.02), floor
    0.30, from the state to the nearest reached cell (BFS order) from which the normal planner returns a route
    (candidates: reached cells with clearance >= 0.85 m, at most 12 tried); the route is that exit path (string-pulled
    with the same radius) followed by the planner's route. Returns what gate0b_expert.fly_one(_build_only) returns."""
    import collections, gate0b_expert as ge
    from score import clearance_fn
    cl = clearance_fn(st.YARD); r_eff = max(0.30, min(0.60, c0 - 0.02))
    tr = np.load(f"{st.SYR}/{st.YARD}/recon/truth.npz"); H = np.asarray(tr["height_top"], float); lo = np.asarray(tr["lo"], float)
    cell = float(tr["cell"]); zg = float(tr["z_ground"]); B, h = 6.0, 0.1
    xs = np.arange(x[0] - B, x[0] + B, h); ys = np.arange(x[1] - B, x[1] + B, h); zs = np.arange(zg - 3.0, zg - 0.4, h)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij"); C = cl(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]).reshape(XX.shape)
    ii = np.clip(((XX - lo[0]) / cell).astype(int), 0, H.shape[0] - 1); jj = np.clip(((YY - lo[1]) / cell).astype(int), 0, H.shape[1] - 1)
    ok = (C >= r_eff) & ~((ZZ < lo[2]) & ((zg - ZZ) < H[ii, jj] + 0.3))
    o = np.array([xs[0], ys[0], zs[0]]); a = tuple(np.clip(np.round((x[:3] - o) / h).astype(int), 0, np.array(ok.shape) - 1))
    if not ok[a]:
        raise RuntimeError("state not free at the fallback radius")
    prev = {a: None}; q = collections.deque([a]); tried = 0
    while q and tried < 12:
        u = q.popleft()
        if C[u] >= 0.85 and u != a:
            tried += 1; p = o + np.array(u) * h
            r = ge.fly_one(dict(m, pos=p.tolist(), _build_only=True, _cache=_CACHE, _relabel=True))
            if not isinstance(r, dict) and r[0].pose is not None:
                path = [u]
                while prev[path[-1]] is not None: path.append(prev[path[-1]])
                P = [o + np.array(v) * h for v in path[::-1]]; P[0] = x[:3].copy()
                out, i = [P[0]], 0
                while i < len(P) - 1:
                    j = len(P) - 1
                    while j > i + 1 and float(cl(P[i] + (P[j] - P[i]) * np.linspace(0, 1, max(2, int(np.linalg.norm(P[j] - P[i]) / 0.05)))[:, None]).min()) < r_eff:
                        j -= 1
                    out.append(P[j]); i = j
                ex = r[0]; ex.route = out[:-1] + [np.asarray(v, float) for v in ex.route]; ex.vv = ex._vertex_speeds()
                PP = np.array(ex.route, float); ex.nom_P = PP; Ls = np.linalg.norm(np.diff(PP[:, :2], axis=0), axis=1); ex.nom_s = np.concatenate([[0.0], np.cumsum(Ls)])
                ex.reset(); return r
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            v = (u[0] + d[0], u[1] + d[1], u[2] + d[2])
            if v not in prev and 0 <= v[0] < ok.shape[0] and 0 <= v[1] < ok.shape[1] and 0 <= v[2] < ok.shape[2] and ok[v]:
                prev[v] = u; q.append(v)
    raise RuntimeError("no planner route from any reachable free cell")


def init():
    import obs_relabel_z as oz, score
    oz.YARD = st.YARD; oz.init_worker(); score.SY = FLAMBE_YARDS


def main():
    seed, tag, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    import obs_relabel_z, score, torch
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    S = [s for s in json.load(open(f"{D}/starts_syn013_400r.json"))["starts"] if s["split"] == "train" and s["id"] not in st.EXCLUDE]
    if os.environ.get("SUBSET"): S = S[:int(os.environ["SUBSET"])]
    t0 = time.time(); R = []
    with ProcessPoolExecutor(NP, mp_context=get_context("fork"), initializer=init) as p:
        for i, r in enumerate(p.map(relabel_one, [(seed, tag, s) for s in S], chunksize=1)):
            R.append(r)
            if i % 20 == 0:
                print(f"[dagger s{seed}{tag}] {i + 1}/{len(S)} {r['id']}: kept {len(r['F'])}/{r['visited']} ({time.time() - t0:.0f}s)", flush=True)
    F = np.concatenate([r["F"] for r in R]); Y = np.concatenate([r["Y"] for r in R])
    vis = sum(r["visited"] for r in R); dc = sum(r["dropped_clearance"] for r in R); dp = sum(r["dropped_no_plan"] for r in R)
    errs = {}
    for r in R:
        for k, v in r["errors"].items(): errs[k] = errs.get(k, 0) + v
    np.savez_compressed(out, F=F, Y=Y, ids=np.array([r["id"] for r in R]))
    log = {"seed": seed, "flight_tag": tag, "flights": len(R), "visited": vis, "kept": int(len(F)), "dropped_clearance_lt_0.30": dc,
           "dropped_no_plan": dp, "no_plan_reasons": errs, "fallback": FALLBACK, "fallback_used": sum(r.get("fallback_used", 0) for r in R),
           "wall_s": round(time.time() - t0)}
    json.dump(log, open(out.replace(".npz", ".json"), "w"), indent=1); print("DAGGER_RELABEL_DONE", json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
