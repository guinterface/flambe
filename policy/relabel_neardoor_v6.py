"""Regenerate expert labels near the delivery point and merge the results into training datasets."""
import json, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_GATE2 = os.environ.get("FLAMBE_GATE2", os.path.join(FLAMBE_ROOT, "research", "02_experiments", "gate2"))
sys.path[:0] = LOCAL_MODULE_PATHS

G = FLAMBE_GATE2
SY = os.path.join(FLAMBE_YARDS, "yards")
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
SY2 = os.path.join(FLAMBE_YARDS_FALLBACK, "yards")
K = 20; R_NEAR = 1.5
XSY = ["syn_034", "syn_035", "syn_036", "syn_037"]; YMAP = {y: y + "_v4" for y in XSY}
CK0 = f"{G}/r78/round0/g2base_student_s0_ep08.pt"
DAG = [f"{G}/r78/round1/s0/fly/dag_s0.npz", f"{G}/r78/round1/s0/fly/rec_s0.npz", f"{G}/f82/n16_v1/s0/fly/dag_s0.npz", f"{G}/f82/n16_v1/s0/fly/rec_s0.npz"]
TAGS = ["r78r1_dag", "r78r1_rec", "f82n16_dag", "f82n16_rec"]


def door(y):
    p = f"{SY}/{y}/goals.json" if os.path.exists(f"{SY}/{y}/goals.json") else f"{SY2}/{y}/goals.json"
    return [g for g in json.load(open(p))["goals"] if g["id"] == "door"][0]


def stream_yards():
    import torch
    return list(torch.load(CK0, map_location="cpu", weights_only=False)["yards"]) + XSY


def stream_ids(y):
    gy = YMAP.get(y, y); E = {r["id"]: r for r in json.load(open(f"{G}/{gy}/expert.json"))["flights"]}
    return gy, [m["id"] for m in json.load(open(f"{G}/{gy}/starts.json"))["starts"] if m["split"] == "train"
                and E.get(m["id"], {}).get("outcome") == "success" and E[m["id"]].get("min_clearance_m", 0) >= 0.30]


def lead(OUT):
    os.makedirs(OUT, exist_ok=True); D, Z = [], []
    for y in stream_yards():
        hz = float(door(y)["hover_ned"][2]); gy, ids = stream_ids(y)
        for i in ids:
            t = np.load(f"{G}/{gy}/tracks/{i}.npz"); m = t["PH"] == 2
            if m.any(): D.append(t["X"][m, 2] - hz); Z.append(t["G"][m, 2] - t["X"][m, 2])
    D, Z = np.concatenate(D), np.concatenate(Z); b = np.floor(D / 0.05).astype(int); T = {}
    for k in np.unique(b): T[int(k)] = {"depth_lo_m": round(k * 0.05, 3), "median_dz_m": float(np.median(Z[b == k])), "n": int((b == k).sum())}
    json.dump({"bin_m": 0.05, "bins": T, "all_median_dz_m": float(np.median(Z)), "frames": int(len(Z))}, open(f"{OUT}/lead_table.json", "w"), indent=1)
    print("NEARDOOR_LEAD " + json.dumps({"frames": int(len(Z)), "all_median": round(float(np.median(Z)), 4),
                                         "by_depth": {v["depth_lo_m"]: round(v["median_dz_m"], 4) for v in T.values() if v["n"] >= 100}}), flush=True)


_CACHE = {}; _LT = None


def winit(yard, lt):
    global _LT
    os.environ["YARD"] = yard; _LT = lt
    import lead_expert_yard as ge, lead_phase_seed, obs_relabel_z as oz, score
    assert ge.YARD == yard
    if not getattr(ge, "_phase_seed", False):
        orig = ge.fly_one
        def fo(m):
            r = orig(m)
            if m.get("_relabel") and m.get("_build_only") and not isinstance(r, dict): lead_phase_seed.seed(r[0], m["pos"])
            return r
        ge.fly_one = fo; ge._phase_seed = True
    if os.environ.get("SEED_XY"): lead_phase_seed.R_XY = float(os.environ["SEED_XY"])
    oz.YARD = yard; oz.init_worker(); score.SY = FLAMBE_YARDS


def lead_at(depth):
    k = str(int(np.floor(depth / 0.05))); B = _LT["bins"]
    if k in B and B[k]["n"] >= 100: return B[k]["median_dz_m"]
    ks = [int(q) for q in B if B[q]["n"] >= 100]; q = min(ks, key=lambda v: abs(v - int(k)))   # nearest populated bin
    return B[str(q)]["median_dz_m"]


def relabel_one(x):
    import obs_relabel_z as oz, goal_expert_v4 as gv, lead_expert_yard as ge, gate0b_student as st
    assert gv.GATE, "goal_expert_v4 is not the gate file with GATE=1"
    x = np.asarray(x, float)
    try:
        r = ge.fly_one(dict(id="nd", split="train", pos=x[:3].tolist(), yaw=0.0, _build_only=True, _relabel=True, _cache=_CACHE))
    except Exception as e:
        return None, f"{type(e).__name__}"[:40]
    if isinstance(r, dict) or r[0].pose is None: return None, "no route"
    ex, cache = r[0], r[2]
    if os.environ.get("SEED_VZ") and ex.phase == "approach":
        import lead_phase_seed as ps
        hv, ld = np.asarray(ex.pose["hover"], float), np.asarray(ex.pose["land"], float)
        if np.linalg.norm(x[:2] - ld[:2]) < ps.R_XY and x[2] >= hv[2] and x[5] > float(os.environ["SEED_VZ"]):
            ex.phase = "descent"; ex.z_set = float(x[2]); ex._seed_vz = True
    if os.environ.get("SEED_HOLD") == "1" and ex.phase == "approach":                # hold-counter fix
        hv = np.asarray(ex.pose["hover"], float); herr = float(st.wrap(float(ex.pose["yaw"]) - float(st.yaw_of(x[None, 6:10])[0])))
        if np.linalg.norm(x[:2] - hv[:2]) < 0.4 and abs(x[2] - hv[2]) < 0.15 and np.linalg.norm(x[3:6]) < 0.2 and abs(herr) < 0.5:
            ex.hold = gv.HOLD_STEPS - 1; ex._seed_hold = True
    seeded = ex.phase + ("_vz" if getattr(ex, "_seed_vz", False) else "") + ("_hold" if getattr(ex, "_seed_hold", False) else "")
    if ex.phase == "descent":
        hz, lz = float(ex.pose["hover"][2]), float(ex.pose["land"][2])
        ex.z_set = min(lz, float(x[2]) + lead_at(float(x[2]) - hz) - gv.ve.V_DESCENT / gv.ve.HZ_CTL)
    XR, GR, PH = [], [], []; u_prev = ge.U0.copy(); xr = x.copy()
    for j in range(K):
        oz.dv.maybe_replan(ex, xr, cache); u = ex.decide(j / 20, xr.copy())
        XR.append(xr.copy()); GR.append(np.asarray(ex._guid, float)); PH.append(gv.PHASE_ID.get(ex.phase, 6))
        xr = oz.DYN.step(xr, u_prev, u); u_prev = u
    XR, GR, PH = np.array(XR), np.array(GR), np.array(PH)
    g = np.c_[GR[:, 0], GR[:, 1], GR[:, 2] - XR[:, 2], st.wrap(GR[:, 3] - st.yaw_of(XR[:, 6:10])), (PH == 2).astype(float)]
    return g.reshape(-1).astype(np.float32), seeded


def yard(y, OUT, NP=8, MAXN=0):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); L = np.asarray(door(y)["land_ned"], float); lt = json.load(open(f"{OUT}/lead_table.json"))
    S = []                                                          # stream near states (id, k, x)
    if y in stream_yards():
        gy, ids = stream_ids(y)
        for i in ids:
            X = np.load(f"{G}/{gy}/tracks/{i}.npz")["X"].astype(np.float64)
            for k in np.flatnonzero(np.linalg.norm(X[:, :3] - L, axis=1) <= R_NEAR): S.append((i, int(k), X[k]))
    Dg = []                                                         # DAG near rows (file index, row, x)
    for fi, p in enumerate(DAG):
        z = np.load(p); m = np.flatnonzero((z["yard"].astype(str) == y) & (np.linalg.norm(z["X"][:, :3].astype(np.float64) - L, axis=1) <= R_NEAR))
        Dg += [(fi, int(r), z["X"][r].astype(np.float64)) for r in m]
    if MAXN: S, Dg = S[:MAXN], Dg[:MAXN]
    xs = [s[2] for s in S] + [d[2] for d in Dg]
    with ProcessPoolExecutor(NP, mp_context=get_context("spawn"), initializer=winit, initargs=(y, lt)) as p:
        R = list(p.map(relabel_one, xs, chunksize=16))
    ok = [r[0] is not None for r in R]; why = {}; seeded = {}
    for r in R:
        if r[0] is None: why[r[1]] = why.get(r[1], 0) + 1
        else: seeded[r[1]] = seeded.get(r[1], 0) + 1
    oldY = lambda fi, row: np.load(DAG[fi])["Y"][row]
    sY = np.stack([R[i][0] for i in range(len(S)) if ok[i]]) if any(ok[:len(S)]) else np.zeros((0, K * 5), np.float32)
    keepS = [i for i in range(len(S)) if ok[i]]; keepD = [j for j in range(len(Dg)) if ok[len(S) + j]]
    dY = np.stack([R[len(S) + j][0] for j in keepD]) if keepD else np.zeros((0, K * 5), np.float32)
    np.savez(f"{OUT}/{y}.tmp.npz", sid=np.array([S[i][0] for i in keepS]), sk=np.array([S[i][1] for i in keepS], np.int32), sY=sY,
             fi=np.array([Dg[j][0] for j in keepD], np.int32), row=np.array([Dg[j][1] for j in keepD], np.int64), dY=dY)
    os.replace(f"{OUT}/{y}.tmp.npz", f"{OUT}/{y}.npz"); w = time.time() - t0
    info = {"yard": y, "stream_states": len(S), "dag_states": len(Dg), "relabelled": int(sum(ok)), "failed_kept_old": why, "seeded_phase": seeded,
            "descent_step0_share": float(np.mean([r[0][4] > 0.5 for r in R if r[0] is not None])) if any(ok) else None,
            "dz_step0_median_descent": float(np.median([r[0][2] for r in R if r[0] is not None and r[0][4] > 0.5])) if any(r[0] is not None and r[0][4] > 0.5 for r in R) else None,
            "wall_s": round(w, 1), "states_per_s": round(len(xs) / max(w, 1e-6), 2), "NP": NP, "MAXN": MAXN}
    json.dump(info, open(f"{OUT}/{y}.json", "w"), indent=1); print("NEARDOOR_YARD " + json.dumps(info), flush=True)


def merge(OUT, DST):
    os.makedirs(DST, exist_ok=True); parts = sorted(f for f in os.listdir(OUT) if f.startswith("syn_") and f.endswith(".npz")); ov = {"yard": [], "id": [], "k": [], "Y": []}
    for fi, p in enumerate(DAG):
        z = np.load(p); Y = z["Y"].copy(); n = 0
        for f in parts:
            q = np.load(f"{OUT}/{f}"); m = q["fi"] == fi; Y[q["row"][m]] = q["dY"][m]; n += int(m.sum())
        changed = np.flatnonzero((Y != z["Y"]).any(1)); rel = set()
        for f in parts:
            q = np.load(f"{OUT}/{f}"); rel |= set(q["row"][q["fi"] == fi].tolist())
        assert set(changed.tolist()) <= rel, f"{TAGS[fi]}: rows changed outside the relabelled set"
        far = np.setdiff1d(np.arange(len(Y)), np.array(sorted(rel), np.int64)); assert np.array_equal(Y[far], z["Y"][far]), "far rows not byte-identical"
        sub, nm = TAGS[fi].rsplit("_", 1); os.makedirs(f"{DST}/{sub}", exist_ok=True)                 # <run tag>/dag_s0.npz | rec_s0.npz
        np.savez_compressed(f"{DST}/{sub}/{nm}_s0.npz", X=z["X"], Y=Y, yard=z["yard"], flight=z["flight"])
        print(f"[neardoor merge] {TAGS[fi]}: {n} rows relabelled, {len(changed)} changed, {len(far)} far rows byte-identical", flush=True)
    for f in parts:
        q = np.load(f"{OUT}/{f}"); ov["yard"] += [f[:-4]] * len(q["sid"]); ov["id"] += q["sid"].tolist(); ov["k"] += q["sk"].tolist(); ov["Y"].append(q["sY"])
    np.savez_compressed(f"{DST}/stream_override.npz", yard=np.array(ov["yard"]), id=np.array(ov["id"]), k=np.array(ov["k"], np.int32),
                        Y=np.concatenate(ov["Y"]) if ov["Y"] else np.zeros((0, K * 5), np.float32))
    print("NEARDOOR_MERGE_DONE " + json.dumps({"stream_override_rows": len(ov["id"]), "yards": len(parts)}), flush=True)


def pool(FD, OUT, NP=8):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    os.makedirs(OUT, exist_ok=True); lt = json.load(open(f"{OUT}/lead_table.json")); F = json.load(open(f"{FD}/fly_s0.json"))["flights"]
    for y in sorted(set(f["yard"] for f in F)):
        if os.path.exists(f"{OUT}/pool_{y}.json"): continue
        t0 = time.time(); ids = sorted(f["id"] for f in F if f["yard"] == y); X, FL = [], []
        for i in ids:
            p = f"{FD}/tracks_s0/{y}/{i}.npz"
            if not os.path.exists(p): continue
            x = np.load(p)["X"].astype(np.float64); X.append(x); FL += [i] * len(x)
        X = np.concatenate(X) if X else np.zeros((0, 10)); 
        with ProcessPoolExecutor(NP, mp_context=get_context("spawn"), initializer=winit, initargs=(y, lt)) as p:
            R = list(p.map(relabel_one, list(X), chunksize=16))
        ok = np.array([r[0] is not None for r in R], bool); why, seeded = {}, {}
        for r in R:
            if r[0] is None: why[r[1]] = why.get(r[1], 0) + 1
            else: seeded[r[1]] = seeded.get(r[1], 0) + 1
        Y = np.stack([r[0] for r in R if r[0] is not None]) if ok.any() else np.zeros((0, K * 5), np.float32)
        np.savez(f"{OUT}/pool_{y}.tmp.npz", X=X[ok].astype(np.float32), Y=Y, fid=np.array(FL)[ok] if len(FL) else np.array([], str))
        os.replace(f"{OUT}/pool_{y}.tmp.npz", f"{OUT}/pool_{y}.npz")
        info = {"yard": y, "flights": len(ids), "states": int(len(X)), "relabelled": int(ok.sum()), "failed_dropped": why, "seeded_phase": seeded,
                "descent_step0_share": float(np.mean(Y[:, 4] > 0.5)) if len(Y) else None, "wall_s": round(time.time() - t0, 1), "NP": NP,
                "SEED_HOLD": os.environ.get("SEED_HOLD", "unset"), "SEED_VZ": os.environ.get("SEED_VZ", "unset"), "SEED_XY": os.environ.get("SEED_XY", "unset")}
        json.dump(info, open(f"{OUT}/pool_{y}.json", "w"), indent=1); print("C13_POOL_YARD " + json.dumps(info), flush=True)
    print("C13_POOL_DONE", flush=True)


def poolmerge(OUT, DST, LINK="../../fly/tracks_s0"):
    parts = sorted(f for f in os.listdir(OUT) if f.startswith("pool_syn_") and f.endswith(".npz")); X, Y, YD, FL, key = [], [], [], [], {}
    for f in parts:
        q = np.load(f"{OUT}/{f}"); y = f[5:-4]
        X.append(q["X"]); Y.append(q["Y"]); YD += [y] * len(q["X"]); FL += [key.setdefault((y, i), len(key)) for i in q["fid"].tolist()]
    os.makedirs(f"{DST}/c13own", exist_ok=True)
    np.savez_compressed(f"{DST}/c13own/dag_s0.npz", X=np.concatenate(X).astype(np.float32), Y=np.concatenate(Y).astype(np.float32), yard=np.array(YD), flight=np.array(FL, np.int32))
    inv = sorted(key.items(), key=lambda kv: kv[1]); assert [v for _, v in inv] == list(range(len(inv)))
    json.dump({"per_flight": [{"yard": y, "id": i} for (y, i), _ in inv], "source": OUT}, open(f"{DST}/c13own/dag_s0.json", "w"), indent=0)
    lk = f"{DST}/c13own/tracks_s0"
    if not os.path.islink(lk): os.symlink(LINK, lk)                                                                        # relative link
    assert os.readlink(lk) == LINK, f"tracks_s0 link {os.readlink(lk)} != {LINK}"
    print("C13_POOLMERGE_DONE " + json.dumps({"rows": len(YD), "yards": len(parts), "flights": len(key), "tracks_link": LINK}), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "lead": lead(a[1])
    elif a[0] == "yard": yard(a[1], a[2], int(a[3]) if len(a) > 3 else 8, int(a[4]) if len(a) > 4 else 0)
    elif a[0] == "merge": merge(a[1], a[2])
    elif a[0] == "pool": pool(a[1], a[2], int(a[3]) if len(a) > 3 else 8)
    elif a[0] == "poolmerge": poolmerge(a[1], a[2], *(a[3:4]))
    elif a[0] == "yards":
        for y in a[1].split(","):
            if not os.path.exists(f"{a[2]}/{y}.json"): yard(y, a[2], int(a[3]) if len(a) > 3 else 8)
        print("NEARDOOR_YARDS_DONE " + a[1], flush=True)
