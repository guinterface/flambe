"""Collect student flights, relabel them with expert guidance, and fine-tune the policy."""
import json, math, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
sys.path[:0] = LOCAL_MODULE_PATHS
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
FAKE = os.environ.get("FAKE_RENDER") == "1"
DEV = os.environ.get("DEV", "cuda")
COPY = ["syn_002", "syn_013", "syn_025", "syn_040"]


def yards():
    import gate2_multi as g2
    Y = os.environ["YARDS"].split(",") if os.environ.get("YARDS") else list(g2.TRAIN_YARDS)
    assert not set(Y) & set(g2.HELD_OUT), "held-out yard"
    return Y


def frame(yard, x):
    if FAKE: return np.full((3, 224, 224), 128, np.uint8)
    import gate2_multi as g2
    return g2.policy_frame(yard, x)


def train_starts(G, yard):
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    S = [dict(s, t_expert_s=E[s["id"]]["t_landed_s"]) for s in json.load(open(f"{G}/{yard}/starts.json"))["starts"]
         if s["split"] == "train" and E.get(s["id"], {}).get("outcome") == "success"]
    return S[:int(os.environ["FLY_N"])] if os.environ.get("FLY_N") else S


# ---------------------------------------------------------------- fly
_M = {}


def fly_one(args):
    """gate2_multi.fly_one (base checkpoints: no aux), with the frame from frame() and every track kept (states only)"""
    G, yard, ck, s, td = args
    p = f"{td}/{yard}/{s['id']}.npz"
    if os.path.exists(p[:-4] + ".json"): return dict(json.load(open(p[:-4] + ".json")), resumed=True)
    import torch, obs_relabel_z as oz, goal_expert_v4 as gv, gate1_student as g1, gate0b_student as st, gate2_multi as g2
    from score import score_track, clearance_fn
    from gate0b_expert import _yaw_quat, U0
    if ck not in _M:
        c = torch.load(ck, map_location="cpu", weights_only=False); assert not c.get("aux"), "base checkpoints only"
        m = g1.build(int(c.get("seed", 0))); m.load_state_dict(c["state"]); _M[ck] = (m.to(DEV).eval(), c)
    m, c = _M[ck]
    L = g2.land(yard); ctl = gv.DirectGoalExpert.__new__(gv.DirectGoalExpert); ctl.prev_oz = 0.0
    x = np.r_[np.asarray(s["pos"], float), np.zeros(3), _yaw_quat(s["yaw"])]; u_prev = U0.copy()
    budget = 1.5 * s["t_expert_s"]; N = int(round(budget * 20)); K = g1.K; w = np.exp(-g1.CHUNK_M * np.arange(K)); hist = []
    P, V, T, LX = [x[:3].copy()], [x[3:6].copy()], [0.0], []
    for k in range(N):
        with torch.no_grad():
            cc = m(torch.tensor(g1.prep(frame(yard, x)[None]), device=DEV)).cpu().numpy()[0] * c["sy"] + c["my"]
        hist.append(cc.reshape(K, 5)); hist = hist[-K:]
        preds = np.array([hist[-1 - j][j] for j in range(len(hist))]); ww = w[:len(preds)][::-1]; g = (preds * ww[:, None]).sum(0) / ww.sum()
        yaw = float(st.yaw_of(x[6:10]))
        u, _ = gv.DirectGoalExpert._u(ctl, x[0:3], x[3:6], x[6:10], g[0:2], float(x[2] + g[2]), float(st.wrap(yaw + g[3])), w_clip=0.3 if g[4] > 0.5 else 0.4)
        LX.append(x.copy()); x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / 20)
    P, V, T = np.array(P), np.array(V), np.array(T); clr = clearance_fn(yard)(P); r = score_track(T, P, V, L, clr, budget)
    os.makedirs(f"{td}/{yard}", exist_ok=True)
    np.savez_compressed(p[:-4] + ".tmp.npz", X=np.array(LX), clr=clr[:len(LX)]); os.replace(p[:-4] + ".tmp.npz", p)
    r = {k: (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, (np.integer, np.bool_)) else v)) for k, v in r.items()}
    r.update({"id": s["id"], "yard": yard, "split": s["split"], "frames": len(LX), "fake_render": FAKE})
    json.dump(r, open(p[:-4] + ".json", "w"), default=float); return r


def fly_winit():
    import obs_relabel_z as oz, score, torch
    oz.init_worker(); score.SY = FLAMBE_YARDS
    if DEV == "cpu": torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))


def fly(G, seed, ck, OUT, NP):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    td = f"{OUT}/tracks_s{seed}"; os.makedirs(td, exist_ok=True)
    YS = yards(); TOT = int(os.environ.get("FLY_TOTAL", "0")); J = []
    for i, y in enumerate(YS):
        S = train_starts(G, y)
        if TOT:                                            # Distribute starts over the yards, evenly spaced within each yard's train starts
            n = TOT // len(YS) + (i < TOT % len(YS)); S = [S[j] for j in np.unique(np.linspace(0, len(S) - 1, n).round().astype(int))]
        J += [(G, y, ck, s, td) for s in S]
    print(f"[g2 dagger fly s{seed}] {ck}: {len(J)} train starts over {len(set(j[1] for j in J))} yards, {NP} processes, dev {DEV}"
          f"{' FAKE_RENDER (grey frames: not a result)' if FAKE else ''}", flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(NP, mp_context=get_context("spawn"), initializer=fly_winit) as p:
        for i, r in enumerate(p.map(fly_one, J, chunksize=1)):
            res.append(r)
            if i % 50 == 0 or len(J) < 50:
                print(f"[g2 dagger fly s{seed}] {i + 1}/{len(J)} {r['yard']} {r['id']} {r['outcome']} {r.get('frames')} frames ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"ckpt": ck, "fake_render": FAKE, "flights": res, "wall_s": round(time.time() - t0)}, open(f"{OUT}/fly_s{seed}.json", "w"), default=float)
    k = {o: sum(r["outcome"] == o for r in res) for o in ("success", "contact", "overshoot", "timeout")}
    print(f"G2_DAGGER_FLY s{seed}: {len(res)} train starts {k}, {sum(r.get('frames', 0) for r in res)} states ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- relabel
_CACHE = {}


def rl_winit(yard):
    os.environ["YARD"] = yard
    import lead_expert_yard as ge                         # YARD read at import: one pool per yard
    import lead_phase_seed, obs_relabel_z as oz, score
    assert ge.YARD == yard
    if not getattr(ge, "_phase_seed", False):             # lead_phase_seed.install, applied to the yard-generic expert
        orig = ge.fly_one
        def fo(m):
            r = orig(m)
            if m.get("_relabel") and m.get("_build_only") and not isinstance(r, dict): lead_phase_seed.seed(r[0], m["pos"])
            return r
        ge.fly_one = fo; ge._phase_seed = True
    oz.YARD = yard; oz.init_worker(); score.SY = FLAMBE_YARDS


def relabel_one(args):
    """gate0b_dagger.relabel_one (FALLBACK off) for one yard; returns the kept indices instead of feature rows"""
    td, yard, m = args
    import obs_relabel_z as oz, goal_expert_v4 as gv, lead_expert_yard as ge, gate0b_student as st
    z = np.load(f"{td}/{yard}/{m['id']}.npz"); X, clr = z["X"], z["clr"][:len(z["X"])]
    idx, Y = [], []; n_clr = n_plan = 0; errs = {}; nr = []
    for k in range(len(X)):
        if clr[k] < 0.30:
            n_clr += 1; continue
        x = X[k].copy()
        try:
            r = ge.fly_one(dict(m, pos=x[:3].tolist(), _build_only=True, _cache=_CACHE, _relabel=True))
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(r, dict) or r[0].pose is None:
            n_plan += 1; key = (r.get("error", "no route") if isinstance(r, dict) else "no route")[:70]; errs[key] = errs.get(key, 0) + 1
            nr.append([k] + [round(float(v), 2) for v in x[:3]] + [round(float(clr[k]), 2), key]); continue   # every drop, state index + NED position + clearance
        ex, cache = r[0], r[2]
        XR, GR, PH = [], [], []; u_prev = ge.U0.copy(); xr = x.copy()
        for j in range(st.K):
            oz.dv.maybe_replan(ex, xr, cache); u = ex.decide(j / 20, xr.copy())
            XR.append(xr.copy()); GR.append(np.asarray(ex._guid, float)); PH.append(gv.PHASE_ID.get(ex.phase, 6))
            xr = oz.DYN.step(xr, u_prev, u); u_prev = u
        XR, GR, PH = np.array(XR), np.array(GR), np.array(PH)
        g = np.c_[GR[:, 0], GR[:, 1], GR[:, 2] - XR[:, 2], st.wrap(GR[:, 3] - st.yaw_of(XR[:, 6:10])), (PH == 2).astype(float)]
        idx.append(k); Y.append(g.reshape(-1).astype(np.float32))
    return {"id": m["id"], "yard": yard, "idx": np.array(idx, int), "Y": np.array(Y, np.float32).reshape(-1, st.K * 5),
            "visited": int(len(X)), "dropped_clearance": n_clr, "dropped_no_plan": n_plan, "errors": errs, "noroute": nr}


def relabel(G, seed, OUT, NP):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    td = f"{OUT}/tracks_s{seed}"; PD = f"{OUT}/relabel_s{seed}_parts"; os.makedirs(PD, exist_ok=True)
    def part_ok(y, i):
        try:
            with np.load(f"{PD}/{y}__{i}.npz") as z: z["X"], z["Y"], z["R"]
            return True
        except Exception: return False
    S = [(y, s) for y in yards() for s in train_starts(G, y) if os.path.exists(f"{td}/{y}/{s['id']}.json")]   # flown starts only
    done = {(y, s["id"]) for y, s in S if part_ok(y, s["id"])}
    print(f"[g2 relabel s{seed}] parts cache {PD}: {len(done)} of {len(S)} flights already done, {len(S) - len(done)} to relabel; {NP} processes per yard", flush=True)
    t0 = time.time(); n = len(done)
    for y in dict.fromkeys(y for y, _ in S):
        todo = [s for yy, s in S if yy == y and (y, s["id"]) not in done]
        if not todo: continue
        with ProcessPoolExecutor(min(NP, len(todo)), mp_context=get_context("spawn"), initializer=rl_winit, initargs=(y,)) as p:
            for r in p.map(relabel_one, [(td, y, s) for s in todo], chunksize=1):
                X = np.load(f"{td}/{y}/{r['id']}.npz")["X"]
                row = {k: r[k] for k in ("id", "yard", "visited", "dropped_clearance", "dropped_no_plan", "errors", "noroute")} | {"kept": int(len(r["idx"]))}
                tmp = f"{PD}/{y}__{r['id']}.tmp.npz"; np.savez(tmp, X=X[r["idx"]].astype(np.float32), Y=r["Y"], R=json.dumps(row, default=int))
                os.replace(tmp, f"{PD}/{y}__{r['id']}.npz"); n += 1
                if n % 100 == 0 or len(S) < 50: print(f"[g2 relabel s{seed}] {n}/{len(S)} {y} {r['id']} kept {row['kept']}/{row['visited']} ({time.time() - t0:.0f}s)", flush=True)
    Xs, Ys, Yd, R = [], [], [], []
    for y, s in S:                                         # S order, with or without a restart
        with np.load(f"{PD}/{y}__{s['id']}.npz") as z: Xs.append(z["X"]); Ys.append(z["Y"]); R.append(json.loads(str(z["R"])))
        Yd += [y] * len(Xs[-1])
    out = f"{OUT}/dag_s{seed}.npz"; X = np.concatenate(Xs).astype(np.float32); Y = np.concatenate(Ys).astype(np.float32)
    np.savez_compressed(out[:-4] + ".tmp.npz", X=X, Y=Y, yard=np.array(Yd), flight=np.concatenate([np.full(len(a), i, np.int32) for i, a in enumerate(Xs)]))
    os.replace(out[:-4] + ".tmp.npz", out)
    log = {"seed": seed, "flights": len(R), "yards": len(set(r["yard"] for r in R)), "visited": sum(r["visited"] for r in R), "kept": int(len(X)),
           "dropped_clearance_lt_0.30": sum(r["dropped_clearance"] for r in R), "dropped_no_plan": sum(r["dropped_no_plan"] for r in R),
           "wall_s": round(time.time() - t0), "per_yard": per_yard(R), "per_flight": R}
    json.dump(log, open(out[:-4] + ".json", "w"))
    for y, v in log["per_yard"].items(): print(f"[g2 relabel s{seed}] {y}: {v}", flush=True)
    print("G2_DAGGER_RELABEL", json.dumps({k: v for k, v in log.items() if k not in ("per_flight", "per_yard")}), flush=True)


def per_yard(R):
    """drop shares per yard"""
    out = {}
    for r in R:
        o = out.setdefault(r["yard"], {"flights": 0, "visited": 0, "kept": 0, "dropped_clearance": 0, "dropped_no_route": 0})
        o["flights"] += 1; o["visited"] += r["visited"]; o["kept"] += r["kept"]; o["dropped_clearance"] += r["dropped_clearance"]; o["dropped_no_route"] += r["dropped_no_plan"]
    for o in out.values(): o["no_route_share"] = round(o["dropped_no_route"] / max(1, o["visited"]), 4); o["clearance_share"] = round(o["dropped_clearance"] / max(1, o["visited"]), 4)
    return out


def stream_ids(G, y):
    """Select evenly spaced training flights using yard_data train ids at index 0/40/80/120/160"""
    E = {r["id"]: r for r in json.load(open(f"{G}/{y}/expert.json"))["flights"]}
    ids = [q["id"] for q in json.load(open(f"{G}/{y}/starts.json"))["starts"] if q["split"] == "train"
           and E.get(q["id"], {}).get("outcome") == "success" and E[q["id"]].get("min_clearance_m", 0) >= 0.30]
    return [ids[i] for i in (0, 40, 80, 120, 160) if i < len(ids)]


# ---------------------------------------------------------------- labelfloor
def labelfloor(G, CK, OUT, NP):
    """relabel the gate 2 rule expert stream flights (5 per yard; YARDS default the 4 copy yards) through relabel_one and score the
    relabel labels against the logged labels (gate0b_student.labels): 1 - R^2 with the checkpoint's my, sy (as olcheck) and first-step
    speed relabel / logged = the label-noise floor for olcheck's DAgger number. Parts cache <out>/floor_parts/<yard>__<id>.npz."""
    import torch, gate0b_student as st
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    YS = os.environ["YARDS"].split(",") if os.environ.get("YARDS") else COPY
    c = torch.load(CK, map_location="cpu", weights_only=False); my, sy = np.asarray(c["my"]), np.asarray(c["sy"]); K = st.K
    td, PD = f"{OUT}/floor_tracks", f"{OUT}/floor_parts"; os.makedirs(PD, exist_ok=True); t0 = time.time(); rows, R = {}, []
    sp = lambda A: float(np.linalg.norm(A.reshape(-1, K, 5)[:, 0, :2], axis=1).sum())
    for y in YS:
        import score; score.SY = FLAMBE_YARDS
        from score import clearance_fn
        SS = {q["id"]: q for q in json.load(open(f"{G}/{y}/starts.json"))["starts"]}; ids = stream_ids(G, y); os.makedirs(f"{td}/{y}", exist_ok=True)
        for i in ids:
            if not os.path.exists(f"{td}/{y}/{i}.npz"):
                X = np.load(f"{G}/{y}/tracks/{i}.npz")["X"]; np.savez(f"{td}/{y}/{i}.tmp.npz", X=X, clr=clearance_fn(y)(X[:, :3])); os.replace(f"{td}/{y}/{i}.tmp.npz", f"{td}/{y}/{i}.npz")
        todo = [i for i in ids if not os.path.exists(f"{PD}/{y}__{i}.npz")]
        print(f"[g2 labelfloor] {y}: {len(ids)} stream flights {ids}, {len(ids) - len(todo)} cached, {len(todo)} to relabel", flush=True)
        if todo:
            with ProcessPoolExecutor(min(NP, len(todo)), mp_context=get_context("spawn"), initializer=rl_winit, initargs=(y,)) as p:
                for r in p.map(relabel_one, [(td, y, SS[i]) for i in todo], chunksize=1):
                    row = {k: r[k] for k in ("id", "yard", "visited", "dropped_clearance", "dropped_no_plan", "errors", "noroute")} | {"kept": int(len(r["idx"]))}
                    np.savez(f"{PD}/{y}__{r['id']}.tmp.npz", idx=r["idx"], Y=r["Y"], R=json.dumps(row, default=int)); os.replace(f"{PD}/{y}__{r['id']}.tmp.npz", f"{PD}/{y}__{r['id']}.npz")
        a = [0.0] * 5
        for i in ids:
            with np.load(f"{PD}/{y}__{i}.npz") as z: idx, Yr, row = z["idx"], z["Y"].astype(np.float64), json.loads(str(z["R"]))
            t = np.load(f"{G}/{y}/tracks/{i}.npz"); Yl = st.labels(t["X"], t["G"], t["PH"]).astype(np.float64)[idx]; R.append(row)
            for j, v in enumerate((float((((Yr - Yl) / sy) ** 2).sum()), float((((my - Yl) / sy) ** 2).sum()), len(Yl), sp(Yr), sp(Yl))): a[j] += v
        rows[y] = {"one_minus_r2": a[0] / a[1], "states": a[2], "speed_relabel_over_logged": a[3] / a[4], "_s": a}
        print(f"[g2 labelfloor] {y}: relabel vs logged 1-R2 {a[0] / a[1]:.3f}, speed {a[3] / a[4]:.2f} ({a[2]} states)", flush=True)
    P = [sum(r["_s"][j] for r in rows.values()) for j in range(5)]
    for r in rows.values(): r.pop("_s")
    rep = {"ckpt_stats": CK, "yards": rows, "per_yard_drops": per_yard(R), "per_flight": R, "wall_s": round(time.time() - t0),
           "pooled": {"one_minus_r2": P[0] / P[1], "states": P[2], "speed_relabel_over_logged": P[3] / P[4]}}
    json.dump(rep, open(f"{OUT}/labelfloor.json", "w"))
    for y, v in rep["per_yard_drops"].items(): print(f"[g2 labelfloor] drops {y}: {v}", flush=True)
    print("G2_LABELFLOOR", json.dumps(rep["pooled"]), flush=True)


# ---------------------------------------------------------------- train
import torch as _torch


class MixStream(_torch.utils.data.IterableDataset):
    """gate2_multi.Stream (AUX none) with each yard's DAgger states appended to its stream frames; yards sharded over workers"""
    def __init__(self, G, yards, my, sy, seed, mb, dag):
        super().__init__(); self.G, self.yards, self.my, self.sy, self.seed, self.mb, self.dag = G, yards, my, sy, seed, mb, dag

    def __iter__(self):
        import torch, gate2_multi as g2
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        torch.set_num_threads(1); mine = self.yards[wid::nw]
        z = np.load(self.dag); DY = z["yard"]; data = {}
        for y in mine:
            X, Y = g2.yard_data(self.G, y)[:2]; s = DY == y
            data[y] = (np.concatenate([X, z["X"][s].astype(np.float64)]), np.concatenate([Y, z["Y"][s]]))
        rng = np.random.default_rng(self.seed * 100 + wid); stride = int(os.environ.get("STRIDE", "4")); ep = 0
        yi = np.concatenate([np.full(len(data[y][0]), j, np.int16) for j, y in enumerate(mine)])
        ki = np.concatenate([np.arange(len(data[y][0]), dtype=np.int32) for y in mine])
        while True:
            sel = np.where(ki % stride == ep % stride)[0]; order = sel[rng.permutation(len(sel))]
            for a in range(0, len(order) - self.mb + 1, self.mb):
                imgs, labs = [], []
                for o in order[a:a + self.mb]:
                    y, k = mine[int(yi[o])], int(ki[o]); X, Y = data[y]; labs.append((Y[k] - self.my) / self.sy); imgs.append(frame(y, X[k]))
                yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32))
            ep += 1


def train(G, seed, CK, DAG, OUT):
    import torch, gate2_multi as g2, gate1_student as g1
    assert not g2.AUX, "base fine-tune only (AUX must be unset)"
    os.makedirs(OUT, exist_ok=True)
    STEPS = int(os.environ.get("STEPS", os.environ.get("MAX_STEPS", "20000"))); NW = int(os.environ.get("RENDER_WORKERS", "4")); MB, B = 8, int(os.environ.get("BATCH", "64"))
    CE = int(os.environ.get("CKPT_EVERY_STEPS", "2500"))
    c = torch.load(CK, map_location="cpu", weights_only=False); assert int(c.get("seed", 0)) == seed, "checkpoint seed"
    torch.manual_seed(2000 + seed); Yd = list(c.get("yards") or g2.TRAIN_YARDS); YS = yards(); assert set(YS) <= set(Yd)
    my, sy = np.asarray(c["my"]), np.asarray(c["sy"])
    Ys = np.concatenate([g2.yard_data(G, y)[1][::10] for y in Yd]); m2, s2 = Ys.mean(0), Ys.std(0) + 1e-6
    assert np.allclose(m2, my) and np.allclose(s2, sy), "label statistics differ from the checkpoint's"
    z = np.load(DAG); nd = {y: int((z["yard"] == y).sum()) for y in YS}; ns = {y: len(g2.yard_data(G, y)[0]) for y in YS}
    print(f"[g2 dagger train s{seed}] from {CK} (step {c.get('step')}); {len(YS)} yards; stream {sum(ns.values())} frames + DAgger {sum(nd.values())} states "
          f"({100 * sum(nd.values()) / (sum(ns.values()) + sum(nd.values())):.1f}% DAgger); stride {os.environ.get('STRIDE', '4')}; {STEPS} steps, batch {B} = {B // MB} x {MB}, "
          f"{NW} render workers; AdamW fresh, enc {g1.ENC_LR:g} head {g1.LR:g} constant; dev {DEV}{' FAKE_RENDER (grey frames: not a result)' if FAKE else ''}", flush=True)
    m = g1.build(seed); m.load_state_dict(c["state"]); m = m.to(DEV).train()
    opt = torch.optim.AdamW([{"params": m.feat.parameters(), "lr": g1.ENC_LR}, {"params": m.head.parameters(), "lr": g1.LR}])
    dl = torch.utils.data.DataLoader(MixStream(G, YS, my, sy, seed, MB, DAG), batch_size=None, num_workers=NW, multiprocessing_context="spawn", prefetch_factor=4)
    it = iter(dl); MEAN = torch.tensor(g1.MEAN, device=DEV); SD = torch.tensor(g1.SD, device=DEV)
    def save(path, step):
        torch.save({"state": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "my": my, "sy": sy, "seed": seed, "yards": Yd,
                    "held_out": g2.HELD_OUT, "n_frames": c.get("n_frames"), "K": g1.K, "CHUNK_M": g1.CHUNK_M, "step": int(c.get("step", 0)) + step,
                    "aux": None, "aux_state": None, "g2_dagger": {"from": CK, "dag": DAG, "steps": step, "fake_render": FAKE, "yards": YS}}, path + ".tmp")
        os.replace(path + ".tmp", path)
    lg = open(f"{OUT}/loss_s{seed}.tsv", "a"); t0 = time.time(); tot, nb = 0.0, 0
    for step in range(1, STEPS + 1):
        parts = [next(it) for _ in range(B // MB)]
        x = (torch.cat([p[0] for p in parts]).to(DEV).float() / 255.0 - MEAN) / SD; y = torch.cat([p[1] for p in parts]).to(DEV)
        with torch.autocast(DEV if DEV == "cuda" else "cpu", dtype=torch.bfloat16):
            out = m(x)
        loss = torch.nn.functional.mse_loss(out.float(), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if step % 100 == 0 or step in (1, 10) or step == STEPS:
            el = time.time() - t0; lg.write(f"{step}\t{el:.0f}\t{tot / nb:.5f}\n"); lg.flush()
            pk = f", peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GB" if DEV == "cuda" else ""
            print(f"[g2 dagger train s{seed}] step {step} loss {tot / nb:.4f} {step / el:.3f} steps/s{pk}", flush=True); tot, nb = 0.0, 0
        if step % CE == 0 and step < STEPS: save(f"{OUT}/student_s{seed}_dag_step{step:05d}.pt", step)
    save(f"{OUT}/student_s{seed}_dag.pt", STEPS)
    print(f"G2_DAGGER_TRAIN_DONE s{seed}: {STEPS} steps, {(time.time() - t0) / 60:.1f} min, saved {OUT}/student_s{seed}_dag.pt", flush=True)


# ---------------------------------------------------------------- olsplit (CPU, from olcheck's per-state npz)
def olsplit(npz):
    """< 1 m DAgger states split by speed ratio v_state / v_expert (expert at the nearest point of the same start's track):
    < 0.6, 0.6-0.9, > 0.9; states where the expert speed is < 0.1 m/s (ratio undefined) are their own row. 1 - R^2 and n per bin."""
    z = np.load(npz); near = z["dist"] < 1.0; ve = z["v_expert"]; r = np.where(ve >= 0.1, z["v_state"] / np.maximum(ve, 1e-9), np.nan)
    rows = [("ratio < 0.6", near & (r < 0.6)), ("0.6-0.9", near & (r >= 0.6) & (r < 0.9)), ("> 0.9", near & (r >= 0.9)), ("expert < 0.1 m/s", near & np.isnan(r))]
    out = {}
    for nm, q in rows:
        n = int(q.sum()); out[nm] = {"n": n, "one_minus_r2": float(z["e"][q].sum() / z["b0"][q].sum()) if n else None,
                                     "speed_pred_over_label": float(z["spP"][q].sum() / z["spY"][q].sum()) if n and z["spY"][q].sum() else None}
        print(f"[g2 olsplit] < 1 m, {nm}: n {n}, 1-R2 {out[nm]['one_minus_r2'] if out[nm]['one_minus_r2'] is None else round(out[nm]['one_minus_r2'], 3)}", flush=True)
    print("G2_OLSPLIT", json.dumps(out), flush=True); return out


# ---------------------------------------------------------------- olcheck
def olcheck(G, CK, DAG, out, NF):
    import torch, gate2_multi as g2, gate1_student as g1, gate0b_student as st
    c = torch.load(CK, map_location="cpu", weights_only=False); m = g1.build(int(c.get("seed", 0))); m.load_state_dict(c["state"]); m = m.to(DEV).eval()
    my, sy = np.asarray(c["my"]), np.asarray(c["sy"]); OS = int(os.environ.get("OL_STRIDE", "1"))   # every OS-th frame on both sides
    print(f"[g2 olcheck] OL_STRIDE {OS}", flush=True)
    def pred(yard, X):
        P = []
        for i in range(0, len(X), 32):
            with torch.no_grad():
                P.append(m(torch.tensor(g1.prep(np.stack([frame(yard, x) for x in X[i:i + 32]])), device=DEV)).float().cpu().numpy() * sy + my)
        return np.concatenate(P)
    K = g1.K; sp = lambda A: float(np.linalg.norm(A.reshape(-1, K, 5)[:, 0, :2], axis=1).sum())     # first-step speed, as gate2_openloop
    def score(P, Y): return float((((P - Y) / sy) ** 2).sum()), float((((my - Y) / sy) ** 2).sum()), len(Y), sp(P), sp(Y)
    z = np.load(DAG); fl = z["flight"]; pick = np.unique(np.linspace(0, fl.max(), min(NF, fl.max() + 1)).round().astype(int)); keep = np.isin(fl, pick); DX, DYl, Dy = z["X"][keep].astype(np.float64), z["Y"][keep].astype(np.float64), z["yard"][keep]
    YS = list(dict.fromkeys(Dy.tolist())); rows = {}
    DD = None; BINS = [(0.0, 1.0, "<1m"), (1.0, 3.0, "1-3m"), (3.0, np.inf, ">3m")]; ACC = {}
    OSP = os.environ.get("OL_SPEED") == "1"               # speed-ratio split + per-state npz (implies OL_DIST)
    if os.environ.get("OL_DIST") == "1" or OSP:           # distance of each DAgger state from the same start's expert track
        PF = json.load(open(DAG[:-4] + ".json"))["per_flight"]; flk = fl[keep]; DD = np.full(len(DX), np.nan); VR = np.full((len(DX), 2), np.nan)
        for f_ in np.unique(flk):
            r_ = PF[int(f_)]; j = np.where(flk == f_)[0]; assert set(Dy[j]) == {r_["yard"]}
            XE_ = np.load(f"{G}/{r_['yard']}/tracks/{r_['id']}.npz")["X"].astype(np.float64); P_ = XE_[:, :3]
            d2 = ((DX[j, None, :3] - P_[None]) ** 2).sum(-1); nn = d2.argmin(1); DD[j] = np.sqrt(d2[np.arange(len(j)), nn])
            VR[j, 0] = np.linalg.norm(DX[j, 3:6], axis=1); VR[j, 1] = np.linalg.norm(XE_[nn, 3:6], axis=1)   # state speed, expert speed at the nearest point
        print(f"[g2 olcheck] OL_DIST: distance to the expert track, median {np.median(DD):.2f} m; states per bin (before stride) "
              + ", ".join(f"{nm} {int(((DD >= lo) & (DD < hi)).sum())}" for lo, hi, nm in BINS), flush=True)
    PS = []                                               # per-state rows (OL_SPEED)
    for y in YS:
        s = np.where(Dy == y)[0][::OS]; Pd = pred(y, DX[s]); a = score(Pd, DYl[s])
        if DD is not None:
            Yd = DYl[s]; e = (((Pd - Yd) / sy) ** 2).sum(1); b0 = (((my - Yd) / sy) ** 2).sum(1)
            spP = np.linalg.norm(Pd.reshape(-1, K, 5)[:, 0, :2], axis=1); spY = np.linalg.norm(Yd.reshape(-1, K, 5)[:, 0, :2], axis=1)
            if OSP: PS.append(dict(yard=np.array([y] * len(s)), idx=s, flight=fl[keep][s], dist=DD[s], v_state=VR[s, 0], v_expert=VR[s, 1], e=e, b0=b0, spP=spP, spY=spY, P=Pd.astype(np.float32)))
            for lo, hi, nm in BINS:
                q = (DD[s] >= lo) & (DD[s] < hi)
                for key in ((y, nm), ("pooled", nm)):
                    A = ACC.setdefault(key, [0.0] * 5); A[0] += float(e[q].sum()); A[1] += float(b0[q].sum()); A[2] += int(q.sum()); A[3] += float(spP[q].sum()); A[4] += float(spY[q].sum())
        E = {r["id"]: r for r in json.load(open(f"{G}/{y}/expert.json"))["flights"]}
        ids = [q["id"] for q in json.load(open(f"{G}/{y}/starts.json"))["starts"] if q["split"] == "train"
               and E.get(q["id"], {}).get("outcome") == "success" and E[q["id"]].get("min_clearance_m", 0) >= 0.30]
        ids = [ids[i] for i in (0, 40, 80, 120, 160) if i < len(ids)]; SX, SY = [], []
        for i in ids:
            t = np.load(f"{G}/{y}/tracks/{i}.npz"); SX.append(t["X"].astype(np.float64)[::OS]); SY.append(st.labels(t["X"], t["G"], t["PH"]).astype(np.float64)[::OS])
        b = score(pred(y, np.concatenate(SX)), np.concatenate(SY))
        rows[y] = {"dagger": {"one_minus_r2": a[0] / a[1], "states": a[2], "speed_pred_over_label": a[3] / a[4]},
                   "stream": {"one_minus_r2": b[0] / b[1], "frames": b[2], "flights": ids, "speed_pred_over_label": b[3] / b[4]}, "_s": (a, b)}
        print(f"[g2 olcheck] {y}: DAgger 1-R2 {a[0] / a[1]:.3f} speed {a[3] / a[4]:.2f} ({a[2]} states) | stream 1-R2 {b[0] / b[1]:.3f} speed {b[3] / b[4]:.2f} ({b[2]} frames, {len(ids)} flights)", flush=True)
    pa = [sum(r["_s"][0][j] for r in rows.values()) for j in range(5)]; pb = [sum(r["_s"][1][j] for r in rows.values()) for j in range(5)]
    for r in rows.values(): r.pop("_s")
    if OSP:
        Z = {k: np.concatenate([d[k] for d in PS]) for k in PS[0]}; np.savez_compressed(out[:-5] + "_perstate.npz", **Z)
        print(f"[g2 olcheck] OL_SPEED: per-state rows saved to {out[:-5]}_perstate.npz ({len(Z['e'])} states)", flush=True)
        olsplit(out[:-5] + "_perstate.npz")
    if DD is not None:
        dist = {f"{k[0]} {k[1]}": {"one_minus_r2": (A[0] / A[1]) if A[1] else None, "n": int(A[2]), "speed_pred_over_label": (A[3] / A[4]) if A[4] else None} for k, A in ACC.items()}
        for k, v in dist.items():
            print(f"[g2 olcheck] DIST {k}: 1-R2 {v['one_minus_r2'] if v['one_minus_r2'] is None else round(v['one_minus_r2'], 3)}, n {v['n']}, "
                  f"speed {v['speed_pred_over_label'] if v['speed_pred_over_label'] is None else round(v['speed_pred_over_label'], 2)}", flush=True)
    rep = {"ckpt": CK, "dag": DAG, "n_flights": NF, "ol_stride": OS, "fake_render": FAKE, "yards": rows,
           **({"by_distance_to_expert_track": dist} if DD is not None else {}),
           "pooled": {"dagger": pa[0] / pa[1], "dagger_states": pa[2], "dagger_speed": pa[3] / pa[4], "stream": pb[0] / pb[1], "stream_frames": pb[2], "stream_speed": pb[3] / pb[4]}}
    print(f"[g2 olcheck] frames used: DAgger {pa[2]}, stream {pb[2]} (stride {OS})", flush=True)
    json.dump(rep, open(out, "w"), indent=1)
    print("G2_OLCHECK", json.dumps(rep["pooled"]) + (" FAKE_RENDER (grey frames: numbers are not a result)" if FAKE else ""), flush=True)


if __name__ == "__main__":     # guard: spawn workers re-import this module
    mode = sys.argv[1]; a = sys.argv[2:]
    if mode == "fly": fly(a[0], int(a[1]), a[2], a[3], int(a[4]) if len(a) > 4 else 8)
    elif mode == "relabel": relabel(a[0], int(a[1]), a[2], int(a[3]) if len(a) > 3 else 32)
    elif mode == "train": train(a[0], int(a[1]), a[2], a[3], a[4])
    elif mode == "labelfloor": labelfloor(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 8)
    elif mode == "olsplit": olsplit(a[0])
    elif mode == "olcheck": olcheck(a[0], a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 50)
    else: sys.exit(__doc__)
