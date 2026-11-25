"""Sample start poses and score expert flights for a configurable yard."""
import json, math, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
FLAMBE_OUTPUT = os.environ.get("FLAMBE_OUTPUT", os.path.join(FLAMBE_ROOT, "out"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

sys.path[:0] = [
    *LOCAL_MODULE_PATHS,
    os.path.join(FLAMBE_YARDS_FALLBACK, "kv14"),
    os.path.join(FLAMBE_TERRA, "goalspec", "v4"),
    os.path.join(FLAMBE_TERRA, "goalspec"),
    FLAMBE_TERRA,
]
YARD = os.environ.get("YARD", "syn_013"); SYR = os.path.join(FLAMBE_YARDS, "yards")
Z0, CLR_MIN, H_MAX, MIN_SEP, MARGIN = -2.15, 1.0, 0.5, 1.0, 1.5
N_SECT, BANDS = 8, [(5.0, 12.0), (12.0, 20.0), (20.0, 40.0)]
HOLD_AFTER, MAX_S = 2.0, 120.0
FINAL_APPROACH = os.environ.get("FINAL_APPROACH", "1") == "1"
R_BODY_TIER = os.environ.get("R_BODY_TIER", "0") == "1"
def _dense(R):
    R = np.asarray(R, float); return np.vstack([R[k] + (R[k + 1] - R[k]) * np.linspace(0, 1, 30)[:, None] for k in range(len(R) - 1)]) if len(R) > 1 else R
CLR_CAP = os.environ.get("CLR_CAP", "1") == "1"; CAP_LO, CLR_LO, CLR_HI = 0.3, 0.5, 0.85   # Scale speed by clearance when enabled.
U0 = np.array([-0.39, 0.0, 0.0, 0.0])     # hover-level first command (data flights start with thrust -0.37 to -0.42)


def draw(out):
    from scipy.spatial import cKDTree
    tr = np.load(f"{SYR}/{YARD}/recon/truth.npz"); G = json.load(open(f"{SYR}/{YARD}/goals.json"))
    hc = np.asarray(G["house_center_ned"], float); ext = np.asarray(tr["extent_blender"], float)
    idx = np.argwhere(np.asarray(tr["occ3d"]).astype(bool)); tree = cKDTree(np.asarray(tr["lo"], float) + (idx + 0.5) * float(tr["cell"]))
    H = np.asarray(tr["height_top"], float); lo, cell = np.asarray(tr["lo"], float), float(tr["cell"])
    def tall_near(p):
        i0, j0 = int((p[0] - lo[0]) / cell), int((p[1] - lo[1]) / cell); r = int(math.ceil(1.0 / cell))
        sub = H[max(i0 - r, 0):i0 + r + 1, max(j0 - r, 0):j0 + r + 1]
        return sub.size == 0 or float(np.nanmax(sub)) > H_MAX
    rng = np.random.default_rng(0); acc, meta = [], []
    strata = [(s, b) for b in range(len(BANDS)) for s in range(N_SECT)]
    per = {st: 0 for st in strata}; tries = {st: 0 for st in strata}; target = int(math.ceil(400 / len(strata)))
    while len(acc) < 400 and any(per[st] < target and tries[st] < 20000 for st in strata):
        for st in strata:
            if per[st] >= target or tries[st] >= 20000 or len(acc) >= 400:
                continue
            tries[st] += 1; s, b = st
            ang = (s + rng.random()) * 2 * math.pi / N_SECT; rad = BANDS[b][0] + rng.random() * (BANDS[b][1] - BANDS[b][0])
            p = np.array([hc[0] + rad * math.cos(ang), hc[1] + rad * math.sin(ang), Z0])
            if not (ext[0] + MARGIN < p[0] < ext[1] - MARGIN and ext[2] + MARGIN < p[1] < ext[3] - MARGIN):
                continue
            if tree.query(p)[0] < CLR_MIN or tall_near(p):
                continue
            if acc and np.min(np.linalg.norm(np.array(acc)[:, :2] - p[:2], axis=1)) < MIN_SEP:
                continue
            acc.append(p); per[st] += 1
            meta.append({"id": f"g{len(acc) - 1:03d}", "pos": p.round(3).tolist(), "yaw": float(rng.uniform(-math.pi, math.pi)),
                         "sector": s, "band": b, "bearing_deg": round(math.degrees(ang) % 360, 1), "radius_m": round(rad, 2),
                         "split": "train" if per[st] % 2 == 1 else "test", "clearance_m": round(float(tree.query(p)[0]), 2)})
    # a stratum that cannot be filled (house footprint, lot edge) leaves its share to the others in a second pass
    while len(acc) < 400:
        st = strata[int(rng.integers(len(strata)))]; s, b = st; tries[st] += 1
        if tries[st] > 200000:
            break
        ang = (s + rng.random()) * 2 * math.pi / N_SECT; rad = BANDS[b][0] + rng.random() * (BANDS[b][1] - BANDS[b][0])
        p = np.array([hc[0] + rad * math.cos(ang), hc[1] + rad * math.sin(ang), Z0])
        if not (ext[0] + MARGIN < p[0] < ext[1] - MARGIN and ext[2] + MARGIN < p[1] < ext[3] - MARGIN) or tree.query(p)[0] < CLR_MIN or tall_near(p):
            continue
        if np.min(np.linalg.norm(np.array(acc)[:, :2] - p[:2], axis=1)) < MIN_SEP:
            continue
        acc.append(p); per[st] += 1
        meta.append({"id": f"g{len(acc) - 1:03d}", "pos": p.round(3).tolist(), "yaw": float(rng.uniform(-math.pi, math.pi)),
                     "sector": s, "band": b, "bearing_deg": round(math.degrees(ang) % 360, 1), "radius_m": round(rad, 2),
                     "split": "train" if sum(m["split"] == "train" for m in meta) < 200 and (len(meta) % 2 == 1) else "test",
                     "clearance_m": round(float(tree.query(p)[0]), 2)})
    ntr = sum(m["split"] == "train" for m in meta)
    for m in meta:                                   # balance to 200/200 if the second pass tilted it
        if ntr > 200 and m["split"] == "train":
            m["split"] = "test"; ntr -= 1
        elif ntr < 200 and m["split"] == "test" and len(meta) - ntr > 200:
            m["split"] = "train"; ntr += 1
    P = np.array([m["pos"] for m in meta]); dmin = min(np.linalg.norm(P[i, :2] - P[j, :2]) for i in range(len(P)) for j in range(i))
    summ = {"n": len(meta), "train": sum(m["split"] == "train" for m in meta), "test": sum(m["split"] == "test" for m in meta),
            "min_sep_m": round(float(dmin), 2), "min_clearance_m": min(m["clearance_m"] for m in meta),
            "per_stratum": {f"s{s}b{b}": per[(s, b)] for s, b in strata}, "house_center_ned": hc.tolist()}
    json.dump({"yard": YARD, "summary": summ, "starts": meta}, open(out, "w"), indent=1); print(json.dumps(summ))


def reroute(starts_in, out):
    """Drop starts with no planner route at sampling time and draw a replacement in the same stratum and
    split (same constraints as draw, seed 1), so train and test stay at 200 each. Dropped starts are logged in the file."""
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import obs_relabel_z as oz, goal_expert_v4 as gv
    from scipy.spatial import cKDTree
    oz.init_worker(); D = json.load(open(starts_in)); S = D["starts"]
    goals = {g["id"]: g for g in json.load(open(f"{SYR}/{YARD}/goals.json"))["goals"]}; cache = {}
    routed = lambda p: gv.plan(oz.GR, goals["door"], np.asarray(p, float), cache) is not None
    tr = np.load(f"{SYR}/{YARD}/recon/truth.npz"); hc = np.asarray(json.load(open(f"{SYR}/{YARD}/goals.json"))["house_center_ned"], float)
    ext = np.asarray(tr["extent_blender"], float); idx = np.argwhere(np.asarray(tr["occ3d"]).astype(bool))
    tree = cKDTree(np.asarray(tr["lo"], float) + (idx + 0.5) * float(tr["cell"])); H = np.asarray(tr["height_top"], float)
    lo, cell = np.asarray(tr["lo"], float), float(tr["cell"]); rng = np.random.default_rng(1); dropped = []
    def tall_near(p):
        i0, j0 = int((p[0] - lo[0]) / cell), int((p[1] - lo[1]) / cell); r = int(math.ceil(1.0 / cell))
        sub = H[max(i0 - r, 0):i0 + r + 1, max(j0 - r, 0):j0 + r + 1]; return sub.size == 0 or float(np.nanmax(sub)) > H_MAX
    for k, m in enumerate(S):
        if routed(m["pos"]):
            continue
        dropped.append(dict(m, reason="no planner route"))
        others = np.array([q["pos"] for j, q in enumerate(S) if j != k])
        for _ in range(100000):
            ang = (m["sector"] + rng.random()) * 2 * math.pi / N_SECT; b = BANDS[m["band"]]; rad = b[0] + rng.random() * (b[1] - b[0])
            p = np.array([hc[0] + rad * math.cos(ang), hc[1] + rad * math.sin(ang), Z0])
            if not (ext[0] + MARGIN < p[0] < ext[1] - MARGIN and ext[2] + MARGIN < p[1] < ext[3] - MARGIN): continue
            if tree.query(p)[0] < CLR_MIN or tall_near(p) or np.min(np.linalg.norm(others[:, :2] - p[:2], axis=1)) < MIN_SEP: continue
            if not routed(p): continue
            S[k] = dict(m, id=m["id"] + "r", pos=p.round(3).tolist(), yaw=float(rng.uniform(-math.pi, math.pi)),
                        bearing_deg=round(math.degrees(ang) % 360, 1), radius_m=round(rad, 2), clearance_m=round(float(tree.query(p)[0]), 2),
                        replaces=m["id"]); break
        else:
            raise RuntimeError(f"no routed replacement for {m['id']}")
    D["dropped"] = dropped; D["summary"]["train"] = sum(q["split"] == "train" for q in S); D["summary"]["test"] = sum(q["split"] == "test" for q in S)
    json.dump(D, open(out, "w"), indent=1); print("REROUTE dropped", [d["id"] for d in dropped], "replaced by", [q["id"] for q in S if "replaces" in q], D["summary"]["train"], D["summary"]["test"])


# The exit search clearance allows for tracking error during the climb.
EXIT_SEARCH = os.environ.get("EXIT_SEARCH", "1") == "1"; EXIT_R, EXIT_STEP, EXIT_BOX = float(os.environ.get("EXIT_R", "0.60")), 0.1, 6.0


def pocket_exit(ex, relabel=False):
    """Search for a finer route out of a narrow pocket. If the route's first segment (grid3d.field's straight 'pocket hop' from a
    start the 0.25 m planning grid sees as cut off) comes closer than EXIT_R to the exact geometry, replace that segment
    by a path found on a 0.1 m grid over the exact occupancy (clearance >= EXIT_R; above the occ3d top, cells under a
    height_top surface + 0.3 m are blocked; 0.4-3.0 m above ground), from the start to the route's first connected
    vertex, string-pulled with the same clearance check; the normal planner's route continues from there. Same rule for
    every flight; a flight whose first segment keeps >= EXIT_R is unchanged."""
    import collections
    from score import clearance_fn
    cl = clearance_fn(YARD); R = [np.asarray(p, float) for p in ex.route]
    if len(R) < 2:
        return
    seg = R[0] + (R[1] - R[0]) * np.linspace(0, 1, 60)[:, None]
    exit_r = EXIT_R
    if relabel:
        # During relabeling, a visited state can itself sit closer than EXIT_R to the geometry. The exit search then
        # runs only when the planner's first segment comes closer than the 0.30 m body, with radius min(EXIT_R, start
        # clearance - 0.02), floor 0.30, so the search starts in free space and only moves away from the obstacle.
        c0 = float(cl(R[0][None])[0])
        if float(cl(seg).min()) >= 0.30:
            return
        exit_r = max(0.30, min(EXIT_R, c0 - 0.02))
    elif float(cl(seg).min()) >= EXIT_R:
        return
    tr = np.load(f"{SYR}/{YARD}/recon/truth.npz"); H = np.asarray(tr["height_top"], float); lo = np.asarray(tr["lo"], float)
    c = float(tr["cell"]); zg = float(tr["z_ground"]); s0, s1 = R[0], R[1]
    xmin, xmax = min(s0[0], s1[0]) - EXIT_BOX, max(s0[0], s1[0]) + EXIT_BOX
    ymin, ymax = min(s0[1], s1[1]) - EXIT_BOX, max(s0[1], s1[1]) + EXIT_BOX
    xs = np.arange(xmin, xmax, EXIT_STEP); ys = np.arange(ymin, ymax, EXIT_STEP); zs = np.arange(zg - 3.0, zg - 0.4, EXIT_STEP)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij"); pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    ok = (cl(pts) >= exit_r).reshape(X.shape)
    ii = np.clip(((X - lo[0]) / c).astype(int), 0, H.shape[0] - 1); jj = np.clip(((Y - lo[1]) / c).astype(int), 0, H.shape[1] - 1)
    ok &= ~((Z < lo[2]) & ((zg - Z) < H[ii, jj] + 0.3))
    o = np.array([xs[0], ys[0], zs[0]]); ix = lambda p: tuple(np.clip(np.round((np.asarray(p) - o) / EXIT_STEP).astype(int), 0, np.array(ok.shape) - 1))
    a, b = ix(s0), ix(s1)
    if not ok[a] or not ok[b]:
        raise RuntimeError(f"exit search: start or first vertex not free at {exit_r} m")
    prev = {a: None}; q = collections.deque([a])
    while q and b not in prev:
        u = q.popleft()
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            v = (u[0] + d[0], u[1] + d[1], u[2] + d[2])
            if v not in prev and 0 <= v[0] < ok.shape[0] and 0 <= v[1] < ok.shape[1] and 0 <= v[2] < ok.shape[2] and ok[v]:
                prev[v] = u; q.append(v)
    if b not in prev:
        raise RuntimeError(f"exit search: no path at {exit_r} m within the box")
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    P = [o + np.array(p) * EXIT_STEP for p in path[::-1]]; P[0], P[-1] = s0, s1
    out, i = [P[0]], 0                                  # string pulling with the exact clearance check
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and float(cl(P[i] + (P[j] - P[i]) * np.linspace(0, 1, max(2, int(np.linalg.norm(P[j] - P[i]) / 0.05)))[:, None]).min()) < exit_r:
            j -= 1
        out.append(P[j]); i = j
    ex.route = out[:-1] + R[1:]; ex.exit_search = len(out) - 1
    ex.vv = ex._vertex_speeds()
    PP = np.array(ex.route, float); ex.nom_P = PP; L = np.linalg.norm(np.diff(PP[:, :2], axis=0), axis=1); ex.nom_s = np.concatenate([[0.0], np.cumsum(L)])


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])        # xyzw, level attitude


def fly_one(m):
    import obs_relabel_z as oz, goal_expert_v4 as gv, visual_expert_syn as ve
    from score import score_track, clearance_fn
    class DoorYaw(oz.ObsExpert):                    # Aim the camera toward the door.
        def _yaw_des(self, pos):
            d = self.pose["look"][:2] - np.asarray(pos)[:2]
            return math.atan2(d[1], d[0]) if float(np.linalg.norm(d)) > 0.5 else self.pose["yaw"]
        def _carrot(self, p):
            # speed cap scaled by the planned clearance, the same rule for every flight.
            # v_cap = CAP_LO when clearance < CLR_LO, rising linearly to the cruise speed V_MAX at CLR_HI. Clearance = the
            # grid's planning clearance at the drone and on the route 0, 0.5, 1.0, 1.5 m ahead (as the v4 cap it tightens).
            c, left, cap = super()._carrot(p)
            if CLR_CAP and len(self.route) > 1:
                ahead = np.array([self._at(self.seg, self._t_on_seg, s)[0] for s in (0.0, 0.5, 1.0, 1.5)] + [np.asarray(p, float)])
                clr = float(self.gr.clearance_plan(ahead).min())
                f = float(np.clip((clr - CLR_LO) / (CLR_HI - CLR_LO), 0.0, 1.0)); cap = min(cap, CAP_LO + f * (gv.V_MAX - CAP_LO))
            return c, left, cap
    assert abs(gv.K_VEL - 1.4) < 1e-9, f"controller gain {gv.K_VEL} != k25zd expert 1.4"
    if R_BODY_TIER and not getattr(gv, "_plan_tier3", False):
        # Try the smaller clearance tier for starts in narrow pockets.
        _g3 = gv.g3
        def plan3(gr, g, p0, cache, soft=gv.SOFT):
            best = None
            for fi, pose in enumerate(_g3.goal_poses(gr, g)):
                if pose is None: continue
                Rt = None
                for r in (_g3.R_FREE, _g3.R_MIN, R_TIER3):
                    k = (g["id"], fi, soft, r)
                    if k not in cache: cache[k] = gr.field(pose["hover"], _g3.ceil_for(pose), r, soft)
                    if cache[k] is not None: Rt = cache[k](np.asarray(p0, float))
                    if Rt is not None:
                        if r != R_TIER3 and float(gr.clearance_plan(_dense(Rt)).min()) < 0.30: Rt = None; continue
                        break
                if Rt is None: continue
                L = _g3.plen(Rt) + float(np.linalg.norm(pose["land"] - pose["hover"]))
                if best is None or L < best[3]: best = (Rt, pose, fi, L)
            return best
        gv.plan = plan3; gv._plan_tier3 = True
    goals = {g["id"]: g for g in json.load(open(oz.dv.SY / YARD / "goals.json"))["goals"]}
    gy = oz.dv.gx.GoalYard(oz.dv.SY / YARD, goals["door"]); gv.K_CT = 0.25
    x = np.r_[np.asarray(m["pos"], float), np.zeros(3), _yaw_quat(m["yaw"])]; cache = m.get("_cache", {}); t0 = time.time()
    try:
        ex = DoorYaw(gy, oz.GR, x[:3].copy(), cache); ex.zmode = "route"
        if FINAL_APPROACH and ex.pose is not None:
            # After the snapped hover pose, approach and descend to the poses in goals.json.
            ex.pose = dict(ex.pose); ex.pose["snapped_land"] = np.asarray(ex.pose["land"], float)
            ex.pose["hover"] = np.asarray(goals["door"]["hover_ned"], float); ex.pose["land"] = np.asarray(goals["door"]["land_ned"], float)
        if EXIT_SEARCH and ex.pose is not None:
            pocket_exit(ex, relabel=bool(m.get("_relabel")))
        ex.reset()
    except Exception as e:
        return {"id": m["id"], "split": m["split"], "error": f"plan: {type(e).__name__}: {e}"}
    if m.get("_build_only"):                         # Return the configured expert without flying it.
        return ex, goals, cache
    u_prev = U0.copy(); P, V, T = [x[:3].copy()], [x[3:6].copy()], [0.0]; k_land = None
    XS, GS, PH = [], [], []                          # state, guidance into the K_VEL 1.4 controller, phase (student labels)
    for k in range(int(MAX_S * ve.HZ_CTL)):
        oz.dv.maybe_replan(ex, x, cache); u = ex.decide(k / ve.HZ_CTL, x.copy())
        XS.append(x.copy()); GS.append(np.asarray(ex._guid, float)); PH.append(gv.PHASE_ID.get(ex.phase, 6))
        if ex.landed and k_land is None:
            k_land = k
        if k_land is not None and (k - k_land) >= HOLD_AFTER * ve.HZ_CTL:
            break
        x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / ve.HZ_CTL)
    P, V, T = np.array(P), np.array(V), np.array(T)
    goal = np.asarray(goals["door"]["land_ned"], float)
    t_exp = (k_land / ve.HZ_CTL) if k_land is not None else MAX_S
    clr = clearance_fn(YARD)(P)
    r = score_track(T, P, V, goal, clr, 1.5 * t_exp)
    # Also score against the expert's snapped landing pose.
    if ex.pose is not None:
        own = np.asarray(ex.pose.get("snapped_land", ex.pose["land"]), float)
        r2 = score_track(T, P, V, own, clr, 1.5 * t_exp)
        r["own_land"] = {"outcome": r2["outcome"], "closest_m": r2["closest_m"], "land_ned": [float(v) for v in own]}
    np.savez_compressed(f"{FLAMBE_OUTPUT}/tracks/{m['id']}.npz", T=T, P=P, V=V, clr=clr, X=np.array(XS), G=np.array(GS), PH=np.array(PH))
    r.update({"id": m["id"], "split": m["split"], "sector": m["sector"], "band": m["band"], "landed": k_land is not None,
              "t_landed_s": None if k_land is None else k_land / ve.HZ_CTL, "phase_end": ex.phase, "wall_s": round(time.time() - t0, 1),
              "final_m": float(np.linalg.norm(P[-1] - goal)), "final_speed": float(np.linalg.norm(V[-1])),
              "exit_search": int(getattr(ex, "exit_search", 0))})
    return r


def init():
    # Load simulator dependencies before forking. Each worker builds its own dynamics and grid.
    import obs_relabel_z as oz, score
    oz.YARD = YARD; oz.init_worker(); score.SY = FLAMBE_YARDS


def fly(starts, out, NP):
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import obs_relabel_z, score                     # parent imports before the fork (see init)
    from multiprocessing import get_context
    S = json.load(open(starts))["starts"]; done = {}
    if os.path.exists(out):
        done = {r["id"]: r for r in json.load(open(out))["flights"]}
    todo = [m for m in S if m["id"] not in done]; t0 = time.time(); os.makedirs(os.path.join(FLAMBE_OUTPUT, "tracks"), exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor   # a dead worker raises BrokenProcessPool instead of hanging
    with ProcessPoolExecutor(NP, mp_context=get_context("fork"), initializer=init) as p:
        for i, r in enumerate(p.map(fly_one, todo, chunksize=1)):
            done[r["id"]] = r
            if i % 10 == 0 or "error" in r:
                print(f"[0b] {len(done)}/{len(S)} {r['id']} {r.get('outcome', r.get('error'))} ({time.time() - t0:.0f}s)", flush=True)
                json.dump({"flights": list(done.values())}, open(out, "w"))
    json.dump({"flights": list(done.values())}, open(out, "w"), indent=0)
    for sp in ("train", "test", "all"):
        R = [r for r in done.values() if sp == "all" or r["split"] == sp]; n = len(R)
        k = {o: sum(r.get("outcome") == o for r in R) for o in ("success", "contact", "overshoot", "timeout")}
        err = sum("error" in r for r in R); anyc = sum(r.get("min_clearance_m", 9) < 0.30 for r in R)
        ko = {o: sum(r.get("own_land", {}).get("outcome") == o for r in R) for o in ("success", "contact", "overshoot", "timeout")}
        print(f"[0b] {sp} vs expert's own land pose (diagnostic): success {ko['success']}/{n} contact {ko['contact']} overshoot {ko['overshoot']} timeout {ko['timeout']}", flush=True)
        print(f"[0b] {sp}: {n} starts | success {k['success']}/{n} ({100 * k['success'] / max(n, 1):.1f}%) | contact {k['contact']} "
              f"overshoot {k['overshoot']} timeout {k['timeout']} | plan errors {err} | any contact (whole flight) {anyc}", flush=True)
    print("GATE0B_EXPERT_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "draw":
        draw(sys.argv[2])
    elif sys.argv[1] == "reroute":
        reroute(sys.argv[2], sys.argv[3])
    else:
        fly(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
