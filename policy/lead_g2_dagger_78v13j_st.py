"""Collect and relabel DAgger flight data, train a visual policy, and evaluate checkpoints."""
import json, math, os, sys, time
import numpy as np
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
FAKE = os.environ.get("FAKE_RENDER") == "1"
AUG = os.environ.get("AUG", "0") == "1"
DEV = os.environ.get("DEV", "cuda")
COPY = ["syn_002", "syn_013", "syn_025", "syn_040"]
NEAR = tuple(float(v) for v in os.environ["NEAR_DOOR"].split(",")) if os.environ.get("NEAR_DOOR") else None   # off by default
assert NEAR is None or (len(NEAR) == 2 and 0 <= NEAR[0] < NEAR[1]), "NEAR_DOOR=<lo>,<hi> metres"
CONC = os.environ.get("CONCEPT_IN") == "1"; CONC_W = float(os.environ.get("CONCEPT_W", "0.1")); NCL = 17   # 3 + 1 + 1 + 12; off by default
_TRUTH = {}
_G1B = None
YMAP = dict(kv.split(":") for kv in os.environ["YARD_DIRS"].split(",")) if os.environ.get("YARD_DIRS") else {}
XSY = os.environ["EXTRA_STREAM_YARDS"].split(",") if os.environ.get("EXTRA_STREAM_YARDS") else []
JOINT = os.environ.get("CONCEPT_JOINT") == "1"; HIST = int(os.environ.get("HIST") or 0); HDT = int(os.environ.get("HIST_DT") or 10)   # all off by default
STIN = os.environ.get("STATE_IN") == "1"; XYF = os.environ.get("XY_FRAME") or "world"; HSRC = os.environ.get("HIST_SRC") or "cache"; HCD = os.environ.get("HIST_CACHE") or ""
assert HIST in (0, 4, 8) and HDT >= 1 and XYF in ("world", "body") and HSRC in ("cache", "render"), "v13h: HIST 4|8, HIST_DT >= 1, XY_FRAME world|body, HIST_SRC cache|render"
HC = bool(HIST) and HSRC == "cache"; assert not HC or HCD, "v13h HIST_SRC=cache needs HIST_CACHE=<dir>"
HPJ = int(os.environ.get("HIST_PROJ") or 0); assert HPJ >= 0 and (not HPJ or HIST > 1), "v13i HIST_PROJ needs HIST"   # off by default
H13 = bool(HIST or STIN or XYF == "body" or JOINT); SV_V, SV_W = 3.0, 1.0                 # wrapper needed; state scales (m/s, rad/s)
_HK = {"x": None, "n": 0, "on": False}                                                  # state handed to the wrapper (one per control step)


def yd(G, y):
    """Load yard data through the optional YARD_DIRS mapping."""
    import gate2_multi as g2
    return g2.yard_data(G, YMAP.get(y, y))
CSTATS = os.environ.get("CONCEPT_STATS") == "1"; assert not CSTATS or CONC, "CONCEPT_STATS needs CONCEPT_IN=1"
CFEAT = os.environ.get("CONCEPT_FEAT", "cls"); assert CFEAT in ("cls", "cls_mean"), "CONCEPT_FEAT: cls or cls_mean"; assert CFEAT == "cls" or CONC, "CONCEPT_FEAT needs CONCEPT_IN=1"
assert not JOINT or (CONC and CFEAT == "cls_mean"), "CONCEPT_JOINT needs CONCEPT_IN=1 CONCEPT_FEAT=cls_mean"


def build_concept(seed, base=None, feat=None):
    """Build a student with a concept head and a policy head that consumes its predictions."""
    import torch, gate1_student as g1
    base = base if base is not None else (_G1B or g1.build)(seed); feat = feat or CFEAT                    # concept-head input
    class ConceptStudent(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.feat = base.feat; torch.manual_seed(5000 + seed)
            s.ch = torch.nn.Sequential(torch.nn.Linear(768 if feat == "cls_mean" else 384, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, NCL))
            s.head = torch.nn.Sequential(torch.nn.Linear(384 + NCL, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, g1.K * 5))
        @staticmethod
        def proc(cr): return torch.cat([torch.nn.functional.normalize(cr[:, :3], dim=1), torch.sigmoid(cr[:, 3:])], 1)
        def forward_all(s, img):
            if feat == "cls_mean":                                                         # one backbone pass, CLS + patch-mean
                h = s.feat.backbone(pixel_values=img).last_hidden_state; fz = h[:, 0]; cr = s.ch(torch.cat([fz, h[:, 1:].mean(1)], 1))
            else: fz = s.feat(img); cr = s.ch(fz)
            ab = os.environ.get("V13_CONCEPT_ABLATE", "")                                   # eval-only concept ablation
            if not ab: return s.head(torch.cat([fz, s.proc(cr).detach()], 1)), cr
            cp = s.proc(cr).detach()
            if ab == "mean": cp = torch.tensor(json.loads(os.environ["V13_CONCEPT_MEAN"]), dtype=cp.dtype, device=cp.device)[None].expand(len(cp), -1)
            elif ab == "lag":
                assert len(cp) == 1, "lag ablation is per flight, batch 1"
                if not hasattr(s, "_buf"): s._buf = []
                s._buf.append(cp); cp = s._buf[max(0, len(s._buf) - 1 - int(os.environ.get("ABL_LAG", "40")))]
            else: raise ValueError(f"V13_CONCEPT_ABLATE={ab}")
            return s.head(torch.cat([fz, cp], 1)), cr
        def forward(s, img): return s.forward_all(img)[0]
    return ConceptStudent()


def concept_patch():
    """make gate1_student.build return the concept-in student (for gate2_multi.fly_one in check workers)"""
    global _G1B
    import gate1_student as g1
    if _G1B is None: _G1B = g1.build; g1.build = lambda seed, state=False: build_concept(seed, _G1B(seed), os.environ.get("V13_CONCEPT_FEAT", "cls"))


def check_winit_concept():
    """Initialize an evaluation worker and install the concept-conditioned student."""
    import gate2_multi as g2
    (check_winit if os.environ.get("TORCH_THREADS") else g2.winit)(); concept_patch()


def concept_frame(yard, x):
    """(policy frame uint8 (3, 224, 224), 21 concept labels float32) at state x"""
    import gate2_multi as g2
    from scipy.spatial.transform import Rotation as Rot
    x = np.asarray(x, float)
    sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.join(FLAMBE_TERRA, "s10")]; import make_labels as mk                    # exact geometry, default 9 x 16 grid (the screen's)
    assert (mk.RH, mk.RW) == (9, 16), f"make_labels grid {mk.RH} x {mk.RW}, not the screen's 9 x 16"
    if yard not in _TRUTH: _TRUTH[yard] = mk.Truth(yard, [g_ for g_ in json.load(open(f"{FLAMBE_YARDS}/yards/{yard}/goals.json"))["goals"] if g_["id"] == "door"][0])
    d, cls = _TRUTH[yard].cast(x[:3], x[6:10]); img = frame(yard, x)
    if yard not in _LAND: _LAND[yard] = g2.land(yard)
    v = _LAND[yard] - x[:3]; dist = float(np.linalg.norm(v)); u = Rot.from_quat(x[6:10]).as_matrix().T @ (v / max(dist, 1e-6))
    dep = np.asarray(d, np.float32)[:9].reshape(3, 3, 4, 4).mean((1, 3)).ravel() / 20.0   # 3 x 4 cells of the 9 x 16 cast
    return img, np.r_[u, min(1.0, math.log1p(dist) / math.log1p(40.0)), float((np.asarray(cls) == 3).any()), dep].astype(np.float32)


def h13_hook(x):
    _HK["x"] = np.asarray(x, float).copy(); _HK["n"] += 1


def h13_yaw(x):
    import gate0b_student as st
    return float(st.yaw_of(np.asarray(x, float)[6:10]))


def h13_state(x, xp):
    """STATE_IN vector (4): body-frame velocity / SV_V, yaw rate (wrapped yaw difference to the previous state x 20 Hz) / SV_W; 0 rate without a previous state"""
    from scipy.spatial.transform import Rotation as Rot
    import gate0b_student as st
    x = np.asarray(x, float); vb = Rot.from_quat(x[6:10]).as_matrix().T @ x[3:6]
    yr = 0.0 if xp is None else float(st.wrap(h13_yaw(x) - h13_yaw(xp))) * 20.0
    return np.r_[vb / SV_V, yr / SV_W].astype(np.float32)


def h13_rot(yk, yaw, sign):
    """rotate the xy pair of each of the K future steps of a K*5 label / command by sign * yaw (sign -1: world -> body, +1: body -> world)"""
    y = np.array(yk, np.float64, copy=True).reshape(-1, 5); c, s = math.cos(sign * yaw), math.sin(sign * yaw)
    vx, vy = y[:, 0].copy(), y[:, 1].copy(); y[:, 0] = c * vx - s * vy; y[:, 1] = s * vx + c * vy
    return y.reshape(np.shape(yk)).astype(np.asarray(yk).dtype)


def h13_body_stats(X, Y):
    """Compute body-frame label statistics with one scale for x and y at each future step."""
    B = np.stack([h13_rot(Y[i], h13_yaw(X[i]), -1) for i in range(len(Y))]).astype(np.float64); my, sy = B.mean(0), B.std(0) + 1e-6
    for j in range(len(my) // 5):
        my[5 * j] = my[5 * j + 1] = 0.0; sy[5 * j] = sy[5 * j + 1] = math.sqrt(float((B[:, 5 * j] ** 2 + B[:, 5 * j + 1] ** 2).mean()) / 2.0) + 1e-6
    return my, sy


def h13_cfg():
    return {"hist": HIST, "hist_dt": HDT, "hist_src": HSRC if HIST else None, "state_in": STIN, "xy_frame": XYF, "joint": JOINT, "conc": CONC, "feat": CFEAT,
            "head_seed": int(os.environ.get("HEAD_SEED", "4000")), **({"hist_proj": HPJ} if HPJ else {})}


def build_h13(seed, inner, cfg):
    """Wrap the student (CamStudent or ConceptStudent); a fresh policy head reading [frames] [+ concepts] [+ state]"""
    import torch, gate1_student as g1
    class H13Student(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.inner = inner; s.cfg = dict(cfg); s.conc = bool(cfg["conc"]); s.nh = int(cfg["hist"]); s.dt = int(cfg["hist_dt"])
            s.stin = bool(cfg["state_in"]); s.body = cfg["xy_frame"] == "body"; s.cm = s.nh > 0 or (s.conc and cfg["feat"] == "cls_mean"); s.joint = bool(cfg["joint"])
            if s.joint:                                                                    # CONCEPT_JOINT: shared trainable layer + fresh concept head (512 in)
                torch.manual_seed(6000 + seed)
                s.sh = torch.nn.Sequential(torch.nn.Linear(768, 512), torch.nn.LayerNorm(512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.LayerNorm(512), torch.nn.GELU())
                s.inner.ch = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, NCL))
            s.pj = int(cfg.get("hist_proj", 0))                                           # past-frame projection
            if s.pj:
                torch.manual_seed(7000 + seed); s.pp = torch.nn.Sequential(torch.nn.Linear(768, s.pj), torch.nn.LayerNorm(s.pj))
            din = (512 if s.joint else (768 if s.nh else 384)) + (s.pj or 768) * max(0, s.nh - 1) + (NCL if s.conc else 0) + (4 if s.stin else 0)
            torch.manual_seed(int(cfg["head_seed"]) + seed)
            s.inner.head = torch.nn.Sequential(torch.nn.Linear(din, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, g1.K * 5))
            s.h13_reset()
        feat = property(lambda s: s.inner.feat); head = property(lambda s: s.inner.head); ch = property(lambda s: s.inner.ch)
        @staticmethod
        def proc(cr): return torch.cat([torch.nn.functional.normalize(cr[:, :3], dim=1), torch.sigmoid(cr[:, 3:])], 1)
        def h13_reset(s): s._buf, s._xp, s._n = [], None, None
        def feats(s, img):                                                                 # (M, 3, 224, 224) -> CLS (M, 384), [CLS, mean] (M, 768) or None
            if s.cm: h = s.inner.feat.backbone(pixel_values=img).last_hidden_state; c0 = h[:, 0]; return c0, torch.cat([c0, h[:, 1:].mean(1)], 1)
            return s.inner.feat(img), None
        def head_in(s, cls_cur, cm_cur, past, st):                                        # past: (B, (N-1)*768) earlier frames, oldest first, or None
            z = s.sh(cm_cur) if s.joint else None; parts = [z if s.joint else (cm_cur if s.nh else cls_cur)]; cr = None
            if s.nh > 1: parts.append(s.pp(past.to(parts[0].dtype).reshape(len(past), s.nh - 1, 768)).reshape(len(past), -1) if s.pj else past.to(parts[0].dtype))
            if s.conc:
                cr = s.inner.ch(z if s.joint else (cm_cur if s.cfg["feat"] == "cls_mean" else cls_cur)); cp = s.proc(cr); parts.append(cp if s.joint else cp.detach())
            if s.stin: parts.append(st)
            return s.inner.head(torch.cat(parts, 1)), cr
        def core(s, imgs, st=None, past=None):                                             # train: imgs (B, N_img, 3, 224, 224): N_img = HIST (render) or 1; past (B, N-1, 768) from the cache
            B, N = imgs.shape[:2]; c0, cm = s.feats(imgs.flatten(0, 1))
            c0 = c0.view(B, N, -1); cm = cm.view(B, N, -1) if cm is not None else None
            if s.nh > 1 and past is None: past = cm[:, :-1].reshape(B, -1)                 # render path: earlier frames from the same batch
            elif past is not None: past = past.reshape(B, -1)
            return s.head_in(c0[:, -1], cm[:, -1] if cm is not None else None, past, st)
        def forward_all(s, img): return s.forward(img), None
        def forward(s, img):                                                               # rollout / check: one frame per control step, batch 1
            assert len(img) == 1 and _HK["x"] is not None, "h13: batch 1 with a state from the frame hook"
            assert s._n is None or _HK["n"] == s._n + 1, f"h13: {_HK['n'] - (s._n or 0)} frames since the last step (one expected)"
            x = _HK["x"]; s._n = _HK["n"]; c0, cm = s.feats(img); past = None
            if s.nh:
                s._buf.append(cm); t = len(s._buf) - 1
                if s.nh > 1: past = torch.cat([s._buf[max(0, t - j * s.dt)] for j in range(s.nh - 1, 0, -1)], 1)
            st = torch.tensor(h13_state(x, s._xp), device=img.device)[None] if s.stin else None; s._xp = x
            out, _ = s.head_in(c0, cm, past, st)
            if s.body:                                                                     # body -> world (+yaw); isotropic xy stats: commutes with de-normalisation
                o = out.reshape(1, -1, 5); yw = h13_yaw(x); c, sn = math.cos(yw), math.sin(yw); vx, vy = o[..., 0].clone(), o[..., 1].clone()
                o = torch.stack([c * vx - sn * vy, sn * vx + c * vy, o[..., 2], o[..., 3], o[..., 4]], -1); out = o.reshape(1, -1)
            return out
    return H13Student()


def h13_from_cfg(seed, cfg):
    import gate1_student as g1
    base = (_G1B or g1.build)(seed)
    return build_h13(seed, build_concept(seed, base, cfg["feat"]) if cfg["conc"] else base, cfg)


def check_winit_h13():
    """Initialize a worker with the checkpoint model and forward rollout state to it."""
    global _G1B
    import gate2_multi as g2, gate1_student as g1
    (check_winit if os.environ.get("TORCH_THREADS") else g2.winit)()
    cfg = json.loads(os.environ["V13H_CFG"])
    if _G1B is None: _G1B = g1.build
    g1.build = lambda seed, state=False: h13_from_cfg(seed, cfg)
    _pf = g2.policy_frame
    def pf(yard, x): h13_hook(x); return _pf(yard, x)
    g2.policy_frame = pf


_HP = {}; _HPN = {}; _HCF = {}                                                          # pools, pool -> cache names, loaded cache files


def h13_stream_pools(G, y):
    """per stream frame of yard y (gate2_multi.yard_data order): (flight arrays, flight index, state index); exact match asserted by the caller"""
    gy = YMAP.get(y, y); E = {r["id"]: r for r in json.load(open(f"{G}/{gy}/expert.json"))["flights"]}
    ids = [m["id"] for m in json.load(open(f"{G}/{gy}/starts.json"))["starts"] if m["split"] == "train"
           and E.get(m["id"], {}).get("outcome") == "success" and E[m["id"]].get("min_clearance_m", 0) >= 0.30]
    P = [np.load(f"{G}/{gy}/tracks/{i}.npz")["X"].astype(np.float64) for i in ids]
    return ids, P, np.concatenate([np.full(len(p), j, np.int32) for j, p in enumerate(P)]), np.concatenate([np.arange(len(p), dtype=np.int32) for p in P])


def h13_dag_pools(spec, G):
    """per row of load_dag(spec): (pool key, state index) into _HP; DAgger rows matched to the student track, recovery rows appended to it"""
    keys, idx = [], []
    for n, p in enumerate(spec.split(",")):
        z = np.load(p); X32 = z["X"].astype(np.float32); fl = z["flight"]; yd_ = z["yard"].astype(str); d = os.path.dirname(p)
        sd = os.path.basename(p)[:-4].split("_")[-1]; td = f"{d}/tracks_{sd}"; rec = os.path.basename(p).startswith("rec_")
        js = json.load(open(p[:-4] + ".json")); FL = [r for r in js["flights"] if "error" not in r] if rec else js["per_flight"]
        K = np.zeros(len(X32), np.int64); I = np.zeros(len(X32), np.int32)
        for f in np.unique(fl):
            r = FL[int(f)]; rows = np.flatnonzero(fl == f); assert (yd_[rows] == r["yard"]).all(), f"h13: {p} flight {f} yard"
            tp = f"{td}/{r['yard']}/{r['id']}.npz"; T = np.load(tp)["X"].astype(np.float64).reshape(-1, 10); key = n * 1_000_000 + int(f)
            _HPN[key] = {"trk": "trk/" + os.path.relpath(tp, G), "rec": (f"rec/{os.path.relpath(p, G)[:-4]}/{int(f)}.npz", len(T)) if rec else None}   # cache names (dumper's)
            if rec:
                _HP[key] = np.concatenate([T, X32[rows].astype(np.float64)]); I[rows] = len(T) + np.arange(len(rows))
            else:
                T32 = T.astype(np.float32); j = 0
                for q in rows:
                    while j < len(T32) and not np.array_equal(T32[j], X32[q]): j += 1
                    assert j < len(T32), f"h13: {p} flight {f} row {q} not found in its track"
                    I[q] = j; j += 1
                _HP[key] = T
            K[rows] = key
        keys.append(K); idx.append(I)
    return np.concatenate(keys), np.concatenate(idx)


def h13_cfeat(key, p):
    """frozen feature (768, float16) of grid state p of pool key, from the HIST cache (dumper names); FAKE caches hold one constant C"""
    N = _HPN[key]; f = N["rec"][0] if N["rec"] and p >= N["rec"][1] else N["trk"]
    if f not in _HCF:
        z = np.load(f"{HCD}/{f}"); assert "C" not in z.files or FAKE, f"v13j: FAKE constant cache {f} refused (FAKE_RENDER is not 1)"
        nexp = len(_HP[key]) if f != N["trk"] or not N["rec"] else N["rec"][1]     # a recovery pool's student-track file holds the track only (length LT)
        assert int(z["n"]) == nexp, f"v13j: cache {f} n {int(z['n'])} != expected length {nexp}"
        _HCF[f] = (z["C"], None, None) if "C" in z.files else (None, z["K"], z["F"])
    C, K, F = _HCF[f]
    if C is not None: return C
    j = int(np.searchsorted(K, p)); assert j < len(K) and K[j] == p, f"h13 cache: state {p} not on the grid of {f}"
    return F[j]


def h13_past(P, i):
    """HIST states of pool P at index i (oldest first, padded with P[0]) and the previous state (None at index 0)"""
    S = [P[max(0, i - j * HDT)] for j in range(HIST - 1, -1, -1)] if HIST else []
    return S, (P[i - 1] if i > 0 else None)


def lab_img(labs, imgs, y, x, yk, my, sy):
    """append one frame's labels and image (CONCEPT_IN: 100 policy labels + 21 concept labels, image from the same render)"""
    if XYF == "body": yk = h13_rot(yk, h13_yaw(x), -1)
    if CONC: im, cl = concept_frame(y, x); labs.append(np.r_[(yk - my) / sy, cl]); imgs.append(im)
    else: labs.append((yk - my) / sy); imgs.append(frame(y, x))


def yards():
    import gate2_multi as g2
    Y = os.environ["YARDS"].split(",") if os.environ.get("YARDS") else list(g2.TRAIN_YARDS)
    assert not set(Y) & set(g2.HELD_OUT), "held-out yard"
    return Y


def frame(yard, x):
    if _HK["on"]: h13_hook(x)                                                           # rollout with an h13 model (off: untouched)
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
        m = build_concept(int(c.get("seed", 0)), feat=c["concept_in"].get("feat", "cls")) if c.get("concept_in") else g1.build(int(c.get("seed", 0)))
        if c.get("h13"): m = h13_from_cfg(int(c.get("seed", 0)), c["h13"]); _HK["on"] = True
        m.load_state_dict(c["state"]); _M[ck] = (m.to(DEV).eval(), c)
    m, c = _M[ck]
    if c.get("h13"): m.h13_reset()                                                      # per-flight buffer
    L = g2.land(yard); ctl = gv.DirectGoalExpert.__new__(gv.DirectGoalExpert); ctl.prev_oz = 0.0
    x = np.r_[np.asarray(s["pos"], float), np.zeros(3), _yaw_quat(s["yaw"])]; u_prev = U0.copy()
    budget = 1.5 * s["t_expert_s"]; N = int(round(budget * 20)); K = g1.K; w = np.exp(-g1.CHUNK_M * np.arange(K)); hist = []
    if s.get("near") is not None:                                                       # NEAR_DOOR start (expert state X[k], remaining budget)
        x = np.load(f"{G}/{yard}/tracks/{s['id']}.npz")["X"][s["near"]["k"]].astype(float).copy(); budget = s["near"]["budget_s"]; N = int(round(budget * 20))
    P, V, T, LX = [x[:3].copy()], [x[3:6].copy()], [0.0], []
    stopc = os.environ.get("FLY_STOP_CONTACT") == "1"; tko = os.environ.get("TAKEOVER") == "1"
    cf = clearance_fn(yard) if (stopc or tko) else None; tk = None; stop_k = None
    if tko:
        route = np.load(f"{G}/{yard}/tracks/{s['id']}.npz")["P"]; TKC, TKD, TKP = (float(os.environ.get(e, d)) for e, d in (("TK_CLR", "0.50"), ("TK_DEV", "1.0"), ("TK_PRED", "0.30")))
    if tko and s.get("near") is not None: route = route[s["near"]["k"]:]               # the rest of the expert track
    for k in range(N):
        if cf is not None:
            c0 = float(np.asarray(cf(x[None, :3])).ravel()[0])
            if tko:
                dv = float(np.linalg.norm(route - x[:3], axis=1).min())
                if c0 < TKC: tk = {"reason": "clr", "k": k, "clr": c0, "dev": dv}
                elif dv > TKD: tk = {"reason": "dev", "k": k, "clr": c0, "dev": dv}
                if tk is not None: break
            if stopc and c0 < 0.30: stop_k = k; break
        with torch.no_grad():
            cc = m(torch.tensor(g1.prep(frame(yard, x)[None]), device=DEV)).cpu().numpy()[0] * c["sy"] + c["my"]
        hist.append(cc.reshape(K, 5)); hist = hist[-K:]
        if tko:
            ch = hist[-1]; Pp = np.c_[x[0] + np.cumsum(ch[:, 0]) / 20, x[1] + np.cumsum(ch[:, 1]) / 20, x[2] + ch[:, 2]]
            cp = float(np.asarray(cf(Pp)).min())
            if cp < TKP: tk = {"reason": "pred", "k": k, "clr": c0, "dev": dv, "pred_clr": cp}; break
        preds = np.array([hist[-1 - j][j] for j in range(len(hist))]); ww = w[:len(preds)][::-1]; g = (preds * ww[:, None]).sum(0) / ww.sum()
        yaw = float(st.yaw_of(x[6:10]))
        u, _ = gv.DirectGoalExpert._u(ctl, x[0:3], x[3:6], x[6:10], g[0:2], float(x[2] + g[2]), float(st.wrap(yaw + g[3])), w_clip=0.3 if g[4] > 0.5 else 0.4)
        LX.append(x.copy()); x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / 20)
    P, V, T = np.array(P), np.array(V), np.array(T); clr = clearance_fn(yard)(P); r = score_track(T, P, V, L, clr, budget)
    os.makedirs(f"{td}/{yard}", exist_ok=True)
    np.savez_compressed(p[:-4] + ".tmp.npz", X=np.array(LX), clr=clr[:len(LX)], **({"XTK": x.copy()} if tk is not None else {})); os.replace(p[:-4] + ".tmp.npz", p)
    r = {k: (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, (np.integer, np.bool_)) else v)) for k, v in r.items()}
    r.update({"id": s["id"], "yard": yard, "split": s["split"], "frames": len(LX), "fake_render": FAKE})
    if s.get("near") is not None: r["near_door"] = s["near"]
    if tk is not None: r.update({"outcome_partial": r.get("outcome"), "outcome": "takeover", "takeover": tk})
    if stop_k is not None: r["stopped_contact_k"] = stop_k
    json.dump(r, open(p[:-4] + ".json", "w"), default=float); return r


def fly_winit():
    import obs_relabel_z as oz, score, torch
    oz.init_worker(); score.SY = FLAMBE_YARDS
    if DEV == "cpu": torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
    elif os.environ.get("TORCH_THREADS"): torch.set_num_threads(int(os.environ["TORCH_THREADS"]))


def check_winit():
    """Initialize an evaluation worker and apply TORCH_THREADS when set."""
    import gate2_multi as g2, torch
    g2.winit()
    if os.environ.get("TORCH_THREADS"): torch.set_num_threads(int(os.environ["TORCH_THREADS"]))


def fly(G, seed, ck, OUT, NP):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    td = f"{OUT}/tracks_s{seed}"; os.makedirs(td, exist_ok=True)
    YS = yards(); TOT = int(os.environ.get("FLY_TOTAL", "0")); J = []
    for i, y in enumerate(YS):
        S = train_starts(G, y)
        if os.environ.get("FLY_IDS"):                      # this round's block of train start ids per yard (json {yard: [ids]})
            want = json.load(open(os.environ["FLY_IDS"]))[y]; byid = {s_["id"]: s_ for s_ in S}
            assert all(w in byid for w in want), f"FLY_IDS: {y} ids not among the expert-successful train starts"; S = [byid[w] for w in want]
        elif TOT:                                          # TOT starts over the yards, evenly spaced within each yard's train starts
            n = TOT // len(YS) + (i < TOT % len(YS)); S = [S[j] for j in np.unique(np.linspace(0, len(S) - 1, n).round().astype(int))]
        if NEAR is not None:                                                           # near-door start per selected start, or drop it
            import gate2_multi as g2; Ld = g2.land(y); S2 = []
            for s_ in S:
                z_ = np.load(f"{G}/{y}/tracks/{s_['id']}.npz"); d_ = np.linalg.norm(z_["P"] - Ld, axis=1); kk = np.flatnonzero((d_ >= NEAR[0]) & (d_ <= NEAR[1]))
                if len(kk):
                    rs = int("".join(c for c in s_["id"] if c.isdigit())) + int(seed); k0 = int(kk[np.random.default_rng(rs).integers(len(kk))])
                    print(f"[g2 dagger fly s{seed}] v12b NEAR_DOOR {y} {s_['id']}: rng seed {rs}, track index k {k0} ({len(kk)} expert states in band), d0 {float(d_[k0]):.2f} m", flush=True); S2.append(dict(s_, near={"rng_seed": rs, "k": k0, "t0": float(z_["T"][k0]), "d0": float(d_[k0]), "budget_s": 1.5 * (s_["t_expert_s"] - float(z_["T"][k0]))}))
            print(f"[g2 dagger fly s{seed}] v12 NEAR_DOOR {NEAR[0]:g}-{NEAR[1]:g} m: {y} {len(S2)} of {len(S)} starts have a near-door expert state", flush=True); S = S2
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
    k = {o: sum(r["outcome"] == o for r in res) for o in ("success", "contact", "overshoot", "timeout") + (("takeover",) if any(r["outcome"] == "takeover" for r in res) else ())}
    if "takeover" in k:
        print(f"[g2 dagger fly s{seed}] takeover reasons " + json.dumps({q: sum(r.get("takeover", {}).get("reason") == q for r in res) for q in ("clr", "dev", "pred")})
              + f"; student frames per flight mean {np.mean([r.get('frames', 0) for r in res]):.0f}, median {np.median([r.get('frames', 0) for r in res]):.0f}", flush=True)
    print(f"G2_DAGGER_FLY s{seed}: {len(res)} train starts {k}, {sum(r.get('frames', 0) for r in res)} states ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- relabel
_CACHE = {}


def rl_winit(yard):
    os.environ["YARD"] = yard
    import lead_expert_yard as ge                         # YARD read at import: one pool per yard
    import lead_phase_seed, obs_relabel_z as oz, score
    assert ge.YARD == yard
    if not getattr(ge, "_phase_seed", False):
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
            nr.append([k] + [round(float(v), 2) for v in x[:3]] + [round(float(clr[k]), 2), key]); continue
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
        with np.load(f"{PD}/{y}__{s['id']}.npz") as z: Xs.append(z["X"].reshape(-1, 10)); Ys.append(z["Y"]); R.append(json.loads(str(z["R"])))   # a 0-frame takeover flight stores X shape (0,)
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


def recover_one(args):
    """Build the yard expert from a takeover state."""
    td, yard, sid, PD = args
    import obs_relabel_z as oz, goal_expert_v4 as gv, lead_expert_yard as ge, visual_expert_syn as ve
    from score import clearance_fn
    x = np.asarray(np.load(f"{td}/{yard}/{sid}.npz")["XTK"], float)
    try:
        b = ge.fly_one(dict(id=sid, split="train", pos=x[:3].tolist(), yaw=0.0, _build_only=True, _relabel=True, _cache=_CACHE))
    except Exception as e:
        b = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(b, dict) or b[0].pose is None:
        return {"id": sid, "yard": yard, "error": (b.get("error", "no route") if isinstance(b, dict) else "no route")[:90]}
    ex, goals, cache = b; u_prev = ge.U0.copy(); cl = clearance_fn(yard)
    XS, GS, PH = [], [], []; k_land = None; first_c = None
    for k in range(int(ge.MAX_S * ve.HZ_CTL)):
        if float(cl(x[None, :3])[0]) < 0.30: first_c = k; break                   # frames kept until its own contact
        oz.dv.maybe_replan(ex, x, cache); u = ex.decide(k / ve.HZ_CTL, x.copy())
        XS.append(x.copy()); GS.append(np.asarray(ex._guid, float)); PH.append(gv.PHASE_ID.get(ex.phase, 6))
        if ex.landed and k_land is None: k_land = k
        if k_land is not None and (k - k_land) >= ge.HOLD_AFTER * ve.HZ_CTL: break
        x = oz.DYN.step(x, u_prev, u); u_prev = u
    p = f"{PD}/{yard}__{sid}.npz"
    np.savez_compressed(p[:-4] + ".tmp.npz", X=np.array(XS).reshape(-1, 10), G=np.array(GS).reshape(-1, 4), PH=np.array(PH, int)); os.replace(p[:-4] + ".tmp.npz", p)
    return {"id": sid, "yard": yard, "frames": len(XS), "landed": k_land is not None, "contact": first_c is not None, "success": k_land is not None and first_c is None}


def recover(G, seed, OUT, NP):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import gate0b_student as st
    td = f"{OUT}/tracks_s{seed}"; PD = f"{OUT}/recover_s{seed}"; os.makedirs(PD, exist_ok=True); t0 = time.time(); res = []; why = {}
    for y in yards():
        J = []
        for s in train_starts(G, y):
            q = f"{td}/{y}/{s['id']}.json"
            if os.path.exists(q):
                r = json.load(open(q))
                if r.get("outcome") == "takeover": J.append(s["id"]); why[(y, s["id"])] = r["takeover"]["reason"]
        if not J: continue
        print(f"[g2 recover s{seed}] {y}: {len(J)} takeover states, {min(NP, len(J))} processes", flush=True)
        with ProcessPoolExecutor(min(NP, len(J)), mp_context=get_context("spawn"), initializer=rl_winit, initargs=(y,)) as p:
            res += list(p.map(recover_one, [(td, y, i, PD) for i in J], chunksize=1))
    ok = [r for r in res if "error" not in r]; Xs, Ys, Yd, Fl = [], [], [], []
    for n, r in enumerate(ok):
        z = np.load(f"{PD}/{r['yard']}__{r['id']}.npz")
        if len(z["X"]): Xs.append(z["X"]); Ys.append(st.labels(z["X"], z["G"], z["PH"])); Yd += [r["yard"]] * len(z["X"]); Fl.append(np.full(len(z["X"]), n, np.int32))
    out = f"{OUT}/rec_s{seed}.npz"
    X = np.concatenate(Xs).astype(np.float32) if Xs else np.zeros((0, 10), np.float32); Y = np.concatenate(Ys).astype(np.float32) if Ys else np.zeros((0, st.K * 5), np.float32)
    np.savez_compressed(out[:-4] + ".tmp.npz", X=X, Y=Y, yard=np.array(Yd), flight=np.concatenate(Fl) if Fl else np.zeros(0, np.int32)); os.replace(out[:-4] + ".tmp.npz", out)
    per = {q: {"n": sum(why[(r["yard"], r["id"])] == q for r in res), "plannable": sum(why[(r["yard"], r["id"])] == q for r in ok)} for q in ("clr", "dev", "pred")}
    log = {"seed": seed, "takeovers": len(res), "plannable": len(ok), "plannable_share": round(len(ok) / max(1, len(res)), 4), "success": sum(r["success"] for r in ok),
           "contact": sum(r["contact"] for r in ok), "frames": int(len(X)), "per_reason": per, "wall_s": round(time.time() - t0), "flights": res}
    json.dump(log, open(out[:-4] + ".json", "w"), default=float)
    print("G2_RECOVER", json.dumps({k: v for k, v in log.items() if k != "flights"}), flush=True)


def check(G, CK, IDS, OUT, NP):
    """copy check on fixed test ids with the gate2_multi.evaluate loop"""
    import gate2_multi as g2
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    W = json.load(open(IDS)); os.makedirs(OUT, exist_ok=True); os.environ["EVAL_OUT"] = OUT; t0 = time.time(); tot = {}
    import torch; _ci = torch.load(CK, map_location="cpu", weights_only=False).get("concept_in"); CKC = bool(_ci)   # concept-in checkpoint -> patched build in the workers
    if CKC: os.environ["V13_CONCEPT_FEAT"] = _ci.get("feat", "cls")                         # spawned workers inherit it
    AB = os.environ.get("CONCEPT_ABLATE", ""); assert AB in ("", "mean", "lag"), "CONCEPT_ABLATE: mean or lag"
    if AB:
        assert CKC, "CONCEPT_ABLATE needs a concept-in checkpoint"; os.environ["V13_CONCEPT_ABLATE"] = AB
        if AB == "mean": assert "mean_pred" in _ci, "CONCEPT_ABLATE=mean needs a checkpoint trained with CONCEPT_STATS=1"; os.environ["V13_CONCEPT_MEAN"] = json.dumps(_ci["mean_pred"])
        print(f"[g2 check] v13d CONCEPT_ABLATE {AB}" + (f" (lag {os.environ.get('ABL_LAG', '40')} steps)" if AB == "lag" else f" (mean_pred over {_ci.get('n_frames')} training frames)"), flush=True)
    if CKC: print(f"[g2 check] v13 concept-in checkpoint: workers build the concept-in student", flush=True)
    _h13 = torch.load(CK, map_location="cpu", weights_only=False).get("h13")
    if _h13:
        assert not AB, "v13h: CONCEPT_ABLATE is refused for an h13 checkpoint"; os.environ["V13H_CFG"] = json.dumps(_h13)
        print(f"[g2 check] v13h checkpoint switches: {_h13}", flush=True)
    for y, ids in W.items():
        EP = os.environ.get("EXPERT", "").replace("{yard}", y) or f"{G}/{y}/expert.json"; SP = os.environ.get("STARTS", "").replace("{yard}", y) or f"{G}/{y}/starts.json"
        E = {r["id"]: r for r in json.load(open(EP))["flights"]}
        S = {s["id"]: dict(s, t_expert_s=E[s["id"]]["t_landed_s"]) for s in json.load(open(SP))["starts"]
             if s["split"] == "test" and E.get(s["id"], {}).get("outcome") == "success"}
        assert all(i in S for i in ids), f"check: {y} ids not among the expert-successful test starts"
        with ProcessPoolExecutor(NP, mp_context=get_context("spawn"), initializer=check_winit_h13 if _h13 else (check_winit_concept if CKC else (check_winit if os.environ.get("TORCH_THREADS") else g2.winit))) as p:
            res = list(p.map(g2.fly_one, [(G, y, CK, S[i]) for i in ids], chunksize=1))
        json.dump({"ckpt": CK, "yard": y, "flights": res}, open(f"{OUT}/eval_{y}.json.tmp", "w")); os.replace(f"{OUT}/eval_{y}.json.tmp", f"{OUT}/eval_{y}.json")
        k = {o: sum(r["outcome"] == o for r in res) for o in ("success", "contact", "overshoot", "timeout")}
        for o, v in k.items(): tot[o] = tot.get(o, 0) + v
        print(f"[g2 check] {y}: {len(res)} test starts {k} ({time.time() - t0:.0f}s)", flush=True)
    print(f"R78_CHECK {CK}: {sum(tot.values())} starts {tot} ({time.time() - t0:.0f}s)", flush=True)


def per_yard(R):
    """drop shares per yard"""
    out = {}
    for r in R:
        o = out.setdefault(r["yard"], {"flights": 0, "visited": 0, "kept": 0, "dropped_clearance": 0, "dropped_no_route": 0})
        o["flights"] += 1; o["visited"] += r["visited"]; o["kept"] += r["kept"]; o["dropped_clearance"] += r["dropped_clearance"]; o["dropped_no_route"] += r["dropped_no_plan"]
    for o in out.values(): o["no_route_share"] = round(o["dropped_no_route"] / max(1, o["visited"]), 4); o["clearance_share"] = round(o["dropped_clearance"] / max(1, o["visited"]), 4)
    return out


def stream_ids(G, y):
    """gate 2 rule: yard_data train ids at index 0/40/80/120/160"""
    E = {r["id"]: r for r in json.load(open(f"{G}/{y}/expert.json"))["flights"]}
    ids = [q["id"] for q in json.load(open(f"{G}/{y}/starts.json"))["starts"] if q["split"] == "train"
           and E.get(q["id"], {}).get("outcome") == "success" and E[q["id"]].get("min_clearance_m", 0) >= 0.30]
    return [ids[i] for i in (0, 40, 80, 120, 160) if i < len(ids)]


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


def load_dag(spec):
    """one npz or a comma list of npz files (X, Y, yard) concatenated in order"""
    if "," not in spec: return np.load(spec)
    Z = [np.load(p) for p in spec.split(",")]
    return {k: np.concatenate([z[k] for z in Z]) for k in ("X", "Y", "yard")}


def dag_src(spec):
    """per DAgger/recovery state, kind (1 DAgger, 2 recovery = file name rec_*) and a flight key (file index, flight)"""
    K, F = [], []
    for n, p in enumerate(spec.split(",")):
        z = np.load(p); m = len(z["X"]); K.append(np.full(m, 2 if os.path.basename(p).startswith("rec_") else 1, np.int8))
        F.append(n * 1_000_000 + (z["flight"].astype(np.int64) if "flight" in z.files else np.zeros(m, np.int64)))
    return np.concatenate(K), np.concatenate(F)


R78 = bool(os.environ.get("MIX") or os.environ.get("STREAM_YARDS"))              # frame tags + share logging only when set
TW = float(os.environ.get("TURN_W") or 0); TURN_LO, TURN_HI = 10.0, 27.0
assert TW >= 0, "TURN_W must be a positive number (unset = off)"
TSW = float(os.environ.get("TURN_SEG_W") or 0); assert TSW >= 0, "TURN_SEG_W must be a positive number (unset = off)"
assert not (TW and TSW), "TURN_W and TURN_SEG_W are exclusive"
WT = TW or TSW                                                                      # the active per-frame weight (0 = off)
DOOR = float(os.environ.get("DOOR_AUX") or 0); DOOR_R = float(os.environ.get("DOOR_R", "27")); assert DOOR >= 0, "DOOR_AUX must be a positive number"   # off by default
UNF = int(os.environ.get("UNFREEZE_LAST") or 0); assert 0 <= UNF <= 12, "UNFREEZE_LAST in 1..12"
_KIND = {"stream": 0, "dagger": 1, "recovery": 2}
DSRC = sorted({_KIND[q.strip()] for q in os.environ["DOOR_SRC"].split(",")}) if os.environ.get("DOOR_SRC") else None
ENC_INIT = os.environ.get("ENC_INIT") or None; assert ENC_INIT in (None, "dinov2"), "ENC_INIT: dinov2 or unset"
HEAD_INIT = os.environ.get("HEAD_INIT") or None; assert HEAD_INIT in (None, "fresh"), "HEAD_INIT: fresh or unset"
FRZ = os.environ.get("FREEZE_ENC") == "1"; assert not (FRZ and UNF), "FREEZE_ENC and UNFREEZE_LAST are exclusive"
ELR_ENV = float(os.environ["ENC_LR"]) if os.environ.get("ENC_LR") else None


def door_t(yard, x):
    """(body-frame unit vector to the door land point, mask 1/0) for one state"""
    from scipy.spatial.transform import Rotation as Rot
    if yard not in _LAND:
        import gate2_multi as g2
        _LAND[yard] = g2.land(yard)
    x = np.asarray(x, float); v = _LAND[yard] - x[:3]; d = float(np.linalg.norm(v))
    if d < 0.5 or d > DOOR_R: return np.zeros(3, np.float32), 0.0
    return (Rot.from_quat(x[6:10]).as_matrix().T @ (v / d)).astype(np.float32), 1.0
SEG_HI, SEG_LO, SEG_MIN_D = 30.0, 10.0, 1.0
_SEG = {}


def seg_mask(X, L):
    """boolean per frame of one expert flight, True inside its turn segment"""
    from scipy.spatial.transform import Rotation as Rot
    X = np.asarray(X, float); d = np.linalg.norm(X[:, :3] - L, axis=1); ok = d >= SEG_MIN_D; m = np.zeros(len(X), bool)
    yaw = Rot.from_quat(X[:, 6:10]).as_euler("ZYX")[:, 0]; brg = np.arctan2(L[1] - X[:, 1], L[0] - X[:, 0])
    e = np.degrees(np.abs((yaw - brg + np.pi) % (2 * np.pi) - np.pi))
    hi = np.flatnonzero(ok & (e > SEG_HI))
    if not len(hi): return m
    lo = np.flatnonzero(ok & (e < SEG_LO) & (np.arange(len(X)) > hi[-1]))
    if len(lo): m[hi[-1] + 1:lo[0] + 1] = True
    return m


def seg_flags(G, yard):
    """turn-segment flag per stream frame of a yard, in gate2_multi.yard_data order (same id rule, same flight order)"""
    if yard not in _SEG:
        import gate2_multi as g2
        E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
        ids = [m["id"] for m in json.load(open(f"{G}/{yard}/starts.json"))["starts"] if m["split"] == "train"
               and E.get(m["id"], {}).get("outcome") == "success" and E[m["id"]].get("min_clearance_m", 0) >= 0.30]
        L = g2.land(yard); _SEG[yard] = np.concatenate([seg_mask(np.load(f"{G}/{yard}/tracks/{i}.npz")["X"], L) for i in ids])
    return _SEG[yard]


def frame_w(G, yard, x, ks):
    """the active weight for one frame; ks = its index in the yard's stream frames, None for a DAgger / recovery state"""
    if TW: return turn_w(yard, x)
    return TSW if ks is not None and seg_flags(G, yard)[ks] else 1.0
_LAND = {}


def turn_w(yard, x):
    """TW if the 3-D distance to the yard's door land point is in [TURN_LO, TURN_HI] m, else 1.0"""
    if yard not in _LAND:
        import gate2_multi as g2
        _LAND[yard] = g2.land(yard)
    d = float(np.linalg.norm(np.asarray(x[:3], float) - _LAND[yard])); return TW if TURN_LO <= d <= TURN_HI else 1.0


_AUG = None


def aug_batch(xb):
    """Apply independently sampled image augmentations to each image in a uint8 batch."""
    global _AUG
    import torch
    if _AUG is None:
        from torchvision.transforms import v2
        _AUG = v2.Compose([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                           v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.2)])
    return torch.stack([_AUG(x) for x in xb])


class MixStream(_torch.utils.data.IterableDataset):
    """gate2_multi.Stream (AUX none) with each yard's DAgger states appended to its stream frames; yards sharded over workers"""
    def __init__(self, G, yards, my, sy, seed, mb, dag):
        super().__init__(); self.G, self.yards, self.my, self.sy, self.seed, self.mb, self.dag = G, yards, my, sy, seed, mb, dag

    def __iter__(self):
        import torch, gate2_multi as g2
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        torch.set_num_threads(1); mine = self.yards[wid::nw]
        z = load_dag(self.dag); DY = z["yard"]; data = {}; SRC = dag_src(self.dag)[0] if R78 else None
        if os.environ.get("MIX"):                         # MIX=5050 (4 + 4) or MIX=<f> (per-frame draw)
            yield from self._mix5050(mine, z, DY, SRC); return
        tags = {}; nst = {}                                                              # nst = stream frames per yard
        for y in mine:
            X, Y = yd(self.G, y)[:2]; s = DY == y
            data[y] = (np.concatenate([X, z["X"][s].astype(np.float64)]), np.concatenate([Y, z["Y"][s]]))
            if TSW: assert len(seg_flags(self.G, y)) == len(X), f"{y}: segment flags {len(seg_flags(self.G, y))} != stream frames {len(X)}"; nst[y] = len(X)
            if R78: tags[y] = np.concatenate([np.zeros(len(X), np.int8), SRC[s]])
        rng = np.random.default_rng(self.seed * 100 + wid); stride = int(os.environ.get("STRIDE", "4")); ep = 0
        yi = np.concatenate([np.full(len(data[y][0]), j, np.int16) for j, y in enumerate(mine)])
        ki = np.concatenate([np.arange(len(data[y][0]), dtype=np.int32) for y in mine])
        while True:
            sel = np.where(ki % stride == ep % stride)[0]; order = sel[rng.permutation(len(sel))]
            for a in range(0, len(order) - self.mb + 1, self.mb):
                imgs, labs = [], []
                tg = []; ws = []; dr = []
                for o in order[a:a + self.mb]:
                    y, k = mine[int(yi[o])], int(ki[o]); X, Y = data[y]; lab_img(labs, imgs, y, X[k], Y[k], self.my, self.sy)
                    if R78: tg.append(int(tags[y][k]))
                    if WT: ws.append(frame_w(self.G, y, X[k], (k if k < nst[y] else None) if TSW else None))
                    if DOOR: dr.append(door_t(y, X[k]))
                if DOOR: yield (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32))) + ((torch.tensor(tg),) if R78 else ()) + ((torch.tensor(ws, dtype=torch.float32),) if WT else ()) + (torch.from_numpy(np.stack([q[0] for q in dr])), torch.tensor([q[1] for q in dr], dtype=torch.float32))
                elif WT: yield (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32))) + ((torch.tensor(tg),) if R78 else ()) + (torch.tensor(ws, dtype=torch.float32),)
                elif R78: yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)), torch.tensor(tg)
                else: yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32))
            ep += 1

    def _mix5050(self, mine, z, DY, SRC):
        import torch, gate2_multi as g2
        rng = np.random.default_rng(self.seed * 100 + 7 + (torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0))
        stride = int(os.environ.get("STRIDE", "4")); h = self.mb // 2
        dy = sorted(set(DY.tolist())) if os.environ.get("STREAM_YARDS") == "ckpt" else mine     # pooled DAgger yards for every worker
        S = {}
        for y in mine:
            Xs_, Ys_ = yd(self.G, y)[:2]; S[y] = (Xs_, Ys_, np.zeros(len(Xs_), np.int8))
            if TSW: assert len(seg_flags(self.G, y)) == len(Xs_), f"{y}: segment flags != stream frames"
        Dd = {y: (z["X"][DY == y].astype(np.float64), z["Y"][DY == y], SRC[DY == y]) for y in dy}
        if HIST or STIN:                                                                          # past-state pools per sample
            hk, hi = h13_dag_pools(self.dag, self.G); Dd = {y: Dd[y] + (hk[DY == y], hi[DY == y]) for y in dy}
            for y in mine:
                ids_, Pl, fj, ij = h13_stream_pools(self.G, y); assert len(fj) == len(S[y][0]) and np.array_equal(np.concatenate(Pl), S[y][0]), f"h13: {y} stream pools != yard_data"
                kb = 2_000_000_000 + int(y[4:]) * 100_000                                         # stream pool keys (DAgger keys < 1e9)
                for j_, p_ in enumerate(Pl): _HP[kb + j_] = p_; _HPN[kb + j_] = {"trk": f"stream/{y}/{ids_[j_]}.npz", "rec": None}
                S[y] = S[y] + (kb + fj.astype(np.int64), ij)
            for y in dy:
                for q in range(len(Dd[y][0])): assert np.array_equal(_HP[int(Dd[y][3][q])][int(Dd[y][4][q])].astype(np.float32), Dd[y][0][q].astype(np.float32)), f"h13: {y} DAgger row {q} pool mismatch"
            if HC:                                                                                # cache: current frames only from grid states
                S = {y: tuple(a[S[y][4] % HDT == 0] for a in S[y]) for y in mine}; Dd = {y: tuple(a[Dd[y][4] % HDT == 0] for a in Dd[y]) for y in dy}
                print(f"[g2 dagger train] v13h HIST cache: grid states only, stream {sum(len(S[y][0]) for y in mine)} / DAgger+recovery {sum(len(Dd[y][0]) for y in dy)} (worker yards)", flush=True)
        Dk = {kd: {y: tuple(a[Dd[y][2] == kd] for a in Dd[y]) for y in dy} for kd in (1, 2)}              # DAgger (1) / recovery (2)
        MX = os.environ["MIX"]; fr = None if MX == "5050" else float(MX); assert fr is None or 0.0 < fr < 1.0, f"MIX={MX}: 5050 or a fraction in (0, 1)"
        side = []
        for src, ys in ((S, mine), (Dk[1], dy), (Dk[2], dy)):
            yi = np.concatenate([np.full(len(src[y][0]), j, np.int16) for j, y in enumerate(ys)]); ki = np.concatenate([np.arange(len(src[y][0]), dtype=np.int32) for y in ys])
            if src is S: assert len(ki), f"MIX={MX}: a worker has no stream frames for yards {ys}"
            side.append((src, ys, yi, ki) if len(ki) else None)
        assert side[1] is not None or side[2] is not None, f"MIX={MX}: no DAgger or recovery states"
        if side[1] is None or side[2] is None: print(f"[g2 dagger train] MIX={MX}: no {'DAgger' if side[1] is None else 'recovery'} states; the other kind takes its draws", flush=True)
        def orders(src, ys, yi, ki):
            ep = 0
            while True:
                sel = np.where(ki % stride == ep % stride)[0]
                for o in sel[rng.permutation(len(sel))]: yield ys[int(yi[o])], int(ki[o]), src
                ep += 1
        a_it = orders(*side[0]); d_it = orders(*side[1]) if side[1] else None; r_it = orders(*side[2]) if side[2] else None
        d_it, r_it = d_it or r_it, r_it or d_it                                                   # an empty kind -> the other takes its draws
        while True:
            imgs, labs, tg = [], [], []
            if fr is None: pick = [a_it] * h + [d_it] * ((self.mb - h) // 2) + [r_it] * (self.mb - h - (self.mb - h) // 2)
            else: pick = [a_it if rng.random() >= fr else (d_it if rng.random() < 0.5 else r_it) for _ in range(self.mb)]
            ws = []; dr = []
            sts = []; pasts = []
            for it_ in pick:
                y, k, src = next(it_); X, Y, T_ = src[y][:3]; lab_img(labs, imgs, y, X[k], Y[k], self.my, self.sy); tg.append(int(T_[k]))
                if HIST or STIN:                                                                  # past frames (rendered) and state
                    Pp = _HP[int(src[y][3][k])]; past, xp = h13_past(Pp, int(src[y][4][k]))
                    if HIST and HC: pasts.append(np.stack([h13_cfeat(int(src[y][3][k]), max(0, int(src[y][4][k]) - j * HDT)) for j in range(HIST - 1, 0, -1)]))
                    elif HIST: imgs[-1] = np.stack([frame(y, q) for q in past[:-1]] + [imgs[-1]])
                    if STIN: sts.append(h13_state(X[k], xp))
                if WT: ws.append(frame_w(self.G, y, X[k], k if src is S else None))
                if DOOR: dr.append(door_t(y, X[k]))
            if DOOR: yield (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)), torch.tensor(tg)) + ((torch.tensor(ws, dtype=torch.float32),) if WT else ()) + (torch.from_numpy(np.stack([q[0] for q in dr])), torch.tensor([q[1] for q in dr], dtype=torch.float32))
            elif WT: yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)), torch.tensor(tg), torch.tensor(ws, dtype=torch.float32)
            elif STIN or HC: yield (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)), torch.tensor(tg)) + ((torch.from_numpy(np.stack(sts)),) if STIN else ()) + ((torch.from_numpy(np.stack(pasts).astype(np.float32)),) if HC else ())
            else: yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)), torch.tensor(tg)


def train(G, seed, CK, DAG, OUT):
    import torch, gate2_multi as g2, gate1_student as g1
    assert not g2.AUX, "base fine-tune only (AUX must be unset)"
    os.makedirs(OUT, exist_ok=True)
    STEPS = int(os.environ.get("STEPS", os.environ.get("MAX_STEPS", "20000"))); NW = int(os.environ.get("RENDER_WORKERS", "4")); MB, B = 8, int(os.environ.get("BATCH", "64"))
    CE = int(os.environ.get("CKPT_EVERY_STEPS", "2500"))
    c = torch.load(CK, map_location="cpu", weights_only=False); assert int(c.get("seed", 0)) == seed, "checkpoint seed"
    torch.manual_seed(2000 + seed); Yd = list(c.get("yards") or g2.TRAIN_YARDS); YS = yards(); assert set(YS) <= set(Yd)
    my, sy = np.asarray(c["my"]), np.asarray(c["sy"])
    Ys = np.concatenate([yd(G, y)[1][::10] for y in Yd]); m2, s2 = Ys.mean(0), Ys.std(0) + 1e-6   # yd (identity unless YARD_DIRS maps a ckpt yard)
    assert np.allclose(m2, my) and np.allclose(s2, sy), "label statistics differ from the checkpoint's"
    if H13 or JOINT:                                                                       # printed only when a switch is on
        print(f"[g2 dagger train s{seed}] v13h switches: CONCEPT_JOINT={int(JOINT)} HIST={HIST} HIST_PROJ={HPJ} HIST_DT={HDT} HIST_SRC={HSRC if HIST else '-'} HIST_CACHE={HCD or '-'} STATE_IN={int(STIN)} XY_FRAME={XYF}", flush=True)
        if HC: print(f"[g2 dagger train s{seed}] v13h HIST cache: past frames from the frozen-feature cache carry NO augmentation (current frame live{' with AUG' if AUG else ''}); eval has none", flush=True)
    if HIST or STIN: assert os.environ.get("MIX") and HEAD_INIT == "fresh" and not (DOOR or WT), "v13h HIST / STATE_IN need MIX, HEAD_INIT=fresh, no DOOR_AUX / TURN_W / TURN_SEG_W"
    if H13: assert HEAD_INIT == "fresh" and not (DOOR or WT), "v13h wrapper needs HEAD_INIT=fresh, no DOOR_AUX / TURN_W / TURN_SEG_W"
    if HC: assert ENC_INIT == "dinov2" and FRZ, "v13j HIST_SRC=cache needs ENC_INIT=dinov2 and FREEZE_ENC=1 (the cache holds frozen DINOv2-small features)"
    if XYF == "body":                                                                      # body-frame label statistics (isotropic xy)
        Xs10 = np.concatenate([yd(G, y)[0][::10] for y in Yd]); my, sy = h13_body_stats(Xs10, Ys)
        print(f"[g2 dagger train s{seed}] v13h XY_FRAME body: xy labels rotated by -yaw at label time, command by +yaw at execution; isotropic xy stats, step-0 scale {sy[0]:.4f} (world sx {s2[0]:.4f} sy {s2[1]:.4f})", flush=True)
    SY_ = Yd if os.environ.get("STREAM_YARDS") == "ckpt" else YS                     # stream yards (default = YARDS, unchanged)
    if XSY:                                                                            # extra stream yards
        assert os.environ.get("STREAM_YARDS") == "ckpt", "EXTRA_STREAM_YARDS needs STREAM_YARDS=ckpt"; assert not set(XSY) & set(g2.HELD_OUT), "held-out yard"
        assert not set(XSY) & set(Yd), "extra stream yard already in the checkpoint list"; assert not (TSW and YMAP), "v13f YARD_DIRS with TURN_SEG_W not supported"
        for y_ in XSY: assert os.path.exists(f"{G}/{YMAP.get(y_, y_)}/expert.json"), f"{y_}: no {YMAP.get(y_, y_)}/expert.json"
        SY_ = list(Yd) + XSY; print(f"[g2 dagger train s{seed}] v13f EXTRA_STREAM_YARDS {XSY} (label dirs {[YMAP.get(y_, y_) for y_ in XSY]}): stream yards {len(SY_)}", flush=True)
    z = load_dag(DAG); nd = {y: int((z["yard"] == y).sum()) for y in YS}; ns = {y: len(yd(G, y)[0]) for y in SY_}
    for y_ in XSY: assert ns[y_] > 0, f"v13f: extra stream yard {y_} has no usable expert flights in gate2/{YMAP.get(y_, y_)} (missing YARD_DIRS?)"
    assert set(np.unique(z["yard"]).tolist()) <= set(YS), "DAgger/recovery yards outside YARDS"
    if R78:                                                                                # the startup line shows the MIX shares
        MXs = os.environ.get("MIX", ""); tgt = ("stream 0.50 / DAgger 0.25 / recovery 0.25 (4 + 2 + 2 per micro-batch)" if MXs == "5050" else
              (f"stream {1 - float(MXs):.2f} / DAgger {float(MXs) / 2:.3f} / recovery {float(MXs) / 2:.3f} (per frame)" if MXs else "pooled by frame count"))
        print(f"[g2 dagger train s{seed}] from {CK} (step {c.get('step')}); stream yards {len(SY_)} ({sum(ns.values())} frames), DAgger/recovery yards "
              f"{len([y for y in YS if nd[y]])} ({sum(nd.values())} states); target shares {tgt}; stride {os.environ.get('STRIDE', '4')}; {STEPS} steps, batch {B} = {B // MB} x {MB}, "
              f"{NW} render workers; AdamW fresh, enc {g1.ENC_LR:g} head {g1.LR:g} constant; dev {DEV}{' FAKE_RENDER (grey frames: not a result)' if FAKE else ''}" + ("; AUG on (train only)" if AUG else ""), flush=True)
    else: print(f"[g2 dagger train s{seed}] from {CK} (step {c.get('step')}); {len(YS)} yards; stream {sum(ns.values())} frames + DAgger {sum(nd.values())} states "
          f"({100 * sum(nd.values()) / (sum(ns.values()) + sum(nd.values())):.1f}% DAgger); stride {os.environ.get('STRIDE', '4')}; {STEPS} steps, batch {B} = {B // MB} x {MB}, "
          f"{NW} render workers; AdamW fresh, enc {g1.ENC_LR:g} head {g1.LR:g} constant; dev {DEV}{' FAKE_RENDER (grey frames: not a result)' if FAKE else ''}"
          + (f"; STREAM_YARDS ckpt: stream yards {len(SY_)}, DAgger/recovery yards {len([y for y in YS if nd[y]])}" if os.environ.get("STREAM_YARDS") == "ckpt" else "")
          + (f"; MIX 5050 (4 stream + 4 DAgger/recovery per micro-batch)" if os.environ.get("MIX") == "5050" else (f"; MIX {os.environ['MIX']} (per-frame DAgger/recovery probability)" if os.environ.get("MIX") else ""))
          + ("; AUG on (train only)" if AUG else ""), flush=True)
    if R78:
        kd, fk = dag_src(DAG); nuq = np.array([0, int((kd == 1).sum()), int((kd == 2).sum())])
        print(f"[g2 dagger train s{seed}] DAgger/recovery set: DAgger {int((kd == 1).sum())} states from {len(set(fk[kd == 1].tolist()))} flights, "
              f"recovery {int((kd == 2).sum())} states from {len(set(fk[kd == 2].tolist()))} flights; stream yards {len(SY_)} ({sum(ns.values())} frames)", flush=True)
        cnt = np.zeros(3, np.int64)
    if TW:                                                                                # the startup line shows the turn weight
        print(f"[g2 dagger train s{seed}] v8 TURN_W {TW:g}: frames {TURN_LO:g}-{TURN_HI:g} m (3-D) from the door land point weighted {TW:g}, others 1; "
              f"loss = weighted mean of per-frame mse", flush=True)
    if TSW:
        print(f"[g2 dagger train s{seed}] TURN_SEG_W {TSW:g}: stream frames inside the expert turn segment (last e > {SEG_HI:g} deg to the first "
              f"later e < {SEG_LO:g} deg, frames < {SEG_MIN_D:g} m from the land point left out) weighted {TSW:g}, others 1; loss = weighted mean of per-frame mse", flush=True)
    if WT: twn = np.zeros(3)                                                              # weighted frames, frames, their weight / weight sum
    m = g1.build(seed); m.load_state_dict(c["state"])
    assert not (DSRC is not None and not R78), "DOOR_SRC needs MIX or STREAM_YARDS (frame tags)"; assert not (DSRC is not None and not DOOR), "DOOR_SRC needs DOOR_AUX"
    if ENC_INIT == "dinov2":                                                              # pretrained DINOv2-small backbone, CLS readout unchanged
        from transformers import AutoModel
        pre = AutoModel.from_pretrained("facebook/dinov2-small").state_dict(); mk = m.feat.backbone.load_state_dict(pre, strict=True)
        print(f"[g2 dagger train s{seed}] v11 ENC_INIT dinov2: feat.backbone <- facebook/dinov2-small ({len(pre)} tensors, strict load {mk})", flush=True)
    if HEAD_INIT == "fresh":                                                              # re-initialised policy head
        torch.manual_seed(int(os.environ.get("HEAD_SEED", "4000")) + seed)                  # HEAD_SEED
        for q in m.head:
            if hasattr(q, "reset_parameters"): q.reset_parameters()
        print(f"[g2 dagger train s{seed}] v11 HEAD_INIT fresh: policy head re-initialised (seed {int(os.environ.get('HEAD_SEED', '4000')) + seed})", flush=True)
        if os.environ.get("HEAD_SEED"): print(f"[g2 dagger train s{seed}] v13e HEAD_SEED {os.environ['HEAD_SEED']}", flush=True)
    if CONC:                                                                              # concept-in student around the (ENC_INIT) encoder
        assert HEAD_INIT == "fresh" and not DOOR, "CONCEPT_IN needs HEAD_INIT=fresh and no DOOR_AUX"
        if CFEAT != "cls": print(f"[g2 dagger train s{seed}] v13c CONCEPT_FEAT {CFEAT}: concept head reads [CLS, patch-mean] (768)", flush=True)
        if CSTATS: csum = torch.zeros(NCL, dtype=torch.float64); cnn = 0; print(f"[g2 dagger train s{seed}] v13d CONCEPT_STATS: mean of the processed concept predictions saved as concept_in.mean_pred", flush=True)
        m = build_concept(seed, m)
        print(f"[g2 dagger train s{seed}] v13 CONCEPT_IN: concept head 384-256-256-{NCL} + policy head ({384}+{NCL})-512-512-{g1.K * 5} (both fresh), predictions detached into the policy; "
              f"loss = policy + {CONC_W:g} x (door cos + dist smooth-L1 + visible BCE + depth L1)", flush=True); cst = np.zeros(5)
    if JOINT: print(f"[g2 dagger train s{seed}] v13h CONCEPT_JOINT: shared layer 768-512-512 (LayerNorm, GELU) on frozen [CLS, patch-mean] feeds a fresh concept head 512-256-256-{NCL} and the policy head; concepts not detached", flush=True)
    if H13:                                                                                # wrapper with the new head width
        m = build_h13(seed, m, h13_cfg())
        print(f"[g2 dagger train s{seed}] v13h H13 wrapper: policy head in {m.inner.head[0].in_features} = {'shared 512' if JOINT else ('[CLS, mean] 768' if HIST else 'CLS 384')}{' + ' + str(HPJ or 768) + ' x ' + str(HIST - 1) + ' past frames' + (' (HIST_PROJ 768-' + str(HPJ) + ' + LayerNorm)' if HPJ else '') if HIST > 1 else ''}"
              f"{' + concepts ' + str(NCL) if CONC else ''}{' + state 4' if STIN else ''}; HIST_DT {HDT} steps; fresh (seed {h13_cfg()['head_seed']} + {seed})", flush=True)
    m = m.to(DEV).train()
    if FRZ:                                                                               # whole encoder frozen, backbone in eval()
        for p_ in m.feat.parameters(): p_.requires_grad_(False)
        m.feat._frozen = True; m.feat.backbone.eval()
        print(f"[g2 dagger train s{seed}] v11 FREEZE_ENC: encoder trainable 0 of {sum(p_.numel() for p_ in m.feat.parameters())} parameters", flush=True)
    ELR = ELR_ENV if ELR_ENV is not None else g1.ENC_LR
    if ELR_ENV is not None: print(f"[g2 dagger train s{seed}] v11 ENC_LR {ELR:g} (default {g1.ENC_LR:g})", flush=True)
    if DSRC is not None: print(f"[g2 dagger train s{seed}] v11 DOOR_SRC {os.environ['DOOR_SRC']}: door-loss mask keeps frame kinds {DSRC} (0 stream, 1 DAgger, 2 recovery)", flush=True)
    if UNF:                                                                               # freeze all but the last UNF blocks + final layernorm
        keep = tuple(f"backbone.encoder.layer.{i}." for i in range(12 - UNF, 12)) + ("backbone.layernorm.",)
        for n_, p_ in m.feat.named_parameters(): p_.requires_grad_(n_.startswith(keep))
        print(f"[g2 dagger train s{seed}] v10 UNFREEZE_LAST {UNF}: encoder trainable {sum(p_.numel() for p_ in m.feat.parameters() if p_.requires_grad)} of "
              f"{sum(p_.numel() for p_ in m.feat.parameters())} parameters (blocks {12 - UNF}-11 + final layernorm)", flush=True)
    dh = None
    if DOOR:                                                                              # fresh door head
        torch.manual_seed(3000 + seed); dh = torch.nn.Sequential(torch.nn.Linear(384, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 3)).to(DEV).train()
        print(f"[g2 dagger train s{seed}] v10 DOOR_AUX {DOOR:g}: door head 384-256-256-3 (lr {g1.LR:g}) on the encoder output, frames {0.5:g}-{DOOR_R:g} m from the door "
              f"land point, loss = policy + {DOOR:g} * mean(1 - cos(pred, body-frame unit door vector))", flush=True)
        dnn = np.zeros(4)                                                                 # masked frames, frames, door loss sum, steps
    if HC:                                                                                 # cache vs live encoder on 8 grid states
        y0 = SY_[0]; ids0, P0, fj0, ij0 = h13_stream_pools(G, y0); kb0 = 2_000_000_000 + int(y0[4:]) * 100_000
        for j_, p_ in enumerate(P0): _HP[kb0 + j_] = p_; _HPN[kb0 + j_] = {"trk": f"stream/{y0}/{ids0[j_]}.npz", "rec": None}
        pick = [(kb0 + j_, k_) for j_ in np.linspace(0, len(P0) - 1, 8).astype(int) for k_ in [(len(P0[j_]) // 2) // HDT * HDT]]
        with torch.no_grad():
            im = torch.tensor(g1.prep(np.stack([frame(y0, _HP[q][k_]) for q, k_ in pick])), device=DEV)
            live = m.feats(im)[1].float().cpu().numpy(); ref = np.stack([h13_cfeat(q, k_) for q, k_ in pick]).astype(np.float32)
        cs = (live * ref).sum(1) / (np.linalg.norm(live, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
        print(f"[g2 dagger train s{seed}] v13j cache check ({y0}, 8 grid states): cosine live vs cache min {cs.min():.5f} median {np.median(cs):.5f}", flush=True)
        assert cs.min() >= 0.99, f"v13j: cache vs live encoder cosine {cs.min():.4f} < 0.99"
        _HCF.clear()
    opt = torch.optim.AdamW([q for q in [{"params": [p_ for p_ in m.feat.parameters() if p_.requires_grad], "lr": ELR}] if q["params"]] + [{"params": m.head.parameters(), "lr": g1.LR}]   # ELR; no empty group
                            + ([{"params": dh.parameters(), "lr": g1.LR}] if DOOR else []) + ([{"params": m.ch.parameters(), "lr": g1.LR}] if CONC else [])
                            + ([{"params": m.sh.parameters(), "lr": g1.LR}] if JOINT else [])
                            + ([{"params": m.pp.parameters(), "lr": g1.LR}] if HPJ else []))
    dnk = np.zeros(3)                                                                     # masked door frames per kind
    dl = torch.utils.data.DataLoader(MixStream(G, SY_, my, sy, seed, MB, DAG), batch_size=None, num_workers=NW, multiprocessing_context="spawn", prefetch_factor=4)
    it = iter(dl); MEAN = torch.tensor(g1.MEAN, device=DEV); SD = torch.tensor(g1.SD, device=DEV)
    def save(path, step):
        torch.save({"state": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "my": my, "sy": sy, "seed": seed, "yards": Yd,
                    "held_out": g2.HELD_OUT, "n_frames": c.get("n_frames"), "K": g1.K, "CHUNK_M": g1.CHUNK_M, "step": int(c.get("step", 0)) + step,
                    "aux": None, "aux_state": None, "g2_dagger": {"from": CK, "dag": DAG, "steps": step, "fake_render": FAKE, "yards": YS,
                    **({"stream_yards": list(SY_)} if os.environ.get("STREAM_YARDS") == "ckpt" else {}), **({"extra_stream_yards": XSY, "yard_dirs": YMAP} if XSY or YMAP else {}),
                    **({"mix": os.environ["MIX"]} if os.environ.get("MIX") else {}), **({"aug": True} if AUG else {}),
                    **({"turn_w": TW, "turn_window_m": [TURN_LO, TURN_HI]} if TW else {}),
                    **({"turn_seg_w": TSW, "turn_seg_rule": [SEG_HI, SEG_LO, SEG_MIN_D]} if TSW else {}),
                    **({"door_aux": DOOR, "door_r_m": DOOR_R} if DOOR else {}), **({"unfreeze_last": UNF} if UNF else {}),
                    **({"door_src": os.environ["DOOR_SRC"]} if DSRC is not None else {}), **({"enc_init": ENC_INIT} if ENC_INIT else {}),
                    **({"head_init": HEAD_INIT} if HEAD_INIT else {}), **({"head_seed": int(os.environ["HEAD_SEED"])} if os.environ.get("HEAD_SEED") else {}), **({"freeze_enc": True} if FRZ else {}), **({"enc_lr": ELR} if ELR_ENV is not None else {})},
                    **({"door_head": {k: v.detach().cpu() for k, v in dh.state_dict().items()}} if DOOR else {}),
                    **({"h13": h13_cfg()} if H13 else {}), **({"concept_joint": True} if JOINT else {}),
                    **({"concept_in": {"ncl": NCL, "w": CONC_W, "detached": not JOINT, "labels": "door unit vector body (3), log1p(d)/log1p(40) (1), door visible (1), exact depth 3x4/20 (12); make_labels 9x16 cast", **({"feat": CFEAT} if CFEAT != "cls" else {}), **({"mean_pred": (csum / max(cnn, 1)).tolist(), "n_frames": int(cnn)} if CSTATS else {})}} if CONC else {})}, path + ".tmp")
        os.replace(path + ".tmp", path)
    lg = open(f"{OUT}/loss_s{seed}.tsv", "a"); t0 = time.time(); tot, nb = 0.0, 0
    for step in range(1, STEPS + 1):
        parts = [next(it) for _ in range(B // MB)]
        xb = torch.cat([p[0] for p in parts])
        if R78: cnt += np.bincount(torch.cat([p[2] for p in parts]).numpy(), minlength=3)[:3]
        if AUG: xb = aug_batch(xb.to(DEV)) if xb.dim() == 4 else aug_batch(xb.to(DEV).flatten(0, 1)).view(xb.shape)   # every frame of a HIST sample
        x = (xb.to(DEV).float() / 255.0 - MEAN) / SD; y = torch.cat([p[1] for p in parts]).to(DEV)
        if CONC: yc = y[:, g1.K * 5:]; y = y[:, :g1.K * 5]                                  # concept labels ride at the end of the label vector
        with torch.autocast(DEV if DEV == "cuda" else "cpu", dtype=torch.bfloat16):
            if DOOR: fz = m.feat(x); out = m.head(fz); dp = dh(fz)                          # one encoder pass feeds both heads
            elif H13: out, cr = m.core(x if x.dim() == 5 else x[:, None], torch.cat([p[3] for p in parts]).to(DEV) if STIN else None, torch.cat([p[-1] for p in parts]).to(DEV) if HC else None)
            elif CONC: out, cr = m.forward_all(x)
            else: out = m(x)
        if WT:
            wb = torch.cat([p[-3 if DOOR else -1] for p in parts]).to(DEV); li = ((out.float() - y) ** 2).reshape(len(y), -1).mean(1)
            loss = (wb * li).sum() / wb.sum(); iw = wb == WT                             # (TURN_W=1: every frame counts as window; a no-op weight)
            twn += [float(iw.sum()), float(len(wb)), float(wb[iw].sum() / wb.sum()) * len(wb)]
        else: loss = torch.nn.functional.mse_loss(out.float(), y)
        if DOOR:                                                                          # door-offset auxiliary loss on masked frames
            du = torch.cat([p[-2] for p in parts]).to(DEV); dm = torch.cat([p[-1] for p in parts]).to(DEV)
            if R78:                                                                       # door mask by frame kind
                tgd = torch.cat([p[2] for p in parts])
                if DSRC is not None: dm = dm * torch.isin(tgd, torch.tensor(DSRC)).to(DEV).float()
                dnk += np.bincount(tgd.numpy()[dm.cpu().numpy() > 0], minlength=3)[:3]
            cs = torch.nn.functional.cosine_similarity(dp.float(), du, dim=1); ld = ((1 - cs) * dm).sum() / dm.sum().clamp(min=1.0)
            lpol = float(loss); loss = loss + DOOR * ld
            dnn += [float(dm.sum()), float(len(dm)), float(ld), 1.0]
            dbe = float(torch.rad2deg(torch.arccos(cs.clamp(-1, 1)))[dm > 0].median()) if bool((dm > 0).any()) else float("nan")
        if CONC:                                                                          # concept losses
            F_ = torch.nn.functional; crf = cr.float(); csu = F_.cosine_similarity(crf[:, :3], yc[:, :3], dim=1)
            um = (yc[:, 3] >= math.log1p(0.5) / math.log1p(40.0)).float()
            lc = [((1 - csu) * um).sum() / um.sum().clamp(min=1.0), F_.smooth_l1_loss(torch.sigmoid(crf[:, 3]), yc[:, 3]), F_.binary_cross_entropy_with_logits(crf[:, 4], yc[:, 4]), F_.l1_loss(torch.sigmoid(crf[:, 5:]), yc[:, 5:])]
            lpol = float(loss); loss = loss + CONC_W * sum(lc)
            if CSTATS:                                                                    # running mean of what the policy saw
                with torch.no_grad(): csum += m.proc(cr.float()).double().sum(0).cpu(); cnn += len(cr)
            cst += [float(torch.rad2deg(torch.arccos(csu.clamp(-1, 1))).median()), float(lc[1]), float(((crf[:, 4] > 0).float() == yc[:, 4]).float().mean()), float(lc[3]), 1.0]
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if step % 100 == 0 or step in (1, 10) or step == STEPS:
            el = time.time() - t0; lg.write(f"{step}\t{el:.0f}\t{tot / nb:.5f}\n"); lg.flush()
            pk = f", peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GB" if DEV == "cuda" else ""
            sh = (f"; realized shares stream {cnt[0] / cnt.sum():.3f} DAgger {cnt[1] / cnt.sum():.3f} recovery {cnt[2] / cnt.sum():.3f} ({int(cnt.sum())} frames); "
                  f"reuse = draws / unique states: DAgger {cnt[1] / max(1, nuq[1]):.3f}, recovery {cnt[2] / max(1, nuq[2]):.3f}") if R78 else ""
            tws = (f"; {'TURN_W window' if TW else 'TURN_SEG_W segment'} frames {twn[0] / twn[1]:.3f}, {'window' if TW else 'segment'} loss-weight share {twn[2] / twn[1]:.3f} ({int(twn[1])} frames since start)") if WT else ""
            dws = (f"; DOOR_AUX policy loss {lpol:.4f} (this step), door loss mean {dnn[2] / dnn[3]:.4f}, masked share {dnn[0] / dnn[1]:.3f}, "
                   f"batch median door bearing error {dbe:.1f} deg (this step)" + (f", masked frames by kind stream/DAgger/recovery {int(dnk[0])}/{int(dnk[1])}/{int(dnk[2])}" if R78 else "")) if DOOR else ""
            cws = (f"; CONCEPT_IN policy loss {lpol:.4f} (this step), door bearing median {cst[0] / cst[4]:.1f} deg, dist L1 {cst[1] / cst[4]:.4f}, visible acc {cst[2] / cst[4]:.3f}, "
                   f"depth L1 {cst[3] / cst[4]:.4f} (means since the last line)") if CONC else ""
            print(f"[g2 dagger train s{seed}] step {step} loss {tot / nb:.4f} {step / el:.3f} steps/s{pk}{sh}{tws}{dws}{cws}", flush=True); tot, nb = 0.0, 0
            if CONC: cst[:] = 0
            if DOOR: dnn[:] = 0; dnk[:] = 0
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
    OSP = os.environ.get("OL_SPEED") == "1"
    if os.environ.get("OL_DIST") == "1" or OSP:
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
    elif mode == "check": check(a[0], a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 8)
    elif mode == "recover": recover(a[0], int(a[1]), a[2], int(a[3]) if len(a) > 3 else 8)
    elif mode == "labelfloor": labelfloor(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 8)
    elif mode == "olsplit": olsplit(a[0])
    elif mode == "olcheck": olcheck(a[0], a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 50)
    else: sys.exit(__doc__)
