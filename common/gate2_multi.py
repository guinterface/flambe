"""Train and evaluate a camera-based flight policy across generated yards."""
import json, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_GATE2 = os.environ.get("FLAMBE_GATE2", os.path.join(FLAMBE_ROOT, "research", "02_experiments", "gate2"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
sys.path[:0] = LOCAL_MODULE_PATHS
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import gate1_student as g1, gate0b_student as st

USABLE = [f"syn_{i:03d}" for i in list(range(1, 14)) + [15, 16, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33] + list(range(39, 50)) + [51]]
HELD_OUT = ["syn_001", "syn_006", "syn_007", "syn_008"]
TRAIN_YARDS = [y for y in USABLE if y not in HELD_OUT]
_R = {}


def renderer(yard):
    if yard not in _R:
        from pathlib import Path
        from figs.render.gsplat_semantic import GSplat
        import visual_expert_syn as ve
        ws = Path(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace")); scene = ve.find_checkpoint(yard)
        cfgs = list((ws / "outputs" / scene).rglob("*.yml")); assert len(cfgs) == 1, cfgs
        frame = json.load(open(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "configs", "frame", "carl.json")))
        cwd = os.getcwd(); os.chdir(ws); gs = GSplat({"name": scene, "path": cfgs[0]}); os.chdir(cwd)
        _R[yard] = (gs, gs.generate_output_camera(frame["camera"]), np.array(frame["camera_to_body_transform"]))
    return _R[yard]


def policy_frame(yard, x):
    from figs.utilities.trajectory_helper import xv_to_T
    gs, cam, T_c2b = renderer(yard); rgb = gs.render_rgb(cam, xv_to_T(np.asarray(x, float)) @ T_c2b)["rgb"]
    rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    rgb = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    return g1.transform_u8(rgb)


def land(yard):
    return np.asarray([g for g in json.load(open(f"{st.SYR}/{yard}/goals.json"))["goals"] if g["id"] == "door"][0]["land_ned"], float)


def mem(out, yards):
    import torch, subprocess
    cap = float(os.environ.get("CAP_GB", "0"))
    if cap > 0: torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    def smi():
        import shutil
        if not shutil.which("nvidia-smi"): return "nvidia-smi not in the container"
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return r.strip()
    rows = []; t0 = time.time()
    for y in yards:
        a = time.time(); renderer(y); torch.cuda.synchronize()
        rows.append({"yard": y, "load_s": round(time.time() - a, 1), "alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                     "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)})
        print("[mem]", json.dumps(rows[-1]), flush=True)
    # frames/s with random yards
    rng = np.random.default_rng(0); n = 0; a = time.time(); tracks = {}
    while time.time() - a < 60:
        y = yards[int(rng.integers(len(yards)))]
        if y not in tracks:
            S = [m["id"] for m in json.load(open(f"{FLAMBE_GATE2}/{y}/starts.json"))["starts"] if m["split"] == "train"][:3]
            tracks[y] = np.concatenate([np.load(f"{FLAMBE_GATE2}/{y}/tracks/{i}.npz")["X"] for i in S])
        X = tracks[y]; policy_frame(y, X[int(rng.integers(len(X)))]); n += 1
    fps = n / (time.time() - a)
    rep = {"yards": len(yards), "rows": rows, "total_alloc_gb": rows[-1]["alloc_gb"], "total_reserved_gb": rows[-1]["reserved_gb"],
           "frames_per_s_one_process_random_yards": round(fps, 1), "load_all_s": round(sum(r["load_s"] for r in rows), 1),
           "nvidia_smi_apps": smi()}
    json.dump(rep, open(out, "w"), indent=1); print("GATE2_MEM", json.dumps({k: v for k, v in rep.items() if k != "rows"}), flush=True)


def fly_one(args):
    G, yard, ck, s = args
    import torch, obs_relabel_z as oz, goal_expert_v4 as gv
    from score import score_track, clearance_fn
    from gate0b_expert import _yaw_quat, U0
    m = g1.build(int(os.environ.get("SEED", "0"))); c = torch.load(ck, weights_only=False)
    if c.get("aux") != "g4": m.load_state_dict(c["state"])
    g4 = c.get("aux") == "g4"
    if g4:                                                    # G4 (oracle, simulation only): RGB + goal-mask input
        g4_enable(m); m.load_state_dict(c["state"])
    dev = "cuda"; m = m.to(dev).eval(); pol = None
    if c.get("aux") == "m1":                                  # memory arm: cached CLS window (M1Cache), same action head
        a_ = M1Net(); a_.load_state_dict(c["aux_state"]); pol = M1Cache(m, a_.to(dev).eval())
    L = land(yard); ctl = gv.DirectGoalExpert.__new__(gv.DirectGoalExpert); ctl.prev_oz = 0.0
    x = np.r_[np.asarray(s["pos"], float), np.zeros(3), _yaw_quat(s["yaw"])]; u_prev = U0.copy()
    budget = 1.5 * s["t_expert_s"]; N = int(round(budget * 20)); K = g1.K; w = np.exp(-g1.CHUNK_M * np.arange(K)); hist = []
    P, V, T, LX = [x[:3].copy()], [x[3:6].copy()], [0.0], []
    for k in range(N):
        with torch.no_grad():
            xin = torch.tensor(g4_prep(policy_frame(yard, x), g4_mask(yard, x)) if g4 else g1.prep(policy_frame(yard, x)[None]), device=dev)
            cc = (pol.step(xin) if pol is not None else m(xin)).cpu().numpy()[0] * c["sy"] + c["my"]
        hist.append(cc.reshape(K, 5)); hist = hist[-K:]
        preds = np.array([hist[-1 - j][j] for j in range(len(hist))]); ww = w[:len(preds)][::-1]; g = (preds * ww[:, None]).sum(0) / ww.sum()
        yaw = float(st.yaw_of(x[6:10]))
        u, _ = gv.DirectGoalExpert._u(ctl, x[0:3], x[3:6], x[6:10], g[0:2], float(x[2] + g[2]), float(st.wrap(yaw + g[3])), w_clip=0.3 if g[4] > 0.5 else 0.4)
        LX.append(x.copy()); x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / 20)
    P, V, T = np.array(P), np.array(V), np.array(T); clr = clearance_fn(yard)(P); r = score_track(T, P, V, L, clr, budget)
    if not r["success"]:
        td = f"{os.environ['EVAL_OUT']}/tracks_{yard}"; os.makedirs(td, exist_ok=True)
        np.savez_compressed(f"{td}/{s['id']}.npz", T=T, P=P, V=V, clr=clr, X=np.array(LX))
    r.update({"id": s["id"], "yard": yard, "split": s["split"]}); return r


def winit():
    import obs_relabel_z as oz, score
    oz.init_worker(); score.SY = FLAMBE_YARDS


def evaluate(G, yard, ck, NP):
    import obs_relabel_z, score, torch
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    S = [dict(s, t_expert_s=E[s["id"]]["t_landed_s"]) for s in json.load(open(f"{G}/{yard}/starts.json"))["starts"]
         if s["split"] == "test" and E.get(s["id"], {}).get("outcome") == "success"]
    if os.environ.get("EVAL_N"): S = S[:int(os.environ["EVAL_N"])]
    t0 = time.time()
    with ProcessPoolExecutor(NP, mp_context=get_context("spawn"), initializer=winit) as p:
        res = list(p.map(fly_one, [(G, yard, ck, s) for s in S], chunksize=1))
    out = os.environ["EVAL_OUT"]; os.makedirs(out, exist_ok=True); json.dump({"ckpt": ck, "yard": yard, "flights": res}, open(f"{out}/eval_{yard}.json", "w"))
    k = {o: sum(r["outcome"] == o for r in res) for o in ("success", "contact", "overshoot", "timeout")}
    print(f"GATE2_EVAL {yard}: {len(res)} test starts (expert-successful) {k} ({time.time() - t0:.0f}s){G4P}", flush=True)


def yard_data(G, yard):
    """train-split, expert-successful, contact-free flights of one yard: states X and K x 5 labels (gate0b_student.labels)"""
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    ids = [m["id"] for m in json.load(open(f"{G}/{yard}/starts.json"))["starts"] if m["split"] == "train"
           and E.get(m["id"], {}).get("outcome") == "success" and E[m["id"]].get("min_clearance_m", 0) >= 0.30]
    X, Y = [], []
    for i in ids:
        z = np.load(f"{G}/{yard}/tracks/{i}.npz"); X.append(z["X"].astype(np.float64)); Y.append(st.labels(z["X"], z["G"], z["PH"]))
    return np.concatenate(X), np.concatenate(Y).astype(np.float32), len(ids)


import torch as _torch


# ---------------- Optional depth and segmentation heads with AUX=r3
AUX = os.environ.get("AUX", "")
AUX_W, AUX_HW, N_CLS = 0.1, 28, 7          # loss = action + AUX_W * (depth L1 on depth/20 + seg CE); targets at 1/8 of 224
_T = {}


def truth(yard):
    """exact-geometry ray caster of terra/s10/make_labels.py (7 classes: sky/far, ground, house, door, window,
    vegetation, other) on a 28 x 28 grid over the policy crop (its RH/RW/DIR_B rebuilt for that grid)"""
    if yard not in _T:
        sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.join(FLAMBE_TERRA, "s10")]; import make_labels as mk
        if mk.RH != AUX_HW:
            mk.RH = mk.RW = AUX_HW; C = mk.C; fr = mk._FR
            US = C["w"] * (0.5 - fr / 2 + (np.arange(AUX_HW) + 0.5) * fr / AUX_HW); VS = C["h"] * (0.5 - fr / 2 + (np.arange(AUX_HW) + 0.5) * fr / AUX_HW)
            UU, VV = np.meshgrid(US, VS)
            mk.DIR_B = np.stack([np.ones_like(UU), (UU - C["cx"]) / C["fx"], (VV - C["cy"]) / C["fy"]], -1).reshape(-1, 3)
        door = [g for g in json.load(open(f"{st.SYR}/{yard}/goals.json"))["goals"] if g["id"] == "door"][0]
        _T[yard] = mk.Truth(yard, door)
    return _T[yard]


def frame_and_targets(yard, x):
    """policy frame (as policy_frame) plus R3 targets at the same pose: depth from the same GSplat render (metric
    depth_raw through the same Resize 256 / CenterCrop 224, area-averaged to 28 x 28, clipped at 20 m) and the
    ray-cast class map (the yard splats have no semantic field, so classes come from the exact geometry)."""
    import cv2
    from figs.utilities.trajectory_helper import xv_to_T
    gs, cam, T_c2b = renderer(yard); out = gs.render_rgb(cam, xv_to_T(np.asarray(x, float)) @ T_c2b)
    rgb = out["rgb"]; rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    rgb = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    d = np.asarray(out["depth_raw"], np.float32); d = cv2.resize(d, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
    d = np.clip(cv2.resize(d, (AUX_HW, AUX_HW), interpolation=cv2.INTER_AREA), 0, 20.0)
    _, cls = truth(yard).cast(np.asarray(x[:3], float), np.asarray(x[6:10], float))
    return g1.transform_u8(rgb), d.astype(np.float32), cls.astype(np.int64)


class R3Head(_torch.nn.Module):
    """patch tokens (B, 256, 384) -> depth (B, 28, 28) and class logits (B, 7, 28, 28)"""
    def __init__(self):
        super().__init__(); nn = _torch.nn
        self.a = nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU()); self.b = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1 + N_CLS, 1))

    def forward(self, tok):
        B = tok.shape[0]; f = tok.transpose(1, 2).reshape(B, 384, 16, 16)
        f = _torch.nn.functional.interpolate(self.a(f), size=(AUX_HW, AUX_HW), mode="bilinear", align_corners=False)
        o = self.b(f); return o[:, 0], o[:, 1:]


def forward_aux(m, aux, x):
    """one trunk pass: CLS -> the unchanged policy head; patch tokens -> R3 heads"""
    h = m.feat.backbone(pixel_values=x).last_hidden_state
    return m.head(h[:, 0]), aux(h[:, 1:])


# ---------------- idea 6 (M1 memory; ViNT 2306.14846 / NoMaD 2310.07896 style), behind AUX=m1 (off by default)
M1_K, M1_GAP, M1_DROP = int(os.environ.get("M1_K", "3")), int(os.environ.get("M1_GAP", "10")), 0.3   # 3 past frames, 0.5 s apart, history dropout
M1_CLIP = os.environ.get("M1_CLIP", "0") == "1"
M1_L = int(os.environ.get("M1_L", str(M1_K + 1))) if M1_CLIP else M1_K + 1     # clip length


def flight_starts(G, yard):
    """per-frame index of its flight's first frame, aligned with yard_data's concatenation"""
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    ids = [m["id"] for m in json.load(open(f"{G}/{yard}/starts.json"))["starts"] if m["split"] == "train"
           and E.get(m["id"], {}).get("outcome") == "success" and E[m["id"]].get("min_clearance_m", 0) >= 0.30]
    fs, n = [], 0
    for i in ids:
        L = len(np.load(f"{G}/{yard}/tracks/{i}.npz")["X"]); fs.append(np.full(L, n, np.int64)); n += L
    return np.concatenate(fs)


class M1Net(_torch.nn.Module):
    """windowed memory: every target t sees its own CLS token and the M1_K previous clip tokens (0.5 s apart), with lag
    embeddings 0..K (not absolute clip positions), so a target's output depends only on its own 4-token window: exactly
    what the deploy-time cache feeds (eval_m1). Padding mask for frames before the clip / flight start."""
    def __init__(self):
        super().__init__(); nn = _torch.nn
        self.lag = nn.Parameter(_torch.zeros(M1_K + 1, 384))           # index K = current frame, 0 = oldest
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(384, 4, 768, 0.1, batch_first=True, norm_first=True), 2)

    def window(self, m, cls, valid):
        """cls (N, K+1, 384), valid (N, K+1) -> (N, K*5): the current (last) position's chunk"""
        h = self.enc(cls + self.lag, src_key_padding_mask=~valid)
        return m.head(h[:, -1])

    def forward(self, m, x, valid, train):
        """x (B, L, 3, H, W) clip frames, valid (B, L) -> (B, L, K*5), each position from its own window"""
        B, Ln = valid.shape; Kp = M1_K + 1
        cls = m.feat(x.flatten(0, 1)).view(B, Ln, 384)
        pad = _torch.zeros(B, M1_K, 384, device=cls.device, dtype=cls.dtype); cp = _torch.cat([pad, cls], 1)
        vp = _torch.cat([_torch.zeros(B, M1_K, dtype=_torch.bool, device=valid.device), valid], 1)
        W = cp.unfold(1, Kp, 1).permute(0, 1, 3, 2).reshape(B * Ln, Kp, 384)      # windows ending at each clip position
        V = vp.unfold(1, Kp, 1).reshape(B * Ln, Kp).clone()
        if train:                                                  # history dropout: drop a target's past tokens with p = 0.3
            drop = _torch.rand(B * Ln, device=V.device) < M1_DROP
            V[drop, :-1] = False
        V[:, -1] = True                                            # the current frame is always present
        return self.window(m, W, V).view(B, Ln, -1)

class M1Cache:
    """deploy-time M1 policy: one trunk pass per step; the CLS tokens of past steps are cached, and the window
    (k - K*GAP, ..., k - GAP, k) goes to M1Net.window, as in training (steps before the flight start are padding)"""
    def __init__(self, m, aux):
        self.m, self.aux, self.cls = m, aux, []

    def step(self, x):
        import torch
        c = self.m.feat(x)[0]; self.cls.append(c); k = len(self.cls) - 1
        idx = [k - j * M1_GAP for j in range(M1_K, -1, -1)]
        W = torch.stack([self.cls[i] if i >= 0 else torch.zeros_like(c) for i in idx])[None]
        V = torch.tensor([[i >= 0 for i in idx]], device=c.device)
        return self.aux.window(self.m, W, V)


def m1_test(G, yard, n_flights=3):
    """cached step-by-step outputs vs the clip-mode forward on the same frames (eval, fp32), 3 recorded train flights"""
    import torch
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = g1.build(0).to(dev).eval(); aux = M1Net().to(dev).eval()
    with torch.no_grad():
        for prm in aux.parameters(): prm.add_(0.02 * torch.randn_like(prm))     # non-trivial lag embeddings / weights
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    ids = [m_["id"] for m_ in json.load(open(f"{G}/{yard}/starts.json"))["starts"] if m_["split"] == "train" and E.get(m_["id"], {}).get("outcome") == "success"][:n_flights]
    worst, ncmp, t0 = 0.0, 0, time.time(); per = {}
    with torch.no_grad():
        for i in ids:
            X = np.load(f"{G}/{yard}/tracks/{i}.npz")["X"]; I = np.stack([policy_frame(yard, x) for x in X])
            xt = torch.tensor(g1.prep(I), device=dev)
            cache = M1Cache(m, aux); A = torch.cat([cache.step(xt[k:k + 1]) for k in range(len(xt))])   # (N, K*5)
            dmax = 0.0; n = 0; span = (M1_L - 1) * M1_GAP
            for k in range(span, len(xt), 7):                      # clips ending at k (every 7th frame)
                idx = list(range(k - span, k + 1, M1_GAP)); out = aux(m, xt[idx][None], torch.ones(1, M1_L, dtype=torch.bool, device=dev), False)[0]
                for j in range(M1_K, M1_L):                       # clip positions with a full K-window inside the clip
                    d = float((out[j] - A[idx[j]]).abs().max()); dmax = max(dmax, d); n += 1
            per[i] = {"frames": int(len(xt)), "comparisons": n, "max_abs_diff": dmax}; worst = max(worst, dmax); ncmp += n
    rep = {"yard": yard, "flights": per, "L": M1_L, "K": M1_K, "gap": M1_GAP, "comparisons": ncmp, "max_abs_diff": worst, "pass": worst < 1e-4,
           "device": dev, "wall_s": round(time.time() - t0, 1)}
    print("GATE2_M1_TEST", json.dumps(rep), flush=True); return rep


# ---------------- Optional simulation-only goal oracle with AUX=g4
# The camera student also gets the exact goal mask: 1 where the pixel's ray first hits the goals.json door box (class 3 of
# the ray caster in make_labels.Truth) or passes within G4_LAND_R of land_ned before its first hit. Computed
# from the TRUE state, so it cannot be deployed; it bounds what perfect goal detection would give the base.
G4P = " [oracle, simulation only]" if AUX == "g4" else ""
G4_CAST, G4_LAND_R = int(os.environ.get("G4_CAST", "28")), 0.5   # 28 x 28 cast (finer than the 16 x 16 patch grid)
_G4 = {}


def g4_truth(yard):
    if yard not in _G4:
        sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.join(FLAMBE_TERRA, "s10")]; import make_labels as mk
        C, fr, n = mk.C, mk._FR, G4_CAST
        US = C["w"] * (0.5 - fr / 2 + (np.arange(n) + 0.5) * fr / n); VS = C["h"] * (0.5 - fr / 2 + (np.arange(n) + 0.5) * fr / n)
        UU, VV = np.meshgrid(US, VS)
        dirs = np.stack([np.ones_like(UU), (UU - C["cx"]) / C["fx"], (VV - C["cy"]) / C["fy"]], -1).reshape(-1, 3)
        door = [g for g in json.load(open(f"{st.SYR}/{yard}/goals.json"))["goals"] if g["id"] == "door"][0]
        _G4[yard] = (mk.Truth(yard, door), dirs, np.asarray(door["land_ned"], float))
    return _G4[yard]


def g4_mask(yard, x):
    """uint8 (224, 224) goal mask (0/1) at state x (oracle, simulation only)"""
    T, dirs, L = g4_truth(yard)                               # puts terra/s10 on the path
    import make_labels as mk
    from scipy.spatial.transform import Rotation
    pos, q = np.asarray(x[:3], float), np.asarray(x[6:10], float)
    keep = (mk.DIR_B, mk.RH, mk.RW); mk.DIR_B, mk.RH, mk.RW = dirs, G4_CAST, G4_CAST      # the caster reads these module globals
    try:
        depth, cls = T.cast(pos, q)
    finally:
        mk.DIR_B, mk.RH, mk.RW = keep
    D = dirs @ Rotation.from_quat(q).as_matrix().T; D /= np.linalg.norm(D, axis=1, keepdims=True)
    t = np.clip(D @ (L - pos), 0, depth.reshape(-1)); land = np.linalg.norm(pos + t[:, None] * D - L, axis=1) < G4_LAND_R
    m = ((cls.reshape(-1) == 3) | land).reshape(G4_CAST, G4_CAST).astype(np.uint8); r = 224 // G4_CAST
    return np.repeat(np.repeat(m, r, 0), r, 1)


def g4_prep(rgb_u8, mask_u8):
    """(3, 224, 224) uint8 + (224, 224) 0/1 -> (1, 4, 224, 224) float32: ImageNet-normalised RGB, mask as 0/1"""
    return np.concatenate([g1.prep(rgb_u8[None]), mask_u8[None, None].astype(np.float32)], 1)


def g4_enable(m):
    """4-channel patch embedding; the mask channel's weights start at 0, so the new student starts as the base network"""
    import torch
    pe = m.feat.backbone.embeddings.patch_embeddings; old = pe.projection
    if old.in_channels == 4: return m
    new = torch.nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :3] = old.weight; new.bias.copy_(old.bias)
    pe.projection = new; pe.num_channels = 4; m.feat.backbone.config.num_channels = 4; return m


def g4_masks(G, yard, out_dir, n=3):
    """mask-on-door check (oracle, simulation only): n frames from the door approach of recorded train flights, saved as
    PNG (rendered frame | frame with the mask in red), plus the share of mask pixels on the door class"""
    import cv2
    E = {r["id"]: r for r in json.load(open(f"{G}/{yard}/expert.json"))["flights"]}
    ids = [m_["id"] for m_ in json.load(open(f"{G}/{yard}/starts.json"))["starts"] if m_["split"] == "train" and E.get(m_["id"], {}).get("outcome") == "success"]
    L = land(yard); os.makedirs(out_dir, exist_ok=True); rep = []
    for i in ids[::max(1, len(ids) // n)][:n]:
        X = np.load(f"{G}/{yard}/tracks/{i}.npz")["X"]; d = np.linalg.norm(X[:, :3] - L, axis=1)
        cand = [k for k in range(len(X)) if 2.5 < d[k] < 6.0] or [int(np.argmin(np.abs(d - 4)))]
        k = max(cand, key=lambda k: int(g4_mask(yard, X[k]).sum()) if k % 5 == 0 else -1)
        f = policy_frame(yard, X[k]).transpose(1, 2, 0); mk_ = g4_mask(yard, X[k]); ov = f.copy()
        ov[mk_ > 0] = (0.45 * ov[mk_ > 0] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        path = f"{out_dir}/g4_mask_{yard}_{i}_k{k}.png"; cv2.imwrite(path, cv2.cvtColor(np.concatenate([f, ov], 1), cv2.COLOR_RGB2BGR))
        rep.append({"png": path, "flight": i, "frame": int(k), "dist_to_land_m": round(float(d[k]), 2), "mask_px": int(mk_.sum())})
    print("GATE2_G4_MASKS [oracle, simulation only]", json.dumps(rep), flush=True); return rep


class Stream(_torch.utils.data.IterableDataset):
    """each loader worker renders only its shard of yards (yards[worker::workers]); yields mini-batches of mb frames"""
    def __init__(self, G, yards, my, sy, seed, mb):
        super().__init__(); self.G, self.yards, self.my, self.sy, self.seed, self.mb = G, yards, my, sy, seed, mb

    def __iter__(self):
        import torch
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        torch.set_num_threads(1); mine = self.yards[wid::nw]; data = {y: yard_data(self.G, y)[:2] for y in mine}
        fst = {y: flight_starts(self.G, y) for y in mine} if AUX == "m1" else None
        for y in mine: renderer(y)
        rng = np.random.default_rng(self.seed * 100 + wid); stride = int(os.environ.get("STRIDE", "1")); ep = 0
        yi = np.concatenate([np.full(len(data[y][0]), j, np.int16) for j, y in enumerate(mine)])        # compact frame index (RAM)
        ki = np.concatenate([np.arange(len(data[y][0]), dtype=np.int32) for y in mine])
        while True:                                   # epoch e: every stride-th frame with offset e % stride, shuffled
            sel = np.where(ki % stride == ep % stride)[0]; order = sel[rng.permutation(len(sel))]
            for a in range(0, len(order) - self.mb + 1, self.mb):
                imgs, labs, deps, segs = [], [], [], []
                for o in order[a:a + self.mb]:
                    y, k = mine[int(yi[o])], int(ki[o]); X, Y = data[y]; labs.append((Y[k] - self.my) / self.sy)
                    if AUX == "m1":                    # frames k - j*GAP (j = K..0); before the flight start -> invalid, padded
                        idx = [k - j * M1_GAP for j in range(M1_L - 1, -1, -1)]; f0 = int(fst[y][k])
                        val = [i >= f0 for i in idx]; idx = [max(i, f0) for i in idx]
                        fr = {i: policy_frame(y, X[i]) for i in set(idx)}
                        imgs.append(np.stack([fr[i] for i in idx])); deps.append(np.array(val))
                        segs.append(np.stack([(Y[i] - self.my) / self.sy for i in idx]).astype(np.float32))
                    elif AUX == "g4":                  # oracle, simulation only: goal mask from the TRUE state
                        imgs.append(np.concatenate([policy_frame(y, X[k]), g4_mask(y, X[k])[None] * np.uint8(255)], 0))
                    elif AUX == "r3":
                        f, dd, cc = frame_and_targets(y, X[k]); imgs.append(f); deps.append(dd); segs.append(cc)
                    else:
                        imgs.append(policy_frame(y, X[k]))
                b = (torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs).astype(np.float32)))
                yield b + ((torch.from_numpy(np.stack(deps)), torch.from_numpy(np.stack(segs))) if AUX in ("r3", "m1") else ())
            ep += 1


def train(G, OUT, seed):
    import torch
    os.makedirs(OUT, exist_ok=True); torch.manual_seed(seed)
    NW, MB, B = int(os.environ.get("RENDER_WORKERS", "8")), 8, int(os.environ.get("BATCH", "64")); MAX_H = float(os.environ.get("MAX_HOURS", "8"))
    MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "0")); STRIDE = int(os.environ.get("STRIDE", "1")); CKPT_MIN = float(os.environ.get("CKPT_MIN", "30")); bench = int(os.environ.get("BENCH_STEPS", "0"))
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
    yards = TRAIN_YARDS if not os.environ.get("YARDS") else os.environ["YARDS"].split(",")
    assert not set(yards) & set(HELD_OUT), "held-out yard in training"
    Ys, nfl, nfr = [], 0, 0
    for y in yards:
        X, Y, n = yard_data(G, y); Ys.append(Y[::10]); nfl += n; nfr += len(X)
    Ys = np.concatenate(Ys); my, sy = Ys.mean(0), Ys.std(0) + 1e-6
    EPOCH_STEPS = int(np.ceil(nfr / STRIDE / B))
    print(f"[gate2 s{seed}] stride {STRIDE}: epoch = {int(np.ceil(nfr / STRIDE))} frames = {EPOCH_STEPS} steps; max epochs {MAX_EPOCHS or 'none'}", flush=True)
    print(f"[gate2 s{seed}] {len(yards)} train yards, {nfl} train flights, {nfr} frames; {NW} render workers (yards sharded), batch {B} = {B // MB} x {MB}", flush=True)
    dev = "cuda"; m = g1.build(seed); step0 = 0
    if AUX == "g4": g4_enable(m); print(f"[gate2 s{seed}]{G4P} AUX=g4: input = RGB + goal mask (goals.json door box + land marker r {G4_LAND_R} m, ray-cast {G4_CAST}x{G4_CAST}, x{224 // G4_CAST} nearest to 224); mask-channel patch weights start at 0", flush=True)
    if os.environ.get("RESUME"):            # weights and step from a checkpoint; the optimizer state was not saved, so AdamW restarts
        c = torch.load(os.environ["RESUME"], weights_only=False); m.load_state_dict(c["state"]); step0 = int(c.get("step", 0))
        assert np.allclose(c["my"], my) and np.allclose(c["sy"], sy), "label statistics differ from the checkpoint's"
        print(f"[gate2 s{seed}] RESUMED from {os.environ['RESUME']} at step {step0} (optimizer state not restored)", flush=True)
    m = m.to(dev)
    aux = R3Head().to(dev) if AUX == "r3" else (M1Net().to(dev) if AUX == "m1" else None)
    groups = [{"params": m.feat.parameters(), "lr": g1.ENC_LR}, {"params": m.head.parameters(), "lr": g1.LR}]
    if aux is not None: groups.append({"params": aux.parameters(), "lr": g1.LR})
    opt = torch.optim.AdamW(groups); print(f"[gate2 s{seed}] AUX={AUX or 'none'}" + (f": loss = action + {AUX_W} x (depth L1 on depth/20 + seg CE), targets {AUX_HW}x{AUX_HW}" if AUX == "r3" else "")
        + (f": K={M1_K} past frames every {M1_GAP} steps ({M1_GAP / 20:.2f} s), history dropout {M1_DROP}, {f'clip mode L={M1_L} (every position a target over its own K-window)' if M1_CLIP else 'naive (last position only)'}, loss = action MSE as base" if AUX == "m1" else ""), flush=True)
    MEAN = torch.tensor(g1.MEAN, device=dev); SD = torch.tensor(g1.SD, device=dev)
    def save(path, step):
        torch.save({"state": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "my": my, "sy": sy, "seed": seed, "yards": yards,
                    "held_out": HELD_OUT, "n_frames": int(nfr), "K": g1.K, "CHUNK_M": g1.CHUNK_M, "step": step, "aux": AUX or None,
                    "aux_state": ({k: v.detach().cpu() for k, v in aux.state_dict().items()} if aux is not None else None)}, path + ".tmp"); os.replace(path + ".tmp", path)
    dl = torch.utils.data.DataLoader(Stream(G, yards, my, sy, seed, MB), batch_size=None, num_workers=NW, multiprocessing_context="spawn", prefetch_factor=4)
    it = iter(dl); t0 = time.time(); last = t0; tot, nb = 0.0, 0; log = open(f"{OUT}/loss_s{seed}.tsv", "a"); step = step0
    while True:
        parts = [next(it) for _ in range(B // MB)]
        if AUX == "m1":
            x = torch.cat([p[0] for p in parts]).to(dev).float() / 255.0; x = (x - MEAN.unsqueeze(0)) / SD.unsqueeze(0)
            val = torch.cat([p[2] for p in parts]).to(dev); yy = torch.cat([p[3] for p in parts]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = aux(m, x, val, True).float()
            if M1_CLIP:                                   # every valid position is a target (its own causal past)
                w = val.float(); loss = (((out - yy) ** 2).mean(-1) * w).sum() / w.sum()
            else:
                loss = torch.nn.functional.mse_loss(out[:, -1], yy[:, -1])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += float(loss); nb += 1
            if step % 100 == 0:
                el = time.time() - t0; log.write(f"{step}\t{el:.0f}\t{tot / nb:.5f}\n"); log.flush()
                print(f"[gate2 s{seed}] step {step} loss {tot / nb:.4f} {(step - step0) / el:.3f} steps/s ({el / 3600:.2f} h this run)", flush=True); tot, nb = 0.0, 0
            if bench and step >= bench:
                el = time.time() - t0; nt = int(val.sum()) if M1_CLIP else len(val)
                print(f"GATE2_BENCH s{seed}: {step / el:.3f} steps/s, {step * len(val) * M1_L / el:.1f} rendered frames/s, {nt} training targets per step, "
                      f"trainer peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True); return
            continue
        x = torch.cat([p[0] for p in parts]).to(dev).float() / 255.0
        x = torch.cat([(x[:, :3] - MEAN) / SD, x[:, 3:]], 1) if AUX == "g4" else (x - MEAN) / SD; y = torch.cat([p[1] for p in parts]).to(dev)
        if aux is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
            loss = torch.nn.functional.mse_loss(out.float(), y)
        else:
            dt = torch.cat([p[2] for p in parts]).to(dev); ct = torch.cat([p[3] for p in parts]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, (dp, cp) = forward_aux(m, aux, x)
            la = torch.nn.functional.mse_loss(out.float(), y); ld = (dp.float() - dt / 20.0).abs().mean(); lc = torch.nn.functional.cross_entropy(cp.float(), ct)
            loss = la + AUX_W * (ld + lc); terms = (float(la), float(ld), float(lc))
            if step % 50 == 0: print(f"[gate2 s{seed}] step {step} terms action {terms[0]:.4f} depth_l1 {terms[1]:.4f} seg_ce {terms[2]:.4f}", flush=True)
        loss = loss; opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += float(loss); nb += 1
        if step % 100 == 0:
            el = time.time() - t0; log.write(f"{step}\t{el:.0f}\t{tot / nb:.5f}\n"); log.flush()
            print(f"[gate2 s{seed}] step {step} loss {tot / nb:.4f} {(step - step0) / el:.3f} steps/s ({el / 3600:.2f} h this run){G4P}", flush=True); tot, nb = 0.0, 0
        if bench and step >= bench:
            el = time.time() - t0; print(f"GATE2_BENCH s{seed}: {step / el:.3f} steps/s, {step * B / el:.1f} frames/s, trainer peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True); return
        if step % EPOCH_STEPS == 0:
            e = step // EPOCH_STEPS; save(f"{OUT}/student_s{seed}_last.pt", step)
            if e % int(os.environ.get("CKPT_EVERY", "1")) == 0: save(f"{OUT}/student_s{seed}_ep{e:02d}.pt", step)
            print(f"GATE2_EPOCH s{seed} ep {e}: step {step}, {(time.time() - t0) / 3600:.2f} h{G4P}", flush=True)
            if MAX_EPOCHS and e >= MAX_EPOCHS: break
        if time.time() - last > CKPT_MIN * 60: save(f"{OUT}/student_s{seed}_last.pt", step); last = time.time()
        if time.time() - t0 > MAX_H * 3600: break
        if MAX_STEPS and step - step0 >= MAX_STEPS:              # smoke tests only (off by default)
            el = time.time() - t0; print(f"GATE2_MAXSTEPS s{seed}: {step - step0} steps, {(step - step0) / el:.3f} steps/s, trainer peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GB{G4P}", flush=True); break
    save(f"{OUT}/student_s{seed}_last.pt", step); print(f"GATE2_TRAIN_DONE s{seed}: {step} steps, {(time.time() - t0) / 3600:.2f} h{G4P}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "mem": mem(sys.argv[2], sys.argv[3:] or TRAIN_YARDS)
    elif sys.argv[1] == "train": train(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "m1test": m1_test(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "g4masks": g4_masks(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "eval": evaluate(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 6)
