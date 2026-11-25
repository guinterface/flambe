"""Render training images and train and evaluate a camera-based flight policy."""
import json, os, sys, time, resource
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

sys.path[:0] = [
    *LOCAL_MODULE_PATHS,
    os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "src"),
    os.path.join(FLAMBE_TERRA, "SousVide_official", "src"),
    os.path.join(FLAMBE_YARDS_FALLBACK, "kv14"),
    os.path.join(FLAMBE_TERRA, "goalspec", "v4"),
    os.path.join(FLAMBE_TERRA, "goalspec"),
    FLAMBE_TERRA,
]
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import gate0b_student as st                       # K, CHUNK_M, labels, land, wrap, yaw_of, EXCLUDE (read only)

K, CHUNK_M, EXCLUDE, YARD = st.K, st.CHUNK_M, st.EXCLUDE, st.YARD
WARM = os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts", "msl_vit_frozen", "roster", "Maverick_vit_ft_goal", "finish_ckpts", "commNet_ep120.pt")
SPLAT = os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace", "outputs", "syn_013", "splatfacto", "2026-10-17_134211", "config.yml")
LR, ENC_LR, EPOCHS, BATCH = 1e-3, 1e-5, int(os.environ.get("EPOCHS", "20")), int(os.environ.get("BATCH", "128"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1); SD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


# ---------------- Optional state input with GATE1_STATE=sv.
# SOUS VIDE's onboard state next to the image: height pz, world velocity vW, attitude q. Never px, py, never a goal.
SV_IDX = [2, 3, 4, 5, 6, 7, 8, 9]                 # state x = [px py pz vx vy vz qx qy qz qw]: pz, vW (3), q (4)
SV_TAG = "reference arm 0c: image + onboard state (pz, vW, q), no x,y, no goal"
assert 0 not in SV_IDX and 1 not in SV_IDX


def sv_state(X):
    """state(s) (..., 10) -> (..., 8) float32 onboard state: pz, vW, q (px, py dropped here and nowhere else read)"""
    return np.asarray(X)[..., SV_IDX].astype(np.float32)


# ---------------- network
def build(seed, state=False):
    import torch
    torch.manual_seed(seed)
    feat = torch.load(WARM, map_location="cpu", weights_only=False).networks["feat"]
    class CamStudent(torch.nn.Module):
        """image only -> K x 5 chunk (standardised); no state, no goal input"""
        def __init__(s):
            super().__init__(); s.feat = feat
            s.head = torch.nn.Sequential(torch.nn.Linear(384, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, K * 5))
        def forward(s, img):                      # img: (B, 3, 224, 224) float, ImageNet-normalised
            return s.head(s.feat(img))
    if state:
        class CamStateStudent(torch.nn.Module):
            """image + onboard state (pz, vW, q; standardised with the training-set statistics in buffers smu, ssd) -> K x 5 chunk.
            The CLS feature and the 8 state numbers are concatenated before the same head (384 + 8 inputs)."""
            def __init__(s):
                super().__init__(); s.feat = feat
                s.head = torch.nn.Sequential(torch.nn.Linear(384 + len(SV_IDX), 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, K * 5))
                s.register_buffer("smu", torch.zeros(len(SV_IDX))); s.register_buffer("ssd", torch.ones(len(SV_IDX)))
            def forward(s, img, state):           # img as CamStudent; state: (B, 8) = sv_state(x)
                return s.head(torch.cat([s.feat(img), (state - s.smu) / s.ssd], 1))
        return CamStateStudent()
    return CamStudent()


def prep(u8):
    """uint8 (B, 3, 224, 224) -> normalised float32 (as the policy transform after the crop)"""
    return (u8.astype(np.float32) / 255.0 - MEAN) / SD


# ---------------- rendering (GPU only: nerfstudio splatfacto calls .cuda() at model init; gsplat rasterises on CUDA)
# The data flights' camera (goalspec/v4/dagger_v4.cmd_render): Simulator(find_checkpoint, "baseline", "carl"),
# gsplat.render_rgb(drone camera, xv_to_T(x) @ T_c2b)["rgb"] (uint8 640 x 360), then the policy transform
# (control/pilot.py: albumentations Resize(256, 256) + CenterCrop(224, 224)) before normalisation.
_R = None


def renderer():
    """the drone camera on the yard splat, as Simulator(find_checkpoint, "baseline", "carl") sets it up for rendering
    (GSplat from the scene's config.yml loaded from the FiGS workspace; camera = frame carl "camera"; T_c2b = its
    "camera_to_body_transform"), without building the acados simulator to avoid parallel workers competing over the shared build. A tiny stand-in with .gsplat keeps render_raw unchanged."""
    global _R
    if _R is None:
        from pathlib import Path
        from figs.render.gsplat_semantic import GSplat
        import visual_expert_syn as ve
        ws = Path(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace")); scene = ve.find_checkpoint(YARD)
        cfgs = list((ws / "outputs" / scene).rglob("*.yml")); assert len(cfgs) == 1, cfgs
        frame = json.load(open(os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "configs", "frame", "carl.json")))
        cwd = os.getcwd(); os.chdir(ws); gs = GSplat({"name": scene, "path": cfgs[0]}); os.chdir(cwd)
        class _S: pass
        sim = _S(); sim.gsplat = gs
        _R = (sim, gs.generate_output_camera(frame["camera"]), np.array(frame["camera_to_body_transform"]))
    return _R


def render_raw(x):
    """state (10,) -> uint8 (360, 640, 3), exactly as the data flights rendered their frames"""
    from figs.utilities.trajectory_helper import xv_to_T
    sim, cam, T_c2b = renderer(); rgb = sim.gsplat.render_rgb(cam, xv_to_T(np.asarray(x, float)) @ T_c2b)["rgb"]
    rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    return rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


_TF = None


def transform_u8(rgb):
    """uint8 H x W x 3 -> uint8 (3, 224, 224): the policy's Resize(256, 256) + CenterCrop(224, 224) (albumentations)"""
    global _TF
    if _TF is None:
        import albumentations as A
        _TF = A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224)])
    return np.ascontiguousarray(_TF(image=rgb)["image"].transpose(2, 0, 1))


def render_u8(x):
    return transform_u8(render_raw(x))


def policy_frame(x):
    """the frame the policy gets at state x in fly mode (and the frame `render` stores): one path for both"""
    return render_u8(x)


def write_ffv1(path, I):
    """uint8 (N, 3, h, w) -> lossless FFV1 (the flags of terra/obs_to_ffv1.py)"""
    import subprocess
    n, _, h, w = I.shape
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "20", "-i", "-",
                        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", path], input=np.ascontiguousarray(I.transpose(0, 2, 3, 1)).tobytes(), capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[-300:]


def read_ffv1(path, shape):
    import subprocess
    h, w = shape[-2], shape[-1]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).transpose(0, 3, 1, 2).copy()


def render(D, tag, ids):
    import shutil
    src = f"{D}/tracks{tag}"; out = os.environ.get("IMG_OUT", f"{D}/gate1/img{tag}"); os.makedirs(out, exist_ok=True)
    if os.path.isfile(ids[0]):                                   # a file with one id per line
        ids = [l.strip() for l in open(ids[0]) if l.strip()]
    ids = sorted(f[:-4] for f in os.listdir(src) if f.endswith(".npz")) if ids == ["ALL"] else ids
    floor = float(os.environ.get("DISK_FLOOR_GB", "0")); cap = float(os.environ.get("CAP_GB", "0"))
    if cap > 0:
        import torch
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time(); n = 0; done = 0
    for i in ids:
        if os.path.exists(f"{out}/{i}.mkv"): continue
        free = shutil.disk_usage(out).free / 1e9
        if floor > 0 and free < floor:
            print(f"[gate1 render] STOP: {free:.0f} GB free on the output disk, floor {floor:.0f} GB; {done} flights written this run", flush=True)
            return
        X = np.load(f"{src}/{i}.npz")["X"]; I = np.stack([policy_frame(x) for x in X]); n += len(I); done += 1
        write_ffv1(f"{out}/{i}.mkv", I)
        assert np.array_equal(read_ffv1(f"{out}/{i}.mkv", I.shape[1:]), I), f"FFV1 round trip failed for {i}"
        print(f"[gate1 render] {i}: {len(I)} frames ({(time.time() - t0) / max(n, 1) * 1000:.1f} ms/frame), {done}/{len(ids)}", flush=True)
    print(f"GATE1_RENDER_DONE {done} flights, {n} frames, {time.time() - t0:.0f} s", flush=True)


# ---------------- training
def train(D, seed):
    import torch
    L = st.land(); E = json.load(open(f"{D}/expert_400e.json"))["flights"]
    ids = sorted(r["id"] for r in E if r["split"] == "train" and r["id"] not in EXCLUDE)
    I, Y = [], []
    for i in ids:
        z = np.load(f"{D}/tracks/{i}.npz"); I.append(read_ffv1(f"{D}/gate1/img/{i}.mkv", (224, 224))); Y.append(st.labels(z["X"], z["G"], z["PH"]))
    for p in [p for p in os.environ.get("EXTRA", "").split(",") if p]:      # relabel sets: Y + rendered frames of their states
        z = np.load(p); I.append(read_ffv1(p.replace(".npz", "_img.mkv"), (224, 224))); Y.append(z["Y"])
    I, Y = np.concatenate(I), np.concatenate(Y).astype(np.float32); my, sy = Y.mean(0), Y.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m = build(seed).to(dev)
    opt = torch.optim.AdamW([{"params": m.feat.parameters(), "lr": ENC_LR}, {"params": m.head.parameters(), "lr": LR}])
    g = np.random.default_rng(seed); n = len(I); t0 = time.time()
    for ep in range(EPOCHS):
        perm = g.permutation(n); tot = 0.0
        for b in range(0, n, BATCH):
            j = perm[b:b + BATCH]; x = torch.tensor(prep(I[j]), device=dev); y = torch.tensor((Y[j] - my) / sy, device=dev)
            loss = torch.nn.functional.mse_loss(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
        print(f"[gate1 s{seed}] epoch {ep + 1}/{EPOCHS} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    tag = os.environ.get("TAG", ""); m = m.cpu()
    torch.save({"state": m.state_dict(), "my": my, "sy": sy, "seed": seed, "train_ids": ids, "n_frames": int(n), "K": K, "CHUNK_M": CHUNK_M},
               f"{D}/gate1/student_s{seed}{tag}.pt")
    print(f"GATE1_TRAINED s{seed}{tag}: {len(ids)} flights, {n} frames, device {dev}", flush=True)


# ---------------- closed loop (GPU: render in the loop)
def fly_one(args):
    D, seed, s = args
    import torch, obs_relabel_z as oz, goal_expert_v4 as gv
    from score import score_track, clearance_fn
    from gate0b_expert import _yaw_quat, U0
    tag = os.environ.get("MODEL_TAG", ""); ck = torch.load(f"{D}/gate1/student_s{seed}{tag}.pt", weights_only=False)
    svm = ck.get("state_in") == "sv"                       # gate 0c reference arm checkpoint: image + onboard state
    m = build(seed, state=svm); m.load_state_dict(ck["state"]); dev = "cuda" if torch.cuda.is_available() else "cpu"; m = m.to(dev).eval()
    L = st.land(); ctl = gv.DirectGoalExpert.__new__(gv.DirectGoalExpert); ctl.prev_oz = 0.0
    x = np.r_[np.asarray(s["pos"], float), np.zeros(3), _yaw_quat(s["yaw"])]; u_prev = U0.copy()
    budget = 1.5 * s["t_expert_s"]; N = int(round(budget * 20)); w = np.exp(-CHUNK_M * np.arange(K)); hist = []
    P, V, T, LX = [x[:3].copy()], [x[3:6].copy()], [0.0], []
    for k in range(N):
        img = policy_frame(x)[None]
        with torch.no_grad():
            c = (m(torch.tensor(prep(img), device=dev), torch.tensor(sv_state(x)[None], device=dev)) if svm else m(torch.tensor(prep(img), device=dev))).cpu().numpy()[0] * ck["sy"] + ck["my"]
        hist.append(c.reshape(K, 5)); hist = hist[-K:]
        preds = np.array([hist[-1 - j][j] for j in range(len(hist))]); ww = w[:len(preds)][::-1]
        g = (preds * ww[:, None]).sum(0) / ww.sum()
        yaw = float(st.yaw_of(x[6:10]))
        u, _ = gv.DirectGoalExpert._u(ctl, x[0:3], x[3:6], x[6:10], g[0:2], float(x[2] + g[2]), float(st.wrap(yaw + g[3])), w_clip=0.3 if g[4] > 0.5 else 0.4)
        LX.append(x.copy()); x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / 20)
    P, V, T = np.array(P), np.array(V), np.array(T); clr = clearance_fn(YARD)(P)
    r = score_track(T, P, V, L, clr, budget)
    td = f"{D}/gate1/tracks_student_s{seed}{os.environ.get('TAG', '')}"; os.makedirs(td, exist_ok=True)
    np.savez_compressed(f"{td}/{s['id']}.npz", T=T, P=P, V=V, clr=clr, X=np.array(LX))   # X + clr: the format relabel_one reads
    r.update({"id": s["id"], "split": s["split"], "sector": s["sector"], "band": s["band"]}); return r


def fly(D, seed, NP):
    import obs_relabel_z, score
    import gate0b_student
    E = {r["id"]: r for r in json.load(open(f"{D}/expert_400e.json"))["flights"]}
    S = json.load(open(f"{D}/starts_syn013_400r.json"))["starts"]
    for s in S: s["t_expert_s"] = E[s["id"]]["t_landed_s"]
    gate0b_student.init(); res = [fly_one((D, seed, s)) for s in S]          # one process: the renderer holds the GPU
    json.dump({"flights": res}, open(f"{D}/gate1/student_s{seed}{os.environ.get('TAG', '')}_flights.json", "w"))
    for sp, keep in (("train-199", lambda r: r["split"] == "train" and r["id"] not in EXCLUDE), ("test-200", lambda r: r["split"] == "test")):
        R = [r for r in res if keep(r)]; k = {o: sum(r["outcome"] == o for r in R) for o in ("success", "contact", "overshoot", "timeout")}
        print(f"[gate1 s{seed}] {sp}: success {k['success']}/{len(R)} | contact {k['contact']} overshoot {k['overshoot']} timeout {k['timeout']}", flush=True)


# ---------------- DAgger relabel hook
def _gd(D):
    """import gate0b_dagger (it reads sys.argv[1] as its data dir at import) without editing it"""
    a = sys.argv[:]; sys.argv = [a[0], D]
    try:
        import gate0b_dagger as gd
    finally:
        sys.argv = a
    gd.D = D; return gd


def relabel_states(D_tracks, seed, tag, starts, NP=1):
    """gate0b_dagger.relabel_one (unchanged) with lead_phase_seed.install(), on flights logged as
    <D_tracks>/tracks_student_s<seed><tag>/<id>.npz (X, clr). Returns per flight: kept states X (for rendering), F, Y, drop counts."""
    import lead_phase_seed
    gd = _gd(D_tracks); gd.init(); lead_phase_seed.install(); L = st.land(); out = []
    for s in starts:
        r = gd.relabel_one((seed, tag, s))
        X = np.load(f"{D_tracks}/tracks_student_s{seed}{tag}/{s['id']}.npz")["X"]; Fv = st.feats(X, L)
        keep = {row.tobytes(): j for j, row in enumerate(Fv)}; idx = np.array([keep[f.tobytes()] for f in r["F"]], int)
        out.append(dict(r, X=X[idx], idx=idx))
    return out


def relabel(D, seed, tag, out):
    S = [s for s in json.load(open(f"{D}/starts_syn013_400r.json"))["starts"] if s["split"] == "train" and s["id"] not in EXCLUDE]
    R = relabel_states(f"{D}/gate1", seed, tag, S)
    np.savez_compressed(out, X=np.concatenate([r["X"] for r in R]), F=np.concatenate([r["F"] for r in R]), Y=np.concatenate([r["Y"] for r in R]))
    log = {k: int(sum(r[k] for r in R)) for k in ("visited", "dropped_clearance", "dropped_no_plan")}; log["kept"] = int(sum(len(r["F"]) for r in R))
    json.dump(log, open(out.replace(".npz", ".json"), "w")); print("GATE1_RELABEL_DONE", json.dumps(log), flush=True)


# ---------------- CPU dry run
def dryrun(D):
    import torch, subprocess
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8"))); torch.manual_seed(0)
    rep = {"cuda_available": torch.cuda.is_available()}
    SV = os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts", "syn_v4", "observation_data", "Maverick_vit_ft_goal_syn", "commNet")
    FL = os.path.join(FLAMBE_YARDS_FALLBACK, "train_v4", "flights", "syn_013")
    courses = os.environ.get("DRY_COURSES", "s001,s003").split(",")
    # 1. frames (already rendered on GPU by an earlier job) + their states; image t pairs with state column t+1 (STATE_SHIFT=1)
    tmp = f"{D}/gate1/dryrun"; os.makedirs(f"{tmp}/tracks_student_s0_dry", exist_ok=True)
    frames, starts = {}, []
    for c in courses:
        pt = f"{SV}/syn_013_{c}__door_canon/observations001.pt"; d = torch.load(pt, map_location="cpu", weights_only=False)
        shp = tuple(d["rgb_shape"]); h, w = shp[-2], shp[-1]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(os.path.dirname(pt), d["rgb_ffv1"]), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).transpose(0, 3, 1, 2)
        T = np.load(f"{FL}/{c}__door/tXUd.npy"); n = min(len(fr), T.shape[1] - 1)
        X = T[1:11, 1:n + 1].T.copy(); fr = fr[:n]
        from score import clearance_fn
        import score; score.SY = FLAMBE_YARDS; clr = clearance_fn(YARD)(X[:, :3])
        np.savez_compressed(f"{tmp}/tracks_student_s0_dry/dry_{c}.npz", X=X, clr=clr)
        frames[c] = fr; starts.append({"id": f"dry_{c}", "split": "train", "sector": 0, "band": 0, "pos": X[0, :3].tolist(), "yaw": float(st.yaw_of(X[0, 6:10]))})
        rep[f"frames_{c}"] = [int(n), list(fr.shape[1:])]
    # 2. labels from the relabel hook at those states (expert replanned from the state, phase seeded)
    t0 = time.time(); R = relabel_states(tmp, 0, "_dry", starts)
    rep["relabel"] = {r["id"]: {"visited": r["visited"], "kept": int(len(r["F"])), "dropped_clearance": r["dropped_clearance"],
                                "dropped_no_plan": r["dropped_no_plan"]} for r in R}; rep["relabel_s"] = round(time.time() - t0, 1)
    I = np.concatenate([frames[c][r["idx"]] for c, r in zip(courses, R)]); Y = np.concatenate([r["Y"] for r in R]).astype(np.float32)
    rep["dataset"] = {"frames": int(len(I)), "img": list(I.shape[1:]), "label": list(Y.shape[1:])}
    # 3. forward + one loss step (CPU), memory and time
    m = build(0); nparam = sum(p.numel() for p in m.parameters()); my, sy = Y.mean(0), Y.std(0) + 1e-6
    opt = torch.optim.AdamW([{"params": m.feat.parameters(), "lr": ENC_LR}, {"params": m.head.parameters(), "lr": LR}])
    for B in (8, 32):
        j = np.random.default_rng(0).choice(len(I), B, replace=False); x = torch.tensor(prep(I[j])); y = torch.tensor((Y[j] - my) / sy)
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; t = time.time()
        loss = torch.nn.functional.mse_loss(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): loss2 = torch.nn.functional.mse_loss(m(x), y)
        rep[f"step_B{B}"] = {"loss_before": round(float(loss), 4), "loss_after_one_step": round(float(loss2), 4), "cpu_s": round(time.time() - t, 2),
                             "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    rep["params_M"] = round(nparam / 1e6, 2)
    # 4. checks (checks.py criteria; checks.py itself is built for SVNet commNets: networks feat/mlp0/mlp1 + objective)
    m.eval(); j = np.linspace(0, len(I) - 1, min(500, len(I))).astype(int); x = torch.tensor(prep(I[j]))
    with torch.no_grad():
        z = torch.cat([m.head[1](m.head[0](m.feat(x[b:b + 50]))) for b in range(0, len(x), 50)])
        c = torch.cat([m(x[b:b + 50]) for b in range(0, len(x), 50)]); cb = m(torch.tensor(prep(np.zeros_like(I[j[:50]]))))
    live = int((z > 0).any(0).sum()); rel = float((cb[:, :5] - c[:50, :5]).abs().mean() / (c[:50, :5].abs().mean() + 1e-9))
    rep["check_live_image"] = {"live_units": live, "units": int(z.shape[1]), "live_frac": round(live / z.shape[1], 3),
                               "blank_image_rel_cmd_change": round(rel, 3), "pass": live / z.shape[1] >= 0.25 and rel > 0.10,
                               "note": "untrained head on the warm-start trunk; re-run after training"}
    import inspect
    sig = list(inspect.signature(m.forward).parameters)
    rep["check_gps_denied"] = {"forward_inputs": sig, "horizontal_position_inputs": [], "goal_inputs": [],
                               "xy_shift_cmd_change": 0.0, "pass": sig == ["img"],
                               "note": "the network reads the image only; a world-frame shift changes no network input"}
    rep["render"] = "not possible on CPU (nerfstudio splatfacto .cuda() at model init, gsplat CUDA rasteriser); frames from cohort syn_v4 (GPU-rendered earlier)"
    json.dump(rep, open(f"{tmp}/dryrun.json", "w"), indent=1); print("GATE1_DRYRUN", json.dumps(rep), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "render": render(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
    elif mode == "train": train(sys.argv[2], int(sys.argv[3]))
    elif mode == "fly": fly(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif mode == "relabel": relabel(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif mode == "dryrun": dryrun(sys.argv[2])
