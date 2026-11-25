"""Train and evaluate a state-only policy using expert guidance."""
import json, math, os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

sys.path[:0] = [
    *LOCAL_MODULE_PATHS,
    os.path.join(FLAMBE_YARDS_FALLBACK, "kv14"),
    os.path.join(FLAMBE_TERRA, "goalspec", "v4"),
    os.path.join(FLAMBE_TERRA, "goalspec"),
    FLAMBE_TERRA,
]
K, CHUNK_M, EPOCHS, EXCLUDE = 20, 0.01, 40, {"g000r"}
YARD = "syn_013"; SYR = os.path.join(FLAMBE_YARDS, "yards")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def land():
    return np.asarray([g for g in json.load(open(f"{SYR}/{YARD}/goals.json"))["goals"] if g["id"] == "door"][0]["land_ned"], float)


def feats(X, L):
    return np.c_[X[:, 0:3] - L, X[:, 3:6], X[:, 6:10]].astype(np.float32)


def labels(X, G, PH):
    g = np.c_[G[:, 0], G[:, 1], G[:, 2] - X[:, 2], wrap(G[:, 3] - yaw_of(X[:, 6:10])), (PH == 2).astype(float)]
    n = len(g); idx = np.minimum(np.arange(n)[:, None] + np.arange(K)[None, :], n - 1)
    return g[idx].reshape(n, -1).astype(np.float32)          # (n, K*5)


def net(seed):
    import torch
    torch.manual_seed(seed)
    W, D = int(os.environ.get("NET_W", "512")), int(os.environ.get("NET_D", "3"))   # Hidden layer width and depth
    NF = int(os.environ.get("FOURIER", "0"))   # NeRF-style encoding [x, sin(2^k pi x), cos(2^k pi x)], k < NF (Tancik 2020); 0 = off
    class Enc(torch.nn.Module):
        def forward(self, x):
            if NF == 0: return x
            f = torch.cat([x * (2.0 ** k) * 3.141592653589793 for k in range(NF)], -1); return torch.cat([x, torch.sin(f), torch.cos(f)], -1)
    L_ = ([Enc()] if NF else []) + [torch.nn.Linear(10 * (1 + 2 * NF), W), torch.nn.GELU()]
    for _ in range(D - 1): L_ += [torch.nn.Linear(W, W), torch.nn.GELU()]
    return torch.nn.Sequential(*L_, torch.nn.Linear(W, K * 5))


def train(D, seed):
    import torch
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    L = land(); E = json.load(open(f"{D}/expert_400e.json"))["flights"]
    ids = sorted(r["id"] for r in E if r["split"] == "train" and r["id"] not in EXCLUDE)
    Xs, Ys = [], []
    for i in ids:
        z = np.load(f"{D}/tracks/{i}.npz"); Xs.append(feats(z["X"], L)); Ys.append(labels(z["X"], z["G"], z["PH"]))
    n_expert = sum(len(x) for x in Xs); extra = [p for p in os.environ.get("EXTRA", "").split(",") if p]
    for p in extra:                                   # DAgger aggregate: relabelled student states (raw feats, encoded K x 5 chunks)
        z = np.load(p); Xs.append(z["F"].astype(np.float32)); Ys.append(z["Y"].astype(np.float32))
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, Yt = torch.tensor((X - mx) / sx).to(dev), torch.tensor((Y - my) / sy).to(dev)
    m = net(seed).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3); n = len(Xt); B = 1024
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * ((n + B - 1) // B)); g = torch.Generator().manual_seed(seed); t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g).to(dev); tot = 0.0
        for b in range(0, n, B):
            j = perm[b:b + B]; loss = torch.nn.functional.mse_loss(m(Xt[j]), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(j)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"[student s{seed}] epoch {ep + 1}/{EPOCHS} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    m = m.cpu(); tag = os.environ.get("TAG", "")
    torch.save({"state": m.state_dict(), "mx": mx, "sx": sx, "my": my, "sy": sy, "train_ids": ids, "n_frames": int(n), "seed": seed,
                "K": K, "CHUNK_M": CHUNK_M, "extra": extra, "n_expert_frames": int(n_expert)}, f"{D}/student_s{seed}{tag}.pt")
    print(f"STUDENT_TRAINED s{seed}{tag}: {len(ids)} expert flights ({n_expert} frames) + {len(extra)} DAgger sets, {n} frames total, excluded {sorted(EXCLUDE)}, device {dev}", flush=True)


_M = {}


def fly_one(args):
    D, seed, m = args
    import torch, obs_relabel_z as oz, goal_expert_v4 as gv
    from score import score_track, clearance_fn
    from gate0b_expert import _yaw_quat, U0
    torch.set_num_threads(1)
    if seed not in _M:
        ck = torch.load(f"{D}/student_s{seed}{os.environ.get('MODEL_TAG', '')}.pt", weights_only=False); nn_ = net(seed); nn_.load_state_dict(ck["state"]); nn_.eval(); _M[seed] = (nn_, ck)
    nn_, ck = _M[seed]; L = land(); HZ = 20
    goals = {g["id"]: g for g in json.load(open(f"{SYR}/{YARD}/goals.json"))["goals"]}
    gy = oz.dv.gx.GoalYard(oz.dv.SY / YARD, goals["door"])
    ctl = gv.DirectGoalExpert.__new__(gv.DirectGoalExpert); ctl.prev_oz = 0.0          # controller state only (no planner)
    x = np.r_[np.asarray(m["pos"], float), np.zeros(3), _yaw_quat(m["yaw"])]; u_prev = U0.copy()
    budget = 1.5 * m["t_expert_s"]; N = int(round(budget * HZ))
    w = np.exp(-CHUNK_M * np.arange(K)); hist = []                                    # chunks made at past steps
    P, V, T = [x[:3].copy()], [x[3:6].copy()], [0.0]; t0 = time.time(); LX, LF, LG, LC = [], [], [], []   # diagnostic log
    for k in range(N):
        f = (feats(x[None], L) - ck["mx"]) / ck["sx"]
        with torch.no_grad():
            c = nn_(torch.tensor(f, dtype=torch.float32)).numpy()[0] * ck["sy"] + ck["my"]
        hist.append(c.reshape(K, 5)); hist = hist[-K:]
        # temporal ensembling: chunk made i steps ago predicts this step at its index i; oldest-first weights exp(-m * age_rank)
        preds = np.array([hist[-1 - j][j] for j in range(len(hist))])        # age j = 0 (newest chunk) .. len-1 (oldest)
        ww = w[:len(preds)][::-1]                                              # oldest-first: the oldest gets exp(0)
        if os.environ.get("ENSEMBLE", "1") == "0": preds, ww = preds[:1], np.ones(1)  # diagnostic: newest chunk[0] only
        g = (preds * ww[:, None]).sum(0) / ww.sum()
        yaw = float(yaw_of(x[6:10])); z_set = float(x[2] + g[2]); yaw_des = float(wrap(yaw + g[3]))
        u, _ = gv.DirectGoalExpert._u(ctl, x[0:3], x[3:6], x[6:10], g[0:2], z_set, yaw_des, w_clip=0.3 if g[4] > 0.5 else 0.4)
        LX.append(x.copy()); LF.append(np.asarray(f, np.float32)[0]); LG.append(np.r_[g, z_set, yaw_des]); LC.append(hist[-1][0])
        x = oz.DYN.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy()); V.append(x[3:6].copy()); T.append((k + 1) / HZ)
    P, V, T = np.array(P), np.array(V), np.array(T); clr = clearance_fn(YARD)(P)
    r = score_track(T, P, V, L, clr, budget)
    tdir = f"{D}/tracks_student_s{seed}{os.environ.get('TAG', '')}"; os.makedirs(tdir, exist_ok=True)
    np.savez_compressed(f"{tdir}/{m['id']}.npz", T=T, P=P, V=V, clr=clr, X=np.array(LX), F=np.array(LF), GA=np.array(LG), C0=np.array(LC))
    r.update({"id": m["id"], "split": m["split"], "sector": m["sector"], "band": m["band"], "wall_s": round(time.time() - t0, 1),
              "final_m": float(np.linalg.norm(P[-1] - L))})
    return r


def init():
    import obs_relabel_z as oz, score
    oz.YARD = YARD; oz.init_worker(); score.SY = FLAMBE_YARDS


def fly(D, seed, NP):
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import obs_relabel_z, score, torch                # parent imports before the fork (gate0b_expert.init note)
    from multiprocessing import get_context
    from concurrent.futures import ProcessPoolExecutor
    E = {r["id"]: r for r in json.load(open(f"{D}/expert_400e.json"))["flights"]}
    S = json.load(open(f"{D}/starts_syn013_400r.json"))["starts"]
    for s in S:
        s["t_expert_s"] = E[s["id"]]["t_landed_s"]
    if os.environ.get("SUBSET"): S = [s for s in S if s["split"] == "test"][:int(os.environ["SUBSET"])]
    tag = os.environ.get("TAG", ""); out = f"{D}/student_s{seed}{tag}_flights.json"; res = []; t0 = time.time()
    with ProcessPoolExecutor(NP, mp_context=get_context("fork"), initializer=init) as p:
        for i, r in enumerate(p.map(fly_one, [(D, seed, s) for s in S], chunksize=1)):
            res.append(r)
            if i % 25 == 0:
                print(f"[student s{seed}] {i + 1}/{len(S)} {r['id']} {r['outcome']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"flights": res}, open(out, "w"))
    for sp, keep in (("train-199 (copy)", lambda r: r["split"] == "train" and r["id"] not in EXCLUDE),
                     ("test-200", lambda r: r["split"] == "test")):
        R = [r for r in res if keep(r)]; n = len(R); k = {o: sum(r["outcome"] == o for r in R) for o in ("success", "contact", "overshoot", "timeout")}
        anyc = sum(r["min_clearance_m"] < 0.30 for r in R)
        print(f"[student s{seed}] {sp}: success {k['success']}/{n} ({100 * k['success'] / max(n, 1):.1f}%) | contact {k['contact']} "
              f"overshoot {k['overshoot']} timeout {k['timeout']} | any contact (whole flight) {anyc}", flush=True)
    print(f"STUDENT_FLY_DONE s{seed}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]))
    else:
        fly(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 8)
