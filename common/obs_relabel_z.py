"""Fly and relabel expert variants with route-based yaw and a fixed cruise altitude."""
import os, sys, re, json, math, time, numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
FLAMBE_YARDS_FALLBACK = os.environ.get("FLAMBE_YARDS_FALLBACK", os.path.join(FLAMBE_ROOT, "flambe_yards"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.join(FLAMBE_TERRA, "goalspec", "v4"), os.path.join(FLAMBE_TERRA, "goalspec"), FLAMBE_TERRA]
import dagger_v4 as dv, goal_expert_v4 as gv, visual_expert_syn as ve, grid3d as g3
from multiprocessing import get_context
DYN = GR = None
def make_dyn(tag):
    """dagger_v4.Dyn with a private acados build dir (the original compiles into a fixed /tmp/w_dyn, which races between processes)"""
    from pathlib import Path
    from figs.simulator import Simulator
    d = dv.Dyn.__new__(dv.Dyn); s_ = Simulator.__new__(Simulator)
    s_.configs_path = Path(sys.modules["figs.simulator"].__file__).parent.parent.parent / "configs"
    s_.conFiG = {"rollout": {}, "drone": {}, "perception": {}}; s_.solver = None
    cwd = os.getcwd(); w = f"/tmp/w_dyn_{tag}"; os.makedirs(w, exist_ok=True); os.chdir(w)
    s_.load_rollout("baseline"); s_.load_frame("carl"); os.chdir(cwd)
    r = s_.conFiG["rollout"]; d.sim = s_; d.n_sub = int(r["frequency"] / ve.HZ_CTL); d.n_delay = int(r["delay"] * r["frequency"])
    d.mu = np.array(r["model_noise"]["mean"]) / r["frequency"]; return d
def init_worker():
    global DYN, GR
    DYN = make_dyn(os.getpid()); GR = g3.Grid(dv.SY / YARD)
CO = os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts", "syn_v4"); FL = FLAMBE_YARDS_FALLBACK; YARD = "syn_013"; YAW_AHEAD = 7.0
VAR = {"base": None, "k25": (0.25, "const"), "k0": (0.0, "const"), "k25z": (0.25, "route"), "k0z": (0.0, "route")}   # (K_CT, cruise z_set mode)
T20 = "s000 s001 s002 s003 s004 s005 s006 s007 s009 s010 s012 s014 s017 s018 s020 s021 s022 s023 s024 s025".split()
class ObsExpert(gv.DirectGoalExpert):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        hv = self.pose["hover"] if self.pose is not None else self.route[-1]
        z = [p[2] for p in self.route if np.linalg.norm(p[:2] - hv[:2]) > 1.5]
        self.z_cruise = float(np.median(z)) if z else float(hv[2]); self.zmode = "const"
        P = np.array(self.route, float); self.nom_P = P; L = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1); self.nom_s = np.concatenate([[0.0], np.cumsum(L)])
    def z_route(self, pos):
        """height of the nominal (initial, never replanned) route at the drone's horizontal projection onto it, mean over +-1 m of arc length"""
        P, S = self.nom_P, self.nom_s; best = None
        for i in range(len(P) - 1):
            a, b = P[i, :2], P[i + 1, :2]; ab = b - a; L = float(np.linalg.norm(ab)) + 1e-9
            t = float(np.clip(np.dot(pos[:2] - a, ab) / L, 0.0, L)); d = float(np.linalg.norm(a + ab / L * t - pos[:2]))
            if best is None or d < best[0]: best = (d, S[i] + t)
        ss = np.clip(best[1] + np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), 0.0, S[-1])
        return float(np.mean(np.interp(ss, S, P[:, 2])))
    def _yaw_des(self, pos):
        look = self._at(self.seg, self._t_on_seg, YAW_AHEAD)[0]; dv_ = look[:2] - pos[:2]; bearing = math.atan2(dv_[1], dv_[0]); face = self.pose["yaw"]
        dh = float(np.linalg.norm(self.pose["look"][:2] - pos[:2])); w = float(np.clip((dh - gv.YAW_BLEND[0]) / (gv.YAW_BLEND[1] - gv.YAW_BLEND[0]), 0.0, 1.0))
        return ve.wrap(face + w * ve.wrap(bearing - face))
    def _u(self, pos, vel, q, v_des_xy, z_set, yaw_des, w_clip=0.4):
        if self.phase == "cruise" and float(np.linalg.norm(np.asarray(pos[:2]) - self.pose["hover"][:2])) > 1.5:
            z_set = self.z_cruise if self.zmode == "const" else self.z_route(pos)
        return super()._u(pos, vel, q, v_des_xy, z_set, yaw_des, w_clip)
def expert(var, p0, cache, gr):
    goals = {g["id"]: g for g in json.load(open(dv.SY / YARD / "goals.json"))["goals"]}; g = goals["door"]; gy = dv.gx.GoalYard(dv.SY / YARD, g)
    if VAR[var] is None: return gv.DirectGoalExpert(gy, gr, np.asarray(p0, float), cache)
    gv.K_CT = VAR[var][0]; ex = ObsExpert(gy, gr, np.asarray(p0, float), cache); ex.zmode = VAR[var][1]; return ex
def closed(var):
    import syn_exact_collision as sx
    yx = sx.ExactYard(YARD); gr = GR; cache = {}; dyn = DYN; res = []
    for s in T20:
        T = np.load(f"{FL}/train_v4/flights/{YARD}/{s}__door/tXUd.npy"); x = T[1:11, 0].copy(); ex = expert(var, x[:3], cache, gr); ex.reset()
        land = np.asarray(ex.pose["land"], float); u_prev = T[11:15, 0].copy(); P = [x[:3].copy()]; rp = 0
        for k in range(1400):
            rp += int(dv.maybe_replan(ex, x, cache)); u = ex.decide(k / ve.HZ_CTL, x.copy())
            if ex.landed: break
            x = dyn.step(x, u_prev, u); u_prev = u; P.append(x[:3].copy())
        P = np.array(P); c = yx.clearance(P); d = np.linalg.norm(P - land, axis=1); hit = np.where(c < 0.30)[0]
        ok = d.min() < 0.5 and not len(hit)
        res.append(ok); print(f"[closed {var}] {s}: {'OK  ' if ok else 'FAIL'} steps {len(P)} landed {ex.landed} closest {d.min():.2f} m final {d[-1]:.2f} | min clearance {c.min():.2f} m"
                              f"{f' first contact t={hit[0]/20:.1f}s' if len(hit) else ''} | replans {rp} | nominal expert flight {T.shape[1]} steps", flush=True)
    print(f"[closed {var}] SUCCESS {sum(res)}/20", flush=True)
def relabel_one(args):
    var, c = args
    if os.path.exists(f"{CO}/obs_labels_{var}/{c}.npz"): return c, -1, -1, 0, None, 0.0
    s, tag = re.match(r"^syn_013_(s\d+)__door_(y13\w+)_canon$", c).groups()
    gr = GR; cache = {}; dyn = DYN; T = np.load(f"{FL}/dagger_v4/flights/{YARD}/{s}__door_{tag}/tXUd.npy")
    X = T[1:11, 1:].T; Us = T[11:15].T; n = len(X); Uprev = np.vstack([Us[0:1], Us[:n - 1]])
    ex = expert(var, T[1:4, 0], cache, gr); t0 = time.time()
    A, C, rp, la, logs, U = dv.relabel_states(ex, X, Uprev, dyn, cache, chunks=True)
    if len(A) < n: A = np.concatenate([A, np.repeat(A[-1:], n - len(A), 0)]); C = np.concatenate([C, np.repeat(C[-1:], n - len(C), 0)])
    os.makedirs(f"{CO}/obs_labels_{var}", exist_ok=True); np.savez(f"{CO}/obs_labels_{var}/{c}.npz", a=A, c=C)
    ref = np.load(f"{CO}/vel_labels_ramp/{c}.npz")["a"]
    return c, n, len(ref), int(rp.sum()), la, time.time() - t0
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "closed":
        init_worker()
        for v in sys.argv[2:]: closed(v)
    else:
        var = sys.argv[2]; OD = os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts", "syn_v4", "observation_data", "Maverick_vit_ft_goal_syn_L8p_G4V4y13de1", "commNet")
        cs = sorted(c for c in os.listdir(OD) if re.match(r"^syn_013_s\d+__door_y13(r1|r2|r3|e1)_canon$", c)); t0 = time.time()
        with get_context("fork").Pool(int(os.environ.get("NP", "16")), initializer=init_worker) as p:
            for c, n, nref, rp, la, dt in p.imap_unordered(relabel_one, [(var, c) for c in cs]):
                print(f"[relabel {var}] {c}: {n} frames (cohort labels {nref}) replans {rp} landed_at {la} {dt:.0f}s", flush=True)
        print(f"[relabel {var}] DONE {len(cs)} courses {time.time()-t0:.0f}s", flush=True)
