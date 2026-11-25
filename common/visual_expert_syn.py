#!/usr/bin/env python3
"""Generate geometric expert flights and rendered observations for synthetic yards."""
import argparse, glob, heapq, json, math, os, sys, time
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

sys.path[:0] = [*LOCAL_MODULE_PATHS, os.path.join(FLAMBE_TERRA, "FiGS-Standalone"), os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "src")]

YARD = os.environ.get("YARD", "syn_002")
SY_ROOT = Path(os.environ.get("SY_ROOT", FLAMBE_YARDS))
WS_OUT = Path(os.environ.get("WS_OUT", os.path.join(FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace", "outputs")))
REG_DIR = Path(os.environ.get("SYN_REG_DIR", os.path.join(FLAMBE_TERRA, "syn_registry")))
YARD_DIR = SY_ROOT / "yards" / YARD
SCENE = YARD
OUT_BASE = WS_OUT / YARD / "visual_expert"

# geometry of the task (metres, NED)
HOVER_AHEAD = 1.0      # hover point: this far in front of the door face
STANDOFF_AHEAD = 3.0   # approach starts here, on the door normal
DOOR_CENTRE_UP = 1.2   # fly height above the doorstep during the approach (door-centre height)
LAND_UP = 0.5          # final height above the doorstep
CRUISE_Z_DEFAULT = -2.15   # NED, 2.15 m above the street (the protocol start height)
INFLATE_M = 0.6        # obstacle inflation radius for planning
BODY_R = 0.30          # body radius for the clearance score
HALF_BAND = 0.45       # occupancy layers within this of the flight height count as obstacles

# drone and control constants (carl frame, same as the msl experts)
MASS, TN, G_NED = 1.144, 27.6, 9.81
F_HOV = -MASS * G_NED / TN
HZ_CTL = 20
K_ALT, K_VZ, K_VEL, K_ATT, K_POS = 0.4, 0.2, 0.8, 3.0, 0.8
OMEGA_CLIP, THRUST_MIN, THRUST_MAX = 1.5, -0.9, 0.0
V_CRUISE, V_APPROACH, V_DESCENT = 1.5, 0.6, 0.3
WP_REACH = 1.2
MAX_EP = 60.0

CAM_CFG = {"height": 360, "width": 640, "channels": 3,
           "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184}


# ── yard description ──────────────────────────────────────────────────────────────

def find_checkpoint(yard: str) -> str:
    cands = sorted(glob.glob(str(WS_OUT / yard / "splatfacto" / "*" / "config.yml")))
    if not cands:
        raise FileNotFoundError(f"no splat for {yard} under {WS_OUT}")
    return f"{yard}/splatfacto/{Path(cands[-1]).parent.name}"


class Yard:
    def __init__(self, yard_dir: Path):
        self.dir = yard_dir
        self.survey = json.load(open(yard_dir / "survey.json"))
        t = np.load(yard_dir / "recon" / "truth.npz")
        self.lo = np.asarray(t["lo"], float)
        self.cell = float(t["cell"])
        self.occ3d = np.asarray(t["occ3d"]).astype(bool)      # (nx, ny, nz), z layers from lo[2] downward in NED
        self.nz = self.occ3d.shape[2]
        self.layer_z = self.lo[2] + (np.arange(self.nz) + 0.5) * self.cell
        I = np.argwhere(self.occ3d)
        self.occ_pts = self.lo + (I + 0.5) * self.cell
        self.tree = cKDTree(self.occ_pts) if len(self.occ_pts) else None
        s = self.survey
        self.face = np.asarray(s["door_face_ned"], float)
        self.step = np.asarray(s["doorstep_ned"], float)
        self.n = np.asarray(s["door_normal_ned"], float); self.n[2] = 0; self.n /= np.linalg.norm(self.n)
        self.fly_z = float(self.step[2] - DOOR_CENTRE_UP)
        self.land_z = float(self.step[2] - LAND_UP)
        self.cruise_z = float(min(CRUISE_Z_DEFAULT, self.fly_z))
        # O2 pilot (2026-10-22): an alias yard (e.g. syn_113 = syn_013) may carry cruise.json {"cruise_z": ...} to fly
        # the same scene at a higher cruise height; yards without the file are unchanged.
        if (yard_dir / "cruise.json").exists():
            cj = json.load(open(yard_dir / "cruise.json"))
            self.cruise_z = float(min(cj["cruise_z"], self.fly_z))
            # plan_z: the slice the route is planned on (default: the cruise slice). Planning on the old 2.15 m
            # slice and flying higher keeps the old sideways route and adds only vertical margin.
            self.plan_z = cj.get("plan_z")
        self.hover = np.array([*(self.face[:2] + HOVER_AHEAD * self.n[:2]), self.fly_z])
        self.goal = np.array([self.hover[0], self.hover[1], self.land_z])        # CANON_DOOR_NED
        self.standoff = np.array([*(self.face[:2] + STANDOFF_AHEAD * self.n[:2]), self.fly_z])
        self.door_yaw = math.atan2(-self.n[1], -self.n[0])                        # looking into the door
        self.starts = {st["id"]: st for st in s["starts"]}

    # occupancy slice for flight at height z (NED)
    def slice_mask(self, z: float, inflate_m: float = INFLATE_M) -> np.ndarray:
        layers = np.where(np.abs(self.layer_z - z) <= HALF_BAND)[0]
        m = self.occ3d[:, :, layers].any(axis=2) if len(layers) else np.zeros(self.occ3d.shape[:2], bool)
        r = int(math.ceil(inflate_m / self.cell))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        return binary_dilation(m, structure=disk)

    def to_cell(self, p):
        return tuple(np.floor((np.asarray(p[:2]) - self.lo[:2]) / self.cell).astype(int))

    def to_xy(self, c):
        return self.lo[:2] + (np.asarray(c) + 0.5) * self.cell

    def clearance(self, P: np.ndarray) -> np.ndarray:
        """distance from each position to the nearest occupied cell centre (3-D)"""
        if self.tree is None:
            return np.full(len(P), np.inf)
        d, _ = self.tree.query(P)
        return d


# ── planning ──────────────────────────────────────────────────────────────────────

def nearest_free(mask, c, max_r=8):
    if not mask[c]:
        return c
    best, bd = None, 1e9
    for di in range(-max_r, max_r + 1):
        for dj in range(-max_r, max_r + 1):
            i, j = c[0] + di, c[1] + dj
            if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and not mask[i, j]:
                d = di * di + dj * dj
                if d < bd:
                    best, bd = (i, j), d
    return best


def astar(mask, start, goal):
    nx, ny = mask.shape
    nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    g = {start: 0.0}; came = {}
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    pq = [(h(start), 0.0, start)]
    closed = set()
    while pq:
        f, gc, c = heapq.heappop(pq)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]; path.append(c)
            return path[::-1]
        if c in closed:
            continue
        closed.add(c)
        for di, dj, w in nbrs:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or mask[n]:
                continue
            if w > 1.0 and (mask[c[0] + di, c[1]] or mask[c[0], c[1] + dj]):
                continue   # no corner cutting
            ng = gc + w
            if ng < g.get(n, 1e18):
                g[n] = ng; came[n] = c
                heapq.heappush(pq, (ng + h(n), ng, n))
    return None


def line_free(mask, a, b, cell, lo, step=0.1):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    for s in np.linspace(0, 1, n):
        p = a + s * (b - a)
        c = tuple(np.floor((p - lo) / cell).astype(int))
        if not (0 <= c[0] < mask.shape[0] and 0 <= c[1] < mask.shape[1]) or mask[c]:
            return False
    return True


def shortcut(points, mask, cell, lo, max_seg=4.0):
    """greedy line-of-sight shortcutting, then resample so no segment exceeds max_seg"""
    out = [points[0]]; i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not line_free(mask, points[i], points[j], cell, lo):
            j -= 1
        out.append(points[j]); i = j
    res = [out[0]]
    for a, b in zip(out[:-1], out[1:]):
        L = np.linalg.norm(np.asarray(b) - np.asarray(a)); k = max(1, int(math.ceil(L / max_seg)))
        for s in range(1, k + 1):
            res.append(tuple(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s / k))
    return [np.asarray(p, float) for p in res]


def plan_route(yard: Yard, start_xy, cruise_z):
    """waypoints (xy) from the start to the standoff at cruise altitude; None if no path"""
    mask = yard.slice_mask(getattr(yard, "plan_z", None) or cruise_z)
    c0 = nearest_free(mask, yard.to_cell(start_xy)); c1 = nearest_free(mask, yard.to_cell(yard.standoff))
    if c0 is None or c1 is None:
        return None, mask
    cells = astar(mask, c0, c1)
    if cells is None:
        return None, mask
    pts = [tuple(start_xy)] + [tuple(yard.to_xy(c)) for c in cells[1:-1]] + [tuple(yard.standoff[:2])]
    wps = shortcut(pts, mask, yard.cell, yard.lo[:2])
    return wps[1:], mask     # drop the start itself


# ── control ───────────────────────────────────────────────────────────────────────

def yaw_of(q_xyzw):
    b1 = Rotation.from_quat(q_xyzw).as_matrix()[:, 0]
    return math.atan2(b1[1], b1[0])


def quat_from_yaw(yaw):
    return Rotation.from_euler("z", yaw).as_quat()   # xyzw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GeometricExpert:
    """privileged expert: waypoints and door pose are known; images are rendered only for the dataset"""
    hz = HZ_CTL

    def __init__(self, yard: Yard, waypoints, cruise_z, verbose=False):
        self.yard = yard; self.door_ned = yard.goal
        self.waypoints = [np.asarray(w, float) for w in waypoints]
        self.cruise_z = cruise_z; self.verbose = verbose
        self.reset()

    def reset(self):
        self.phase = "cruise"; self.wp = 0; self._landed = False
        self.prev_oz = 0.0; self.hold_steps = 0; self.z_set = None
        self.step = 0; self.last_log = -1.0; self.counts = {}

    @property
    def landed(self):
        return self._landed

    def _u(self, pos, vel, q, v_des_xy, z_set, yaw_des, w_clip=0.4):
        a = K_VEL * (v_des_xy - vel[:2])
        z_des = np.array([-a[0], -a[1], G_NED]); z_des /= np.linalg.norm(z_des)
        R = Rotation.from_quat(q).as_matrix()
        e = np.cross(R[:, 2], z_des)
        wx = float(np.clip(K_ATT * np.dot(e, R[:, 0]), -w_clip, w_clip))
        wy = float(np.clip(K_ATT * np.dot(e, R[:, 1]), -w_clip, w_clip))
        f = F_HOV - K_ALT * (pos[2] - z_set) - K_VZ * vel[2]
        hdg_err = wrap(yaw_des - yaw_of(q))
        oz = float(np.clip(0.7 * hdg_err, -0.8, 0.8))
        oz = float(np.clip(oz, self.prev_oz - 0.12, self.prev_oz + 0.12)); self.prev_oz = oz
        u = np.array([float(np.clip(f, THRUST_MIN, THRUST_MAX)), wx, wy, float(np.clip(oz, -OMEGA_CLIP, OMEGA_CLIP))])
        return u, hdg_err

    def decide(self, t, xcr, rgb=None, depth=None):
        pos, vel, q = xcr[0:3], xcr[3:6], xcr[6:10]
        y = self.yard
        if self.phase == "cruise":
            tgt = self.waypoints[self.wp]
            d = np.linalg.norm(tgt - pos[:2])
            if d < WP_REACH and self.wp < len(self.waypoints) - 1:
                self.wp += 1; tgt = self.waypoints[self.wp]; d = np.linalg.norm(tgt - pos[:2])
            last = self.wp == len(self.waypoints) - 1
            if last and d < 0.5:
                self.phase = "approach"
            unit = (tgt - pos[:2]) / (d + 1e-6)
            yaw_des = math.atan2(unit[1], unit[0]) if (not last or d > 1.5) else y.door_yaw
            vmax = V_CRUISE if not last else min(V_CRUISE, 0.8)
            speed = min(vmax, 0.5 * d + 0.3)
            hdg = wrap(yaw_des - yaw_of(q))
            speed *= max(0.0, math.cos(hdg)) if abs(hdg) > 0.25 else 1.0
            z_set = self.cruise_z if not last else y.fly_z
            u, _ = self._u(pos, vel, q, speed * unit, z_set, yaw_des)
        elif self.phase == "approach":
            tgt = y.hover[:2]; dvec = tgt - pos[:2]; d = np.linalg.norm(dvec)
            speed = min(V_APPROACH, 0.5 * d + 0.05)
            v_des = speed * dvec / (d + 1e-6)
            u, _ = self._u(pos, vel, q, v_des, y.fly_z, y.door_yaw)
            if d < 0.15 and np.linalg.norm(vel[:2]) < 0.15:
                self.hold_steps += 1
                if self.hold_steps >= 10:
                    self.phase = "descent"; self.z_set = pos[2]
            else:
                self.hold_steps = 0
        else:  # descent
            self.z_set = min(y.land_z, self.z_set + V_DESCENT / HZ_CTL)
            v_des = np.clip(K_POS * (y.hover[:2] - pos[:2]), -0.3, 0.3)
            u, _ = self._u(pos, vel, q, v_des, self.z_set, y.door_yaw, w_clip=0.3)
            if pos[2] >= y.land_z - 0.05:
                self._landed = True
        self.counts[self.phase] = self.counts.get(self.phase, 0) + 1
        if self.verbose and (t - self.last_log >= 1.0):
            dg = np.linalg.norm(y.goal - pos)
            print(f"  t={t:5.1f}s {self.phase:8s} wp={self.wp}/{len(self.waypoints)} pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}) "
                  f"v={np.linalg.norm(vel[:2]):.2f} d_goal={dg:.2f}")
            self.last_log = t
        self.step += 1
        return u


# ── episode ───────────────────────────────────────────────────────────────────────

def perturb_state(xcr, rng, pos, vel, yaw):
    """SINGER-style state perturbation: position, velocity and yaw jitter, applied in place"""
    xcr[0:2] += rng.uniform(-pos, pos, 2); xcr[2] += rng.uniform(-pos / 2, pos / 2)
    xcr[3:5] += rng.uniform(-vel, vel, 2); xcr[5] += rng.uniform(-vel / 2, vel / 2)
    if yaw > 0:
        q = Rotation.from_euler("z", rng.uniform(-yaw, yaw)) * Rotation.from_quat(xcr[6:10])
        xcr[6:10] = q.as_quat()
    return xcr


def run_episode(sim, x0, tf, expert, T_c2b, want_images=True, perturb=None):
    """perturb: None (nominal) or dict(period_s, pos_m, vel_ms, yaw_rad, kick_ms or 0, band=(lo, hi), rng)"""
    from figs.utilities.trajectory_helper import xv_to_T, obedient_quaternion as oq
    hz_sim = sim.conFiG["rollout"]["frequency"]; t_dly = sim.conFiG["rollout"]["delay"]
    mu = np.array(sim.conFiG["rollout"]["model_noise"]["mean"]) / hz_sim
    sd = np.array(sim.conFiG["rollout"]["model_noise"]["std"]) / hz_sim
    nx = sim.conFiG["drone"]["nx"]; cam_cfg = sim.conFiG["drone"]["camera"]
    n_sim2ctl = int(hz_sim / HZ_CTL); n_delay = int(t_dly * hz_sim)
    Nctl = int(tf * HZ_CTL); Nsim = Nctl * n_sim2ctl   # exact multiple, so the last control index stays in range
    Tro = np.zeros(Nctl); Xro = np.zeros((nx, Nctl + 1)); Uro = np.zeros((4, Nctl)); rgbs = []
    Xro[:, 0] = x0; xcr = x0.copy(); xpr = x0.copy()
    ucm = np.array([F_HOV, 0, 0, 0.0]); udl = np.stack([ucm, ucm], 1)
    camera = sim.gsplat.generate_output_camera(cam_cfg); expert.reset(); n_actual = Nctl
    kicked = False; door_xy = np.asarray(expert.door_ned[:2], float)
    per_steps = int(perturb["period_s"] * hz_sim) if perturb else 0
    for i in range(Nsim):
        tcr = i / hz_sim
        if perturb and i > 0 and per_steps and i % per_steps == 0 and expert.phase != "descent":
            scale = 1.0 if expert.phase == "cruise" else 0.5
            perturb_state(xcr, perturb["rng"], scale * perturb["pos_m"], scale * perturb["vel_ms"], scale * perturb["yaw_rad"])
            xcr[6:10] = oq(xcr[6:10], xpr[6:10]); xpr = xcr.copy()
        if perturb and perturb.get("kick_ms", 0) > 0 and not kicked:
            dv = door_xy - xcr[:2]; dd = float(np.linalg.norm(dv))
            if perturb["band"][0] <= dd <= perturb["band"][1]:
                xcr[3:5] = perturb["kick_ms"] * dv / (dd + 1e-6); kicked = True
        if i % n_sim2ctl == 0:
            k = i // n_sim2ctl
            if want_images:
                rgb = sim.gsplat.render_rgb(camera, xv_to_T(xcr) @ T_c2b)["rgb"]; rgbs.append(rgb)
            ucm = expert.decide(tcr, xcr)
            udl[:, 0] = udl[:, 1]; udl[:, 1] = ucm
            Tro[k] = tcr; Uro[:, k] = ucm
        uin = udl[:, 0] if (i % n_sim2ctl) < n_delay else udl[:, 1]
        xcr = sim.solver.simulate(x=xcr, u=uin) + np.random.normal(mu, sd)
        xcr[6:10] = oq(xcr[6:10], xpr[6:10]); xpr = xcr.copy()
        if i % n_sim2ctl == 0:
            Xro[:, i // n_sim2ctl + 1] = xcr
        if expert.landed:
            n_actual = i // n_sim2ctl + 1
            break
    return Tro, Xro, Uro, rgbs, n_actual


def save_episode(out_dir: Path, Tro, Xro, Uro, rgbs, T_c2b, cam_cfg, n_actual, video=False):
    from figs.utilities.trajectory_helper import xv_to_T
    out_dir.mkdir(parents=True, exist_ok=True)
    N = min(n_actual, Uro.shape[1])
    np.save(out_dir / "tXUd.npy", np.vstack([Tro[:N].reshape(1, -1), Xro[:, :N], Uro[:, :N]]))
    frames = [{"file_path": f"images/frame_{k:05d}.png", "transform_matrix": (xv_to_T(Xro[:, k]) @ T_c2b).tolist()} for k in range(N)]
    json.dump({"fl_x": cam_cfg["fx"], "fl_y": cam_cfg["fy"], "cx": cam_cfg["cx"], "cy": cam_cfg["cy"],
               "w": cam_cfg["width"], "h": cam_cfg["height"], "camera_model": "OPENCV", "frames": frames},
              open(out_dir / "transforms.json", "w"))
    if rgbs:
        import cv2
        (out_dir / "images").mkdir(exist_ok=True)
        for k, rgb in enumerate(rgbs[:N]):
            cv2.imwrite(str(out_dir / "images" / f"frame_{k:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if video:
            h, w = rgbs[0].shape[:2]
            vw = cv2.VideoWriter(str(out_dir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(HZ_CTL), (w, h))
            for rgb in rgbs[:N]:
                vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            vw.release()
    return N


def write_registry(yard: Yard, ckpt: str):
    REG_DIR.mkdir(parents=True, exist_ok=True)
    reg = {"yard": YARD, "ckpt": ckpt, "frame": "NED", "door_ned": yard.goal.tolist(), "hover_ned": yard.hover.tolist(),
           "standoff_ned": yard.standoff.tolist(), "fly_z": yard.fly_z, "land_z": yard.land_z, "cruise_z": yard.cruise_z,
           "door_face_ned": yard.face.tolist(), "doorstep_ned": yard.step.tolist(), "door_normal_ned": yard.n.tolist(),
           "door_facing": yard.survey.get("door_facing"), "r_det_m": yard.survey.get("r_det_m"),
           "starts": [{"id": s["id"], "pos_ned": s["pos_ned"], "yaw_ned_deg": s["yaw_ned_deg"], "valid": s["valid"],
                       "dist_to_door_m": s.get("dist_to_door_m")} for s in yard.survey["starts"]]}
    json.dump(reg, open(REG_DIR / f"{YARD}.json", "w"), indent=1)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--starts", nargs="+"); ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--include_invalid", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--rollouts", type=int, default=1, help="per start: 1 nominal + (K-1) perturbed siblings <id>_p<k>")
    ap.add_argument("--max_starts", type=int, default=0, help="use only the first N starts (by id)")
    ap.add_argument("--perturb_seed", type=int, default=0)
    args = ap.parse_args()

    yard = Yard(YARD_DIR)
    ckpt = find_checkpoint(YARD) if not args.plan_only else "(plan only)"
    reg = write_registry(yard, ckpt)
    print(f"[syn] {YARD} ckpt={ckpt} facing={yard.survey.get('door_facing')}")
    print(f"[syn] goal (CANON_DOOR_NED)={np.round(yard.goal, 3).tolist()} hover={np.round(yard.hover, 2).tolist()} "
          f"standoff={np.round(yard.standoff, 2).tolist()} fly_z={yard.fly_z:.2f} land_z={yard.land_z:.2f} cruise_z={yard.cruise_z:.2f}")

    ids = list(yard.starts)
    if args.starts:
        ids = args.starts
    elif args.preview:
        ids = [next(i for i in ids if yard.starts[i]["valid"])]
    if not args.include_invalid:
        ids = [i for i in ids if yard.starts[i]["valid"]]
    if args.max_starts > 0:
        ids = ids[:args.max_starts]
    print(f"[syn] starts: {len(ids)} rollouts per start: {args.rollouts}")

    # plans first (CPU), so a yard with no route fails before the GPU is touched
    plans = {}
    for sid in ids:
        st = yard.starts[sid]; p0 = np.asarray(st["pos_ned"], float)
        cz = yard.cruise_z; wps = None
        for cz_try in (cz, cz - 0.75, cz - 1.5):
            wps, _ = plan_route(yard, p0[:2], cz_try)
            if wps is not None:
                cz = cz_try; break
        if wps is None:
            print(f"  {sid}: NO ROUTE at any altitude, skipped"); continue
        L = sum(np.linalg.norm(b - a) for a, b in zip([p0[:2]] + wps[:-1], wps))
        plans[sid] = (wps, cz, L)
        if args.plan_only or args.preview:
            print(f"  {sid}: {len(wps)} waypoints, path {L:.1f} m, cruise_z {cz:.2f}: " + " > ".join(f"({w[0]:.1f},{w[1]:.1f})" for w in wps))
    if args.plan_only:
        return

    from figs.simulator import Simulator
    t0 = time.time(); sim = Simulator(ckpt, "baseline", "carl"); print(f"[syn] simulator loaded in {time.time()-t0:.1f}s")
    T_c2b = np.array(sim.conFiG["drone"]["T_c2b"]); cam_cfg = sim.conFiG["drone"]["camera"]
    summ_path = OUT_BASE / "fleet_summary.jsonl"; OUT_BASE.mkdir(parents=True, exist_ok=True)
    done = 0
    rng = np.random.default_rng(1000 + args.perturb_seed)
    for sid, (wps, cz, L) in plans.items():
      for r in range(max(1, args.rollouts)):
        name = sid if r == 0 else f"{sid}_p{r}"
        out_dir = OUT_BASE / name
        if (out_dir / "tXUd.npy").exists() and not args.overwrite:
            print(f"  {name}: exists, skip"); continue
        st = yard.starts[sid]; p0 = np.asarray(st["pos_ned"], float); yaw0 = math.radians(st["yaw_ned_deg"]); v0 = np.zeros(3)
        perturb = None; kick = 0.0
        if r > 0:
            p0 = p0 + np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)])
            yaw0 += math.radians(rng.uniform(-25, 25)); v0 = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0])
            kick = float(rng.choice([0.0, 1.5, 2.0]))
            perturb = {"period_s": 2.0, "pos_m": 0.4, "vel_ms": 0.4, "yaw_rad": 0.2, "kick_ms": kick, "band": (1.5, 3.0), "rng": rng}
        x0 = np.array([*p0, *v0, *quat_from_yaw(yaw0)])
        tf = float(min(MAX_EP, 1.3 * (L / 1.2) + 2.0 / V_APPROACH + 1.0 + LAND_UP / V_DESCENT + 8.0 + (6.0 if r else 0.0)))
        expert = GeometricExpert(yard, wps, cz, verbose=args.preview)
        te = time.time()
        Tro, Xro, Uro, rgbs, n_act = run_episode(sim, x0, tf, expert, T_c2b, want_images=not args.no_images, perturb=perturb)
        N = save_episode(out_dir, Tro, Xro, Uro, rgbs, T_c2b, cam_cfg, n_act, video=args.preview)
        P = Xro[0:3, :N].T; clr = yard.clearance(P)
        d_goal = float(np.linalg.norm(Xro[0:3, N] - yard.goal))
        s = {"yard": YARD, "start": name, "base_start": sid, "rollout": r, "perturbed": r > 0, "kick_ms": kick,
             "frames": int(N), "duration_s": float(Tro[N - 1]) if N else 0.0, "landed": bool(expert.landed),
             "final_to_goal_m": d_goal, "min_clearance_m": float(clr.min()), "collided": bool(clr.min() < BODY_R),
             "path_m": float(L), "cruise_z": cz, "n_waypoints": len(wps), "phases": expert.counts, "wall_s": time.time() - te}
        json.dump(s, open(out_dir / "summary.json", "w"), indent=1)
        with open(summ_path, "a") as fp:
            fp.write(json.dumps(s) + "\n")
        done += 1
        print(f"  {name}: {N} frames {s['duration_s']:.1f}s landed={s['landed']} final_to_goal={d_goal:.2f} m "
              f"min_clear={s['min_clearance_m']:.2f} m kick={kick} ({s['wall_s']:.0f}s)")
    print(f"[syn] done {done} starts -> {OUT_BASE}")


if __name__ == "__main__":
    main()
