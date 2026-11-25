"""Initialize the relabeling expert's flight phase from the visited state's geometry."""
import numpy as np
R_XY, DZ, R_SEG = 0.30, 0.05, 0.30


def _dseg(p, a, b):
    ab = b - a; t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)); return float(np.linalg.norm(a + t * ab - p))


def seed(ex, pos):
    if ex.pose is None: return "none"
    hv, ld = np.asarray(ex.pose["hover"], float), np.asarray(ex.pose["land"], float); sh = np.asarray(ex.route[-1], float)
    p = np.asarray(pos, float)
    if np.linalg.norm(p[:2] - ld[:2]) < R_XY and p[2] > hv[2] + DZ:
        ex.phase = "descent"; ex.z_set = float(p[2]); return "descent"
    if _dseg(p, sh, hv) < R_SEG:
        ex.phase = "approach"; ex.hold = 0; return "approach"
    return "cruise"


def install():
    import gate0b_expert as ge
    if getattr(ge, "_phase_seed", False): return
    orig = ge.fly_one
    def fly_one(m):
        r = orig(m)
        if m.get("_relabel") and m.get("_build_only") and not isinstance(r, dict):
            seed(r[0], m["pos"])
        return r
    ge.fly_one = fly_one; ge._phase_seed = True
