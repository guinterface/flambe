"""Build 3-D yard occupancy grids and plan collision-aware routes to goals."""
import json, math, os
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt, binary_closing, generate_binary_structure
from scipy.spatial import cKDTree

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))

R_FREE = 0.85          # metres to the nearest occupied cell centre (~0.75 m to the surface)
R_MIN = 0.65           # fallback for narrow passages (~0.55 m to the surface, 0.25 m over the body)
KEEP = 1.25            # string pulling keeps up to this much of the grid path's clearance
BACK_COST = 10.0       # edge cost multiplier in the back yard (behind the house mid-depth line, outside the house)
MESH = Path(os.path.join(FLAMBE_YARDS, "v4_mesh"))   # collider-surface samples (mesh_pts.py): exact geometry for scoring
CEIL_DEFAULT = 3.0     # m above ground for transit
FLOOR_H = 1.15         # transit floor; landings descend from the hover
HOVER_MIN, HOVER_MAX = 1.2, 3.0
SPLAT_MIN_PTS = 10**9  # Minimum sample density for splat-only occupancy.
VEH_D, VEH_H = 1.5, 1.2  # m: vehicle end pose, distance from the car's surface and height
OBJ_MIN_H = 0.9        # m, lowest final hover beside an object
DOOR_D = 1.6           # m door standoff (hover and land point)
ROOF_T = 0.8           # m, roof/eave thickness assumed under the top-down roof surface
CAM = {"w": 640, "h": 360, "fx": 462.956, "fy": 463.002}


class Grid:
    def __init__(self, ydir: Path, top_h=6.5):
        t = np.load(ydir / "recon" / "truth.npz")
        occ = np.asarray(t["occ3d"]).astype(bool); self.lo = np.asarray(t["lo"], float); self.c = c = float(t["cell"])
        nx, ny, nb = occ.shape
        h_top = np.asarray(t["height_top"], float); ex = np.asarray(t["extent_blender"], float)
        # extra layers above the band, centres at lo_z - (k+0.5)c
        n_up = int(math.ceil((top_h + self.lo[2]) / c))
        xs = self.lo[0] + (np.arange(nx) + .5) * c; ys = self.lo[1] + (np.arange(ny) + .5) * c
        hx = np.clip(((xs - ex[0]) / c).astype(int), 0, h_top.shape[0] - 1); hy = np.clip(((-ys - ex[2]) / c).astype(int), 0, h_top.shape[1] - 1)
        H = h_top[hx][:, hy]
        inside = ((xs - ex[0]) >= 0)[:, None] & ((-ys - ex[2]) >= 0)[None, :]
        H = np.where(inside, H, 0.0)
        up_z = self.lo[2] - (np.arange(n_up) + .5) * c                     # NED centres, going up
        up = H[:, :, None] > (-up_z[None, None, :] - 0.05)
        # eaves: a roof cell with no wall under it (band top layers empty) is an overhang, solid only in a ROOF_T slab
        # under the roof surface; otherwise the column is solid (walls, trunks). Vegetation stays solid to the band (unknown crown base).
        sem = np.asarray(t["sem_top"]); names = json.loads(str(t["sem_names"])); roof_id = int([k for k, v in names.items() if v == "roof"][0])
        S = np.where(inside, sem[hx][:, hy], 0)
        eave = (S == roof_id) & ~occ[:, :, :2].any(2)
        slab = (-up_z[None, None, :]) > (H[:, :, None] - ROOF_T)
        up = np.where(eave[:, :, None], up & slab, up)
        # full grid indexed k=0 top .. bottom, NED z increasing
        self.occ = np.concatenate([up[:, :, ::-1], occ], axis=2)
        self.z0 = self.lo[2] - n_up * c                                    # NED z of the grid top edge
        self.nz = self.occ.shape[2]
        self.zc = self.z0 + (np.arange(self.nz) + .5) * c
        self.mesh = None
        if (MESH / f"{ydir.name}.npz").exists():
            # exact geometry. Planning voxels = collider-surface samples U occ3d band U vegetation columns of the height
            # map (crowns down to the band, conservative); the eave heuristic above is replaced. Scoring = distance to the samples.
            M = np.load(MESH / f"{ydir.name}.npz")["pts"].astype(float); self.mesh = M
            I = np.floor((M - np.array([self.lo[0], self.lo[1], self.z0])) / c).astype(int)
            k = np.all((I >= 0) & (I < np.array(self.occ.shape)), 1); vox = np.zeros(self.occ.shape, bool); vox[tuple(I[k].T)] = True
            veg = (S == int([k2 for k2, v in names.items() if v == "vegetation"][0]))
            up_veg = (H[:, :, None] > (-up_z[None, None, :] - 0.05)) & veg[:, :, None]
            self.occ = vox | np.concatenate([up_veg[:, :, ::-1], occ], axis=2)
        splat = np.zeros(self.occ.shape, bool)
        if self.mesh is not None and (MESH / f"{ydir.name}_splat.npz").exists():
            # Add dense rendered surfaces after morphological closing.
            Sp = np.load(MESH / f"{ydir.name}_splat.npz")["pts"].astype(float)
            if len(Sp):
                I = np.floor((Sp - np.array([self.lo[0], self.lo[1], self.z0])) / c).astype(int)
                k = np.all((I >= 0) & (I < np.array(self.occ.shape)), 1); u, cnt = np.unique(I[k], axis=0, return_counts=True)
                u = u[cnt >= SPLAT_MIN_PTS]; splat[tuple(u.T)] = True
        # Add rendered samples near exact surfaces as planning obstacles after closing.
        for fz in sorted(MESH.glob(f"{ydir.name}_fuzz_*.npz")) if self.mesh is not None else []:
            Fp = np.load(fz)["pts"].astype(float)
            if not len(Fp): continue
            I = np.floor((Fp - np.array([self.lo[0], self.lo[1], self.z0])) / c).astype(int)
            k = np.all((I >= 0) & (I < np.array(self.occ.shape)), 1); u, cnt = np.unique(I[k], axis=0, return_counts=True)
            splat[tuple(u[cnt >= 3].T)] = True
        # planning occupancy: holes in sparse crowns/bushes closed (up to ~0.75 m), so routes go round trees, not through them
        self.occ_plan = self.occ | binary_closing(self.occ, generate_binary_structure(3, 1), iterations=3) | splat
        self.occ = self.occ | splat
        # free space not connected to the street side (house interiors, closed sheds) counts as solid
        from scipy.ndimage import label
        lab, _ = label(~self.occ_plan); st = json.load(open(ydir / "survey.json"))["starts"]
        keep = {lab[self.idx(np.asarray(s0["pos_ned"], float))] for s0 in st} - {0}
        self.occ_plan |= ~np.isin(lab, list(keep))
        self.clr = distance_transform_edt(~self.occ_plan, sampling=c)     # m to nearest occupied (planning) voxel centre
        mk = lambda I: cKDTree(np.c_[self.lo[0] + (I[:, 0] + .5) * c, self.lo[1] + (I[:, 1] + .5) * c, self.z0 + (I[:, 2] + .5) * c])
        self.tree = cKDTree(self.mesh) if self.mesh is not None else mk(np.argwhere(self.occ)); self.tree_plan = mk(np.argwhere(self.occ_plan))
        self._graphs = {}
        # back yard: behind the back facade line
        sv = json.load(open(ydir / "survey.json")); hx0, hx1, hy0, hy1 = sv["house_rect_ned"][:4]
        hcx = (hx0 + hx1) / 2; sx = float(np.mean(sv["street_x_ned"])); fs = np.sign(sx - hcx) or -1.0
        X = self.lo[0] + (np.arange(nx) + .5) * c; Y = self.lo[1] + (np.arange(ny) + .5) * c
        xb = hcx - fs * abs(hx1 - hx0) / 2                    # back facade line; side yards (beside the house) are not back yard
        behind = ((X - xb) * fs < 0)[:, None] & np.ones(ny, bool)[None, :]
        inside_h = ((X >= min(hx0, hx1)) & (X <= max(hx0, hx1)))[:, None] & ((Y >= min(hy0, hy1)) & (Y <= max(hy0, hy1)))[None, :]
        self.back = behind & ~inside_h

    def in_back(self, P):
        P = np.atleast_2d(P); i = np.clip(((P[:, 0] - self.lo[0]) / self.c).astype(int), 0, self.back.shape[0] - 1)
        j = np.clip(((P[:, 1] - self.lo[1]) / self.c).astype(int), 0, self.back.shape[1] - 1); return self.back[i, j]

    # ---- geometry ----
    def idx(self, p):
        p = np.asarray(p, float)
        return (int((p[0] - self.lo[0]) / self.c), int((p[1] - self.lo[1]) / self.c), int((p[2] - self.z0) / self.c))

    def pos(self, i):
        return np.array([self.lo[0] + (i[0] + .5) * self.c, self.lo[1] + (i[1] + .5) * self.c, self.z0 + (i[2] + .5) * self.c])

    def clearance(self, P):
        """scoring clearance: distance to the collider surface samples (exact geometry); without them, occupancy cell centres"""
        return self.tree.query(np.atleast_2d(P))[0]

    def clearance_plan(self, P):
        return self.tree_plan.query(np.atleast_2d(P))[0]

    def seg_clear(self, a, b, r, step=0.08):
        a = np.asarray(a, float); b = np.asarray(b, float); n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        return bool(self.clearance_plan(a + np.linspace(0, 1, n)[:, None] * (b - a)).min() >= r)

    # ---- graph ----
    def graph(self, ceil_h, r, soft=0.0, avoid_back=True):
        """soft > 0: edge cost x (1 + soft * max(0, 1 - clr) / (1 - r)), routes keep ~1 m clearance when it is cheap"""
        key = (round(ceil_h, 2), r, soft, avoid_back)
        if key in self._graphs: return self._graphs[key]
        kz = np.where((-self.zc <= ceil_h) & (-self.zc >= FLOOR_H))[0]; k0, k1 = kz.min(), kz.max() + 1
        Cf = self.clr[:, :, k0:k1]; free = Cf >= r
        nx, ny, nz = free.shape; ids = -np.ones(free.shape, int); ids[free] = np.arange(free.sum())
        rows, cols, w = [], [], []
        for di in (0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if (di, dj, dk) <= (0, 0, 0) and not (di == 0 and (dj, dk) > (0, 0)): continue
                    if di == 0 and dj == 0 and dk <= 0: continue
                    if di == 0 and dj == -1: continue
                    a = ids[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), max(0, -dk):nz - max(0, dk)]
                    b = ids[max(0, di):nx - max(0, -di) or None, max(0, dj):ny - max(0, -dj) or None, max(0, dk):nz - max(0, -dk) or None]
                    m = (a >= 0) & (b >= 0)
                    L = self.c * math.sqrt(di * di + dj * dj + 1.5 * dk * dk)    # vertical moves cost 1.22x (no needless climbs)
                    ww = np.full(m.sum(), L)
                    if soft:
                        ca = Cf[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), max(0, -dk):nz - max(0, dk)][m]
                        cb = Cf[max(0, di):nx - max(0, -di) or None, max(0, dj):ny - max(0, -dj) or None, max(0, dk):nz - max(0, -dk) or None][m]
                        ww = ww * (1 + soft * np.clip(1 - 0.5 * (ca + cb), 0, None) / (1 - r))
                    if avoid_back:
                        Bk = np.broadcast_to(self.back[:, :, None], free.shape)
                        ba = Bk[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), max(0, -dk):nz - max(0, dk)][m]
                        bb = Bk[max(0, di):nx - max(0, -di) or None, max(0, dj):ny - max(0, -dj) or None, max(0, dk):nz - max(0, -dk) or None][m]
                        ww = ww * np.where(ba | bb, BACK_COST, 1.0)
                    rows.append(a[m]); cols.append(b[m]); w.append(ww)
        rows = np.concatenate(rows); cols = np.concatenate(cols); w = np.concatenate(w)
        G = sparse.coo_matrix((w, (rows, cols)), shape=(free.sum(),) * 2).tocsr()
        G = G + G.T
        self._graphs[key] = (G, ids, k0, r)
        return self._graphs[key]

    def snap(self, ids, k0, p, max_r=8):
        i = np.array(self.idx(p)); i[2] -= k0
        best = None
        for d in range(max_r + 1):
            for di in range(-d, d + 1):
                for dj in range(-d, d + 1):
                    for dk in range(-d, d + 1):
                        if max(abs(di), abs(dj), abs(dk)) != d: continue
                        j = i + (di, dj, dk)
                        if np.all(j >= 0) and np.all(j < ids.shape) and ids[tuple(j)] >= 0:
                            e = di * di + dj * dj + dk * dk
                            if best is None or e < best[0]: best = (e, tuple(j))
            if best: return best[1]
        return None

    def field(self, goal_p, ceil_h, r=R_FREE, soft=0.0, avoid_back=True):
        """distance/predecessor field from the goal pose; returns a router(start)->polyline or None"""
        G, ids, k0, r = self.graph(ceil_h, r, soft, avoid_back)
        gi = self.snap(ids, k0, goal_p, max_r=3)    # <= 0.75 m, the last leg to the exact pose is a straight line
        if gi is None: return None
        D, Pr = dijkstra(G, indices=int(ids[gi]), return_predecessors=True)
        flat = np.argwhere(ids >= 0)          # id -> cell (ids are assigned in argwhere order)
        reach = np.isfinite(D)

        def route(p_start):
            si = self.snap(ids, k0, p_start)
            if si is None: return None
            n = int(ids[si])
            if not reach[n]:        # start in a pocket cut off by the margin -> nearest cell that reaches the goal, if the
                i0 = np.array(si); best = None   # straight hop to it keeps >= 0.35 m to the true surface
                for di in range(-8, 9):
                    for dj in range(-8, 9):
                        for dk in range(-3, 4):
                            j = i0 + (di, dj, dk)
                            if np.all(j >= 0) and np.all(j < ids.shape) and ids[tuple(j)] >= 0 and reach[ids[tuple(j)]]:
                                e = di * di + dj * dj + dk * dk
                                if best is None or e < best[0]: best = (e, tuple(j))
                if best is None: return None
                q = self.pos((best[1][0], best[1][1], best[1][2] + k0)); a = np.asarray(p_start, float)
                L = np.linspace(0, 1, 30)[:, None]
                if self.clearance(a + L * (q - a)).min() < 0.35: return None
                n = int(ids[best[1]])
            cells = [n]
            while cells[-1] != int(ids[gi]): cells.append(int(Pr[cells[-1]]))
            P = [self.pos((*flat[k][:2], flat[k][2] + k0)) for k in cells]
            P = [np.asarray(p_start, float)] + P[1:-1] + [np.asarray(goal_p, float)]
            return self.pull(P, r * 0.95, keep=KEEP if soft else 0.0)
        return route

    def pull(self, P, r, keep=0.9):
        """greedy string pulling; a shortcut i->j must keep the clearance the grid path had there (capped at keep),
        so routes planned with the soft cost do not get pulled back against walls"""
        cl = self.clearance_plan(np.asarray(P)); out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self.seg_clear(P[i], P[j], max(r, 0.95 * min(keep, float(cl[i:j + 1].min())))): j -= 1
            out.append(P[j]); i = j
        return out


def plen(P):
    P = np.asarray(P); return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def goal_poses(gr: Grid, g: dict):
    """per face: (hover, land, yaw, look_at). Windows: hover at the window centre height, far enough out that the whole
    pane fits in the 640x360 frame with ~15% margin (1.2..3.0 m), nudged out until free. Doors/objects: goals.json
    hover/land (land lowered only while clearance >= 0.5 m)."""
    out = []
    for f in g.get("faces", [g]):
        n = np.asarray(f["normal_ned"], float); n[2] = 0; n /= np.linalg.norm(n); face = np.asarray(f["face_ned"], float)
        yaw = math.atan2(-n[1], -n[0])
        if g["kind"] == "window":
            lo, hi = np.asarray(g["lo_ned"]), np.asarray(g["hi_ned"]); tang = np.array([-n[1], n[0], 0])
            w = abs((hi - lo) @ tang); h = abs(hi[2] - lo[2])
            d = max(w / 2 * CAM["fx"] / (CAM["w"] / 2), h / 2 * CAM["fy"] / (CAM["h"] / 2)) * 1.15
            d = float(np.clip(d, HOVER_MIN, HOVER_MAX)); c = np.array([face[0], face[1], g["center_ned"][2]])
            hv = None
            # candidates: distance near the framing distance, and a height offset of up to
            # 0.6 m as long as the window centre stays within 15 deg of the optical axis; preference: small offset, then framing
            cands = sorted(((dd, dz) for dd in np.arange(max(1.0, d - 0.5), d + 1.6, 0.1) for dz in np.arange(-0.6, 0.61, 0.2)
                            if math.degrees(math.atan2(abs(dz), dd)) <= 15.0), key=lambda v: (round(abs(v[1]), 1), abs(v[0] - d)))
            P = np.array([c + dd * n + np.array([0, 0, dz]) for dd, dz in cands]); cl = gr.clearance_plan(P)
            ok = np.where(cl >= 1.0)[0]
            if len(ok): hv = P[ok[0]]
            else:
                ok = np.where(cl >= R_MIN)[0]  # else the first (smallest offset) pose with >= R_MIN
                if len(ok): hv = P[ok[0]]
            if hv is None: out.append(None); continue
            out.append({"hover": hv, "land": hv.copy(), "yaw": yaw, "look": c, "dist": float(np.linalg.norm(hv - c))})
        else:
            hv = np.asarray(f["hover_ned"], float); ld = np.asarray(f["land_ned"], float)
            if g["kind"] == "door":     # stand off DOOR_D from the door; closer only if not clear
                H2 = [np.array([face[0] + dd * n[0], face[1] + dd * n[1], hv[2]]) for dd in np.arange(DOOR_D, 0.95, -0.1)]
                cl = gr.clearance_plan(np.array(H2)); okk = np.where(cl >= R_FREE)[0]
                h2 = H2[okk[0]] if len(okk) else H2[int(np.argmax(cl))]     # farthest clear standoff, else the roomiest
                ld = np.array([h2[0], h2[1], ld[2]]); hv = h2
            look = np.asarray(g.get("center_ned", face), float)
            if g["kind"] != "door":
                # props, mailboxes, vehicles and garage doors end like windows,
                # beside the object at about its own height with it framed, no descent. Height = object centre, but >= OBJ_MIN_H
                # (above the splat grass); far enough out that the centre is within 15 deg of the optical axis; stepped out if not clear.
                d0 = float(np.linalg.norm(hv[:2] - face[:2])); fc = float(np.linalg.norm(face[:2] - look[:2])); hv = None
                h0 = max(-look[2], OBJ_MIN_H)
                if g["kind"] == "vehicle":   # end ~1.5 m from the car's surface, above its centre, car framed
                    d0 = max(d0, VEH_D); h0 = max(h0, VEH_H)
                for h, ang, span in [(h0, 15.0, 1.0)] + [(hh, 18.0, 2.0) for hh in sorted({max(-look[2], OBJ_MIN_H), 1.1, 1.35})]:
                    d1 = max(d0, (h + look[2]) / math.tan(math.radians(ang)) - fc)
                    cands = [np.array([face[0] + dd * n[0], face[1] + dd * n[1], -h]) for dd in np.arange(d1, d1 + span + 0.01, 0.1)]
                    cl = gr.clearance_plan(np.array(cands)); okk = np.where(cl >= R_FREE)[0]
                    if not len(okk): okk = np.where(cl >= R_MIN)[0]
                    if len(okk): hv = cands[okk[0]]; break
                if hv is None: out.append(None); continue
                out.append({"hover": hv, "land": hv.copy(), "yaw": yaw, "look": look, "dist": float(np.linalg.norm(hv[:2] - face[:2]))})
                continue
            if gr.clearance(hv)[0] < R_MIN: out.append(None); continue
            zs = np.arange(hv[2], ld[2] + 1e-6, 0.05); P = np.c_[np.full(len(zs), hv[0]), np.full(len(zs), hv[1]), zs]
            ok = np.where(np.minimum.accumulate(gr.clearance(P)) >= 0.5)[0]
            land = P[ok[-1]] if len(ok) else hv.copy()
            out.append({"hover": hv, "land": land, "yaw": yaw, "look": look, "dist": float(np.linalg.norm(hv[:2] - face[:2]))})
    return out


def ceil_for(pose):
    return max(CEIL_DEFAULT, -pose["hover"][2] + 0.5)


def visible(gr: Grid, eye, q, pts, need=2, min_px=10.0, end_margin=0.4, step=0.1):
    """vectorised twin of goal_expert.visible on the full 3-D grid (true occupancy): >= need goal points inside the
    640x360 frame with a clear ray (0.3 m from the eye to end_margin before the point) and projected extent >= min_px."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(q).as_matrix(); f, r = R[:, 0], R[:, 1]; d = np.cross(f, r)
    V = np.asarray(pts, float) - eye; z = V @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        u = CAM["fx"] * (V @ r) / z + 323.076; w = CAM["fy"] * (V @ d) / z + 181.184
    inf = (z > 0.1) & (u >= 0) & (u < CAM["w"]) & (w >= 0) & (w < CAM["h"])
    if inf.sum() < need:
        return False, None
    Vi = V[inf]; L = np.linalg.norm(Vi, axis=1); S = np.arange(0.3, max(0.31, L.max() - end_margin), step)
    X = eye + Vi[:, None, :] * (S[None, :, None] / L[:, None, None]); valid = S[None, :] < (L[:, None] - end_margin)
    I = np.floor((X - np.array([gr.lo[0], gr.lo[1], gr.z0])) / gr.c).astype(int)
    inb = np.all((I >= 0) & (I < np.array(gr.occ.shape)), axis=2)
    Ic = np.clip(I, 0, np.array(gr.occ.shape) - 1); hit = gr.occ[Ic[..., 0], Ic[..., 1], Ic[..., 2]] & inb & valid
    clear = ~hit.any(1)
    if clear.sum() < need:
        return False, None
    ui, wi = u[inf], w[inf]; ext = float(max(np.ptp(ui), np.ptp(wi)))
    return ext >= min_px, [float(ui[clear].mean()), float(wi[clear].mean())]


def los(gr: Grid, eye, pts, need=2, end_margin=0.4, step=0.1):
    """line of sight (any heading): >= need goal points with a clear ray through the true occupancy"""
    V = np.asarray(pts, float) - eye; L = np.linalg.norm(V, axis=1); S = np.arange(0.3, max(0.31, L.max() - end_margin), step)
    X = eye + V[:, None, :] * (S[None, :, None] / L[:, None, None]); valid = S[None, :] < (L[:, None] - end_margin)
    I = np.floor((X - np.array([gr.lo[0], gr.lo[1], gr.z0])) / gr.c).astype(int)
    inb = np.all((I >= 0) & (I < np.array(gr.occ.shape)), axis=2); Ic = np.clip(I, 0, np.array(gr.occ.shape) - 1)
    hit = gr.occ[Ic[..., 0], Ic[..., 1], Ic[..., 2]] & inb & valid
    return int((~hit.any(1)).sum()) >= need
