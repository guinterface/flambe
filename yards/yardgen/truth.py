"""Exact ground truth from the authored scene: surface samples -> occupancy grid (kt3 format, NED), top-down semantics,
door survey, street starts with door-visibility flags, collider GLB, init point cloud for splat training."""
import os, json, math
import bpy
import numpy as np
from mathutils import Vector
from .common import SEM, SEM_NAMES, SOLID, ned, collection

SIM_CAM = {"width": 640, "height": 360, "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184}


def scene_objects(include_hidden=False):
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.hide_render: continue
        if o.name.startswith("lib_"): continue
        out.append(o)
    return out


def sample_surfaces(density=250.0, max_per_obj=600000, classes=None, rng=None):
    """Return (points Nx3 float32 world, cls N uint8). density = samples per m^2."""
    rng = rng or np.random.RandomState(0)
    dg = bpy.context.evaluated_depsgraph_get()
    P, C = [], []
    for ob in scene_objects():
        if classes is not None and ob.pass_index not in classes: continue
        ev = ob.evaluated_get(dg)
        try:
            me = ev.to_mesh()
        except Exception:
            continue
        if me is None or len(me.polygons) == 0:
            ev.to_mesh_clear(); continue
        me.calc_loop_triangles()
        nv = len(me.vertices); V = np.empty(nv * 3, dtype=np.float64); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        nt = len(me.loop_triangles); T = np.empty(nt * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", T); T = T.reshape(-1, 3)
        M = np.array(ev.matrix_world)
        Vw = V @ M[:3, :3].T + M[:3, 3]
        a, b, c = Vw[T[:, 0]], Vw[T[:, 1]], Vw[T[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tot = area.sum()
        if tot <= 0: ev.to_mesh_clear(); continue
        n = int(min(max_per_obj, max(len(T), tot * density)))
        idx = rng.choice(len(T), size=n, p=area / tot)
        r1, r2 = rng.rand(n, 1), rng.rand(n, 1); s = np.sqrt(r1)
        pts = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
        P.append(pts.astype(np.float32)); C.append(np.full(n, ob.pass_index, dtype=np.uint8))
        ev.to_mesh_clear()
    return (np.concatenate(P), np.concatenate(C)) if P else (np.zeros((0, 3), np.float32), np.zeros(0, np.uint8))


def occupancy(P, C, ground_z, extent, cell=0.25, band=(0.25, 2.5)):
    """kt3-compatible truth grid in NED. extent = (x0,x1,y0,y1) in Blender coords (metres)."""
    x0, x1, y0, y1 = extent
    solid = np.isin(C, list(SOLID))
    Q = P[solid]
    zb = Q[:, 2] - ground_z
    m = (zb >= band[0]) & (zb <= band[1]) & (Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 1] >= y0) & (Q[:, 1] <= y1)
    Q = Q[m]
    N = np.stack([Q[:, 0], -Q[:, 1], -Q[:, 2]], 1)              # NED
    zg = -ground_z
    lo = np.array([x0, -y1, zg - band[1] - 0.01]); hi = np.array([x1, -y0, zg - band[0] + 0.01])
    shape = np.ceil((hi - lo) / cell).astype(int) + 1
    idx = np.floor((N - lo) / cell).astype(int)
    idx = np.clip(idx, 0, shape - 1)
    flat, cnt = np.unique(np.ravel_multi_index(idx.T, shape), return_counts=True)
    occ3 = np.zeros(shape, dtype=bool); occ3.flat[flat] = True
    occ2 = occ3.any(axis=2)
    return dict(occ=flat, occ_any=flat, count=cnt, lo=lo, shape=shape, cell=cell, z_ground=zg, n_gauss=len(N), mincount=1, occ3d=occ3, occ2d=occ2)


def topdown_semantics(extent, cell=0.25, z_from=60.0):
    """Ray-cast straight down per cell: first-hit class and height above the street surface."""
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    sem = np.zeros((nx, ny), np.uint8); hgt = np.zeros((nx, ny), np.float32)
    dg = bpy.context.evaluated_depsgraph_get(); sc = bpy.context.scene
    for i in range(nx):
        x = x0 + (i + 0.5) * cell
        for j in range(ny):
            y = y0 + (j + 0.5) * cell
            hit, loc, nrm, fi, ob, mw = sc.ray_cast(dg, Vector((x, y, z_from)), Vector((0, 0, -1)), distance=200)
            if hit:
                sem[i, j] = ob.pass_index; hgt[i, j] = loc.z
    return sem, hgt


def _cam_axes(yaw, pitch):
    f = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
    r = f.cross(Vector((0, 0, 1))).normalized(); u = r.cross(f).normalized()
    return f, r, u


def in_frustum(eye, yaw, pitch, pt, cam=SIM_CAM):
    f, r, u = _cam_axes(yaw, pitch); d = Vector(pt) - Vector(eye)
    z = d.dot(f)
    if z <= 0.05: return False
    px = cam["fx"] * d.dot(r) / z + cam["cx"]; py = cam["fy"] * (-d.dot(u)) / z + cam["cy"]
    return 0 <= px < cam["width"] and 0 <= py < cam["height"]


def door_points(door):
    fx, fy, fz = door["face"]; nx, ny = door["normal"]; w, h = door["width"], door["height"]
    tx, ty = -ny, nx
    pts = []
    for a in (-0.4, 0.0, 0.4):
        for b in (0.1, 0.5, 0.9):
            pts.append((fx + nx * 0.05 + tx * a * w, fy + ny * 0.05 + ty * a * w, fz + b * h))
    return pts


def door_visible(eye, yaw, pitch, door, cam=SIM_CAM):
    """(in_frustum_any, line_of_sight_any, both_any) over 9 door points."""
    dg = bpy.context.evaluated_depsgraph_get(); sc = bpy.context.scene
    fr = los = both = False
    for pt in door_points(door):
        f_ = in_frustum(eye, yaw, pitch, pt, cam)
        d = Vector(pt) - Vector(eye); L = d.length
        hit, loc, nrm, fi, ob, mw = sc.ray_cast(dg, Vector(eye), d.normalized(), distance=L + 0.5)
        l_ = bool(hit) and ob is not None and ob.pass_index == SEM["door"] and (loc - Vector(eye)).length > L - 0.35
        fr |= f_; los |= l_; both |= (f_ and l_)
    return fr, los, both


def make_starts(lot, door, cam_pitch_deg, rng, r_det=11.1, n_pos=24):
    """24 street-side start positions x 3 yaw offsets; door must be > r_det away; log door visibility at t0."""
    from .lot import STREET_X0, STREET_X1, GROUND_Z
    gz = GROUND_Z; z = gz + 2.0
    xs = [STREET_X1 + 0.6, STREET_X1 - 2.0, STREET_X1 - 5.0]          # verge, near lane, far lane
    ys = np.linspace(lot.y0 - 7.0, lot.y1 + 7.0, n_pos // len(xs))
    pitch = math.radians(-cam_pitch_deg)
    starts = []; k = 0
    dx_, dy_, _ = door["doorstep"]
    for x in xs:
        for y in ys:
            # base heading: along the street toward the door's side (the drone would fly along the street first)
            base = math.pi / 2 if dy_ > y else -math.pi / 2
            for off in (0.0, math.radians(60), math.radians(-60)):
                yaw = base + off
                eye = (float(x), float(y), z)
                dist = math.hypot(dx_ - x, dy_ - y)
                fr, los, both = door_visible(eye, yaw, pitch, door)
                starts.append({"id": f"s{k:03d}", "pos_ned": ned(eye), "yaw_ned_deg": math.degrees(-yaw) % 360, "yaw_offset_deg": round(math.degrees(off)),
                               "dist_to_door_m": round(dist, 2), "beyond_r_det": dist > r_det, "door_in_frustum": fr, "door_line_of_sight": los,
                               "door_visible_t0": both, "valid": (dist > r_det) and not both})
                k += 1
    return starts


def export_collider(path, exclude_classes=(SEM["vegetation"],)):
    """GLB of all solid geometry (vegetation replaced by trunk+crown proxies), no materials."""
    objs = [o for o in scene_objects() if o.pass_index in SOLID or o.pass_index in (SEM["steps"], SEM["porch"])]
    sel = [o for o in objs if o.pass_index not in exclude_classes]
    proxies = []
    pc = collection("collider_proxies")
    from .common import cylinder
    for o in objs:
        if o.pass_index in exclude_classes:
            bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
            xs = [v.x for v in bb]; ys = [v.y for v in bb]; zs = [v.z for v in bb]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2; h = max(zs) - min(zs); r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
            if h < 1.5:
                proxies.append(cylinder("vprox", (cx, cy, min(zs) + h / 2), r, h, pc, None, SEM["vegetation"], n=10))
            else:
                proxies.append(cylinder("vprox_trunk", (cx, cy, min(zs) + h * 0.2), max(0.12, r * 0.08), h * 0.4, pc, None, SEM["vegetation"], n=8))
                proxies.append(cylinder("vprox_crown", (cx, cy, min(zs) + h * 0.65), r, h * 0.7, pc, None, SEM["vegetation"], n=12))
    for o in bpy.data.objects: o.select_set(False)
    for o in sel + proxies: o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format='GLB', export_materials='NONE', export_apply=True, export_yup=True)
    for o in proxies: bpy.data.objects.remove(o, do_unlink=True)
    return len(sel), len(proxies)


def write_ply(path, P, C, n_max=400000, rng=None):
    rng = rng or np.random.RandomState(1)
    if len(P) > n_max:
        sel = rng.choice(len(P), n_max, replace=False); P, C = P[sel], C[sel]
    pal = {SEM["lawn"]: (60, 110, 40), SEM["street"]: (50, 50, 50), SEM["sidewalk"]: (150, 150, 145), SEM["driveway"]: (140, 140, 135), SEM["path"]: (150, 130, 110),
           SEM["wall"]: (180, 160, 140), SEM["roof"]: (90, 70, 60), SEM["door"]: (120, 30, 30), SEM["window"]: (80, 100, 120), SEM["vegetation"]: (40, 90, 30),
           SEM["fence"]: (200, 200, 190), SEM["prop"]: (120, 120, 120), SEM["neighbor"]: (170, 160, 150), SEM["furniture"]: (90, 90, 90), SEM["porch"]: (140, 120, 100),
           SEM["steps"]: (150, 150, 145), SEM["trim"]: (220, 220, 215), SEM["vehicle"]: (100, 100, 110), SEM["soil"]: (90, 70, 50)}
    cols = np.array([pal.get(int(c), (128, 128, 128)) for c in C], dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]; rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        f.write(rec.tobytes())
    return len(P)


def export_truth(out_dir, lot, door, extent, cam_pitch_deg, rng, cell=0.25):
    os.makedirs(out_dir + "/recon", exist_ok=True)
    P, C = sample_surfaces(density=250.0, rng=rng.np)
    from .lot import GROUND_Z
    occ = occupancy(P, C, GROUND_Z, extent, cell=cell)
    sem, hgt = topdown_semantics(extent, cell=cell)
    np.savez_compressed(out_dir + "/recon/truth.npz", occ=occ["occ"], occ_any=occ["occ_any"], count=occ["count"], lo=occ["lo"], shape=occ["shape"],
                        cell=occ["cell"], z_ground=occ["z_ground"], n_gauss=occ["n_gauss"], mincount=occ["mincount"], occ3d=occ["occ3d"], occ2d=occ["occ2d"],
                        sem_top=sem, height_top=hgt, extent_blender=np.array(extent), sem_names=json.dumps(SEM_NAMES), source="synthetic_exact")
    n_ply = write_ply(out_dir + "/recon/surface_points.ply", P, C, rng=rng.np)
    ncol, nprox = export_collider(out_dir + "/recon/collider.glb")
    return {"n_surface_samples": int(len(P)), "occupied_cells": int(len(occ["occ"])), "grid_shape": occ["shape"].tolist(), "ply_points": n_ply, "collider_objects": ncol, "veg_proxies": nprox}
