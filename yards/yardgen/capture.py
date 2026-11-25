"""Cameras, trajectories (coverage pass, street passes, approaches, expert delivery paths) and the render loop with
RGB + depth + semantic-index passes. Writes nerfstudio transforms.json and poses.csv (NED) per set."""
import os, json, math, csv, time
import bpy
import numpy as np
from mathutils import Vector, Matrix
from .common import look_at_matrix, yaw_pitch_matrix, ned, SEM

CAPTURE_CAM = {"width": 1280, "height": 720, "fx": 830.0, "fy": 830.0, "cx": 640.0, "cy": 360.0}      # HFOV 75.4 deg
SIM_CAM = {"width": 640, "height": 360, "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184}      # FiGS deploy camera


def make_camera(cam):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("cam"); ob = bpy.data.objects.new("cam", cd); sc.collection.objects.link(ob); sc.camera = ob
    set_intrinsics(ob, cam); return ob


def set_intrinsics(ob, cam):
    sc = bpy.context.scene; cd = ob.data
    W, H = cam["width"], cam["height"]
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0; cd.lens = cam["fx"] * 36.0 / W
    cd.shift_x = (W / 2 - cam["cx"]) / W; cd.shift_y = (cam["cy"] - H / 2) / W
    sc.render.pixel_aspect_x = 1.0; sc.render.pixel_aspect_y = cam["fx"] / cam["fy"]
    cd.clip_start, cd.clip_end = 0.05, 500.0


def render_settings(samples=128, denoise=True, device_name_contains="RTX PRO 6000"):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; cy = sc.cycles
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
    for d in prefs.devices: d.use = (d.type == 'OPTIX')
    cy.device = 'GPU'; cy.samples = samples; cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.02
    cy.use_denoising = denoise; cy.denoiser = 'OPTIX'; cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    cy.max_bounces = 6; cy.diffuse_bounces = 3; cy.glossy_bounces = 3; cy.transmission_bounces = 2; cy.transparent_max_bounces = 8
    cy.caustics_reflective = False; cy.caustics_refractive = False; cy.sample_clamp_indirect = 10.0; cy.blur_glossy = 1.0
    sc.render.use_persistent_data = True
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'; sc.render.image_settings.color_depth = '8'
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'None'; sc.view_settings.exposure = 0.0
    sc.render.film_transparent = False
    sc.render.use_motion_blur = False
    vl = bpy.context.view_layer; vl.use_pass_z = True; vl.use_pass_object_index = True


def setup_pass_output(dirpath):
    """Compositor: depth + object index to EXR files named depth####.exr / index####.exr in dirpath."""
    sc = bpy.context.scene; sc.use_nodes = True; nt = sc.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    rl = nt.nodes.new('CompositorNodeRLayers'); comp = nt.nodes.new('CompositorNodeComposite')
    nt.links.new(rl.outputs['Image'], comp.inputs['Image'])
    fo = nt.nodes.new('CompositorNodeOutputFile'); fo.base_path = dirpath
    fo.format.file_format = 'OPEN_EXR'; fo.format.color_depth = '32'; fo.format.exr_codec = 'ZIP'; fo.format.color_mode = 'RGB'
    fo.file_slots.clear(); fo.file_slots.new('depth'); fo.file_slots.new('index')
    nt.links.new(rl.outputs['Depth'], fo.inputs[0]); nt.links.new(rl.outputs['IndexOB'], fo.inputs[1])
    return fo


# ---------------- trajectory primitives (each yields (pos, yaw, pitch) tuples) ----------------
def _resample(poly, ds):
    pts = [Vector(p) for p in poly]; out = [pts[0]]; acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = (b - a).length
        if L < 1e-6: continue
        n = int(L / ds)
        for k in range(1, n + 1): out.append(a + (b - a) * (k * ds / L))
    return out


def ring(cx, cy, hx, hy, z, pitch_deg, ds=0.5, look="in", target_z=None, yaw_jitter=0.0, rng=None):
    """Rectangle loop with rounded corners around centre (cx,cy) with half-extents hx,hy at height z."""
    r = min(hx, hy) * 0.5; poly = []
    corners = [(cx + hx - r, cy + hy - r, 0), (cx - hx + r, cy + hy - r, math.pi / 2), (cx - hx + r, cy - hy + r, math.pi), (cx + hx - r, cy - hy + r, 3 * math.pi / 2)]
    for (ox, oy, a0) in corners:
        for k in range(7):
            a = a0 + k * (math.pi / 2) / 6; poly.append((ox + r * math.cos(a), oy + r * math.sin(a), z))
    poly.append(poly[0])
    frames = []
    tz = target_z if target_z is not None else z - 1.0
    for p in _resample(poly, ds):
        if look == "in":
            yaw = math.atan2(cy - p.y, cx - p.x)
        else:
            yaw = math.atan2(p.y - cy, p.x - cx)
        if yaw_jitter and rng is not None: yaw += rng.u(-yaw_jitter, yaw_jitter)
        frames.append((p, yaw, math.radians(pitch_deg)))
    return frames


def line(a, b, z, yaw, pitch_deg, ds=0.5):
    return [(p, yaw, math.radians(pitch_deg)) for p in _resample([(a[0], a[1], z), (b[0], b[1], z)], ds)]


def line_lookat(a, b, z, target, pitch_bias_deg=0.0, ds=0.5):
    out = []
    for p in _resample([(a[0], a[1], z), (b[0], b[1], z)], ds):
        d = Vector(target) - p; yaw = math.atan2(d.y, d.x); pitch = math.atan2(d.z, math.hypot(d.x, d.y)) + math.radians(pitch_bias_deg)
        out.append((p, yaw, pitch))
    return out


def coverage_plan(lot, door, rng):
    """The splat substrate: everything an exploration policy can reach at 1.5-3.5 m, plus two higher rings for the roof."""
    from .lot import GROUND_Z, STREET_X0, STREET_X1
    gz = GROUND_Z
    hx0, hx1, hy0, hy1 = lot.house.main_rect
    if lot.house.wing_rect:
        wx0, wx1, wy0, wy1 = lot.house.wing_rect
        bx0, bx1, by0, by1 = min(hx0, wx0), max(hx1, wx1), min(hy0, wy0), max(hy1, wy1)
    else:
        bx0, bx1, by0, by1 = hx0, hx1, hy0, hy1
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2; hx, hy = (bx1 - bx0) / 2, (by1 - by0) / 2
    F = []
    # 1) inward loop at 2 m, 4 m clearance
    F += ring(cx, cy, hx + 4.0, hy + 4.0, gz + 2.0, -8, ds=0.55, look="in", yaw_jitter=0.15, rng=rng)
    # 2) inward loop at 2.8 m, 6.5 m clearance, slightly steeper
    F += ring(cx, cy, hx + 6.5, hy + 6.5, gz + 2.8, -14, ds=0.7, look="in", yaw_jitter=0.15, rng=rng)
    # 3) outward loop at 2 m (what the drone sees looking away from the house: fences, neighbours, street)
    F += ring(cx, cy, hx + 5.0, hy + 5.0, gz + 2.0, -5, ds=0.9, look="out", yaw_jitter=0.3, rng=rng)
    # 4) lawn lines across the front yard, looking along the line both ways
    for xf in (hx0 - 3.0, hx0 - 6.5):
        F += line((xf, lot.y0 + 1.0), (xf, lot.y1 - 1.0), gz + 1.8, math.pi / 2, -12, ds=0.6)
        F += line((xf, lot.y1 - 1.0), (xf, lot.y0 + 1.0), gz + 1.8, -math.pi / 2, -12, ds=0.6)
    # 5) street passes: sidewalk and near lane, looking along the street both ways, and toward the house, and across
    for xs in (STREET_X1 + 0.9, STREET_X1 - 2.5):
        F += line((xs, lot.y0 - 10), (xs, lot.y1 + 10), gz + 2.0, math.pi / 2, -8, ds=0.8)
        F += line((xs, lot.y1 + 10), (xs, lot.y0 - 10), gz + 2.0, -math.pi / 2, -8, ds=0.8)
    F += line((STREET_X1 - 2.5, lot.y0 - 8), (STREET_X1 - 2.5, lot.y1 + 8), gz + 2.0, 0.0, -6, ds=0.7)                # toward the house
    F += line((STREET_X1 + 0.9, lot.y0 - 8), (STREET_X1 + 0.9, lot.y1 + 8), gz + 2.0, math.pi, -6, ds=1.0)             # across the street
    # 6) approach to the door, twice, converging on the door face
    ap = lot.house.entry["approach_from"]; fpt = door["face"]
    tgt = (fpt[0], fpt[1], fpt[2] + 1.0)
    for lat in (-0.6, 0.6):
        a = (STREET_X1 + 1.2, lot.path_entry_y + lat)
        b = (ap[0] + (door["normal"][0]) * 0.0, ap[1] + lat * 0.3)
        F += line_lookat(a, b, gz + 1.8, tgt, ds=0.45)
        F += line_lookat(b, (fpt[0] + door["normal"][0] * 1.3, fpt[1] + door["normal"][1] * 1.3), gz + 1.4, tgt, ds=0.3)
    # 7) two higher rings for the roof (the drone will not fly here, but the splat must render the roof from the street)
    F += ring(cx, cy, hx + 7.0, hy + 7.0, gz + 5.5, -32, ds=1.2, look="in")
    F += ring(cx, cy, hx + 3.0, hy + 3.0, gz + 8.0, -55, ds=1.2, look="in")
    # Cover recessed porches from multiple heights and angles, including views from inside the alcove.
    if os.environ.get("YARDGEN_PORCH", "1") == "1":
        F += porch_views(door, gz)
    # 8) backyard lines (behind the house)
    for xb in (hx1 + 3.0, hx1 + 7.0):
        if xb < lot.lot_x1 - 1.5:
            F += line((xb, lot.y0 + 1.0), (xb, lot.y1 - 1.0), gz + 1.8, math.pi / 2, -10, ds=0.7)
            F += line((xb, lot.y1 - 1.0), (xb, lot.y0 + 1.0), gz + 1.8, -math.pi / 2, -10, ds=0.7)
    return F


def porch_views(door, gz):
    """camera frames (pos, yaw, pitch) covering the door, its alcove / porch walls and ceiling, from 4 m out to 0.8 m in front"""
    fx, fy, fz = door["face"]; nx, ny = door["normal"]; tx, ty = -ny, nx; w = door.get("width", 1.0); out = []
    def look(p, t):
        d = Vector(t) - Vector(p); return (Vector(p), math.atan2(d.y, d.x), math.atan2(d.z, math.hypot(d.x, d.y)))
    door_c = (fx, fy, fz + 1.0)
    for h in (1.2, 1.6, 2.0):                                        # dolly in, camera on the door
        for k in range(14):
            dd = 4.0 - k * (3.2 / 13)
            out.append(look((fx + nx * dd, fy + ny * dd, gz + h), door_c))
    for side in (-1, 1):                                             # obliques from outside any alcove (recess <= 2.6 m): at the door, at the alcove side wall
        for dd in (2.8, 3.6, 4.5):
            for h in (1.3, 1.9):
                p = (fx + nx * dd + tx * side * (w / 2 + 0.6), fy + ny * dd + ty * side * (w / 2 + 0.6), gz + h)
                out.append(look(p, door_c))
                out.append(look(p, (fx + nx * 0.6 - tx * side * (w / 2 + 0.5), fy + ny * 0.6 - ty * side * (w / 2 + 0.5), fz + 1.2)))
    for h in (1.2, 1.7):                                             # from inside the porch: looking out, up at the ceiling
        for dd in (0.8, 1.4):
            p = (fx + nx * dd, fy + ny * dd, gz + h)
            out.append(look(p, (fx + nx * (dd + 4), fy + ny * (dd + 4), gz + h - 0.3)))
            out.append(look(p, (fx + nx * (dd + 1.5), fy + ny * (dd + 1.5), gz + h + 1.6)))
            for side in (-1, 1):
                out.append(look(p, (fx + nx * dd + tx * side * 3, fy + ny * dd + ty * side * 3, gz + h)))
    return out


# ---------------- expert delivery paths ----------------
def _smooth_path(wps, v=1.0, hz=20.0, ease=1.5):
    """Catmull-Rom through waypoints, arc-length resampled at constant speed v with cosine ease-out at the end."""
    W = [Vector(w) for w in wps]
    P = [W[0]] + W + [W[-1]]
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(40):
            t = k / 40.0; t2, t3 = t * t, t * t * t
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    dense.append(W[-1])
    seg = [(b - a).length for a, b in zip(dense[:-1], dense[1:])]; S = np.concatenate([[0], np.cumsum(seg)]); L = S[-1]
    # speed profile: constant v, decelerating over the last 'ease' metres to 0.15 m/s
    out = []; s = 0.0; t = 0.0; dt = 1.0 / hz
    while s < L:
        rem = L - s; vs = v if rem > ease else max(0.15, v * rem / ease)
        pos = Vector(np.array([np.interp(s, S, [d[i] for d in dense]) for i in range(3)]))
        out.append((t, pos, vs)); s += vs * dt; t += dt
    out.append((t, W[-1], 0.0))
    return out


def expert_paths(lot, door, rng, starts):
    """Three delivery flights: street start -> above the path -> in front of the door -> descend; 20 Hz poses."""
    from .lot import GROUND_Z, STREET_X1
    gz = GROUND_Z
    ap = lot.house.entry["approach_from"]; land = lot.house.entry["landing"]
    nx, ny = door["normal"]; fx, fy, fz = door["face"]
    hover = (fx + nx * 1.0, fy + ny * 1.0, fz + 1.2)        # 1 m in front of the door, 1.2 m above the landing
    final = (fx + nx * 0.9, fy + ny * 0.9, fz + 0.5)
    valid = [s for s in starts if s["valid"]]
    picks = [valid[len(valid) // 6], valid[-len(valid) // 6], valid[len(valid) // 2]] if len(valid) >= 3 else valid
    paths = []
    for k, s in enumerate(picks[:3]):
        sx, sy, sz = s["pos_ned"][0], -s["pos_ned"][1], -s["pos_ned"][2]
        wps = [(sx, sy, sz), (STREET_X1 + 0.8, lot.path_entry_y + (0.0 if k != 2 else 2.5), gz + 2.0)]
        if k == 2:   # detour variant: swing wide around the front-yard clutter before lining up
            wps.append((lot.lot_x0 + 3.5, lot.path_entry_y + 3.0 * (1 if lot.path_entry_y < 0 else -1), gz + 2.2))
        wps += [((lot.lot_x0 + ap[0]) / 2, (lot.path_entry_y + ap[1]) / 2, gz + 1.9), (ap[0], ap[1], gz + 1.7), hover, final]
        samples = _smooth_path(wps, v=1.0, hz=20.0)
        # yaw: along velocity, blending to door-facing in the last 4 m; hover 3 s at the end facing the door
        door_yaw = math.atan2(-ny, -nx)
        rows = []
        for i, (t, pos, vs) in enumerate(samples):
            nxt = samples[min(i + 5, len(samples) - 1)][1]; d = nxt - pos
            yaw_v = math.atan2(d.y, d.x) if d.length > 0.05 else door_yaw
            rem = (Vector(final) - pos).length
            w = min(1.0, max(0.0, (4.0 - rem) / 3.0))
            dy = (door_yaw - yaw_v + math.pi) % (2 * math.pi) - math.pi
            yaw = yaw_v + w * dy
            rows.append((t, pos, yaw))
        tl = rows[-1][0]
        for j in range(1, 61): rows.append((tl + j / 20.0, Vector(final), door_yaw))
        paths.append({"name": f"path_{k + 1:02d}", "start": s["id"], "rows": rows, "waypoints": wps})
    return paths


# ---------------- rendering ----------------
def _quat_ned_from_yaw(yaw_blender):
    psi = -yaw_blender   # NED yaw (north toward east)
    return (0.0, 0.0, math.sin(psi / 2), math.cos(psi / 2))   # xyzw about z_down


def write_poses_csv(path, rows, cam_pitch_deg):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for (t, pos, yaw) in rows:
            n = ned(pos); q = _quat_ned_from_yaw(yaw)
            w.writerow([f"{t:.3f}", f"{n[0]:.4f}", f"{n[1]:.4f}", f"{n[2]:.4f}", f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}"])


def render_frames(frames, out_dir, cam, samples=128, passes=True, name_fmt="frame_{:05d}", log_every=10, tag=""):
    """frames: list of (pos, yaw, pitch) or 4x4 c2w matrices. Writes images/, passes/, transforms.json."""
    os.makedirs(out_dir + "/images", exist_ok=True)
    if passes: os.makedirs(out_dir + "/passes", exist_ok=True)
    sc = bpy.context.scene; cam_ob = sc.camera
    set_intrinsics(cam_ob, cam); sc.cycles.samples = samples
    fo = setup_pass_output(out_dir + "/passes") if passes else None
    if not passes: sc.use_nodes = False
    tf = {"w": cam["width"], "h": cam["height"], "fl_x": cam["fx"], "fl_y": cam["fy"], "cx": cam["cx"], "cy": cam["cy"], "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
          "camera_model": "OPENCV", "frames": [], "frame_convention": "camera_to_world OpenGL (-Z fwd, +Y up) in the GS frame = Blender world (X north, Y west, Z up)"}
    t0 = time.time()
    for i, fr in enumerate(frames):
        if isinstance(fr, Matrix): m = fr
        else:
            pos, yaw, pitch = fr; m = yaw_pitch_matrix(pos, yaw, pitch)
        cam_ob.matrix_world = m
        sc.frame_set(i)
        fn = name_fmt.format(i)
        sc.render.filepath = f"{out_dir}/images/{fn}.png"
        bpy.ops.render.render(write_still=True)
        tf["frames"].append({"file_path": f"images/{fn}.png", "transform_matrix": [list(map(float, r)) for r in m], "depth_file_path": f"passes/depth{i:04d}.exr" if passes else None, "index_file_path": f"passes/index{i:04d}.exr" if passes else None})
        if (i + 1) % log_every == 0 or i == len(frames) - 1:
            el = time.time() - t0; print(f"[render{tag}] {i + 1}/{len(frames)}  {el / (i + 1):.2f} s/frame  eta {el / (i + 1) * (len(frames) - i - 1) / 60:.1f} min", flush=True)
    json.dump(tf, open(out_dir + "/transforms.json", "w"), indent=1)
    return tf
