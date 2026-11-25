"""Generate synthetic yards, render camera views, and export ground truth.
Supports rotating the lot about a configurable pivot."""
import sys, os, json, time, math, argparse, random, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bpy
import numpy as np
from mathutils import Vector, Matrix
from yardgen.common import RNG, reset_scene, collection, ned, SEM
from yardgen import materials as M, assets as A
from yardgen.house import house_params
from yardgen.lot import Lot, lot_params, GROUND_Z, STREET_X0, STREET_X1
from yardgen import truth as T
from yardgen import capture as C

FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))

ROOT = FLAMBE_YARDS


class Layout:   # the turn, in the Blender frame (X north, Y west, Z up); NED yaw R = Blender angle -R about +Z
    def __init__(self, R_deg, pivot_b, source):
        self.R = float(R_deg) % 360.0; self.source = source; q = self.R / 90.0
        if abs(q - round(q)) < 1e-12: cR, sR = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(round(q)) % 4]   # exact for multiples of 90 deg
        else: cR, sR = math.cos(math.radians(self.R)), math.sin(math.radians(self.R))
        self.c, self.s = float(cR), float(-sR); self.phi = -math.radians(self.R)   # Blender CCW angle phi = -R
        self.cx, self.cy = float(pivot_b[0]), float(pivot_b[1])
        m = Matrix.Identity(4); m[0][0], m[0][1], m[1][0], m[1][1] = self.c, -self.s, self.s, self.c
        m[0][3] = self.cx - (self.c * self.cx - self.s * self.cy); m[1][3] = self.cy - (self.s * self.cx + self.c * self.cy); self.M = m

    def p(self, v):   # Blender point (2 or 3 components) -> same type
        x, y = float(v[0]) - self.cx, float(v[1]) - self.cy
        o = (self.c * x - self.s * y + self.cx, self.s * x + self.c * y + self.cy) + tuple(float(t) for t in v[2:])
        return Vector(o) if isinstance(v, Vector) else o

    def d(self, v):   # Blender direction
        x, y = float(v[0]), float(v[1]); return (self.c * x - self.s * y, self.s * x + self.c * y) + tuple(float(t) for t in v[2:])

    def frame(self, fr):   # (pos, yaw, pitch) or 4x4 c2w Matrix
        if isinstance(fr, Matrix): return self.M @ fr
        pos, yaw, pitch = fr; return (Vector(self.p(pos)), yaw + self.phi, pitch)

    def frames(self, F): return [self.frame(fr) for fr in F]

    def ned_p(self, n): b = self.p((n[0], -n[1]) + tuple(n[2:])); return [float(b[0]), float(-b[1])] + [float(t) for t in n[2:]]

    def ned_d(self, n): b = self.d((n[0], -n[1]) + tuple(n[2:])); return [float(b[0]), float(-b[1])] + [float(t) for t in n[2:]]

    def ned_deg(self, a): return float((float(a) + self.R) % 360.0)

    def ned_box(self, x0, x1, y0, y1):   # NED axis-aligned rect -> (corners turned, their axis-aligned box [x0, x1, y0, y1])
        Q = [self.ned_p([x, y]) for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        return Q, [min(q[0] for q in Q), max(q[0] for q in Q), min(q[1] for q in Q), max(q[1] for q in Q)]

    def turn_scene(self):
        n = 0
        for ob in bpy.data.objects:
            if ob.type == 'CAMERA' or ob.name.startswith("lib_") or ob.parent is not None: continue   # lib_ = hidden instancing sources (stay at identity, as built)
            ob.matrix_world = self.M @ ob.matrix_world; n += 1
        bpy.context.view_layer.update(); return n


def turned_survey(LY, sv, lot):   # survey of the turned yard from the survey of the lot's own frame
    o = dict(sv)
    o["doorstep_ned"] = LY.ned_p(sv["doorstep_ned"]); o["door_face_ned"] = LY.ned_p(sv["door_face_ned"]); o["door_normal_ned"] = LY.ned_d(sv["door_normal_ned"])
    o["street_heading_deg"] = LY.ned_deg(sv["street_heading_deg"])
    lr = sv["lot_rect_ned"]; scorn, sbox = LY.ned_box(STREET_X0, STREET_X1, lr[2], lr[3])   # the street strip beside the lot
    o["street_x_ned"] = [sbox[0], sbox[1]]
    o["start_ned"] = LY.ned_p(sv["start_ned"]); o["start_yaw_deg"] = LY.ned_deg(sv["start_yaw_deg"])
    o["starts"] = [dict(s, pos_ned=LY.ned_p(s["pos_ned"]), yaw_ned_deg=LY.ned_deg(s["yaw_ned_deg"])) for s in sv["starts"]]
    hr = sv["house_rect_ned"]; hcorn, hbox = LY.ned_box(*hr[:4]); lcorn, lbox = LY.ned_box(*lr[:4])
    o["house_rect_ned"] = hbox; o["lot_rect_ned"] = lbox
    o["street_y_ned"] = [sbox[2], sbox[3]]
    o["layout"] = {"yaw_deg": LY.R, "yaw_source": LY.source, "pivot_ned": [LY.cx, -LY.cy], "convention": "p_ned = Rz(yaw) (p_local - pivot) + pivot, z unchanged, yaw_ned = yaw_local + yaw",
                   "street_heading_local_deg": sv["street_heading_deg"], "street_x_local_ned": sv["street_x_ned"], "house_rect_local_ned": hr, "lot_rect_local_ned": lr,
                   "house_corners_ned": hcorn, "lot_corners_ned": lcorn, "street_strip_corners_ned": scorn,
                   "front_ned": LY.ned_d([-1.0, 0.0]), "street_dir_ned": LY.ned_d([0.0, 1.0]),
                   "note": "house_rect_ned, lot_rect_ned, street_x_ned, street_y_ned are axis-aligned boxes of the turned shapes (rot_survey convention). "
                           "terra/goalspec goals.py house_frame, goal_expert.py, v4/grid3d.py back-yard rule and v4/tag_goals.py read them as if the street were at constant x; "
                           "exact for yaw 0 / 180, approximate otherwise (front side sign right for |yaw| < 45 deg)."}
    return o


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--yard", default="syn_001"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stratum", default="recessed", choices=["street", "recessed", "side", "corner"])
    ap.add_argument("--render", default="preview", help="comma list of: preview,capture,paths,starts,none")
    ap.add_argument("--samples", type=int, default=128); ap.add_argument("--path_samples", type=int, default=64)
    ap.add_argument("--cam_pitch", type=float, default=10.0, help="deploy camera mount pitch, degrees down")
    ap.add_argument("--path_img_hz", default="10,2,2", help="image rate per path (poses are always 20 Hz)")
    ap.add_argument("--out", default=None); ap.add_argument("--no_truth", action="store_true"); ap.add_argument("--save_blend", action="store_true", default=True)
    ap.add_argument("--max_frames", type=int, default=0, help="debug: cap frames per render set")
    ap.add_argument("--layout_yaw", default="0", help="NED yaw (deg) of the whole built lot about the pivot, or 'random' (seeded); 0 = yardgen.py")
    ap.add_argument("--layout_pivot", default="house", choices=["house", "origin"], help="pivot of the turn")
    ap.add_argument("--cycles_device", default="gpu", choices=["gpu", "cpu"], help="cpu = Cycles on CPU with OpenImageDenoise")
    ap.add_argument("--no_save_blend", action="store_true", help="do not write scene.blend")
    ap.add_argument("--capture_passes", type=int, default=1, choices=[0, 1], help="0 = capture RGB only (no depth/index EXR, 2.2 GB)")
    return ap.parse_args(argv)


def main():
    a = parse(); t0 = time.time()
    out = a.out or f"{ROOT}/yards/{a.yard}"; os.makedirs(out, exist_ok=True)
    if a.layout_yaw == "random": yaw_R, yaw_src = round(random.Random(zlib.crc32(f"f140_layout_yaw:{a.seed}".encode())).uniform(0.0, 360.0), 3), "random"   # own stream
    else: yaw_R, yaw_src = float(a.layout_yaw) % 360.0, "given"
    if a.cycles_device == "cpu":   # Use CPU Cycles with OpenImageDenoise.
        _rs = C.render_settings
        def _rs_cpu(samples=128, denoise=True, device_name_contains=None):
            try: _rs(samples=samples, denoise=denoise)
            except TypeError: pass   # stops at the OptiX denoiser line; the rest is set here as there
            sc = bpy.context.scene; cy = sc.cycles; cy.device = 'CPU'; cy.samples = samples; cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.02
            cy.use_denoising = denoise; cy.denoiser = 'OPENIMAGEDENOISE'; cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
            cy.max_bounces = 6; cy.diffuse_bounces = 3; cy.glossy_bounces = 3; cy.transmission_bounces = 2; cy.transparent_max_bounces = 8
            cy.caustics_reflective = False; cy.caustics_refractive = False; cy.sample_clamp_indirect = 10.0; cy.blur_glossy = 1.0
            sc.render.use_persistent_data = True
            sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'; sc.render.image_settings.color_depth = '8'
            sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'None'; sc.view_settings.exposure = 0.0
            sc.render.film_transparent = False; sc.render.use_motion_blur = False
            vl = bpy.context.view_layer; vl.use_pass_z = True; vl.use_pass_object_index = True
            print(f"[yardgen_v2] Cycles device {cy.device}, denoiser {cy.denoiser}, samples {cy.samples}, threads {sc.render.threads} ({sc.render.threads_mode})", flush=True)
        C.render_settings = _rs_cpu
    rng = RNG(a.seed)
    reset_scene()
    hp = house_params(rng.sub("house"), a.stratum); lp = lot_params(rng.sub("lot"), a.stratum, hp)
    print(f"[yardgen] {a.yard} seed={a.seed} stratum={a.stratum} hdri={lp['hdri']} wall={hp['wall_tex']} roof={hp['roof_tex']}", flush=True)
    lot = Lot(rng.sub("build"), lp, hp, collection("lot"))
    door = lot.house.door
    print(f"[yardgen] built in {time.time() - t0:.1f}s: {len(bpy.data.objects)} objects; door face {tuple(round(v, 2) for v in door['face'])} normal {door['normal']}", flush=True)
    cam = C.make_camera(C.CAPTURE_CAM); C.render_settings(samples=a.samples)

    # ---- survey + starts ----
    pitch = a.cam_pitch
    starts = T.make_starts(lot, door, pitch, rng.sub("starts"))
    nvalid = sum(s["valid"] for s in starts)
    street_heading_ned = 90.0   # street runs east-west; heading along +east
    survey = {"doorstep_ned": ned(door["doorstep"]), "door_face_ned": ned(door["face"]), "door_normal_ned": [door["normal"][0], -door["normal"][1], 0.0],
              "door_width_m": door["width"], "door_height_m": door["height"], "door_facing": a.stratum, "method": "synthetic_exact",
              "ground_z_ned": -GROUND_Z, "street_heading_deg": street_heading_ned, "street_x_ned": [STREET_X0, STREET_X1],
              "start_ned": starts[0]["pos_ned"], "start_yaw_deg": starts[0]["yaw_ned_deg"], "starts": starts,
              "n_starts": len(starts), "n_valid_starts": nvalid, "r_det_m": 11.1,
              "house_rect_ned": [lot.house.main_rect[0], lot.house.main_rect[1], -lot.house.main_rect[3], -lot.house.main_rect[2]],
              "lot_rect_ned": [lot.lot_x0, lot.lot_x1, -lot.y1, -lot.y0]}
    LY = None; starts_w = starts   # world-frame starts (= starts when not turned)
    if yaw_R != 0.0:   # turn the built scene; everything below is written in the turned (world) frame
        hx0, hx1, hy0, hy1 = lot.house.main_rect
        LY = Layout(yaw_R, ((hx0 + hx1) / 2, (hy0 + hy1) / 2) if a.layout_pivot == "house" else (0.0, 0.0), yaw_src)
        n_turned = LY.turn_scene()
        survey = turned_survey(LY, survey, lot); starts_w = survey["starts"]
        print(f"[yardgen_v2] layout yaw {LY.R} deg ({yaw_src}) about pivot NED ({LY.cx:.3f}, {-LY.cy:.3f}): {n_turned} objects turned; street heading {survey['street_heading_deg']:.3f} deg", flush=True)
    json.dump(survey, open(out + "/survey.json", "w"), indent=1)
    meta = {"yard_id": a.yard, "date": time.strftime("%Y-%m-%d"), "vehicle": "synthetic (Blender 4.5 Cycles/OptiX)", "seed": a.seed,
            "camera": {**C.SIM_CAM, "pitch_deg": pitch, "hfov_deg": 2 * math.degrees(math.atan(C.SIM_CAM["width"] / 2 / C.SIM_CAM["fx"]))},
            "capture_camera": {**C.CAPTURE_CAM, "hfov_deg": 2 * math.degrees(math.atan(C.CAPTURE_CAM["width"] / 2 / C.CAPTURE_CAM["fx"]))},
            "door_facing": a.stratum, "frame": "NED = (X, -Y, -Z) of the Blender/GS frame; GS frame == Blender world; ground at z_ned = -0.15 (street surface z_ned = 0)",
            "lot_params": {k: v for k, v in lp.items()}, "house_params": {k: v for k, v in hp.items()}, "notes": "procedurally generated; all labels exact"}
    if LY is not None:
        meta["layout"] = {"yaw_deg": LY.R, "yaw_source": yaw_src, "pivot": a.layout_pivot, "pivot_ned": [LY.cx, -LY.cy], "objects_turned": n_turned, "generator": "synth_yards/tools/yardgen_v2.py"}
    json.dump(meta, open(out + "/meta.json", "w"), indent=1, default=str)
    print(f"[yardgen] starts: {nvalid}/{len(starts)} valid (beyond r_det and door not visible at t0)", flush=True)

    # ---- truth ----
    ext = (STREET_X0 - 4.0, lot.lot_x1 + 2.0, lot.y0 - 10.0, lot.y1 + 10.0)
    if LY is not None:   # axis-aligned world box covering the turned extent
        Q = [LY.p((x, y)) for x in ext[:2] for y in ext[2:]]; ext = (min(q[0] for q in Q), max(q[0] for q in Q), min(q[1] for q in Q), max(q[1] for q in Q))
    if not a.no_truth:
        tt = time.time(); info = T.export_truth(out, lot, door, ext, pitch, rng.sub("truth"))
        print(f"[yardgen] truth exported in {time.time() - tt:.1f}s: {info}", flush=True)
        meta["truth"] = info; json.dump(meta, open(out + "/meta.json", "w"), indent=1, default=str)

    if a.save_blend and not a.no_save_blend:   # --no_save_blend
        bpy.ops.wm.save_as_mainfile(filepath=out + "/scene.blend", compress=True)

    sets = [s for s in a.render.split(",") if s and s != "none"]
    cap = lambda F: F[:a.max_frames] if a.max_frames else F
    turn = (lambda F: F) if LY is None else LY.frames

    # ---- preview: a dozen diagnostic views ----
    if "preview" in sets:
        F = []
        hx0, hx1, hy0, hy1 = lot.house.main_rect; cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
        tgt = (door["face"][0], door["face"][1], door["face"][2] + 1.0)
        from mathutils import Vector
        views = [((STREET_X1 - 2.0, cy - 6, GROUND_Z + 2.0), (cx, cy, 2.0)), ((STREET_X1 + 0.8, lot.path_entry_y, GROUND_Z + 1.8), tgt),
                 ((lot.lot_x0 + 4.0, lot.path_entry_y + 1.0, GROUND_Z + 1.7), tgt), ((door["face"][0] + door["normal"][0] * 3.0, door["face"][1] + door["normal"][1] * 3.0, GROUND_Z + 1.5), tgt),
                 ((hx0 - 9, hy1 + 6, GROUND_Z + 2.5), (cx, cy, 2.5)), ((hx1 + 8, hy0 - 6, GROUND_Z + 2.5), (cx, cy, 2.5)),
                 ((STREET_X1 - 6, cy + 14, GROUND_Z + 2.0), (STREET_X1 - 6, cy - 10, 1.5)), ((STREET_X1 + 0.8, cy, GROUND_Z + 2.0), (STREET_X0 - 12, cy, 1.5)),
                 ((cx - 2, cy, GROUND_Z + 28.0), (cx, cy + 0.01, 0.0)), ((hx0 - 20, cy - 16, GROUND_Z + 9.0), (cx, cy, 2.0)),
                 ((hx1 + 4, cy, GROUND_Z + 1.8), (hx1, cy, 1.5)), ((hx0 - 4.0, hy0 - 2.0, GROUND_Z + 1.6), (hx0 + 2, hy0 - 2.0, 1.2))]
        if LY is not None: views = [(LY.p(e), LY.p(t)) for (e, t) in views]
        from yardgen.common import look_at_matrix
        F = [look_at_matrix(e, t) for (e, t) in views]
        C.render_frames(cap(F), out + "/preview", C.CAPTURE_CAM, samples=a.samples, passes=False, name_fmt="view_{:02d}", log_every=4, tag=":preview")

    # ---- capture (splat substrate) ----
    if "capture" in sets:
        F = turn(C.coverage_plan(lot, door, rng.sub("cov")))   # plan in the lot frame, then turned
        print(f"[yardgen] capture plan: {len(F)} frames", flush=True)
        tf = C.render_frames(cap(F), out + "/capture", C.CAPTURE_CAM, samples=a.samples, passes=bool(a.capture_passes), tag=":capture")   # --capture_passes
        # nerfstudio needs the point cloud next to transforms.json
        import shutil
        if os.path.exists(out + "/recon/surface_points.ply"):
            shutil.copy(out + "/recon/surface_points.ply", out + "/capture/sparse_pc.ply"); tf["ply_file_path"] = "sparse_pc.ply"
            json.dump(tf, open(out + "/capture/transforms.json", "w"), indent=1)
        rows = []
        for i, fr in enumerate(cap(F)):
            rows.append((i * 0.5, fr[0], fr[1]))
        C.write_poses_csv(out + "/capture/poses.csv", rows, pitch)

    # ---- expert delivery paths ----
    if "paths" in sets:
        paths = C.expert_paths(lot, door, rng.sub("paths"), starts)   # lot frame (local starts)
        if LY is not None:   # turn poses and waypoints
            paths = [dict(P, rows=[(t, LY.p(pos), yaw + LY.phi) for (t, pos, yaw) in P["rows"]], waypoints=[LY.p(w) for w in P["waypoints"]]) for P in paths]
        hz = [float(h) for h in a.path_img_hz.split(",")]
        for k, P in enumerate(paths):
            d = f"{out}/{P['name']}"; os.makedirs(d, exist_ok=True)
            C.write_poses_csv(d + "/poses.csv", P["rows"], pitch)
            json.dump({"start": P["start"], "waypoints_blender": P["waypoints"], "n_poses": len(P["rows"]), "pose_hz": 20, "image_hz": hz[min(k, len(hz) - 1)]}, open(d + "/path.json", "w"), indent=1)
            step = max(1, int(round(20.0 / hz[min(k, len(hz) - 1)])))
            frames = [(pos, yaw, math.radians(-pitch)) for (t, pos, yaw) in P["rows"][::step]]
            print(f"[yardgen] {P['name']}: {len(P['rows'])} poses, {len(frames)} images", flush=True)
            C.render_frames(cap(frames), d, C.SIM_CAM, samples=a.path_samples, passes=True, tag=f":{P['name']}")

    # ---- start views (what the policy sees at t0 from every start) ----
    if "starts" in sets:
        frames = []
        for s in starts_w:   # world-frame starts
            x, y, z = s["pos_ned"][0], -s["pos_ned"][1], -s["pos_ned"][2]
            frames.append(((x, y, z), -math.radians(s["yaw_ned_deg"]), math.radians(-pitch)))
        C.render_frames(cap(frames), out + "/starts", C.SIM_CAM, samples=a.path_samples, passes=True, name_fmt="s{:03d}", tag=":starts")

    print(f"[yardgen] done {a.yard} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
