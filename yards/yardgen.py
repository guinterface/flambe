"""Synthetic residential yard generator (run inside Blender):
  blender -b --factory-startup -noaudio -P yardgen.py -- --yard syn_001 --seed 1 --stratum recessed --render capture,paths,preview
Outputs terra-compatible yard folders under synth_yards/yards/<yard_id>/ (meta.json, survey.json, recon/, capture/, path_XX/)."""
import sys, os, json, time, math, argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bpy
import numpy as np
from yardgen.common import RNG, reset_scene, collection, ned, SEM
from yardgen import materials as M, assets as A
from yardgen.house import house_params
from yardgen.lot import Lot, lot_params, GROUND_Z, STREET_X0, STREET_X1
from yardgen import truth as T
from yardgen import capture as C

ROOT = FLAMBE_YARDS


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
    return ap.parse_args(argv)


def main():
    a = parse(); t0 = time.time()
    out = a.out or f"{ROOT}/yards/{a.yard}"; os.makedirs(out, exist_ok=True)
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
    json.dump(survey, open(out + "/survey.json", "w"), indent=1)
    meta = {"yard_id": a.yard, "date": time.strftime("%Y-%m-%d"), "vehicle": "synthetic (Blender 4.5 Cycles/OptiX)", "seed": a.seed,
            "camera": {**C.SIM_CAM, "pitch_deg": pitch, "hfov_deg": 2 * math.degrees(math.atan(C.SIM_CAM["width"] / 2 / C.SIM_CAM["fx"]))},
            "capture_camera": {**C.CAPTURE_CAM, "hfov_deg": 2 * math.degrees(math.atan(C.CAPTURE_CAM["width"] / 2 / C.CAPTURE_CAM["fx"]))},
            "door_facing": a.stratum, "frame": "NED = (X, -Y, -Z) of the Blender/GS frame; GS frame == Blender world; ground at z_ned = -0.15 (street surface z_ned = 0)",
            "lot_params": {k: v for k, v in lp.items()}, "house_params": {k: v for k, v in hp.items()}, "notes": "procedurally generated; all labels exact"}
    json.dump(meta, open(out + "/meta.json", "w"), indent=1, default=str)
    print(f"[yardgen] starts: {nvalid}/{len(starts)} valid (beyond r_det and door not visible at t0)", flush=True)

    # ---- truth ----
    ext = (STREET_X0 - 4.0, lot.lot_x1 + 2.0, lot.y0 - 10.0, lot.y1 + 10.0)
    if not a.no_truth:
        tt = time.time(); info = T.export_truth(out, lot, door, ext, pitch, rng.sub("truth"))
        print(f"[yardgen] truth exported in {time.time() - tt:.1f}s: {info}", flush=True)
        meta["truth"] = info; json.dump(meta, open(out + "/meta.json", "w"), indent=1, default=str)

    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=out + "/scene.blend", compress=True)

    sets = [s for s in a.render.split(",") if s and s != "none"]
    cap = lambda F: F[:a.max_frames] if a.max_frames else F

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
        from yardgen.common import look_at_matrix
        F = [look_at_matrix(e, t) for (e, t) in views]
        C.render_frames(cap(F), out + "/preview", C.CAPTURE_CAM, samples=a.samples, passes=False, name_fmt="view_{:02d}", log_every=4, tag=":preview")

    # ---- capture (splat substrate) ----
    if "capture" in sets:
        F = C.coverage_plan(lot, door, rng.sub("cov"))
        print(f"[yardgen] capture plan: {len(F)} frames", flush=True)
        tf = C.render_frames(cap(F), out + "/capture", C.CAPTURE_CAM, samples=a.samples, passes=True, tag=":capture")
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
        paths = C.expert_paths(lot, door, rng.sub("paths"), starts)
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
        for s in starts:
            x, y, z = s["pos_ned"][0], -s["pos_ned"][1], -s["pos_ned"][2]
            frames.append(((x, y, z), -math.radians(s["yaw_ned_deg"]), math.radians(-pitch)))
        C.render_frames(cap(frames), out + "/starts", C.SIM_CAM, samples=a.path_samples, passes=True, name_fmt="s{:03d}", tag=":starts")

    print(f"[yardgen] done {a.yard} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
