#!/usr/bin/env python3
"""Convert canonical visual-expert trajectories to the SousVide rollout format."""
import argparse
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VE_DIR = os.path.join(
    FLAMBE_TERRA, "FiGS-Standalone", "3dgs", "workspace", "outputs",
    "msl_001", "visual_expert")
COHORTS = os.path.join(FLAMBE_TERRA, "SousVide_official", "cohorts")

# our canonical T_C2B == carl.json (verified). Kept here only for a guard assert.
CANON_T_C2B = np.array([
    [0.0,  0.0, -1.0,  0.10],
    [1.0,  0.0,  0.0, -0.03],
    [0.0, -1.0,  0.0, -0.01],
    [0.0,  0.0,  0.0,  1.00],
])


# ---------------------------------------------------------------------------
# Pure-python repack (no official deps) -- this is the selftest-able core.
# ---------------------------------------------------------------------------
def repack_tXUd(tXUd: np.ndarray, trim_static: bool = True,
                keep_hover_frames: int = 0):
    """
    Pure repack of a (15, N) tXUd into the official trajectory arrays.

    Returns dict with Tro(N,), Xro(N,10), Uro(N,4), Wro(N,6), tXUd_field(N,11),
    plus "N_raw" (input length), "N" (after trim), and "keep_idx" (the raw column
    indices retained, IN ORDER -- pass this to load_images_rgb so the frames stay
    aligned with the trajectory).
    Quaternion is copied DIRECTLY (XYZW -> XYZW), no reorder.

    trim_static: our save_episode writes ALL Nctl control frames, but the flight
      actually ENDS at the landing index; the tail is padded with frozen-time
      duplicate frames (e.g. mid_center: active 0..373, then 526 dup frames with
      t==18.65, dt==0).  Feeding ALL those frozen frames into ORCA replay biases
      histNet/commNet toward one hover-at-door state, so by default we trim
      everything after the last STRICTLY-increasing time step.  The kept slice
      ends at the last real motion sample (inclusive).  Set False to keep the raw
      padded length.

    keep_hover_frames: retention mode for the terminal "stop and hold at the door"
      behavior (default 0 == legacy trim, drop ALL frozen padding).  When >0 (and
      trim_static is True) we keep the FULL decelerate/descent ramp (every
      real-motion frame) AND additionally retain up to `keep_hover_frames` frames
      EVENLY SUBSAMPLED from the frozen-time hover tail.  This re-introduces the
      "arrived, zero velocity, holding" label the policy must imitate WITHOUT
      letting the (hundreds of) identical hover frames dominate the dataset.  The
      kept frozen frames carry dt==0 in the EKF history block (pilot.py records dt
      as a *feature*, never a divisor), so this is safe for obs-gen / svu.
    """
    assert tXUd.ndim == 2 and tXUd.shape[0] == 15, \
        f"tXUd must be (15,N), got {tXUd.shape}"
    N_raw = tXUd.shape[1]
    assert N_raw >= 2, f"need at least 2 columns, got {N_raw}"

    if trim_static:
        dt_raw = np.diff(tXUd[0, :])
        good = np.where(dt_raw > 0)[0]
        # keep up to (and including) the last sample that advanced time
        last = int(good[-1]) + 1 if len(good) else N_raw - 1
        keep = last + 1                       # end of real-motion ramp (exclusive)
        active_idx = np.arange(keep)
        frozen_idx = np.arange(keep, N_raw)   # frozen-time hover/hold padding tail
        if keep_hover_frames > 0 and len(frozen_idx) > 0:
            # full ramp + an evenly-spaced subsample of the frozen hover tail
            n_hov = min(int(keep_hover_frames), len(frozen_idx))
            pick = np.unique(
                np.linspace(0, len(frozen_idx) - 1, n_hov).round().astype(int))
            hover_idx = frozen_idx[pick]
            keep_idx = np.concatenate([active_idx, hover_idx])
            print(f"  [trim+hover] {N_raw} raw cols -> {keep} ramp + "
                  f"{len(hover_idx)} downsampled hover (of {len(frozen_idx)} "
                  f"frozen); dropped {N_raw - len(keep_idx)}")
        else:
            keep_idx = active_idx
            if keep < N_raw:
                print(f"  [trim] {N_raw} raw cols -> {keep} active (dropped "
                      f"{N_raw-keep} frozen-time padding frames)")
    else:
        keep_idx = np.arange(N_raw)
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    tXUd = tXUd[:, keep_idx]
    N = tXUd.shape[1]

    Tro = np.ascontiguousarray(tXUd[0, :]).astype(np.float64)        # (N,)
    pos = tXUd[1:4, :].T                                            # (N,3)
    vel = tXUd[4:7, :].T                                            # (N,3)
    quat = tXUd[7:11, :].T                                          # (N,4) XYZW DIRECT
    Xro = np.ascontiguousarray(np.hstack([pos, vel, quat])).astype(np.float64)  # (N,10)
    Uro = np.ascontiguousarray(tXUd[11:15, :].T).astype(np.float64)  # (N,4) [nf,wx,wy,wz]
    Wro = np.zeros((N, 6), dtype=np.float64)                        # no external wrench

    # tXUd field for the dict = (N,11) = [t,pos,vel,quat_xyzw]  (drop control rows;
    # the canonical reference uses 11 columns)
    tXUd_field = np.ascontiguousarray(
        np.hstack([Tro.reshape(-1, 1), pos, vel, quat])).astype(np.float64)  # (N,11)

    # ---- invariant asserts (array lengths, quaternion norm, monotonic time) ----
    assert Xro.shape == (N, 10), Xro.shape
    assert Uro.shape == (N, 4), Uro.shape
    assert Wro.shape == (N, 6), Wro.shape
    assert tXUd_field.shape == (N, 11), tXUd_field.shape
    qn = np.linalg.norm(quat, axis=1)
    assert np.allclose(qn, 1.0, atol=1e-3), \
        f"quaternions not unit norm: min={qn.min():.4f} max={qn.max():.4f}"
    dt = np.diff(Tro)
    n_flat = int(np.sum(dt <= 0))
    if n_flat:
        print(f"  [warn] {n_flat} non-increasing time step(s) remain after trim "
              f"(interior frozen frames?) -- check the source trajectory")

    return {
        "Tro": Tro, "Xro": Xro, "Uro": Uro, "Wro": Wro,
        "tXUd_field": tXUd_field, "N": N, "N_raw": N_raw,
        "keep_idx": keep_idx,
    }


def _pad_n_plus_1(arr2d_or_1d):
    """Append a duplicate of the last row/element -> length N+1 (for svu only)."""
    a = arr2d_or_1d
    if a.ndim == 1:
        return np.concatenate([a, a[-1:]], axis=0)
    return np.concatenate([a, a[-1:, :]], axis=0)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_images_rgb(img_dir: str, N: int, keep_idx=None) -> np.ndarray:
    """
    Load frame_NNNNN.png in STRICT numeric order, convert BGR->RGB, stack.
    Returns (N, 360, 640, 3) uint8.

    keep_idx: optional 1-D array of RAW frame indices to retain (in order), as
      returned by repack_tXUd -- keeps images aligned with a subsampled trajectory
      (e.g. keep_hover_frames).  When None, the first N frames are used (the legacy
      contiguous-prefix behavior).
    """
    import cv2
    paths = sorted(glob.glob(os.path.join(img_dir, "frame_*.png")))
    assert len(paths) > 0, f"no frames in {img_dir}"
    # sort by the integer index, NOT lexically, to be safe about width changes
    def idx_of(p):
        return int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
    paths = sorted(paths, key=idx_of)
    # the frame indices MUST be contiguous 0..len-1 (time order == tXUd order)
    idxs = [idx_of(p) for p in paths]
    assert idxs == list(range(len(idxs))), \
        f"frame indices not contiguous 0..{len(idxs)-1}: {idxs[:5]}...{idxs[-3:]}"
    if keep_idx is None:
        assert len(paths) >= N, f"have {len(paths)} frames, need {N}"
        if len(paths) > N:
            print(f"  [warn] {len(paths)} frames > N={N}; truncating to N")
        sel = list(range(N))
    else:
        sel = [int(i) for i in np.asarray(keep_idx).ravel()]
        assert len(sel) == N, f"keep_idx len {len(sel)} != N {N}"
        assert max(sel) < len(paths), \
            f"keep_idx max {max(sel)} >= available frames {len(paths)}"

    frames = []
    for i in sel:
        p = paths[i]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)  # (H,W,3) uint8 BGR
        assert bgr is not None, f"failed to read {p}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    rgb = np.stack(frames, axis=0)
    assert rgb.dtype == np.uint8, f"rgb dtype {rgb.dtype} != uint8"
    assert rgb.ndim == 4 and rgb.shape[0] == N and rgb.shape[3] == 3, rgb.shape
    return rgb


# ---------------------------------------------------------------------------
# Full conversion (needs the official `figs` + `sousvide` packages -> Docker)
# ---------------------------------------------------------------------------
def convert_start(start: str, use_svu: bool = True, compress: bool = False,
                  cohort: str = None, keep_hover_frames: int = 0,
                  scene: str = None, ve_dir: str = None, door_ned=None):
    """
    scene:    optional scene id (e.g. "msl_002"). When given, the emitted course dir
              is f"{scene}_{start}_canon" so that pooling MANY scenes' starts into ONE
              shared cohort yields FLAT, UNIQUE course names (obs-gen discovers courses
              via os.listdir(rollout_data) and requires unique leaf names). When None the
              legacy prefix "msl_001" is used -> byte-identical default behavior.
    ve_dir:   optional visual_expert root override (…/outputs/<scene>/visual_expert). When
              None the module-level msl_001 VE_DIR is used -> byte-identical default.
    door_ned: optional per-scene door target [x,y,z] in aligned NED. TRACEABILITY ONLY --
              the policy has NO goal/objective input (visual-servoing), and the door is
              already baked into the expert tXUd trajectory, so this value is logged but
              NOT injected into the saved rollout schema (keeps observation generator schema).
    """
    import torch

    ve_root = ve_dir if ve_dir is not None else VE_DIR
    course_scene = scene if scene is not None else "msl_001"
    if door_ned is not None:
        print(f"  [meta] scene={course_scene} start={start} door_ned={list(door_ned)} "
              f"(traceability only; not used in conversion)")

    src = os.path.join(ve_root, start)
    txud_path = os.path.join(src, "tXUd.npy")
    img_dir = os.path.join(src, "images")
    assert os.path.isfile(txud_path), f"missing {txud_path}"

    tXUd = np.load(txud_path)
    packed = repack_tXUd(tXUd, keep_hover_frames=keep_hover_frames)
    N = packed["N"]
    Tro, Xro, Uro, Wro = packed["Tro"], packed["Xro"], packed["Uro"], packed["Wro"]

    rgb = load_images_rgb(img_dir, N, keep_idx=packed["keep_idx"])

    # --- frame / params / Wrs / FOro via the official svu (Docker only) ---
    if use_svu:
        import sys as _sys
        for _p in (os.path.join(FLAMBE_TERRA, "SousVide_official", "src"), os.path.join(FLAMBE_TERRA, "SousVide_official", "FiGS", "src")):
            if _p not in _sys.path:
                _sys.path[:0] = [_p]
        _sys.path[:0] = LOCAL_MODULE_PATHS
        import figs.utilities.config_helper as ch
        import sousvide.utilities.sousvide_utilities as svu
        frame = ch.get_config("carl", "frames")
        bframe = ch.get_config("carl", "frames")
        # guard: the canonical camera mount really is carl's mount
        c2b = np.array(frame["camera_to_body_transform"])
        assert np.allclose(c2b, CANON_T_C2B, atol=1e-6), \
            "carl camera_to_body_transform != canonical T_C2B"

        prms = svu.compute_prms(frame)
        # svu expects len(Xro)==len(Uro)+1 (Xro[:-1] in compute_FOro). Our Xro is
        # length N == len(Uro); pad an (N+1)th duplicate state/time for svu only.
        Xro_pad = _pad_n_plus_1(Xro)   # (N+1,10)
        Tro_pad = _pad_n_plus_1(Tro)   # (N+1,)
        Wro_pad = _pad_n_plus_1(Wro)   # (N+1,6) ; compute_Wrs loops range(Uro.shape[0])=N
        Wrs = svu.compute_Wrs(Xro_pad, Uro, Wro_pad, frame, bframe)
        FOro = svu.compute_FOro(Tro_pad, Xro_pad, Uro, Wro_pad, frame)
    else:
        prms, Wrs, FOro, frame = None, None, None, "carl"

    Ndata = int(Uro.shape[0])  # == N
    # All the per-timestep arrays the obs-generator iterates MUST be equal length
    assert len(Tro) == N == len(Xro) == Uro.shape[0] == rgb.shape[0] == Ndata, (
        len(Tro), len(Xro), Uro.shape[0], rgb.shape[0], Ndata)

    trajectory = {
        "Tro": Tro, "Xro": Xro, "Uro": Uro, "Wro": Wro,
        "params": prms, "Wrs": Wrs, "FOro": FOro,
        "tXUd": packed["tXUd_field"],
        "Ndata": Ndata,
        "Tsol": np.zeros((4, Ndata)),
        "rollout_id": "000000",
        "frame": frame,
    }
    images = {"rgb": rgb, "depth": None, "rollout_id": "000000"}

    # --- write to official cohort layout ---
    # course leaf name is scene-prefixed so a MULTI-scene pooled cohort stays collision-free
    # (default course_scene == "msl_001" -> unchanged legacy name "msl_001_<start>_canon").
    course = f"{course_scene}_{start}_canon"
    if cohort is None:
        cohort = f"{course_scene.replace('_', '')}_{start}_canon"
    dset = os.path.join(COHORTS, cohort, "rollout_data", course)
    traj_dir = os.path.join(dset, "trajectories")
    imgs_dir = os.path.join(dset, "images")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)

    # official save_rollouts stores a LIST of trajectory dicts and a LIST? -> see
    # rollout_generator.save_rollouts: torch.save(Trajectories,...) where
    # Trajectories is a python list. Images is also a list of image dicts.
    torch.save([trajectory], os.path.join(traj_dir, "001.pt"))
    torch.save([images], os.path.join(imgs_dir, "001.pt"))

    print(f"  wrote {traj_dir}/001.pt  (N={N}, svu={'on' if use_svu else 'OFF'})")
    print(f"  wrote {imgs_dir}/001.pt  rgb {rgb.shape} {rgb.dtype}")
    return dset


# ---------------------------------------------------------------------------
# Selftest (NO official deps; exercises the pure repack + invariants)
# ---------------------------------------------------------------------------
def _synthetic_tXUd(N=50):
    from scipy.spatial.transform import Rotation
    t = np.arange(N) * 0.05
    pos = np.linspace([5, -8, -1], [8, -21, -0.4], N).T          # (3,N)
    vel = np.gradient(pos, axis=1) / 0.05                         # (3,N)
    yaws = np.linspace(0.0, -1.4, N)
    quat = Rotation.from_euler("z", yaws).as_quat().T            # (4,N) XYZW
    thrust = np.full(N, -0.41)
    omega = np.zeros((3, N))
    tXUd = np.vstack([t.reshape(1, -1), pos, vel, quat,
                      thrust.reshape(1, -1), omega])             # (15,N)
    assert tXUd.shape == (15, N)
    return tXUd, quat.T, yaws


def run_selftest():
    from scipy.spatial.transform import Rotation
    print("[selftest] building synthetic (15,50) tXUd ...")
    tXUd, quat_xyzw, yaws = _synthetic_tXUd(50)
    out = repack_tXUd(tXUd)
    N = out["N"]
    assert N == 50

    # shapes
    assert out["Tro"].shape == (50,)
    assert out["Xro"].shape == (50, 10)
    assert out["Uro"].shape == (50, 4)
    assert out["Wro"].shape == (50, 6)
    assert out["tXUd_field"].shape == (50, 11)
    print("[selftest] shapes OK")

    # quat copied DIRECTLY (no reorder): Xro[:,6:10] must equal tXUd rows 7-10
    assert np.allclose(out["Xro"][:, 6:10], quat_xyzw), "quat was reordered!"
    # and decode to the intended yaw (proves XYZW slotting is correct)
    dec_yaw = Rotation.from_quat(out["Xro"][:, 6:10]).as_euler("xyz")[:, 2]
    assert np.allclose(dec_yaw, yaws, atol=1e-6), "decoded yaw != intended"
    print("[selftest] quat XYZW-direct + decode OK")

    # quat unit norm
    qn = np.linalg.norm(out["Xro"][:, 6:10], axis=1)
    assert np.allclose(qn, 1.0, atol=1e-6)
    print("[selftest] quat norm OK")

    # Uro = [nf,wx,wy,wz] from rows 11..14
    assert np.allclose(out["Uro"][:, 0], -0.41)
    assert np.allclose(out["Uro"][:, 1:], 0.0)
    print("[selftest] Uro mapping OK")

    # tXUd_field = [t,pos,vel,quat]
    assert np.allclose(out["tXUd_field"][:, 0], tXUd[0])
    assert np.allclose(out["tXUd_field"][:, 1:4], tXUd[1:4].T)
    assert np.allclose(out["tXUd_field"][:, 4:7], tXUd[4:7].T)
    assert np.allclose(out["tXUd_field"][:, 7:11], quat_xyzw)
    print("[selftest] tXUd_field mapping OK")

    # Length invariant (simulate a matching rgb / Ndata)
    Ndata = out["Uro"].shape[0]
    rgb_fake = np.zeros((N, 360, 640, 3), dtype=np.uint8)
    assert len(out["Tro"]) == N == len(out["Xro"]) == out["Uro"].shape[0] \
        == rgb_fake.shape[0] == Ndata
    print("[selftest] R9 length invariant OK")

    # N+1 pad helper used for svu
    Xpad = _pad_n_plus_1(out["Xro"])
    Tpad = _pad_n_plus_1(out["Tro"])
    assert Xpad.shape == (N + 1, 10) and np.allclose(Xpad[-1], out["Xro"][-1])
    assert Tpad.shape == (N + 1,) and Tpad[-1] == out["Tro"][-1]
    # this is exactly what compute_FOro consumes: Xpad[:-1] aligns with Uro
    assert Xpad[:-1].shape[0] == out["Uro"].shape[0]
    print("[selftest] svu N+1 pad alignment OK")

    # rejects a wxyz-looking input only via norm? -> verify bad shape rejected
    try:
        repack_tXUd(np.zeros((14, 10)))
        raise AssertionError("should have rejected (14,N)")
    except AssertionError as e:
        if "should have rejected" in str(e):
            raise
    print("[selftest] bad-shape rejection OK")

    # keep_hover_frames: full ramp + downsampled frozen tail, aligned keep_idx
    tXUd2, _, _ = _synthetic_tXUd(50)
    frozen = np.repeat(tXUd2[:, -1:], 60, axis=1)   # 60 frozen-time dup cols
    frozen[0, :] = tXUd2[0, -1]                      # freeze time EXACTLY (dt==0)
    tXUd2 = np.hstack([tXUd2, frozen])               # (15,110): 50 ramp + 60 frozen
    base = repack_tXUd(tXUd2, keep_hover_frames=0)   # legacy: trims all frozen
    assert base["N"] == 50, base["N"]
    assert np.array_equal(base["keep_idx"], np.arange(50))
    hov = repack_tXUd(tXUd2, keep_hover_frames=40)
    assert 50 < hov["N"] <= 90, hov["N"]             # kept some, not all 60 frozen
    assert hov["N"] < 110, hov["N"]                  # downsampled (not raw length)
    assert np.array_equal(hov["keep_idx"][:50], np.arange(50))  # full ramp intact
    assert hov["keep_idx"].min() == 0 and hov["keep_idx"].max() == 109
    assert np.all(np.diff(hov["keep_idx"]) > 0)      # ordered -> image alignment ok
    assert (hov["N"] - 50) <= 40                     # hover count capped by param
    print("[selftest] keep_hover_frames ramp+downsampled-tail OK")

    print("\n[selftest] ALL PASS")
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="mid_center",
                    help="named start dir under visual_expert/ (default mid_center)")
    ap.add_argument("--selftest", action="store_true",
                    help="run pure-python selftest (no official deps) and exit")
    ap.add_argument("--no-svu", action="store_true",
                    help="skip svu.compute_Wrs/FOro/prms (Wrs/FOro/params = None)")
    ap.add_argument("--compress", action="store_true", help="(reserved)")
    ap.add_argument("--cohort", default=None,
                    help="target cohort name (default msl001_<start>_canon); set to merge many starts into one cohort")
    ap.add_argument("--keep-hover-frames", type=int, default=0, dest="keep_hover_frames",
                    help="retain up to N evenly-subsampled frozen-time hover frames after the "
                         "real-motion ramp so 'stop and hold at door' is imitated "
                         "(default 0 = drop all frozen padding, legacy behavior)")
    ap.add_argument("--scene", default=None,
                    help="scene id (e.g. msl_002); sets the course prefix to {scene}_<start>_canon "
                         "so many scenes pool into ONE cohort without name collisions "
                         "(default None = legacy msl_001 prefix)")
    ap.add_argument("--ve-dir", default=None, dest="ve_dir",
                    help="override visual_expert root (.../outputs/<scene>/visual_expert); "
                         "default None = module msl_001 VE_DIR")
    ap.add_argument("--door-ned", default=None, dest="door_ned",
                    help="per-scene door target 'x,y,z' in aligned NED (traceability log only)")
    args = ap.parse_args()

    door_ned = None
    if args.door_ned is not None:
        door_ned = [float(v) for v in args.door_ned.split(",")]

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    print(f"converting start='{args.start}' (svu={'OFF' if args.no_svu else 'on'}) "
          f"scene={args.scene or 'msl_001(default)'} cohort={args.cohort or 'default'} "
          f"keep_hover_frames={args.keep_hover_frames} ...")
    convert_start(args.start, use_svu=not args.no_svu, compress=args.compress,
                  cohort=args.cohort, keep_hover_frames=args.keep_hover_frames,
                  scene=args.scene, ve_dir=args.ve_dir, door_ned=door_ned)
    print("done.")


if __name__ == "__main__":
    main()
