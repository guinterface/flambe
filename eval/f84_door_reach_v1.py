"""Check whether the front door is reachable from generated starts in the yard occupancy grid."""
import json, os, sys
import numpy as np
from scipy.ndimage import distance_transform_edt, label

R_PASS = float(os.environ.get("R_PASS", "0.65")); PROBE_D = float(os.environ.get("PROBE_D", "1.0")); H_PROBE = float(os.environ.get("H_PROBE", "1.2"))
H_LO, H_HI = 0.6, 2.4


def check(yd):
    sv = json.load(open(f"{yd}/survey.json")); t = np.load(f"{yd}/recon/truth.npz")
    occ = np.asarray(t["occ3d"]).astype(bool); lo = np.asarray(t["lo"], float); c = float(t["cell"]); zg = float(sv.get("ground_z_ned", t["z_ground"]))
    nx, ny, nz = occ.shape; zc = lo[2] + (np.arange(nz) + 0.5) * c; hgt = zg - zc                      # height above ground of each layer centre
    clr = distance_transform_edt(~occ, sampling=c)
    trav = (~occ) & (clr >= R_PASS) & ((hgt >= H_LO) & (hgt <= H_HI))[None, None, :]
    lab, n = label(trav)
    I = np.argwhere(trav); P = np.c_[lo[0] + (I[:, 0] + .5) * c, lo[1] + (I[:, 1] + .5) * c, zc[I[:, 2]]]
    def comp(p, rmax=1.0):
        if len(P) == 0: return 0, None
        d = np.linalg.norm(P - p, axis=1); j = int(np.argmin(d))
        return (int(lab[tuple(I[j])]), float(d[j])) if d[j] <= rmax else (0, float(d[j]))
    probe = np.asarray(sv["doorstep_ned"], float) + PROBE_D * np.asarray(sv["door_normal_ned"], float); probe[2] = zg - H_PROBE
    kd, dd = comp(probe)
    ks = {}
    for s in sv["starts"]:
        if not s.get("valid", True): continue
        k, d = comp(np.asarray(s["pos_ned"], float), 1.0)
        if k: ks[k] = ks.get(k, 0) + 1
    ok = kd != 0 and kd in ks
    size = int((lab == kd).sum()) if kd else 0
    return ok, (f"facing {sv.get('door_facing')}, door probe {'voxel' if kd else 'NO traversable voxel'} within {dd:.2f} m, door component {kd} ({size} voxels), "
                f"start components {dict(sorted(ks.items(), key=lambda kv: -kv[1])[:3])} of {sum(ks.values())} valid starts mapped, R_PASS {R_PASS:g} m")


def main(dirs):
    bad = 0
    for yd in dirs:
        y = os.path.basename(os.path.normpath(yd))
        try: ok, msg = check(yd)
        except Exception as e: ok, msg = False, f"ERROR {type(e).__name__}: {e}"
        bad += not ok; print(f"DOOR_REACH {y} {'PASS' if ok else 'REJECT'} ({msg})", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
