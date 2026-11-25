"""Truth audit per yard: door face distance to the nearest occupied band cell, doorstep free, start counts.
usage: python audit_yards.py [yard ...]   (defaults to all yards/syn_*)"""
import json, sys, glob, os
import numpy as np
SY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yards = sys.argv[1:] or sorted(glob.glob(f"{SY}/yards/syn_*"))
print(f"{'yard':8s} {'stratum':9s} {'door-occ m':>10s} {'hover3d m':>10s} {'valid':>6s} {'occ%':>5s} verdict")
for y in yards:
    y = y if os.path.isdir(y) else f"{SY}/yards/{y}"
    name = os.path.basename(y)
    try:
        t = np.load(f"{y}/recon/truth.npz"); s = json.load(open(f"{y}/survey.json")); m = json.load(open(f"{y}/meta.json"))
    except Exception as e:
        print(f"{name:8s} missing ({e.__class__.__name__})"); continue
    cell = float(t["cell"]); origin = np.asarray(t["lo"])[:2]; occ2 = np.asarray(t["occ2d"]).astype(bool)
    ii, jj = np.nonzero(occ2)
    cx = origin[0] + (ii + 0.5) * cell; cy = origin[1] + (jj + 0.5) * cell
    def dmin(p):
        return float(np.sqrt((cx - p[0])**2 + (cy - p[1])**2).min()) if len(cx) else float("nan")
    d_face = dmin(s["door_face_ned"])
    lo3 = np.asarray(t["lo"]); o3 = np.asarray(t["occ3d"]).astype(bool); I = np.argwhere(o3)
    C = lo3 + (I + 0.5) * cell; hp = np.asarray(s["doorstep_ned"], float) + np.array([0, 0, -0.5])  # hover 0.5 m above the doorstep
    d_step = float(np.sqrt(((C - hp) ** 2).sum(1)).min()) if len(C) else float("nan")  # 3D clearance of the hover point
    valid = sum(1 for st in s["starts"] if st.get("valid"))
    stratum = s.get("door_facing") or s.get("stratum") or (s.get("door") or {}).get("stratum") or m.get("stratum") or (m.get("params") or {}).get("stratum") or (m.get("house") or {}).get("stratum", "?")
    ok = d_face <= 0.5 and d_step >= 0.30 and valid >= 36
    print(f"{name:8s} {str(stratum):9s} {d_face:10.2f} {d_step:10.2f} {valid:6d} {100*occ2.mean():5.1f} {'ok' if ok else 'CHECK'}")
