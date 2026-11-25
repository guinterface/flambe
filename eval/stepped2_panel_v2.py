"""Summarize stepped-start evaluation results in a Markdown report."""
import json, os, sys
from math import comb
import numpy as np

FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_GATE2 = os.environ.get("FLAMBE_GATE2", os.path.join(FLAMBE_ROOT, "research", "02_experiments", "gate2"))

M, FULL, DM, DP = sys.argv[1:5]; OB = os.path.join(FLAMBE_GATE2, "stepped_v2b"); ST = OB + "/final"
YU = ["syn_001", "syn_006", "syn_007", "syn_008"]; YC = ["syn_002", "syn_016", "syn_018", "syn_040"]; ORDER = ["0", "1", "2", "3", "4", "0T", "1T"]
sp = lambda w, l: 1.0 if w + l == 0 else min(1.0, 2 * sum(comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l))
post = lambda f: f["success"] and f.get("t_contact") is not None and f["t_contact"] > f["t_success"]
lab = lambda y, s: (str(s) + "T") if y in YC else str(s)


def load(d):
    out = {}
    for y in YU + YC:
        L = np.asarray([g for g in json.load(open(f"{FLAMBE_YARDS}/yards/{y}/goals.json"))["goals"] if g["id"] == "door"][0]["land_ned"], float)
        for f in json.load(open(f"{OB}/{d}/eval_{y}.json"))["flights"]:
            P = np.load(f"{OB}/{d}/tracks_{y}/{f['id']}.npz")["P"]; out[(y, f["id"])] = dict(f, end_m=float(np.linalg.norm(P[-1] - L)))
    return out


def distinct(P):
    k = []
    for p in P:
        if all(np.linalg.norm(np.asarray(p) - np.asarray(q)) >= 0.5 for q in k): k.append(p)
    return len(k)


S = {(y, s["id"]): s for y in YU + YC for s in json.load(open(f"{ST}/stepped2_{y}.json"))["starts"]}; step = {k: lab(k[0], s["step"]) for k, s in S.items()}
A = load(DM); assert set(A) == set(S), "eval ids != stepped2 ids"
T = [f"# Stepped test v2 starts, {M} s0 (no rule applied; landings = score rule successes, frames not viewed yet)", "",
     "Spec score_1121/STEPPED_TEST.md 7d19e6b7. Steps 0-3: ring starts on the door's approach side (0: 2-3 m, 1: 4-6 m, 2/3: 8-12 m, horizontal from the land "
     "point, at hover height); step 4 = v1's step 4 (25 of the 796 per yard); 0T / 1T = steps 0 / 1 on training yards syn_002/016/018/040. n = flights; distinct = "
     "starts >= 0.5 m apart (use as n for intervals). End = distance from the track's last state to the land point.", "",
     "Selection note: every start was kept only if the expert lands from it; on syn_007 the expert landed from 63.8% of the ring candidates (146/229; syn_006 94.8%, others 100%), so syn_007's starts are filtered toward easier ones.", "",
     "| Step | n | distinct | Landings | No later contact | End <= 1.5 m | End <= 1.0 m | End <= 0.5 m | Closest <= 1.5 m | Closest <= 0.5 m | Median closest (m) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for s in ORDER:
    ks = [k for k in A if step[k] == s]
    if not ks: continue
    F = [A[k] for k in ks]; cl = sorted(f["closest_m"] for f in F); dn = sum(distinct([S[k]["pos"] for k in ks if k[0] == y]) for y in YU + YC)
    T.append(f"| {s} | {len(F)} | {dn} | {sum(f['success'] for f in F)} | {sum(f['success'] and not post(f) for f in F)} | {sum(f['end_m'] <= 1.5 for f in F)} | "
             f"{sum(f['end_m'] <= 1.0 for f in F)} | {sum(f['end_m'] <= 0.5 for f in F)} | {sum(f['closest_m'] <= 1.5 for f in F)} | {sum(f['closest_m'] <= 0.5 for f in F)} | {cl[len(cl) // 2]:.2f} |")
rows = [t for t in T if t.startswith("| ") and not t.startswith("| Step")]
B = {(y, f["id"]): f for y in YU for f in json.load(open(f"{FLAMBE_GATE2}/{FULL}/eval_{y}.json"))["flights"]} if FULL else {}
k4 = [k for k in A if step[k] == "4" and k in B]; w = sum(A[k]["success"] and not B[k]["success"] for k in k4); l = sum(B[k]["success"] and not A[k]["success"] for k in k4)
tail = [f"Step 4 vs the full unseen run ({FULL}, same ids, {len(k4)} flights): {sum(B[k]['success'] for k in k4)} landings there vs {sum(A[k]['success'] for k in k4)} here; differ {w + l}"]
if DP:
    P_ = load(DP)
    for s in ORDER:
        ks = [k for k in A if step[k] == s]
        if not ks: continue
        w = sum(A[k]["success"] and not P_[k]["success"] for k in ks); l = sum(P_[k]["success"] and not A[k]["success"] for k in ks)
        cw = sum(A[k]["end_m"] < P_[k]["end_m"] for k in ks); cf = sum(A[k]["end_m"] > P_[k]["end_m"] for k in ks)
        tail.append(f"Step {s} {M} vs previous ({DP}): landings wins {w} losses {l} (p {sp(w, l):.4f}); ends closer {cw} / farther {cf} (p {sp(cw, cf):.3g})")
open(f"{OB}/stepped2_{M}.md", "w").write("\n".join(T + [""] + tail) + "\n")
print("STEPPED2_PANEL " + M + " " + " | ".join(rows) + " | " + " | ".join(tail))
