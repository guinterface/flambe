"""Generate stepped evaluation starts and select candidates the expert can complete."""
import json, math, os, sys
import numpy as np

FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

SEED = 20261122; NPER = 25; NCAND = int(os.environ.get("NCAND", "80")); MINSEP = {0: 0.5, 1: 0.5, 2: 1.0}
HELD = ["syn_001", "syn_006", "syn_007", "syn_008"]; CTRL = os.environ.get("CTRL", "syn_002,syn_013,syn_025,syn_040").split(",")
BANDS = {0: (2.0, 3.0), 1: (4.0, 6.0), 2: (8.0, 12.0)}; CODE = {0: 0, 1: 1, 2: 2, 3: 2}


def door(y):
    return [g for g in json.load(open(f"{FLAMBE_YARDS}/yards/{y}/goals.json"))["goals"] if g["id"] == "door"][0]


def steps_of(y): return (0, 1, 2, 3) if y in HELD else (0, 1)


def ring(y, band_step, cf):
    g = door(y); L = np.asarray(g["land_ned"], float); n = np.asarray(g["normal_ned"], float); a0 = math.atan2(n[1], n[0]); z = float(g["hover_ned"][2])
    lo, hi = BANDS[band_step]; rng = np.random.default_rng([SEED, int(y[4:]), CODE[band_step], 0]); out = []; tries = 0
    while len(out) < NCAND and tries < 200000:
        tries += 1; r = float(rng.uniform(lo, hi)); off = float(rng.uniform(-75, 75)); a = a0 + math.radians(off)
        p = np.array([L[0] + r * math.cos(a), L[1] + r * math.sin(a), z])
        if float(cf(p[None])[0]) < 1.0 or any(np.linalg.norm(p - q[0]) < MINSEP[band_step] for q in out): continue
        out.append((p, r, off))
    return out, tries


def cands(G, OUT, y):
    import score
    sys.path.insert(0, os.path.join(FLAMBE_TERRA, "s10")); import make_labels as mk
    from gate0b_expert import _yaw_quat
    score.SY = FLAMBE_YARDS; cf = score.clearance_fn(y); g = door(y); L = np.asarray(g["land_ned"], float); tr = mk.Truth(y, g)
    C, info = [], {}
    pos = {s: ring(y, s, cf) for s in sorted({min(s, 2) for s in steps_of(y)})}
    for st in steps_of(y):
        P, tries = pos[min(st, 2)]; hr = np.random.default_rng([SEED, int(y[4:]), st, 1]); n = 0
        for i, (p, r, off) in enumerate(P):
            b = math.degrees(math.atan2(L[1] - p[1], L[0] - p[0]))
            yo = 0.0 if st <= 1 else (float(hr.uniform(-15, 15)) if st == 2 else float(hr.choice([-1.0, 1.0]) * hr.uniform(90, 180)))
            yaw = math.radians(((b + yo + 180.0) % 360.0) - 180.0)
            if st == 3:
                _, c = tr.cast(p, np.asarray(_yaw_quat(yaw), float))
                if (np.asarray(c) == 3).any(): continue
            C.append({"id": f"v2s{st}_{i:03d}", "pos": [float(v) for v in p], "yaw": yaw, "split": "test", "sector": -1, "band": -1, "step": st,
                      "r_m": r, "bearing_off_deg": off, "yaw_off_deg": yo, "dist3d_m": float(np.linalg.norm(p - L))}); n += 1
        info[st] = {"positions": len(P), "draws": tries, "candidates": n}
    os.makedirs(f"{OUT}/{y}", exist_ok=True); json.dump({"yard": y, "seed": SEED, "starts": C}, open(f"{OUT}/{y}/cands.json", "w"), indent=0)
    print(f"STEPPED2_CANDS {y}: {json.dumps(info)}", flush=True)


def distinct(P, d=0.5):
    k = []
    for p in P:
        if all(np.linalg.norm(np.asarray(p) - np.asarray(q)) >= d for q in k): k.append(p)
    return len(k)


def scan(G, OUT, ys):
    """counts of ring positions (steps 0 and 1 rules) per yard, for choosing control yards (spec: >= 20 at 0.5 m for both 0T and 1T)"""
    import score
    score.SY = FLAMBE_YARDS
    for y in ys.split(","):
        cf = score.clearance_fn(y); print(f"STEPPED2_SCAN {y}: step0 {len(ring(y, 0, cf)[0])} step1 {len(ring(y, 1, cf)[0])}", flush=True)


def select(G, OUT, y):
    C = json.load(open(f"{OUT}/{y}/cands.json"))["starts"]; X = {r["id"]: r for r in json.load(open(f"{OUT}/{y}/expert_cands.json"))["flights"]}
    pick, ex, info = [], [], {}
    for st in steps_of(y):
        cs = [c for c in C if c["step"] == st]; ok = [c for c in cs if X.get(c["id"], {}).get("outcome") == "success" and X[c["id"]].get("t_landed_s") is not None]
        assert len(ok) >= 1, f"{y} step {st}: no candidate passes"
        sel = ok[:NPER]; pick += sel; ex += [X[c["id"]] for c in sel]
        info[st] = {"candidates": len(cs), "expert_pass": len(ok), "tried_until": cs.index(sel[-1]) + 1, "distinct_0p5m": distinct([c["pos"] for c in sel]),
                    "mean_r_m": round(float(np.mean([c["r_m"] for c in sel])), 2), "median_t_expert_s": float(np.median([X[c["id"]]["t_landed_s"] for c in sel]))}
    if y in HELD:                                                                     # Include step 4 on held-out yards.
        v1 = json.load(open(f"{G}/stepped_v1/stepped_{y}.json"))["starts"]; e1 = {r["id"]: r for r in json.load(open(f"{G}/stepped_v1/expert_stepped_{y}.json"))["flights"]}
        s4 = [s for s in v1 if s["step"] == 4]; assert len(s4) == NPER
        pick += s4; ex += [e1[s["id"]] for s in s4]; info[4] = {"from": "stepped_v1 step 4", "n": len(s4)}
    assert len({p["id"] for p in pick}) == len(pick)
    assert len(ex) == len(pick) and all(e["id"] == p["id"] for e, p in zip(ex, pick)), "expert records do not line up with the starts"
    os.makedirs(f"{OUT}/final", exist_ok=True)
    json.dump({"yard": y, "seed": SEED, "spec": "score_1121/STEPPED_TEST.md 7d19e6b7 (v2)", "starts": pick}, open(f"{OUT}/final/stepped2_{y}.json", "w"), indent=0)
    json.dump({"flights": ex}, open(f"{OUT}/final/expert_stepped2_{y}.json", "w"), indent=0)
    json.dump({y: [p["id"] for p in pick]}, open(f"{OUT}/final/ids_stepped2_{y}.json", "w"))
    print(f"STEPPED2_SELECT {y}: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    {"cands": cands, "select": select, "scan": scan}[sys.argv[1]](*sys.argv[2:5])
