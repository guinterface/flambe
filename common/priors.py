"""Fit goal location priors in the house frame using training yards."""
import json, re, os
from pathlib import Path
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))

ROOT = Path(os.path.join(FLAMBE_YARDS, "yards"))
HELD = {1, 6, 7, 8}                 # base scenes held out from the start
HAS_FLEET = {10, 20, 30}            # seed ends in 0 but already has expert fleets: treated as TRAINING


def base(y: str):
    m = re.match(r"syn_(\d+)$", y)
    if not m:
        return None
    n = int(m.group(1))
    return n - 100 if n > 100 else n   # aliases (syn_113 = syn_013 flown higher) share their base scene's split


def held_out(y: str) -> bool:
    b = base(y)
    return b is None or b in HELD or (b % 10 == 0 and b not in HAS_FLEET)


def kmeans(X, k, it=50, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return C, np.bincount(lab, minlength=k) / len(X)


def key_of(g):
    if g["kind"] == "door":
        return "door"
    if g["kind"] == "window":
        return "window/" + g["side"]
    return "object/" + g["name"].split(" at the ")[0]


def fit(pts):
    out = {}
    for key, P in pts.items():
        X = np.asarray(P, float); k = min(3, len(X))
        C, w = kmeans(X, k)
        out[key] = [{"w": float(wi), "ab": c.tolist()} for c, wi in zip(C, w) if wi > 0]
        out[key + "#n"] = len(X)
    return out


def main():
    per = {}
    for d in sorted(ROOT.iterdir()):
        f = d / "goals.json"
        if held_out(d.name) or not f.exists():
            continue
        for g in json.load(open(f))["goals"]:
            per.setdefault(base(d.name), []).append((key_of(g), g["house_frame"]))

    def pool(exclude=None):
        pts = {}
        for b, items in per.items():
            if b == exclude:
                continue
            for k, ab in items:
                pts.setdefault(k, []).append(ab); pts.setdefault("_any", []).append(ab)
        return pts
    # leave-one-scene-out: a training yard's expert never searches with priors that saw its own goals
    out = {"all": fit(pool()), "loo": {}}
    for d in sorted(ROOT.iterdir()):
        b = base(d.name)
        if b in per:
            out["loo"][d.name] = fit(pool(exclude=b))
    json.dump(out, open(ROOT.parent / "goal_priors.json", "w"), indent=1)
    print("scenes:", len(per), {k: v for k, v in out["all"].items() if k.endswith("#n")})


if __name__ == "__main__":
    main()
