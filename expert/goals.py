"""Goal set for one generated yard: which places the drone can be asked to reach, where it delivers, and how it is told.

Runs in figs-sv (numpy, scipy) next to visual_expert_syn.py. Input: yards/<yard>/{survey.json, recon/truth.npz,
goals_raw.json (export_goals_blender.py)}. Output: yards/<yard>/goals.json, one record per goal:
  id, kind, phrase_key, face_ned, normal_ned, hover_ned, land_ned, standoff_ned, points_ned (for visibility tests),
  instructions {train: [...], test: [...]}, reachable_starts (ids of valid starts with a route), house_frame (a, b) position.

Goal selection rules:
- the door's delivery point is the existing expert's (1 m in front of the face, land 0.5 m above the step);
- other goals: hover 1.2 m in front of the object's face that a planner can reach, at 1.2 m above ground (windows: at the
  window's centre height, clamped to the flight band), no descent for windows, descent to 0.5 m for objects on the ground;
- a goal is kept only if its phrase is unique in the yard (two trash cans: both get a side qualifier or are dropped) and a
  route exists from at least half of the valid starts;
- instructions come from templates; two templates per kind are held out for the phrasing test and never used in training.
"""
from __future__ import annotations

import json, math, os, re, sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
FLAMBE_TERRA = os.environ.get("FLAMBE_TERRA", os.path.join(FLAMBE_ROOT, "terra"))

sys.path[:0] = [*LOCAL_MODULE_PATHS, FLAMBE_TERRA]
import visual_expert_syn as ve   # noqa: E402

NAMES = {  # asset id -> what a person would call it
    "metal_trash_can": "trash can", "trashbag": "trash bag", "wooden_picnic_table": "picnic table",
    "outdoor_table_chair_set_01": "table and chairs", "stone_fire_pit": "fire pit", "wooden_ladder": "ladder",
    "watering_can_metal_01": "watering can", "garden_gnome": "garden gnome", "wooden_crate_01": "wooden crate",
    "plastic_crate_01": "plastic crate", "wicker_basket_01": "wicker basket", "old_tyre": "old tyre",
    "cement_bag": "cement bags", "compost_bags": "compost bags", "propane_tank": "propane tank",
    "garden_sprinkler_01": "sprinkler", "exterior_aircon_unit": "air conditioner", "planter_box_02": "planter box",
    "garden_hose_wall_mounted": "garden hose", "covered_car": "covered car", "mail_box": "mailbox",
    "garage_door": "garage door", "potted_plant": "potted plant", "utility_box": "utility box", "cardboard_box": "cardboard box",
    "planter_box": "planter box",
}
TEMPLATES = {  # last two of each list are the held-out phrasing set
    "door": ["deliver to the front door", "go to the front door", "land at the door", "fly to the entrance",
             "drop it at the front door", "take it to the door"],
    "window": ["fly to the {q} window", "go near the {q} window", "hover by the {q} window", "approach the {q} window",
               "head to the {q} window", "stop next to the {q} window"],
    "object": ["deliver next to the {n}", "go to the {n}", "land by the {n}", "fly to the {n}",
               "drop it next to the {n}", "take it to the {n}"],
}
MIN_START_DIST = 8.0   # m: a start nearer than this to the goal is trivial for that goal
MIN_SIZE = 0.35        # m, largest box side a goal object needs
HOVER_UP = 1.2          # m above lot ground for non-door goals
HOVER_GAP = 1.2         # m from the object's face
STANDOFF_GAP = 3.0      # m, where the straight approach starts (as the door expert)
WIN_Z = (-2.3, -1.0)    # NED band for window hover heights relative to ground (above ground 1.0 to 2.3 m)


def house_frame(survey):
    """Origin at the house centre; a = toward the street (front), b = lateral. Normalised by the half extents."""
    hx0, hx1, hy0, hy1 = survey["house_rect_ned"][:4]
    c = np.array([(hx0 + hx1) / 2, (hy0 + hy1) / 2]); half = np.array([abs(hx1 - hx0) / 2, abs(hy1 - hy0) / 2])
    sx = float(np.mean(survey["street_x_ned"]))
    front = np.array([np.sign(sx - c[0]) or -1.0, 0.0])       # streets run along y in every generated lot
    lat = np.array([-front[1], front[0]])
    def to_hf(p):
        d = np.asarray(p[:2], float) - c
        return [float(d @ front / half[0]), float(d @ lat / half[1])]
    return to_hf, front, lat, c


def side_word(ab):
    a, b = ab
    if abs(a) >= abs(b):
        return "front" if a > 0 else "back"
    return "left" if b > 0 else "right"


def face_points(lo, hi, normal, n=3):
    """grid of points on the box face that looks along `normal` (for visibility), plus the centre"""
    lo, hi = np.asarray(lo), np.asarray(hi); c = (lo + hi) / 2; ext = (hi - lo) / 2
    nx, ny = normal[0], normal[1]
    face_c = c + np.array([nx * ext[0], ny * ext[1], 0.0])
    t = np.array([-ny, nx, 0.0]); w = abs(t[0]) * ext[0] + abs(t[1]) * ext[1]
    pts = [face_c + t * w * a + np.array([0, 0, ext[2] * b]) for a in np.linspace(-0.7, 0.7, n) for b in np.linspace(-0.7, 0.7, n)]
    return face_c, [p.tolist() for p in pts]


def box_points(lo, hi):
    """points spread over the whole box (corners pulled 15% in, face centres, centre): visible from any side"""
    lo, hi = np.asarray(lo), np.asarray(hi); c = (lo + hi) / 2; e = (hi - lo) / 2 * 0.85
    pts = [c + e * np.array(s) for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))]
    pts += [c + e * np.array(s) for s in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1))] + [c]
    return [p.tolist() for p in pts]


def _occ(yard, mask, xy):
    i = tuple(np.clip(yard.to_cell(xy), 0, np.array(mask.shape) - 1))
    return bool(mask[i])


def candidate_normals(k=16):
    return [np.array([math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k), 0.0]) for i in range(k)]


def reachable(yard, mask, standoff_xy, starts, max_starts=12):
    c1 = ve.nearest_free(mask, yard.to_cell(standoff_xy))
    if c1 is None:
        return []
    ok = []
    for st in starts[:max_starts]:
        c0 = ve.nearest_free(mask, yard.to_cell(st["pos_ned"][:2]))
        if c0 is not None and ve.astar(mask, c0, c1) is not None:
            ok.append(st["id"])
    return ok


def build(yard_dir: Path) -> dict:
    yard = ve.Yard(yard_dir); s = yard.survey
    raw = json.load(open(yard_dir / "goals_raw.json"))["items"] if (yard_dir / "goals_raw.json").exists() else []
    to_hf, front, lat, hc = house_frame(s)
    gz = float(s["ground_z_ned"])
    valid = [st for st in s["starts"] if st.get("valid")]
    rng = np.random.default_rng(0); rng.shuffle(valid)
    goals = []

    # the door: exactly the existing expert's geometry
    d_hf = to_hf(yard.face)
    goals.append({"id": "door", "kind": "door", "phrase_key": "door", "name": "front door",
                  "face_ned": yard.face.tolist(), "normal_ned": yard.n.tolist(), "hover_ned": yard.hover.tolist(),
                  "land_ned": yard.goal.tolist(), "standoff_ned": yard.standoff.tolist(),
                  "points_ned": [list(p) for p in _door_pts(yard)], "house_frame": d_hf, "center_ned": yard.face.tolist(),
                  "door_facing": s.get("door_facing")})

    hover_z = gz - HOVER_UP
    fly_mask = yard.slice_mask(getattr(yard, "plan_z", None) or yard.cruise_z)     # the slice routes are flown on
    for it in raw:
        lo, hi = np.asarray(it["lo_ned"]), np.asarray(it["hi_ned"]); c = (lo + hi) / 2
        if it["kind"] == "window":
            # facade normal: the thin axis of the box, pointing away from the house centre
            ext = hi - lo; ax = 0 if ext[0] < ext[1] else 1
            n = np.zeros(3); n[ax] = np.sign(c[ax] - hc[ax]) or 1.0
            normals = [n]; z = float(np.clip(c[2], gz + WIN_Z[0], gz + WIN_Z[1])); name = None
        else:
            normals = candidate_normals(); z = hover_z
            a = re.sub(r"^h_", "", re.sub(r"_?\d+_?$", "", it["asset"].rstrip("_")))
            name = NAMES.get(a) or NAMES.get(it["asset"].rstrip("_")) or a.replace("_", " ")
            if max(hi - lo) < MIN_SIZE:
                continue      # too small to see from the street (e.g. a 0.2 m tap): not a fair camera goal
        hov_mask = yard.slice_mask(z)
        faces = []
        for n in normals:
            face_c, _ = face_points(lo, hi, n)
            hov = face_c[:2] + HOVER_GAP * n[:2]; sto = face_c[:2] + STANDOFF_GAP * n[:2]
            if _occ(yard, hov_mask, hov) or _occ(yard, hov_mask, sto) or _occ(yard, fly_mask, sto):
                continue
            if not ve.line_free(hov_mask, sto, hov, yard.cell, yard.lo[:2]):
                continue      # the straight approach from standoff to hover must be clear at hover height
            ok = reachable(yard, fly_mask | hov_mask, sto, valid)   # Use the same union of slices as the expert planner.
            if len(ok) < max(1, len(valid[:12]) // 2):
                continue
            land_z = z if it["kind"] == "window" else gz - 0.5
            faces.append({"normal_ned": n.tolist(), "face_ned": face_c.tolist(), "hover_ned": [*hov.tolist(), z],
                          "land_ned": [*hov.tolist(), land_z], "standoff_ned": [*sto.tolist(), z], "reachable_starts": ok})
        if not faces:
            continue
        # keep faces at least 45 deg apart (16 candidates collapse to the distinct sides of the object)
        dist = []
        for f in faces:
            if all(np.dot(f["normal_ned"], g["normal_ned"]) < 0.7 for g in dist):
                dist.append(f)
        f0 = dist[0]
        goals.append({"id": f"{it['kind']}:{it['object']}", "kind": it["kind"], "asset": it["asset"], "name": name,
                      "center_ned": c.tolist(), "lo_ned": lo.tolist(), "hi_ned": hi.tolist(), "faces": dist,
                      **{k: f0[k] for k in ("face_ned", "normal_ned", "hover_ned", "land_ned", "standoff_ned")},
                      "points_ned": box_points(lo, hi), "house_frame": to_hf(c)})

    # phrases: windows by facade side and left-to-right order; objects by name, qualified by side when repeated
    for g in goals[1:]:
        g["side"] = side_word(g["house_frame"])
    by = {}
    for g in goals[1:]:
        key = ("window", g["side"]) if g["kind"] == "window" else ("object", g["name"])
        by.setdefault(key, []).append(g)
    keep = [goals[0]]
    for (k, v), gs in by.items():
        if k == "window":
            nrm = np.asarray(gs[0]["normal_ned"][:2]); right = np.array([nrm[1], -nrm[0]])
            gs.sort(key=lambda g: float(np.asarray(g["center_ned"][:2]) @ right))   # leftmost first, as seen from outside
            where = "front" if v == "front" else "back" if v == "back" else f"{v}-side"
            ords = ["first", "second", "third", "fourth", "fifth", "sixth"]
            order = [""] if len(gs) == 1 else ["left", "right"] if len(gs) == 2 else ["leftmost", "middle", "rightmost"] if len(gs) == 3 \
                else [f"{ords[i]} from the left" for i in range(min(len(gs), len(ords)))]
            quals = [(f"{o} {where}" if "from the left" not in o else f"{where} {o}") if o else where for o in order]
            for g, q in zip(gs, quals):
                g["qualifier"] = q; g["phrase_key"] = f"window/{q}"; keep.append(g)
        else:
            if len(gs) == 1:
                gs[0]["phrase_key"] = f"object/{v}"; keep.append(gs[0])
            else:   # repeated object: keep those whose side is unique
                sides = [g["side"] for g in gs]
                for g in gs:
                    if sides.count(g["side"]) == 1:
                        g["name"] = f"{v} at the {g['side']} of the house"; g["phrase_key"] = f"object/{g['name']}"; keep.append(g)
    # starts that make a goal trivial are not used for it: closer than MIN_START_DIST, or the goal already visible at t0
    import goal_expert as gx
    gyards = {}
    for g in keep:
        gy = gyards.setdefault(g["id"], gx.GoalYard(yard_dir, g))
        pts = [np.asarray(p) for p in g["points_ned"]]
        use = []
        for st in s["starts"]:
            if not st.get("valid"):
                continue
            p0 = np.asarray(st["pos_ned"], float)
            if np.linalg.norm(p0[:2] - np.asarray(g.get("center_ned", g["face_ned"]))[:2]) < MIN_START_DIST:
                continue
            q0 = ve.quat_from_yaw(gx.start_yaw(s, st))
            if gx.visible(gy, p0, q0, pts)[0]:
                continue
            use.append(st["id"])
        g["starts"] = use
    keep = [g for g in keep if g["starts"]]
    for g in keep:
        t = TEMPLATES["door" if g["kind"] == "door" else "window" if g["kind"] == "window" else "object"]
        txt = [x.format(q=g.get("qualifier", ""), n=g.get("name", "")) for x in t]
        g["instructions"] = {"train": txt[:-2], "test": txt[-2:]}
    return {"yard": yard_dir.name, "house_center_ned": hc.tolist(), "front_ned": front.tolist(), "goals": keep}


def _door_pts(yard):
    f, n = yard.face, yard.n; t = np.array([-n[1], n[0], 0.0])
    w = float(yard.survey.get("door_width_m", 0.9)); h = float(yard.survey.get("door_height_m", 2.0))
    return [f + n * 0.05 + t * a * w + np.array([0, 0, -b * h]) for a in (-0.4, 0.0, 0.4) for b in (0.1, 0.5, 0.9)]


if __name__ == "__main__":
    root = Path(os.path.join(FLAMBE_YARDS, "yards"))
    for y in sys.argv[1:]:
        out = build(root / y)
        json.dump(out, open(root / y / "goals.json", "w"), indent=1)
        print(y, len(out["goals"]), "goals:", ", ".join(g["phrase_key"] for g in out["goals"]))
