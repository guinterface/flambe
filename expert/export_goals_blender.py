"""Export goal candidates from a yard's scene.blend (run inside Blender, no GPU):
  blender -b yards/<yard>/scene.blend --factory-startup -noaudio -P expert/export_goals_blender.py -- <out.json>
Writes every object that can be a delivery goal, with its world box in NED (X, -Y, -Z of the Blender frame):
  windows ('glass' objects of the house), yard props (pass_index 12), furniture (mailbox), vehicles, the garage door.
The front door itself comes from survey.json (goals.py adds it)."""
import bpy, json, sys, re
from mathutils import Vector

SEM = {"door": 8, "window": 9, "prop": 12, "furniture": 14, "vehicle": 19}
out = sys.argv[sys.argv.index("--") + 1]


def ned(v):
    return [float(v.x), float(-v.y), float(-v.z)]


def box(o):
    P = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p.x for p in P), min(p.y for p in P), min(p.z for p in P)))
    hi = Vector((max(p.x for p in P), max(p.y for p in P), max(p.z for p in P)))
    a, b = ned(lo), ned(hi)   # NED flips y and z, so sort again
    return [min(a[i], b[i]) for i in range(3)], [max(a[i], b[i]) for i in range(3)]


items = []
for o in bpy.data.objects:
    if o.type != "MESH" or not o.users_collection:
        continue
    col = o.users_collection[0].name
    base = re.sub(r"(\.\d+|_\d{2,}_?)$", "", o.name)
    kind = None
    if col == "house" and o.pass_index == SEM["window"] and base.startswith("glass"):
        kind = "window"
    elif col == "house" and "garage" in o.name.lower() and o.pass_index == SEM["door"]:
        kind = "garage_door"
    elif col in ("props", "house") and o.pass_index == SEM["prop"]:
        kind = "prop"
    elif col == "props" and o.pass_index == SEM["furniture"] and base.startswith("mail_box"):
        kind = "mailbox"
    elif col == "props" and o.pass_index == SEM["vehicle"]:
        kind = "vehicle"
    if kind is None:
        continue
    lo, hi = box(o)
    items.append({"kind": kind, "asset": base, "object": o.name, "lo_ned": lo, "hi_ned": hi})
json.dump({"items": items}, open(out, "w"), indent=1)
print(f"[export_goals] {len(items)} candidates -> {out}")
