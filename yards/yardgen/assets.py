"""Imported glTF assets (Poly Haven + marble_worlds/assets): cached import, linked-duplicate instancing, footprint-aware placement."""
import bpy, os, json, math
from mathutils import Vector, Matrix
from .common import PH, MARBLE_ASSETS, link, collection
from .materials import manifest

_LIB = {}       # asset id -> source object (hidden, in collection 'lib')
_STATS = {}
# assets that ship several copies in one file: keep only the largest connected part
KEEP_LARGEST = {"fire_hydrant", "exterior_aircon_unit", "rollershutter_window_01", "rollershutter_window_02", "fir_tree_01",
                "pine_sapling_medium", "fir_sapling_medium", "fir_sapling", "wooden_barrels_01", "rock_moss_set_01", "modular_street_seating"}
# vegetation assets that ship a flat ground patch under the trunk: drop faces that never rise above 12 cm
STRIP_GROUND = {"island_tree_01", "island_tree_02", "island_tree_03", "tree_small_02", "jacaranda_tree", "fir_tree_01", "pine_tree_01",
                "searsia_lucida", "searsia_burchellii", "shrub_02", "wild_rooibos_bush", "tree_stump_01"}


def _find_gltf(aid):
    rec = manifest().get(aid)
    if rec and rec.get("kind") == "model":
        p = os.path.join(PH, rec["path"])
        if os.path.exists(p): return p
    d = os.path.join(MARBLE_ASSETS, aid)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith((".gltf", ".glb")): return os.path.join(d, f)
    return None


def available(ids):
    return [a for a in ids if _find_gltf(a) is not None]


def _import(aid):
    """Import once; join into one mesh object; origin at footprint centre on the ground (z = min). Returns the source object."""
    if aid in _LIB: return _LIB[aid]
    path = _find_gltf(aid)
    if path is None: raise FileNotFoundError(aid)
    lib = collection("lib")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    new_names = [o.name for o in new]
    meshes = [o for o in new if o.type == 'MESH']
    for o in new:
        for c in list(o.users_collection): c.objects.unlink(o)
        lib.objects.link(o)
    # apply hierarchy transforms, join meshes
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    with bpy.context.temp_override(selected_objects=meshes, active_object=meshes[0], selected_editable_objects=meshes):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if len(meshes) > 1: bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    src.parent = None; src.matrix_world = Matrix.Identity(4)
    for nm in new_names:
        o = bpy.data.objects.get(nm)
        if o is not None and o is not src: bpy.data.objects.remove(o, do_unlink=True)
    # shift geometry so origin = (bbox centre x, y, min z)
    vs = [Vector(c) for c in src.bound_box]
    mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    off = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
    src.data.transform(Matrix.Translation(-off)); src.data.update()
    src.name = f"lib_{aid}"; src.hide_render = True; src.hide_viewport = True; src.hide_set(True)
    _STATS[aid] = {"size": [mx.x - mn.x, mx.y - mn.y, mx.z - mn.z], "faces": len(src.data.polygons)}
    if aid in KEEP_LARGEST: _keep_largest_part(src)
    if aid in STRIP_GROUND: _strip_ground(src)
    _LIB[aid] = src; return src


def _keep_largest_part(src):
    import bmesh
    bm = bmesh.new(); bm.from_mesh(src.data); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    seen = set(); comps = []
    for f in bm.faces:
        if f.index in seen: continue
        stack = [f]; comp = []
        while stack:
            g = stack.pop()
            if g.index in seen: continue
            seen.add(g.index); comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen: stack.append(h)
        comps.append(comp)
    if len(comps) > 1:
        # merge parts whose bounding boxes overlap the largest (an asset's own sub-meshes), delete the rest
        big = max(comps, key=len)
        def bbox(c):
            xs = [v.co.x for f in c for v in f.verts]; ys = [v.co.y for f in c for v in f.verts]
            return min(xs), max(xs), min(ys), max(ys)
        bx = bbox(big); keep = set(id(f) for f in big)
        for c in comps:
            if c is big: continue
            b = bbox(c)
            if b[0] < bx[1] and b[1] > bx[0] and b[2] < bx[3] and b[3] > bx[2]:
                keep.update(id(f) for f in c)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if id(f) not in keep], context='FACES')
        bm.to_mesh(src.data); src.data.update()
    bm.free()
    _recentre(src)


def _strip_ground(src, zmax=0.12):
    import bmesh
    bm = bmesh.new(); bm.from_mesh(src.data)
    low = [f for f in bm.faces if max(v.co.z for v in f.verts) < zmax]
    if low and len(low) < len(bm.faces):
        bmesh.ops.delete(bm, geom=low, context='FACES'); bm.to_mesh(src.data); src.data.update()
    bm.free()
    _recentre(src)


def _recentre(src):
    vs = [Vector(c) for c in src.bound_box]
    mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    off = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
    src.data.transform(Matrix.Translation(-off)); src.data.update()
    aid = src.name[4:] if src.name.startswith("lib_") else src.name
    _STATS[aid] = {"size": [mx.x - mn.x, mx.y - mn.y, mx.z - mn.z], "faces": len(src.data.polygons)}


def size(aid):
    _import(aid); return _STATS[aid]["size"]


def place(aid, col, location, yaw=0.0, target_h=None, scale=None, sem=0, name=None):
    """Linked duplicate of the library mesh at location (ground point), yaw about Z, scaled to target height or by scale."""
    src = _import(aid)
    ob = bpy.data.objects.new(name or f"{aid}_{len(col.objects):03d}", src.data); link(ob, col)
    s = 1.0
    if target_h is not None: s = target_h / max(1e-6, _STATS[aid]["size"][2])
    if scale is not None: s = scale
    ob.scale = (s, s, s); ob.rotation_euler = (0, 0, yaw); ob.location = location
    ob.pass_index = sem; ob.hide_render = False; ob.hide_viewport = False
    return ob


def footprint(aid, s=1.0):
    sz = _STATS[aid]["size"] if aid in _STATS else size(aid)
    return sz[0] * s, sz[1] * s
