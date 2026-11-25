"""Shared helpers for the synthetic-yard generator (runs inside Blender's Python).

Frame convention (chosen so that the authored scene IS the FiGS Gaussian-splat frame):
  Blender world:  X = north (street -> house direction), Y = west, Z = up, metres.
  FiGS NED:       (x_n, y_e, z_d) = (X, -Y, -Z).   nerfstudio/GS frame = T_w2g @ NED = (X, Y, Z) = Blender frame.
  The street runs east-west south of the lot (negative X); the house sits at positive X.
"""
import bpy, bmesh, math, os, random, zlib
import numpy as np
from mathutils import Vector, Matrix, Euler

FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
FLAMBE_YARDS = os.environ.get("FLAMBE_YARDS", os.path.join(FLAMBE_ROOT, "synth_yards"))
ROOT = FLAMBE_YARDS
PH = ROOT + "/assets/polyhaven"
MARBLE_ASSETS = os.environ.get("MARBLE_ASSETS", os.path.join(FLAMBE_ROOT, "marble_worlds", "assets"))

# semantic classes -> pass_index (0 is reserved for 'unlabelled')
SEM = {"lawn": 1, "street": 2, "sidewalk": 3, "driveway": 4, "path": 5, "wall": 6, "roof": 7, "door": 8, "window": 9,
       "vegetation": 10, "fence": 11, "prop": 12, "neighbor": 13, "furniture": 14, "porch": 15, "steps": 16, "soil": 17,
       "trim": 18, "vehicle": 19}
SEM_NAMES = {v: k for k, v in SEM.items()}
# classes that are solid obstacles for the occupancy truth (everything except flat ground surfaces)
SOLID = {SEM[k] for k in ("wall", "roof", "door", "window", "vegetation", "fence", "prop", "neighbor", "furniture", "porch", "steps", "trim", "vehicle")}


class RNG:
    def __init__(self, seed):
        self.r = random.Random(seed); self.np = np.random.RandomState(seed)
    def u(self, a, b): return self.r.uniform(a, b)
    def i(self, a, b): return self.r.randint(a, b)
    def p(self, prob): return self.r.random() < prob
    def choice(self, seq): return self.r.choice(list(seq))
    def shuffle(self, seq): s = list(seq); self.r.shuffle(s); return s
    def sub(self, tag): return RNG((self.r.getrandbits(32) ^ zlib.crc32(str(tag).encode())) & 0xFFFFFFFF)


def ned(v):
    """Blender (X north, Y west, Z up) -> NED list."""
    return [float(v[0]), float(-v[1]), float(-v[2])]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'; sc.unit_settings.scale_length = 1.0
    return sc


def collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def link(obj, col):
    for c in list(obj.users_collection): c.objects.unlink(obj)
    col.objects.link(obj); return obj


def mesh_obj(name, verts, faces, col, mat=None, sem=0, smooth=False):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces]); me.update()
    if smooth:
        for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new(name, me); link(ob, col)
    if mat is not None: me.materials.append(mat)
    ob.pass_index = sem; return ob


def box(name, center, size, col, mat=None, sem=0, rot_z=0.0):
    """Axis-aligned box (optionally yawed about Z) with metric object coordinates (scale applied)."""
    cx, cy, cz = center; sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz), (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    ob = mesh_obj(name, verts, faces, col, mat, sem)
    ob.location = (cx, cy, cz); ob.rotation_euler = (0, 0, rot_z); return ob


def prism_gable(name, center, size, pitch_h, col, mat=None, sem=0, ridge_axis='y'):
    """A gable roof volume: rectangular base (sx, sy) at z=center.z-? -> ridge pitch_h above the eave. ridge runs along ridge_axis."""
    cx, cy, cz = center; sx, sy, sz = size; hx, hy = sx / 2, sy / 2
    if ridge_axis == 'y':
        verts = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0), (0, -hy, pitch_h), (0, hy, pitch_h)]
        faces = [(0, 3, 2, 1), (0, 1, 4), (2, 3, 5), (1, 2, 5, 4), (3, 0, 4, 5)]
    else:
        verts = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0), (-hx, 0, pitch_h), (hx, 0, pitch_h)]
        faces = [(0, 3, 2, 1), (0, 4, 3), (1, 2, 5), (0, 1, 5, 4), (2, 3, 4, 5)]
    ob = mesh_obj(name, verts, faces, col, mat, sem); ob.location = (cx, cy, cz); return ob


def prism_hip(name, center, size, pitch_h, col, mat=None, sem=0):
    cx, cy, cz = center; sx, sy, sz = size; hx, hy = sx / 2, sy / 2
    r = max(0.0, (max(sx, sy) - min(sx, sy)) / 2)
    if sx >= sy: ridge = [(-r, 0, pitch_h), (r, 0, pitch_h)]
    else: ridge = [(0, -r, pitch_h), (0, r, pitch_h)]
    verts = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0), ridge[0], ridge[1]]
    if sx >= sy: faces = [(0, 3, 2, 1), (0, 1, 5, 4), (2, 3, 4, 5), (1, 2, 5), (3, 0, 4)]
    else: faces = [(0, 3, 2, 1), (0, 1, 4), (1, 2, 5, 4), (2, 3, 5), (3, 0, 4, 5)]
    ob = mesh_obj(name, verts, faces, col, mat, sem); ob.location = (cx, cy, cz); return ob


def cylinder(name, center, radius, height, col, mat=None, sem=0, n=24, axis='z'):
    verts = []; faces = []
    for i in range(n):
        a = 2 * math.pi * i / n; verts.append((radius * math.cos(a), radius * math.sin(a), -height / 2))
    for i in range(n):
        a = 2 * math.pi * i / n; verts.append((radius * math.cos(a), radius * math.sin(a), height / 2))
    for i in range(n):
        j = (i + 1) % n; faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1))); faces.append(tuple(range(n, 2 * n)))
    ob = mesh_obj(name, verts, faces, col, mat, sem, smooth=True); ob.location = center
    if axis == 'x': ob.rotation_euler = (0, math.pi / 2, 0)
    if axis == 'y': ob.rotation_euler = (math.pi / 2, 0, 0)
    return ob


def plane(name, center, size, col, mat=None, sem=0, rot_z=0.0, subdiv=1):
    cx, cy, cz = center; sx, sy = size; hx, hy = sx / 2, sy / 2
    verts, faces = [], []
    for j in range(subdiv + 1):
        for i in range(subdiv + 1):
            verts.append((-hx + sx * i / subdiv, -hy + sy * j / subdiv, 0))
    for j in range(subdiv):
        for i in range(subdiv):
            a = j * (subdiv + 1) + i; faces.append((a, a + 1, a + subdiv + 2, a + subdiv + 1))
    ob = mesh_obj(name, verts, faces, col, mat, sem); ob.location = (cx, cy, cz); ob.rotation_euler = (0, 0, rot_z); return ob


def boolean_cut(target, cutters):
    """Subtract cutter objects from target (exact solver), apply, delete cutters."""
    for c in cutters:
        m = target.modifiers.new("cut", 'BOOLEAN'); m.operation = 'DIFFERENCE'; m.solver = 'EXACT'; m.object = c
        c.hide_render = True; c.hide_viewport = True
    bpy.context.view_layer.objects.active = target
    with bpy.context.temp_override(object=target, active_object=target, selected_objects=[target]):
        for m in list(target.modifiers): bpy.ops.object.modifier_apply(modifier=m.name)
    for c in cutters: bpy.data.objects.remove(c, do_unlink=True)


def set_pass(objs, sem):
    for o in objs: o.pass_index = sem


def look_at_matrix(eye, target, up=(0, 0, 1)):
    """Camera-to-world (OpenGL: -Z forward, +Y up) looking from eye to target."""
    eye = Vector(eye); f = (Vector(target) - eye).normalized()
    if abs(f.dot(Vector(up))) > 0.999: up = (1, 0, 0)
    r = f.cross(Vector(up)).normalized(); u = r.cross(f).normalized()
    m = Matrix.Identity(4)
    m.col[0][:3] = r; m.col[1][:3] = u; m.col[2][:3] = -f; m.col[3][:3] = eye
    return m


def yaw_pitch_matrix(eye, yaw, pitch):
    """Camera-to-world from position, yaw (rad, 0 = +X north, CCW positive toward +Y west) and pitch (rad, negative = down)."""
    f = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
    return look_at_matrix(eye, Vector(eye) + f)
