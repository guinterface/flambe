"""PBR materials from the Poly Haven library (box-projected, metric tiling, no UVs needed) + simple procedural materials."""
import bpy, json, os, math
from .common import PH, ROOT

_MANIFEST = None
_CACHE = {}


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        p = PH + "/MANIFEST.json"
        _MANIFEST = json.load(open(p)) if os.path.exists(p) else {}
    return _MANIFEST


def textures_by_role(role):
    """Texture ids with the given role that are actually on disk."""
    out = []
    for aid, rec in manifest().items():
        if rec.get("kind") == "texture" and rec.get("role") == role and "Diffuse" in rec.get("maps", {}):
            if os.path.exists(os.path.join(PH, rec["maps"]["Diffuse"])): out.append(aid)
    return sorted(out)


def _img(path, srgb):
    img = bpy.data.images.get(os.path.basename(path))
    if img is None:
        img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
    return img


def pbr(aid, tile_m=None, tint=None, rough_mult=1.0, name=None, bump=0.0):
    """Material from a Poly Haven texture id. tile_m = physical size of one texture repeat (metres)."""
    key = (aid, tile_m, tint, rough_mult, name, bump)
    if key in _CACHE: return _CACHE[key]
    rec = manifest()[aid]; maps = {k: os.path.join(PH, v) for k, v in rec["maps"].items()}
    if tile_m is None:
        tile_m = 2.0
    mat = bpy.data.materials.new(name or f"pbr_{aid}"); mat.use_nodes = True
    nt = mat.node_tree; nodes = nt.nodes; links = nt.links
    bsdf = nodes["Principled BSDF"]; out = nodes["Material Output"]
    tc = nodes.new("ShaderNodeTexCoord"); mp = nodes.new("ShaderNodeMapping"); mp.vector_type = 'TEXTURE'
    mp.inputs["Scale"].default_value = (tile_m, tile_m, tile_m)
    links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tex(path, srgb):
        n = nodes.new("ShaderNodeTexImage"); n.image = _img(path, srgb); n.projection = 'BOX'; n.projection_blend = 0.25
        n.interpolation = 'Linear'; links.new(mp.outputs["Vector"], n.inputs["Vector"]); return n

    d = tex(maps["Diffuse"], True)
    if tint is not None:
        mix = nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs[0].default_value = 1.0
        links.new(d.outputs["Color"], mix.inputs[6]); mix.inputs[7].default_value = (*tint, 1.0)
        links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    else:
        links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
    if "Rough" in maps:
        r = tex(maps["Rough"], False)
        if rough_mult != 1.0:
            mm = nodes.new("ShaderNodeMath"); mm.operation = 'MULTIPLY'; mm.inputs[1].default_value = rough_mult; mm.use_clamp = True
            links.new(r.outputs["Color"], mm.inputs[0]); links.new(mm.outputs[0], bsdf.inputs["Roughness"])
        else:
            links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
    elif "arm" in maps:
        a = tex(maps["arm"], False); sep = nodes.new("ShaderNodeSeparateColor"); links.new(a.outputs["Color"], sep.inputs[0])
        links.new(sep.outputs[1], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.8
    if "nor_gl" in maps:
        n = tex(maps["nor_gl"], False); nm = nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0
        links.new(n.outputs["Color"], nm.inputs["Color"]); links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if bump > 0 and "Displacement" in maps:
        dsp = tex(maps["Displacement"], False); dn = nodes.new("ShaderNodeDisplacement"); dn.inputs["Scale"].default_value = bump
        dn.inputs["Midlevel"].default_value = 0.5; links.new(dsp.outputs["Color"], dn.inputs["Height"]); links.new(dn.outputs[0], out.inputs["Displacement"])
        mat.displacement_method = 'BUMP'
    bsdf.inputs["Specular IOR Level"].default_value = 0.4
    _CACHE[key] = mat; return mat


def flat(name, rgb, rough=0.6, metallic=0.0, spec=0.5, emission=0.0):
    key = ("flat", name, tuple(rgb), rough, metallic, spec, emission)
    if key in _CACHE: return _CACHE[key]
    mat = bpy.data.materials.new(name); mat.use_nodes = True; b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic; b.inputs["Specular IOR Level"].default_value = spec
    if emission > 0:
        b.inputs["Emission Color"].default_value = (*rgb, 1.0); b.inputs["Emission Strength"].default_value = emission
    _CACHE[key] = mat; return mat


def glass(name="window_glass", tint=(0.35, 0.42, 0.45)):
    key = ("glass", name)
    if key in _CACHE: return _CACHE[key]
    mat = bpy.data.materials.new(name); mat.use_nodes = True; b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tint, 1.0); b.inputs["Roughness"].default_value = 0.05
    b.inputs["Metallic"].default_value = 0.6; b.inputs["Specular IOR Level"].default_value = 1.0
    _CACHE[key] = mat; return mat


def leaf(name="hedge_leaf", rgb=(0.08, 0.22, 0.06)):
    """Procedural dense-foliage look for hedges: noise-modulated green with soft bump."""
    key = ("leaf", name, rgb)
    if key in _CACHE: return _CACHE[key]
    mat = bpy.data.materials.new(name); mat.use_nodes = True; nt = mat.node_tree; n = nt.nodes; l = nt.links
    b = n["Principled BSDF"]; tc = n.new("ShaderNodeTexCoord"); nz = n.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60
    nz.inputs["Detail"].default_value = 8; nz.inputs["Roughness"].default_value = 0.7
    l.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = n.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].color = (rgb[0] * 0.4, rgb[1] * 0.4, rgb[2] * 0.4, 1)
    ramp.color_ramp.elements[1].color = (rgb[0] * 1.6, rgb[1] * 1.5, rgb[2] * 1.2, 1)
    l.new(nz.outputs["Fac"], ramp.inputs[0]); l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    bump = n.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.8; bump.inputs["Distance"].default_value = 0.05
    l.new(nz.outputs["Fac"], bump.inputs["Height"]); l.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.75
    _CACHE[key] = mat; return mat


def lane_paint(name="lane_paint"):
    return flat(name, (0.85, 0.82, 0.6), rough=0.7)


def hdri_mean_luminance(img):
    """Mean luminance of the upper hemisphere (what lights the ground), from a downsampled read of the pixels."""
    import numpy as np
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(px); px = px.reshape(h, w, 4)
    top = px[h // 2:, ::4, :3]                      # rows are bottom-up in Blender: upper half = sky
    lum = 0.2126 * top[..., 0] + 0.7152 * top[..., 1] + 0.0722 * top[..., 2]
    return float(np.clip(lum, 0, 50).mean())


def world_hdri(path, rotation_deg=0.0, strength=1.0, target_lum=0.9):
    """HDRI sky; strength is normalised so the upper-hemisphere mean luminance equals target_lum * strength
    (overcast maps come out much darker than clear ones; a real camera would auto-expose)."""
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; n = nt.nodes; l = nt.links
    for x in list(n): n.remove(x)
    out = n.new("ShaderNodeOutputWorld"); bg = n.new("ShaderNodeBackground"); env = n.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path, check_existing=True)
    try:
        ml = hdri_mean_luminance(env.image); strength = strength * target_lum / max(ml, 1e-3); strength = float(min(max(strength, 0.3), 5.0))
        print(f"[sky] {os.path.basename(path)} mean upper lum {ml:.3f} -> strength {strength:.2f}", flush=True)
    except Exception as e:
        print("[sky] luminance normalisation failed:", e)
    mp = n.new("ShaderNodeMapping"); tc = n.new("ShaderNodeTexCoord"); mp.inputs["Rotation"].default_value = (0, 0, math.radians(rotation_deg))
    l.new(tc.outputs["Generated"], mp.inputs["Vector"]); l.new(mp.outputs["Vector"], env.inputs["Vector"])
    l.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = strength; l.new(bg.outputs[0], out.inputs[0])
    return w


def world_sky(sun_elev_deg=45, sun_rot_deg=120, strength=1.0):
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; n = nt.nodes; l = nt.links
    for x in list(n): n.remove(x)
    out = n.new("ShaderNodeOutputWorld"); bg = n.new("ShaderNodeBackground"); sky = n.new("ShaderNodeTexSky")
    sky.sky_type = 'NISHITA'; sky.sun_elevation = math.radians(sun_elev_deg); sky.sun_rotation = math.radians(sun_rot_deg)
    sky.sun_intensity = 1.0; sky.sun_size = math.radians(0.6); sky.altitude = 50; sky.air_density = 1.0; sky.dust_density = 1.5
    l.new(sky.outputs[0], bg.inputs["Color"]); bg.inputs["Strength"].default_value = strength * 0.03; l.new(bg.outputs[0], out.inputs[0])
    return w
