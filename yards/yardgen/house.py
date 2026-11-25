"""Parametric detached house: solid masses with boolean openings, trim, porch, steps, roofs, garage, props on the facade.
The house faces -X (south, toward the street). Door strata: street | recessed | side | corner."""
import math
import bpy
from mathutils import Vector
from .common import SEM, box, prism_gable, prism_hip, cylinder, plane, boolean_cut, collection, link
from . import materials as M
from . import assets as A

STORY_H = 2.85
WALL_T = 0.30


def house_params(rng, stratum, lod="full"):
    p = {}
    p["stratum"] = stratum
    p["W"] = rng.u(10.5, 14.5)          # along Y (parallel to street)
    p["D"] = rng.u(8.5, 11.5)           # along X
    p["stories"] = 2 if rng.p(0.55) else 1
    p["plinth"] = rng.u(0.30, 0.65)     # floor height above ground
    p["roof"] = rng.choice(["gable", "hip", "gable", "hip"])
    p["pitch_deg"] = rng.u(22, 34)
    p["overhang"] = rng.u(0.45, 0.7)
    p["garage"] = rng.p(0.75)
    p["garage_side"] = rng.choice([-1, 1])    # +1 = +Y (west) side
    p["garage_W"] = rng.u(3.8, 6.2) if rng.p(0.5) else rng.u(6.0, 7.0)
    p["garage_D"] = rng.u(6.0, 7.5)
    p["garage_front_dx"] = rng.u(-1.5, 1.0)   # garage front relative to house front (negative = protrudes toward street)
    p["door_w"], p["door_h"] = rng.u(0.9, 1.05), rng.u(2.05, 2.2)
    p["door_color"] = rng.choice([(0.55, 0.06, 0.05), (0.05, 0.08, 0.25), (0.03, 0.03, 0.03), (0.85, 0.85, 0.82), (0.05, 0.22, 0.12), (0.45, 0.28, 0.12), (0.6, 0.45, 0.1)])
    p["trim_color"] = rng.choice([(0.92, 0.92, 0.9), (0.85, 0.85, 0.8), (0.15, 0.15, 0.15), (0.35, 0.3, 0.25)])
    p["window_w"], p["window_h"], p["sill"] = rng.u(1.0, 1.35), rng.u(1.25, 1.55), rng.u(0.85, 1.0)
    p["shutters"] = rng.p(0.45)
    p["chimney"] = rng.p(0.5)
    p["gutters"] = rng.p(0.7)
    p["porch_railing"] = rng.p(0.6)
    p["canopy"] = rng.p(0.6)
    p["recess"] = rng.u(1.6, 2.6)
    p["wing_protrude"] = rng.u(3.0, 4.5)
    p["door_side"] = rng.choice([-1, 1])      # for side/corner strata: which Y side
    p["lod"] = lod
    # texture picks
    walls = M.textures_by_role("wall"); roofs = M.textures_by_role("roof")
    p["wall_tex"] = rng.choice(walls) if walls else None
    p["wall_tile"] = rng.u(1.6, 2.6)
    p["roof_tex"] = rng.choice(roofs) if roofs else None
    p["accent_tex"] = rng.choice(walls) if walls else None
    return p


class House:
    """Builds the house; records door geometry for the survey."""

    def __init__(self, rng, p, col, front_x=10.0, cy=0.0, prefix="h"):
        self.rng, self.p, self.col, self.prefix = rng, p, col, prefix
        self.fx, self.cy = front_x, cy
        self.objs = []
        self.door = None     # dict(face=(x,y,z_floor), normal=(nx,ny), step_point, stratum, width)
        self.openings = []   # (facade, y_or_x_center, width) to avoid windows there
        self.keepout = []    # rectangles (x0,x1,y0,y1) on the ground where nothing may be placed
        self.build()

    # ---------- materials ----------
    def mats(self):
        p = self.p
        self.m_wall = M.pbr(p["wall_tex"], tile_m=p["wall_tile"]) if p["wall_tex"] else M.flat("wall", (0.7, 0.65, 0.55))
        self.m_roof = M.pbr(p["roof_tex"], tile_m=1.2) if p["roof_tex"] else M.flat("roof", (0.25, 0.22, 0.2))
        self.m_trim = M.flat("trim", p["trim_color"], rough=0.5)
        self.m_door = M.flat("door", p["door_color"], rough=0.35)
        self.m_glass = M.glass()
        self.m_curtain = M.flat("curtain", (0.82, 0.8, 0.74), rough=0.9)
        self.m_dark = M.flat("dark_interior", (0.03, 0.03, 0.03), rough=0.9)
        self.m_conc = M.pbr("brushed_concrete", tile_m=2.0) if "brushed_concrete" in M.manifest() else M.flat("concrete", (0.55, 0.55, 0.52))
        self.m_deck = M.pbr("wood_floor_deck", tile_m=2.0) if "wood_floor_deck" in M.manifest() else M.flat("deck", (0.45, 0.32, 0.2))
        self.m_garage = M.pbr("wooden_garage_door", tile_m=2.4) if "wooden_garage_door" in M.manifest() else M.flat("garage", (0.8, 0.8, 0.78))
        self.m_brick = M.pbr("red_brick_03", tile_m=1.5) if "red_brick_03" in M.manifest() else M.flat("brick", (0.5, 0.2, 0.15))
        self.m_metal = M.flat("metal_dark", (0.15, 0.15, 0.16), rough=0.4, metallic=0.8)
        self.m_mat = M.flat("doormat", (0.25, 0.2, 0.15), rough=0.95)

    # ---------- helpers ----------
    def add(self, ob):
        self.objs.append(ob); return ob

    def z_top(self, stories=None):
        return self.p["plinth"] + (stories or self.p["stories"]) * STORY_H

    # ---------- main ----------
    def build(self):
        p, rng = self.p, self.rng
        self.mats()
        fx, cy, W, D = self.fx, self.cy, p["W"], p["D"]
        x0, x1, y0, y1 = fx, fx + D, cy - W / 2, cy + W / 2
        self.main_rect = (x0, x1, y0, y1)
        ztop = self.z_top()
        # ---- plinth / foundation strip (visible band below the wall) ----
        self.add(box(f"{self.prefix}_foundation", (fx + D / 2, cy, p["plinth"] / 2 - 0.05), (D + 0.1, W + 0.1, p["plinth"] + 0.1), self.col, self.m_conc, SEM["wall"]))
        # ---- main mass ----
        mass = box(f"{self.prefix}_mass", (fx + D / 2, cy, (p["plinth"] + ztop) / 2), (D, W, ztop - p["plinth"]), self.col, self.m_wall, SEM["wall"])
        self.add(mass)
        cutters = []
        # ---- garage / wing ----
        self.wing_rect = None
        if p["garage"] or p["stratum"] == "corner":
            gs = p["garage_side"] if p["stratum"] != "corner" else p["door_side"] * -1
            gW, gD = p["garage_W"], p["garage_D"]
            gfx = fx + (p["garage_front_dx"] if p["stratum"] != "corner" else -p["wing_protrude"])
            gy0, gy1 = (y1, y1 + gW) if gs > 0 else (y0 - gW, y0)
            gcy = (gy0 + gy1) / 2
            gz = p["plinth"] + STORY_H * (1 if p["stratum"] != "corner" or not p["garage"] else 1)
            self.wing_rect = (gfx, gfx + gD, gy0, gy1)
            self.add(box(f"{self.prefix}_wing_found", (gfx + gD / 2, gcy, p["plinth"] / 2 - 0.05), (gD + 0.1, gW + 0.1, p["plinth"] + 0.1), self.col, self.m_conc, SEM["wall"]))
            wing = box(f"{self.prefix}_wing", (gfx + gD / 2, gcy, (p["plinth"] + gz) / 2), (gD, gW, gz - p["plinth"]), self.col, self.m_wall, SEM["wall"])
            self.add(wing)
            self.wing = wing; self.wing_z = gz; self.wing_side = gs; self.wing_front = gfx; self.wing_cy = gcy
            if p["garage"]:
                # garage door pocket on the wing front
                gdw = min(gW - 1.0, 2.5 if gW < 5.5 else 4.9); gdh = 2.15
                cut = box("cut_g", (gfx + 0.06, gcy, p["plinth"] + gdh / 2 - 0.02), (0.4, gdw, gdh), self.col)
                self.wing_cutters = [cut]
                self.add(box(f"{self.prefix}_garage_door", (gfx + 0.18, gcy, p["plinth"] + gdh / 2 - 0.02), (0.06, gdw - 0.04, gdh - 0.04), self.col, self.m_garage, SEM["door"]))
                self.garage_door = (gfx, gcy, gdw)
                self.frame_opening(gfx, gcy, p["plinth"] - 0.02, gdw, gdh, facing=-1, axis='x', depth=0.0)
                # driveway apron keep-out in front of the garage
                self.keepout.append((gfx - 7.0, gfx, gcy - gdw / 2 - 0.3, gcy + gdw / 2 + 0.3))
            else:
                self.wing_cutters = []
            # wing roof
            self.roof(gfx, gfx + gD, gy0, gy1, gz, kind="hip" if p["roof"] == "hip" else "gable", ridge_axis='x', name="wing_roof")
        # ---- door + porch by stratum ----
        st = p["stratum"]
        if st == "street":
            self.door_front(mass, cutters)
        elif st == "recessed":
            self.door_recessed(mass, cutters)
        elif st == "side":
            self.door_side(mass, cutters)
        elif st == "corner":
            self.door_corner(cutters)
        # ---- windows ----
        self.windows(mass, cutters)
        boolean_cut(mass, cutters)
        if self.wing_rect is not None and self.wing_cutters:
            boolean_cut(self.wing, self.wing_cutters)
        # ---- main roof ----
        self.roof(x0, x1, y0, y1, ztop, kind=p["roof"], ridge_axis='y', name="roof")
        if p["chimney"]:
            cxp = x0 + D * rng.u(0.3, 0.7); cyp = y0 + 0.9 if rng.p(0.5) else y1 - 0.9
            self.add(box(f"{self.prefix}_chimney", (cxp, cyp, ztop + 1.2), (0.7, 0.9, 3.2), self.col, self.m_brick, SEM["wall"]))
        # ---- facade props ----
        self.facade_props()

    # ---------- openings ----------
    def frame_opening(self, x, yc, zb, w, h, facing=-1, axis='x', depth=0.12, trim_t=0.09, sill=False):
        """Trim frame around an opening in a wall whose outward normal is 'facing' along 'axis'. zb = bottom z."""
        t = trim_t; d = 0.05
        if axis == 'x':
            px = x - facing * (-d / 2) if False else x + facing * d / 2
            self.add(box("fr_l", (px, yc - w / 2 - t / 2, zb + h / 2), (d, t, h + 2 * t), self.col, self.m_trim, SEM["trim"]))
            self.add(box("fr_r", (px, yc + w / 2 + t / 2, zb + h / 2), (d, t, h + 2 * t), self.col, self.m_trim, SEM["trim"]))
            self.add(box("fr_t", (px, yc, zb + h + t / 2), (d, w + 2 * t, t), self.col, self.m_trim, SEM["trim"]))
            if sill: self.add(box("fr_s", (x + facing * 0.06, yc, zb - 0.03), (0.14, w + 2 * t + 0.06, 0.06), self.col, self.m_trim, SEM["trim"]))
        else:
            py = x + facing * d / 2   # here 'x' is the wall plane y, yc is the x-centre
            self.add(box("fr_l", (yc - w / 2 - t / 2, py, zb + h / 2), (t, d, h + 2 * t), self.col, self.m_trim, SEM["trim"]))
            self.add(box("fr_r", (yc + w / 2 + t / 2, py, zb + h / 2), (t, d, h + 2 * t), self.col, self.m_trim, SEM["trim"]))
            self.add(box("fr_t", (yc, py, zb + h + t / 2), (w + 2 * t, d, t), self.col, self.m_trim, SEM["trim"]))
            if sill: self.add(box("fr_s", (yc, x + facing * 0.06, zb - 0.03), (w + 2 * t + 0.06, 0.14, 0.06), self.col, self.m_trim, SEM["trim"]))

    def window(self, wall_x, yc, zb, w, h, facing, axis, cutters, shutters=False):
        """Pocket + glass + curtain + frame. wall_x = outer wall plane coordinate; yc = centre along the wall."""
        depth = 0.22
        if axis == 'x':
            c = (wall_x - facing * (depth / 2 - 0.02), yc, zb + h / 2); s = (depth, w, h)
            g = (wall_x - facing * 0.10, yc, zb + h / 2); gs = (0.02, w - 0.02, h - 0.02)
            cu = (wall_x - facing * 0.19, yc, zb + h / 2); cs = (0.01, w, h)
        else:
            c = (yc, wall_x - facing * (depth / 2 - 0.02), zb + h / 2); s = (w, depth, h)
            g = (yc, wall_x - facing * 0.10, zb + h / 2); gs = (w - 0.02, 0.02, h - 0.02)
            cu = (yc, wall_x - facing * 0.19, zb + h / 2); cs = (w, 0.01, h)
        cutters.append(box("cut_w", c, s, self.col))
        self.add(box("glass", g, gs, self.col, self.m_glass, SEM["window"]))
        self.add(box("curtain", cu, cs, self.col, self.m_curtain if self.rng.p(0.7) else self.m_dark, SEM["window"]))
        # mullions
        if axis == 'x':
            self.add(box("mull_v", (wall_x - facing * 0.09, yc, zb + h / 2), (0.03, 0.04, h), self.col, self.m_trim, SEM["trim"]))
            self.add(box("mull_h", (wall_x - facing * 0.09, yc, zb + h * 0.55), (0.03, w, 0.04), self.col, self.m_trim, SEM["trim"]))
        else:
            self.add(box("mull_v", (yc, wall_x - facing * 0.09, zb + h / 2), (0.04, 0.03, h), self.col, self.m_trim, SEM["trim"]))
            self.add(box("mull_h", (yc, wall_x - facing * 0.09, zb + h * 0.55), (w, 0.03, 0.04), self.col, self.m_trim, SEM["trim"]))
        self.frame_opening(wall_x, yc, zb, w, h, facing=facing, axis=axis, sill=True)
        if shutters:
            sw = 0.35
            if axis == 'x':
                for sgn in (-1, 1):
                    self.add(box("shutter", (wall_x + facing * 0.03, yc + sgn * (w / 2 + 0.09 + sw / 2), zb + h / 2), (0.04, sw, h + 0.1), self.col, self.m_door, SEM["trim"]))
            else:
                for sgn in (-1, 1):
                    self.add(box("shutter", (yc + sgn * (w / 2 + 0.09 + sw / 2), wall_x + facing * 0.03, zb + h / 2), (sw, 0.04, h + 0.1), self.col, self.m_door, SEM["trim"]))

    def door_panel(self, wall_x, yc, zb, facing, axis, cutters, pocket=0.25):
        """Door pocket + panel + frame + hardware + number plaque + sconce + doormat. Records self.door."""
        p = self.p; w, h = p["door_w"], p["door_h"]
        if axis == 'x':
            cutters.append(box("cut_d", (wall_x - facing * (pocket / 2 - 0.02), yc, zb + h / 2), (pocket, w, h), self.col))
            self.add(box("door", (wall_x - facing * 0.17, yc, zb + h / 2), (0.05, w - 0.02, h - 0.02), self.col, self.m_door, SEM["door"]))
            for k in range(2):   # raised panels
                self.add(box("panel", (wall_x - facing * 0.195, yc, zb + 0.55 + k * 0.95), (0.012, w * 0.6, 0.7), self.col, self.m_door, SEM["door"]))
            self.add(box("knob", (wall_x - facing * 0.22, yc + w * 0.38, zb + 1.02), (0.06, 0.06, 0.06), self.col, self.m_metal, SEM["door"]))
            self.add(box("plaque", (wall_x + facing * 0.012, yc + w / 2 + 0.35, zb + 1.7), (0.02, 0.28, 0.16), self.col, M.flat("plaque", (0.1, 0.1, 0.1), 0.4, 0.5), SEM["trim"]))
            self.add(box("doormat", (wall_x - facing * 0.45, yc, zb + 0.012), (0.55, 0.85, 0.024), self.col, self.m_mat, SEM["porch"]))
            self.add(box("threshold", (wall_x - facing * 0.11, yc, zb + 0.02), (0.22, w + 0.1, 0.04), self.col, self.m_metal, SEM["door"]))
            normal = (float(facing), 0.0); face = (wall_x - facing * 0.17, yc, zb)
            # sconce
            if A.available(["security_light", "industrial_wall_lamp", "street_lamp_02"]):
                aid = self.rng.choice(A.available(["security_light", "industrial_wall_lamp", "street_lamp_02"]))
                A.place(aid, self.col, (wall_x + facing * 0.12, yc - w / 2 - 0.35, zb + 1.95), yaw=math.pi if facing > 0 else 0.0, target_h=0.3, sem=SEM["trim"])
        else:
            cutters.append(box("cut_d", (yc, wall_x - facing * (pocket / 2 - 0.02), zb + h / 2), (w, pocket, h), self.col))
            self.add(box("door", (yc, wall_x - facing * 0.17, zb + h / 2), (w - 0.02, 0.05, h - 0.02), self.col, self.m_door, SEM["door"]))
            for k in range(2):
                self.add(box("panel", (yc, wall_x - facing * 0.195, zb + 0.55 + k * 0.95), (w * 0.6, 0.012, 0.7), self.col, self.m_door, SEM["door"]))
            self.add(box("knob", (yc + w * 0.38, wall_x - facing * 0.22, zb + 1.02), (0.06, 0.06, 0.06), self.col, self.m_metal, SEM["door"]))
            self.add(box("plaque", (yc + w / 2 + 0.35, wall_x + facing * 0.012, zb + 1.7), (0.28, 0.02, 0.16), self.col, M.flat("plaque", (0.1, 0.1, 0.1), 0.4, 0.5), SEM["trim"]))
            self.add(box("doormat", (yc, wall_x - facing * 0.45, zb + 0.012), (0.85, 0.55, 0.024), self.col, self.m_mat, SEM["porch"]))
            self.add(box("threshold", (yc, wall_x - facing * 0.11, zb + 0.02), (w + 0.1, 0.22, 0.04), self.col, self.m_metal, SEM["door"]))
            normal = (0.0, float(facing)); face = (yc, wall_x - facing * 0.17, zb)
            if A.available(["security_light", "industrial_wall_lamp", "street_lamp_02"]):
                aid = self.rng.choice(A.available(["security_light", "industrial_wall_lamp", "street_lamp_02"]))
                A.place(aid, self.col, (yc - w / 2 - 0.35, wall_x + facing * 0.12, zb + 1.95), yaw=math.pi / 2 if facing > 0 else -math.pi / 2, target_h=0.3, sem=SEM["trim"])
        self.frame_opening(wall_x, yc, zb, w, h, facing=facing, axis=axis, trim_t=0.11)
        # the point on the landing 0.5 m in front of the door face, at floor level
        step_pt = (face[0] + normal[0] * 0.5 + (0 if axis == 'x' else 0), face[1] + normal[1] * 0.5, zb)
        if axis == 'x': step_pt = (face[0] + normal[0] * 0.5, face[1], zb)
        else: step_pt = (face[0], face[1] + normal[1] * 0.5, zb)
        self.door = {"face": face, "normal": normal, "doorstep": step_pt, "width": w, "height": h, "stratum": p["stratum"], "axis": axis, "facing": facing}
        self.openings.append((axis, facing, wall_x, yc if axis == 'x' else yc, w + 1.0))

    # ---------- strata ----------
    def steps_to(self, landing_x0, landing_x1, yc, width, facing_axis='x', facing=-1, top_z=None):
        """Concrete steps descending from the landing edge toward 'facing' direction, centred at yc."""
        top_z = top_z if top_z is not None else self.p["plinth"]
        n = max(1, int(math.ceil(top_z / 0.17))); rise = top_z / n; tread = 0.30
        for k in range(n):
            zk = top_z - rise * (k + 1); hgt = top_z - rise * k
            if facing_axis == 'x':
                xc = landing_x0 + facing * (tread * (k + 0.5))
                self.add(box("step", (xc, yc, hgt / 2 - 0.001), (tread + 0.02, width, hgt), self.col, self.m_conc, SEM["steps"]))
            else:
                yk = landing_x0 + facing * (tread * (k + 0.5))
                self.add(box("step", (yc, yk, hgt / 2 - 0.001), (width, tread + 0.02, hgt), self.col, self.m_conc, SEM["steps"]))
        return n * tread

    def door_front(self, mass, cutters):
        p, rng = self.p, self.rng
        x0, x1, y0, y1 = self.main_rect
        yd = self.cy + rng.u(-0.3, 0.3) * p["W"]
        yd = min(max(yd, y0 + 2.0), y1 - 2.0)
        zb = p["plinth"]
        self.door_panel(x0, yd, zb, facing=-1, axis='x', cutters=cutters, pocket=0.25)
        # stoop landing
        ld, lw = rng.u(1.2, 1.6), rng.u(1.6, 2.2)
        self.add(box("stoop", (x0 - ld / 2, yd, zb / 2 - 0.001), (ld, lw, zb), self.col, self.m_conc, SEM["porch"]))
        run = self.steps_to(x0 - ld, None, yd, lw, 'x', -1)
        self.keepout.append((x0 - ld - run - 2.5, x0, yd - lw / 2 - 0.8, yd + lw / 2 + 0.8))
        self.entry = {"approach_from": (x0 - ld - run - 1.0, yd, 0.0), "landing": (x0 - ld / 2, yd, zb)}
        if p["canopy"]:
            cw, cd = lw + 0.6, 1.3; zc = zb + p["door_h"] + 0.45
            self.add(box("canopy", (x0 - cd / 2 + 0.05, yd, zc), (cd, cw, 0.12), self.col, self.m_trim, SEM["trim"]))
            for sgn in (-1, 1):
                self.add(box("bracket", (x0 - 0.3, yd + sgn * (cw / 2 - 0.1), zc - 0.35), (0.5, 0.08, 0.08), self.col, self.m_trim, SEM["trim"]))
                self.add(box("bracket2", (x0 - 0.06, yd + sgn * (cw / 2 - 0.1), zc - 0.35), (0.08, 0.08, 0.7), self.col, self.m_trim, SEM["trim"]))
            if rng.p(0.5):
                prism_gable("canopy_roof", (x0 - cd / 2 + 0.05, yd, zc + 0.06), (cd, cw, 0), 0.4, self.col, self.m_roof, SEM["roof"], ridge_axis='x').pass_index = SEM["roof"]
        # railing on the stoop steps (one side)
        if p["porch_railing"]:
            self.railing_line((x0 - ld - run, yd + lw / 2 - 0.05, 0.0), (x0, yd + lw / 2 - 0.05, zb), height=0.95)

    def door_recessed(self, mass, cutters):
        p, rng = self.p, self.rng
        x0, x1, y0, y1 = self.main_rect
        zb = p["plinth"]; rec = p["recess"]
        pw = rng.u(3.2, 4.6)
        yd = self.cy + rng.u(-0.25, 0.25) * p["W"]; yd = min(max(yd, y0 + pw / 2 + 0.8), y1 - pw / 2 - 0.8)
        ceil_z = zb + STORY_H - 0.15
        # cut the porch volume out of the mass (full first storey height up to the porch ceiling)
        cutters.append(box("cut_porch", (x0 + rec / 2 - 0.01, yd, (zb - 0.2 + ceil_z) / 2), (rec + 0.02, pw, ceil_z - zb + 0.2), self.col))
        # porch floor (deck or concrete) and ceiling board
        m_floor = self.m_deck if rng.p(0.5) else self.m_conc
        self.add(box("porch_floor", (x0 + rec / 2, yd, zb - 0.06), (rec + 0.02, pw - 0.02, 0.12), self.col, m_floor, SEM["porch"]))
        self.add(box("porch_ceiling", (x0 + rec / 2, yd, ceil_z + 0.02), (rec + 0.02, pw - 0.02, 0.06), self.col, self.m_trim, SEM["trim"]))
        # recessed back wall gets the door
        self.door_panel(x0 + rec, yd + rng.u(-0.6, 0.6), zb, facing=-1, axis='x', cutters=cutters, pocket=0.25)
        # side windows inside the porch (optional)
        if rng.p(0.5):
            self.window(x0 + rec, yd + (pw / 2 - 0.9) * (1 if self.door["face"][1] < yd else -1), zb + p["sill"], 0.8, 1.2, -1, 'x', cutters)
        # columns at the porch front
        for sgn in (-1, 1):
            cx, cyc = x0 + 0.18, yd + sgn * (pw / 2 - 0.2)
            self.add(box("column", (cx, cyc, (zb + ceil_z) / 2), (0.24, 0.24, ceil_z - zb), self.col, self.m_trim, SEM["porch"]))
            self.add(box("col_base", (cx, cyc, zb + 0.12), (0.32, 0.32, 0.24), self.col, self.m_trim, SEM["porch"]))
        # steps centred on the door, descending toward the street from the porch front edge
        sw = rng.u(1.4, 2.0); ys = self.door["face"][1]
        run = self.steps_to(x0, None, ys, sw, 'x', -1)
        # railing along the porch front except the step opening
        if p["porch_railing"]:
            for (ya, yb) in ((yd - pw / 2 + 0.3, ys - sw / 2), (ys + sw / 2, yd + pw / 2 - 0.3)):
                if yb - ya > 0.5: self.railing_line((x0 + 0.18, ya, zb), (x0 + 0.18, yb, zb), height=0.95)
        self.keepout.append((x0 - run - 2.5, x0 + rec, yd - pw / 2 - 0.5, yd + pw / 2 + 0.5))
        self.entry = {"approach_from": (x0 - run - 1.0, ys, 0.0), "landing": (x0 + rec / 2, ys, zb)}
        # porch furniture
        avail = A.available(["painted_wooden_bench", "plastic_monobloc_chair_01", "potted_plant_01", "potted_plant_02", "planter_box_01", "cardboard_box_01"])
        if avail:
            free_side = 1 if self.door["face"][1] < yd else -1
            if "painted_wooden_bench" in avail and pw > 3.6 and rng.p(0.7):
                A.place("painted_wooden_bench", self.col, (x0 + rec - 0.45, yd + free_side * (pw / 2 - 1.1), zb), yaw=0.0, sem=SEM["furniture"])
            if "potted_plant_01" in avail: A.place(rng.choice([a for a in avail if "pot" in a or "planter" in a]), self.col, (x0 + 0.7, yd - free_side * (pw / 2 - 0.5), zb), yaw=rng.u(0, 6.28), sem=SEM["prop"])
            if "cardboard_box_01" in avail and rng.p(0.3): A.place("cardboard_box_01", self.col, (x0 + rec - 0.5, self.door["face"][1] + free_side * 0.9, zb), yaw=rng.u(0, 6.28), sem=SEM["prop"])

    def door_side(self, mass, cutters):
        p, rng = self.p, self.rng
        x0, x1, y0, y1 = self.main_rect
        s = p["door_side"]; wall_y = y1 if s > 0 else y0
        xd = x0 + rng.u(2.0, 4.0); zb = p["plinth"]
        self.door_panel(wall_y, xd, zb, facing=s, axis='y', cutters=cutters, pocket=0.25)
        ld, lw = rng.u(1.1, 1.4), rng.u(1.5, 1.9)
        self.add(box("stoop", (xd, wall_y + s * ld / 2, zb / 2 - 0.001), (lw, ld, zb), self.col, self.m_conc, SEM["porch"]))
        run = self.steps_to(wall_y + s * ld, None, xd, lw, 'y', s)
        if p["canopy"]:
            zc = zb + p["door_h"] + 0.45; cw, cd = lw + 0.6, 1.2
            self.add(box("canopy", (xd, wall_y + s * (cd / 2 - 0.05), zc), (cw, cd, 0.12), self.col, self.m_trim, SEM["trim"]))
        self.keepout.append((xd - lw / 2 - 0.8, xd + lw / 2 + 0.8, min(wall_y, wall_y + s * (ld + run + 2.0)), max(wall_y, wall_y + s * (ld + run + 2.0))))
        self.entry = {"approach_from": (xd, wall_y + s * (ld + run + 1.0), 0.0), "landing": (xd, wall_y + s * ld / 2, zb)}
        self.side_path = {"side": s, "x_door": xd}

    def door_corner(self, cutters):
        """L-plan: the wing protrudes toward the street; the door sits on the wing's inner side wall, facing across the notch."""
        p, rng = self.p, self.rng
        x0, x1, y0, y1 = self.main_rect
        gfx, gx1, gy0, gy1 = self.wing_rect
        s = self.wing_side              # wing on this Y side; door faces -s (into the notch, parallel to the street)
        wall_y = gy0 if s > 0 else gy1  # inner wall of the wing
        xd = gfx + rng.u(1.2, min(2.6, (x0 - gfx) - 1.0)); zb = p["plinth"]
        self.door_panel(wall_y, xd, zb, facing=-s, axis='y', cutters=self.wing_cutters, pocket=0.25)
        ld, lw = rng.u(1.1, 1.4), rng.u(1.5, 1.9)
        self.add(box("stoop", (xd, wall_y - s * ld / 2, zb / 2 - 0.001), (lw, ld, zb), self.col, self.m_conc, SEM["porch"]))
        run = self.steps_to(wall_y - s * ld, None, xd, lw, 'y', -s)
        zc = zb + p["door_h"] + 0.45
        self.add(box("canopy", (xd, wall_y - s * 0.55, zc), (lw + 0.6, 1.2, 0.12), self.col, self.m_trim, SEM["trim"]))
        self.keepout.append((gfx - 1.0, x0, min(wall_y, wall_y - s * (ld + run + 2.5)), max(wall_y, wall_y - s * (ld + run + 2.5))))
        self.entry = {"approach_from": (xd, wall_y - s * (ld + run + 1.0), 0.0), "landing": (xd, wall_y - s * ld / 2, zb)}

    def railing_line(self, a, b, height=0.95):
        ax, ay, az = a; bx, by, bz = b
        L = math.hypot(bx - ax, by - ay); n = max(2, int(L / 1.2) + 1)
        for i in range(n):
            t = i / (n - 1); x, y, z = ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t
            self.add(box("post", (x, y, z + height / 2), (0.08, 0.08, height), self.col, self.m_trim, SEM["porch"]))
        # top rail: a thin box rotated to the segment (approximate with yaw)
        yaw = math.atan2(by - ay, bx - ax); mz = (az + bz) / 2 + height
        slope_len = math.hypot(L, bz - az)
        r = box("rail", ((ax + bx) / 2, (ay + by) / 2, mz), (slope_len, 0.06, 0.05), self.col, self.m_trim, SEM["porch"], rot_z=yaw)
        if abs(bz - az) > 1e-3: r.rotation_euler = (0, -math.atan2(bz - az, L), yaw)
        self.add(r)
        for k in range(int(L / 0.14)):
            t = (k + 0.5) / int(L / 0.14); x, y, z = ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t
            self.add(box("baluster", (x, y, z + height / 2), (0.025, 0.025, height - 0.05), self.col, self.m_trim, SEM["porch"]))

    # ---------- windows ----------
    def windows(self, mass, cutters):
        p, rng = self.p, self.rng
        x0, x1, y0, y1 = self.main_rect
        ww, wh = p["window_w"], p["window_h"]
        door_face = self.door["face"] if self.door else None

        def blocked(axis, facing, wall_c, c, w):
            # avoid door openings, porch inset, wing attachment
            for (ax_, fc_, wc_, cc_, ow) in self.openings:
                if ax_ == axis and fc_ == facing and abs(wc_ - wall_c) < 0.05 and abs(cc_ - c) < (w + ow) / 2: return True
            if p["stratum"] == "recessed" and axis == 'x' and facing == -1 and abs(wall_c - x0) < 0.05:
                yd = self.entry["landing"][1]
                if abs(c - yd) < 2.8: return True
            if self.wing_rect is not None:
                gfx, gx1, gy0, gy1 = self.wing_rect
                if axis == 'y':
                    wy = y1 if facing > 0 else y0
                    if abs(wall_c - wy) < 0.05 and ((facing > 0 and gy0 <= wy + 0.01) or (facing < 0 and gy1 >= wy - 0.01)) and gfx - 0.5 < c < gx1 + 0.5: return True
            return False

        for story in range(p["stories"]):
            zb = p["plinth"] + story * STORY_H + p["sill"]
            # front (-x) and back (+x)
            for (wall_x, facing) in ((x0, -1), (x1, 1)):
                L = y1 - y0; n = max(1, int((L - 1.2) / 3.0)); 
                for i in range(n):
                    c = y0 + L * (i + 0.5) / n
                    w = ww
                    if facing == -1 and story == 0 and rng.p(0.4): w = ww * 1.7   # living-room window
                    if blocked('x', facing, wall_x, c, w): continue
                    self.window(wall_x, c, zb, w, wh, facing, 'x', cutters, shutters=p["shutters"] and facing == -1)
            # sides (±y)
            for (wall_y, facing) in ((y0, -1), (y1, 1)):
                L = x1 - x0; n = max(1, int((L - 1.2) / 3.2))
                for i in range(n):
                    c = x0 + L * (i + 0.5) / n
                    if blocked('y', facing, wall_y, c, ww): continue
                    if rng.p(0.85): self.window(wall_y, c, zb, ww, wh, facing, 'y', cutters)
        # wing side windows (one) + a small front window if no garage door
        if self.wing_rect is not None and p["lod"] == "full":
            gfx, gx1, gy0, gy1 = self.wing_rect
            wy = gy1 if self.wing_side > 0 else gy0
            if rng.p(0.7):
                self.window(wy, (gfx + gx1) / 2, p["plinth"] + p["sill"], ww, wh * 0.8, self.wing_side, 'y', self.wing_cutters)

    # ---------- roofs ----------
    def roof(self, x0, x1, y0, y1, ztop, kind="gable", ridge_axis='y', name="roof"):
        p = self.p; ov = p["overhang"]
        sx, sy = (x1 - x0) + 2 * ov, (y1 - y0) + 2 * ov
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = (sx if ridge_axis == 'y' else sy) / 2
        ph = half * math.tan(math.radians(p["pitch_deg"]))
        # eave board / fascia volume
        self.add(box(f"{self.prefix}_{name}_eave", (cx, cy, ztop + 0.08), (sx, sy, 0.16), self.col, self.m_trim, SEM["roof"]))
        if kind == "gable":
            r = prism_gable(f"{self.prefix}_{name}", (cx, cy, ztop + 0.16), (sx, sy, 0), ph, self.col, self.m_roof, SEM["roof"], ridge_axis=ridge_axis)
            # gable-end wall triangles (same wall material) flush with the walls
            if ridge_axis == 'y':
                for (yy, sgn) in ((y0, -1), (y1, 1)):
                    self.add(self._tri_wall(name, (x0, x1), yy, ztop, half - ov, ph, 'y'))
            else:
                for xx in (x0, x1):
                    self.add(self._tri_wall(name, (y0, y1), xx, ztop, half - ov, ph, 'x'))
        else:
            r = prism_hip(f"{self.prefix}_{name}", (cx, cy, ztop + 0.16), (sx, sy, 0), ph, self.col, self.m_roof, SEM["roof"])
        self.add(r)
        if p["gutters"]:
            for (c, s) in (((cx, y0 - ov + 0.02, ztop + 0.1), (sx, 0.12, 0.1)), ((cx, y1 + ov - 0.02, ztop + 0.1), (sx, 0.12, 0.1))) if ridge_axis == 'y' else (((x0 - ov + 0.02, cy, ztop + 0.1), (0.12, sy, 0.1)), ((x1 + ov - 0.02, cy, ztop + 0.1), (0.12, sy, 0.1))):
                self.add(box("gutter", c, s, self.col, self.m_trim, SEM["trim"]))
            for (xx, yy) in ((x0 + 0.1, y0 + 0.1), (x0 + 0.1, y1 - 0.1), (x1 - 0.1, y0 + 0.1), (x1 - 0.1, y1 - 0.1)):
                self.add(cylinder("downspout", (xx - (0.2 if xx < cx else -0.2), yy - (0.0), (p["plinth"] + ztop) / 2), 0.045, ztop - p["plinth"], self.col, self.m_trim, SEM["trim"], n=10))
        return r

    def _tri_wall(self, name, span, at, ztop, half, ph, axis):
        from .common import mesh_obj
        a, b = span; mid = (a + b) / 2
        if axis == 'y':
            verts = [(a + 0.01, at, ztop), (b - 0.01, at, ztop), (mid, at, ztop + ph * (half / (half + self.p["overhang"])) * 1.0)]
            verts2 = [(v[0], v[1] + (0.28 if at < self.cy else -0.28), v[2]) for v in verts]
        else:
            verts = [(at, a + 0.01, ztop), (at, b - 0.01, ztop), (at, mid, ztop + ph * (half / (half + self.p["overhang"])))]
            verts2 = [(v[0] + (0.28 if at < (self.main_rect[0] + self.main_rect[1]) / 2 else -0.28), v[1], v[2]) for v in verts]
        vs = verts + verts2; faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
        ob = mesh_obj(f"{self.prefix}_{name}_gable", vs, faces, self.col, self.m_wall, SEM["wall"]); return ob

    # ---------- facade props ----------
    def facade_props(self):
        p, rng = self.p, self.rng
        if p["lod"] != "full": return
        x0, x1, y0, y1 = self.main_rect
        # AC unit + hose + utility box on a side wall away from the door
        side = -p["door_side"] if p["stratum"] in ("side", "corner") else rng.choice([-1, 1])
        wy = y1 if side > 0 else y0
        yaw = math.pi / 2 if side > 0 else -math.pi / 2
        xs = rng.shuffle([x0 + 1.5, x0 + 4.0, x0 + 6.5])
        if "exterior_aircon_unit" in A.available(["exterior_aircon_unit"]):
            A.place("exterior_aircon_unit", self.col, (xs[0], wy + side * 0.45, 0.0), yaw=yaw, sem=SEM["prop"])
        if "garden_hose_wall_mounted_01" in A.available(["garden_hose_wall_mounted_01"]) and rng.p(0.7):
            A.place("garden_hose_wall_mounted_01", self.col, (xs[1], wy + side * 0.15, 0.9), yaw=yaw, sem=SEM["prop"])
        if rng.p(0.6):
            ub = A.available(["utility_box_01", "utility_box_02"])
            if ub: A.place(rng.choice(ub), self.col, (xs[2], wy + side * 0.12, 1.2), yaw=yaw, sem=SEM["prop"])
        # planters flanking the entry
        if self.door and p["stratum"] in ("street", "side", "corner"):
            pots = A.available(["planter_box_01", "planter_box_02", "planter_box_03", "potted_plant_01", "potted_plant_02", "planter_pot_clay"])
            if pots:
                fx_, fy_, fz_ = self.door["face"]; nx, ny = self.door["normal"]
                for sgn in (-1, 1):
                    if rng.p(0.75):
                        px_ = fx_ + nx * 0.5 + (-ny) * sgn * (self.door["width"] / 2 + 0.45)
                        py_ = fy_ + ny * 0.5 + (nx) * sgn * (self.door["width"] / 2 + 0.45)
                        A.place(rng.choice(pots), self.col, (px_, py_, fz_), yaw=rng.u(0, 6.28), sem=SEM["prop"])
