"""Lot layout around the house: street, sidewalks, curbs, driveway, front path, lawn, fences/hedges, trees, props, neighbours, sky."""
import math
import bpy
from mathutils import Vector
from .common import SEM, box, plane, cylinder, collection, link, prism_gable
from . import materials as M
from . import assets as A
from .house import House, house_params

GROUND_Z = 0.15          # lot ground (lawn/sidewalk) above the street surface (z = 0)
HEDGE_SPECIES = ["searsia_lucida"]
TREE_SPECIES = ["jacaranda_tree", "island_tree_01", "island_tree_02", "island_tree_03", "tree_small_02"]
SHRUB_SPECIES = ["searsia_lucida", "island_tree_03"]
LAWN_TINTS = [(0.55, 0.85, 0.4), (0.6, 0.9, 0.45), (0.7, 0.9, 0.55), (0.8, 0.9, 0.65)]
STREET_X0, STREET_X1 = -12.5, -4.5   # street spans these X; near curb at -4.5, far curb at -12.5
SIDEWALK_W = 1.5
VERGE_W = 1.2            # grass strip between curb and sidewalk


def lot_params(rng, stratum, hp=None):
    p = {"stratum": stratum}
    footprint_w = (hp["W"] + (hp["garage_W"] if (hp["garage"] or stratum == "corner") else 0.0)) if hp else 14.0
    p["W"] = footprint_w + rng.u(7.0, 12.0)      # lot width along Y: house + wing + side setbacks of 3.5-6 m each
    p["depth"] = rng.u(32.0, 42.0)              # lot depth along X from the sidewalk
    p["setback"] = rng.u(8.0, 12.0)             # front facade X
    p["path_w"] = rng.u(1.1, 1.4)
    p["path_kind"] = rng.choice(["straight", "straight", "bent"])
    p["front_hedge"] = rng.choice([None, None, "hedge", "picket"])
    p["side_fence"] = rng.choice(["picket", "wood", "hedge", "wood", None])
    p["rear_fence"] = rng.choice(["wood", "wood", "hedge"])
    p["n_front_trees"] = rng.i(1, 2)
    p["n_back_trees"] = rng.i(1, 3)
    p["street_tree"] = rng.p(0.65)
    p["car"] = rng.p(0.6)
    p["bins"] = rng.p(0.85)
    p["lamp"] = rng.p(0.75)
    p["hydrant"] = rng.p(0.4)
    p["lane_marks"] = rng.p(0.6)
    lawns = M.textures_by_role("lawn"); pav = M.textures_by_role("paving"); drv = M.textures_by_role("driveway"); road = M.textures_by_role("road")
    p["lawn_tex"] = rng.choice(lawns) if lawns else None
    p["lawn_tint"] = rng.choice(LAWN_TINTS)
    p["grass_clumps"] = rng.p(0.85)
    p["path_tex"] = rng.choice(pav) if pav else None
    p["drive_tex"] = rng.choice(drv) if drv else None
    p["road_tex"] = rng.choice(road) if road else None
    p["sidewalk_tex"] = rng.choice(["concrete_pavement", "concrete_pavement_03", "brushed_concrete"])
    hd = [k for k, v in M.manifest().items() if v.get("kind") == "hdri"]
    p["hdri"] = rng.choice(hd) if hd else None
    p["hdri_rot"] = rng.u(0, 360)
    p["hdri_strength"] = rng.u(0.9, 1.3)
    return p


class Lot:
    def __init__(self, rng, lp, hp, col):
        self.rng, self.lp, self.hp, self.col = rng, lp, hp, col
        self.props_col = collection("props"); self.veg_col = collection("vegetation"); self.ground_col = collection("ground")
        self.nb_col = collection("neighbors"); self.house_col = collection("house")
        self.build()

    # ----- geometry -----
    def build(self):
        lp, hp, rng = self.lp, self.hp, self.rng
        W, Dl, sb = lp["W"], lp["depth"], lp["setback"]
        self.y0, self.y1 = -W / 2, W / 2
        self.lot_x0 = STREET_X1 + VERGE_W + SIDEWALK_W     # front lot line (after verge + sidewalk)
        self.lot_x1 = self.lot_x0 + Dl
        gz = GROUND_Z
        # materials
        m_lawn = M.pbr(lp["lawn_tex"], tile_m=3.0, tint=lp.get("lawn_tint")) if lp["lawn_tex"] else M.flat("lawn", (0.15, 0.3, 0.08))
        m_road = M.pbr(lp["road_tex"], tile_m=6.0) if lp["road_tex"] else M.flat("asphalt", (0.12, 0.12, 0.12))
        m_side = M.pbr(lp["sidewalk_tex"], tile_m=1.5) if lp["sidewalk_tex"] in M.manifest() else M.flat("sidewalk", (0.6, 0.6, 0.58))
        m_curb = M.flat("curb", (0.55, 0.55, 0.52), rough=0.8)
        m_path = M.pbr(lp["path_tex"], tile_m=1.2) if lp["path_tex"] else M.flat("path", (0.5, 0.45, 0.4))
        m_drive = M.pbr(lp["drive_tex"], tile_m=2.5) if lp["drive_tex"] else M.flat("drive", (0.5, 0.5, 0.48))
        self.m_lawn = m_lawn
        # ---- ground: big lawn plane everywhere, street cut as a lower plane ----
        plane("ground_far", (0, 0, gz), (260, 260), self.ground_col, m_lawn, SEM["lawn"])
        # street (lower) with shoulders; the lawn plane above it is hidden by a street box filling the trench
        plane("street", ((STREET_X0 + STREET_X1) / 2, 0, 0.0), (STREET_X1 - STREET_X0, 240), self.ground_col, m_road, SEM["street"])
        # cut the lawn over the street: we simply put the lawn plane as two half planes instead
        bpy.data.objects.remove(bpy.data.objects["ground_far"], do_unlink=True)
        plane("ground_north", ((STREET_X1 + 130) / 2, 0, gz), (130 - STREET_X1, 260), self.ground_col, m_lawn, SEM["lawn"])
        plane("ground_south", ((STREET_X0 - 130) / 2, 0, gz), (STREET_X0 + 130, 260), self.ground_col, m_lawn, SEM["lawn"])
        # curbs (with a gap for the driveway on the near side)
        self.drive_y = None
        if hp.get("garage") and hasattr(self, "house") is False:
            pass
        # sidewalks both sides
        plane("sidewalk_near", (STREET_X1 + VERGE_W + SIDEWALK_W / 2, 0, gz + 0.004), (SIDEWALK_W, 240), self.ground_col, m_side, SEM["sidewalk"])
        plane("sidewalk_far", (STREET_X0 - VERGE_W - SIDEWALK_W / 2, 0, gz + 0.004), (SIDEWALK_W, 240), self.ground_col, m_side, SEM["sidewalk"])
        # lane marks
        if lp["lane_marks"]:
            mp = M.lane_paint()
            for k in range(-40, 40):
                plane("lane", ((STREET_X0 + STREET_X1) / 2, k * 3.0, 0.003), (0.12, 1.5), self.ground_col, mp, SEM["street"])
        # ---- the house ----
        wing_w = hp["garage_W"] if (hp["garage"] or hp["stratum"] == "corner") else 0.0
        wing_side = hp["garage_side"] if hp["stratum"] != "corner" else -hp["door_side"]
        self.house = House(rng.sub("house"), hp, self.house_col, front_x=sb, cy=-wing_side * wing_w / 2, prefix="h")
        door = self.house.door; entry = self.house.entry
        # ---- driveway ----
        if hp.get("garage"):
            gfx, gcy, gdw = self.house.garage_door
            dw = gdw + 0.8; self.drive_y = gcy
            plane("driveway", ((STREET_X1 + gfx) / 2, gcy, gz + 0.006), (gfx - STREET_X1, dw), self.ground_col, m_drive, SEM["driveway"])
            # apron ramp from street level to the lot level
            ramp = box("apron", (STREET_X1 + 0.45, gcy, gz / 2), (0.9, dw + 0.4, gz), self.ground_col, m_curb, SEM["driveway"])
            self.drive_rect = (STREET_X1, gfx, gcy - dw / 2, gcy + dw / 2)
        else:
            self.drive_rect = None
        self.curbs(m_curb)
        # ---- front path ----
        self.path_rects = []
        ap = entry["approach_from"]
        pw = lp["path_w"]
        sx = self.lot_x0 - 0.1   # start at the sidewalk edge
        if hp["stratum"] in ("street", "recessed"):
            yd = ap[1]
            if lp["path_kind"] == "straight" or self.drive_rect is None:
                plane("path", ((sx + ap[0]) / 2, yd, gz + 0.008), (ap[0] - sx + 0.4, pw), self.ground_col, m_path, SEM["path"])
                self.path_rects.append((sx, ap[0] + 0.2, yd - pw / 2, yd + pw / 2))
                self.path_entry_y = yd
            else:
                # path branches off the driveway: along the driveway edge then across to the door
                gx0, gx1, gy0, gy1 = self.drive_rect
                side_y = gy1 + pw / 2 if gcy < yd else gy0 - pw / 2
                xb = ap[0] - 1.5
                plane("path_a", ((sx + xb) / 2, side_y, gz + 0.008), (xb - sx, pw), self.ground_col, m_path, SEM["path"])
                plane("path_b", (xb, (side_y + yd) / 2, gz + 0.008), (pw, abs(yd - side_y) + pw), self.ground_col, m_path, SEM["path"])
                plane("path_c", ((xb + ap[0]) / 2, yd, gz + 0.008), (ap[0] - xb + 0.4, pw), self.ground_col, m_path, SEM["path"])
                self.path_rects += [(sx, xb, side_y - pw / 2, side_y + pw / 2), (xb - pw / 2, xb + pw / 2, min(side_y, yd), max(side_y, yd)), (xb, ap[0] + 0.2, yd - pw / 2, yd + pw / 2)]
                self.path_entry_y = side_y
        else:
            # side / corner: path from the sidewalk to the front corner, then along the house side to the door approach point
            s = 1 if ap[1] > 0 else -1
            xa, ya = ap[0], ap[1]
            yline = ya + s * 0.0 if hp["stratum"] == "corner" else ya
            # straight run from the sidewalk up to the approach point x, at the approach y
            if hp["stratum"] == "side":
                yline = ya
                plane("path", ((sx + xa) / 2, yline, gz + 0.008), (xa - sx + pw, pw), self.ground_col, m_path, SEM["path"])
                self.path_rects.append((sx, xa + pw / 2, yline - pw / 2, yline + pw / 2))
                # connector from the run to the stoop steps (perpendicular)
                self.path_entry_y = yline
            else:
                plane("path", ((sx + xa) / 2, ya, gz + 0.008), (xa - sx + pw, pw), self.ground_col, m_path, SEM["path"])
                self.path_rects.append((sx, xa + pw / 2, ya - pw / 2, ya + pw / 2))
                self.path_entry_y = ya
        # ---- grass clumps on the front and back lawn (hair particles instancing a grass asset) ----
        if lp.get("grass_clumps"):
            self.grass_clumps(m_lawn)
        # ---- fences / hedges ----
        self.fences()
        # ---- vegetation ----
        self.vegetation()
        # ---- props ----
        self.props()
        # ---- neighbours + background ----
        self.neighbors()
        # ---- sky ----
        if lp["hdri"]:
            import os
            M.world_hdri(os.path.join(M.PH, M.manifest()[lp["hdri"]]["path"]), lp["hdri_rot"], lp["hdri_strength"])
        else:
            M.world_sky(rng.u(35, 60), rng.u(0, 360))

    def grass_clumps(self, m_lawn):
        """Instanced grass clumps over the lot lawn via a hair particle system (keeps the lawn plane as the renderable ground)."""
        rng = self.rng; gz = GROUND_Z
        species = A.available(["grass_medium_02"])
        if not species: return
        aid = rng.choice(species); src = A._import(aid)
        hx0, hx1, hy0, hy1 = self.house.main_rect
        # emitter: the lot area minus nothing (clumps under paths are hidden by the paving planes sitting 8 mm above the lawn: avoid by lowering clumps)
        em = plane("lawn_emitter", ((self.lot_x0 + self.lot_x1) / 2, 0, gz + 0.001), (self.lot_x1 - self.lot_x0, self.y1 - self.y0), self.veg_col, m_lawn, SEM["lawn"], subdiv=8)
        ps = em.modifiers.new("grass", 'PARTICLE_SYSTEM').particle_system; st = ps.settings
        area = (self.lot_x1 - self.lot_x0) * (self.y1 - self.y0)
        st.type = 'HAIR'; st.count = int(area * 2.5); st.hair_length = 1.0; st.use_advanced_hair = True
        st.render_type = 'OBJECT'; st.instance_object = src; st.particle_size = 1.0; st.size_random = 0.5
        st.use_rotations = True; st.rotation_mode = 'OB_Z'; st.phase_factor_random = 2.0; st.use_rotation_instance = True
        st.emit_from = 'FACE'; st.use_emit_random = True; st.distribution = 'RAND'; ps.seed = rng.i(0, 10000)
        self.grass_src = src

    def curbs(self, m_curb):
        gz = GROUND_Z
        segs_near = [(-120, 120)]
        if self.drive_rect is not None:
            gx0, gx1, gy0, gy1 = self.drive_rect
            segs_near = [(-120, gy0 - 0.2), (gy1 + 0.2, 120)]
        for (a, b) in segs_near:
            box("curb_near", (STREET_X1 + 0.1, (a + b) / 2, gz / 2), (0.2, b - a, gz), self.ground_col, m_curb, SEM["sidewalk"])
        box("curb_far", (STREET_X0 - 0.1, 0, gz / 2), (0.2, 240, gz), self.ground_col, m_curb, SEM["sidewalk"])

    # ----- keep-out logic -----
    def free(self, x, y, r=0.8):
        """True if (x,y) is clear of the house, driveway, paths and entry keep-outs by margin r."""
        rects = [self.house.main_rect] + ([self.house.wing_rect] if self.house.wing_rect else []) + self.house.keepout + self.path_rects
        if self.drive_rect: rects.append(self.drive_rect)
        for (x0, x1, y0, y1) in rects:
            if x0 - r < x < x1 + r and y0 - r < y < y1 + r: return False
        return True

    # ----- fences -----
    def fence_run(self, a, b, kind, name="fence"):
        ax, ay = a; bx, by = b; L = math.hypot(bx - ax, by - ay); yaw = math.atan2(by - ay, bx - ax)
        gz = GROUND_Z; col = self.props_col
        if kind == "hedge":
            # a hedge = a dense row of real shrub instances (overlapping, jittered), one species per run
            species = A.available(HEDGE_SPECIES)
            if not species:
                h = self.rng.u(0.9, 1.6); t = self.rng.u(0.6, 0.9)
                box(name, ((ax + bx) / 2, (ay + by) / 2, gz + h / 2), (L, t, h), self.veg_col, M.leaf(), SEM["vegetation"], rot_z=yaw); return
            aid = self.rng.choice(species); h = self.rng.u(0.9, 1.5)
            fw, fd = A.footprint(aid, h / max(1e-6, A.size(aid)[2])); step = max(0.35, 0.55 * max(fw, fd))
            n = max(1, int(L / step))
            for k in range(n):
                t = (k + 0.5) / n; x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                A.place(aid, self.veg_col, (x + self.rng.u(-0.1, 0.1), y + self.rng.u(-0.1, 0.1), gz - 0.05), yaw=self.rng.u(0, 6.28), target_h=h * self.rng.u(0.85, 1.15), sem=SEM["vegetation"])
            return
        m_wood = M.pbr("weathered_planks", tile_m=1.5) if "weathered_planks" in M.manifest() else M.flat("fence_wood", (0.45, 0.35, 0.22))
        m_paint = M.flat("fence_paint", (0.9, 0.9, 0.86), rough=0.6)
        if kind == "picket":
            h = 1.0; n = int(L / 0.16)
            for k in range(n):
                t = (k + 0.5) / n; x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                box("picket", (x, y, gz + h / 2), (0.02, 0.08, h), col, m_paint, SEM["fence"], rot_z=yaw)
            for zz in (0.3, 0.8):
                box("rail", ((ax + bx) / 2, (ay + by) / 2, gz + zz), (L, 0.03, 0.08), col, m_paint, SEM["fence"], rot_z=yaw)
            for k in range(int(L / 2.4) + 1):
                t = min(1.0, k * 2.4 / L); x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                box("post", (x, y, gz + 0.55), (0.1, 0.1, 1.1), col, m_paint, SEM["fence"], rot_z=yaw)
        elif kind == "wood":
            h = 1.8
            box("fence_panel", ((ax + bx) / 2, (ay + by) / 2, gz + h / 2), (L, 0.04, h), col, m_wood, SEM["fence"], rot_z=yaw)
            for k in range(int(L / 2.4) + 1):
                t = min(1.0, k * 2.4 / L); x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                box("post", (x, y, gz + h / 2 + 0.05), (0.1, 0.1, h + 0.1), col, m_wood, SEM["fence"], rot_z=yaw)
            box("cap", ((ax + bx) / 2, (ay + by) / 2, gz + h + 0.02), (L, 0.09, 0.04), col, m_wood, SEM["fence"], rot_z=yaw)
        elif kind == "chainlink":
            if A.available(["modular_chainlink_fence"]):
                seg = A.size("modular_chainlink_fence")[0]; n = max(1, int(L / seg))
                for k in range(n):
                    t = (k + 0.5) / n; x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                    A.place("modular_chainlink_fence", col, (x, y, gz), yaw=yaw, sem=SEM["fence"])

    def fences(self):
        lp, rng = self.lp, self.rng
        hx0 = self.house.main_rect[0]
        # side boundaries from a little in front of the house to the rear
        if lp["side_fence"]:
            for yy in (self.y0, self.y1):
                self.fence_run((hx0 - 1.0, yy), (self.lot_x1, yy), lp["side_fence"])
        # rear
        self.fence_run((self.lot_x1, self.y0), (self.lot_x1, self.y1), lp["rear_fence"])
        # front low hedge / picket with a gap at the path (and the driveway)
        if lp["front_hedge"]:
            xf = self.lot_x0 + 0.4; gaps = [(self.path_entry_y - lp["path_w"] / 2 - 0.4, self.path_entry_y + lp["path_w"] / 2 + 0.4)]
            if self.drive_rect: gaps.append((self.drive_rect[2] - 0.3, self.drive_rect[3] + 0.3))
            gaps.sort(); y = self.y0
            for (ga, gb) in gaps:
                if ga - y > 0.8: self.fence_run((xf, y), (xf, ga), lp["front_hedge"])
                y = max(y, gb)
            if self.y1 - y > 0.8: self.fence_run((xf, y), (xf, self.y1), lp["front_hedge"])

    # ----- vegetation -----
    def vegetation(self):
        lp, rng = self.lp, self.rng
        gz = GROUND_Z
        trees = A.available(TREE_SPECIES)
        shrubs = A.available(SHRUB_SPECIES)
        hx0, hx1, hy0, hy1 = self.house.main_rect
        placed = []

        def try_place(xr, yr, aid, h, tries=40, clear=1.0, sem=SEM["vegetation"]):
            for _ in range(tries):
                x, y = rng.u(*xr), rng.u(*yr)
                if not self.free(x, y, clear): continue
                if any(math.hypot(x - px, y - py) < pr + clear for (px, py, pr) in placed): continue
                ob = A.place(aid, self.veg_col, (x, y, gz), yaw=rng.u(0, 6.28), target_h=h, sem=sem)
                fw, fd = A.footprint(aid, ob.scale[0]); placed.append((x, y, max(fw, fd) / 2)); return ob
            return None

        # front-yard trees
        for _ in range(lp["n_front_trees"]):
            if trees: try_place((self.lot_x0 + 2.0, hx0 - 3.0), (self.y0 + 1.5, self.y1 - 1.5), rng.choice(trees), rng.u(5.0, 9.0), clear=2.0)
        # back-yard trees
        for _ in range(lp["n_back_trees"]):
            if trees: try_place((hx1 + 3.0, self.lot_x1 - 2.0), (self.y0 + 1.5, self.y1 - 1.5), rng.choice(trees), rng.u(5.0, 10.0), clear=2.0)
        # street tree in the verge
        if lp["street_tree"] and trees:
            yv = rng.u(self.y0 + 1, self.y1 - 1)
            if self.drive_rect is None or not (self.drive_rect[2] - 2 < yv < self.drive_rect[3] + 2):
                ob = A.place(rng.choice(trees), self.veg_col, (STREET_X1 + VERGE_W / 2, yv, gz), yaw=rng.u(0, 6.28), target_h=rng.u(5, 8), sem=SEM["vegetation"])
                placed.append((STREET_X1 + VERGE_W / 2, yv, 1.5))
        # foundation planting along the front facade
        if shrubs:
            y = hy0 + 0.8
            while y < hy1 - 0.8:
                if self.free(hx0 - 0.75, y, 0.6) and rng.p(0.8):
                    A.place(rng.choice(shrubs), self.veg_col, (hx0 - 0.75, y, gz), yaw=rng.u(0, 6.28), target_h=rng.u(0.5, 1.1), sem=SEM["vegetation"])
                y += rng.u(1.2, 2.0)
            # a few loose shrubs in the yard
            for _ in range(rng.i(2, 5)):
                try_place((self.lot_x0 + 1.0, hx0 - 1.5), (self.y0 + 0.8, self.y1 - 0.8), rng.choice(shrubs), rng.u(0.6, 1.4), clear=0.8)
        self.tree_positions = placed

    # ----- props -----
    def props(self):
        lp, rng = self.lp, self.rng; gz = GROUND_Z
        col = self.props_col
        hx0, hx1, hy0, hy1 = self.house.main_rect
        # mailbox at the curb near the path or driveway
        ym = self.path_entry_y + rng.choice([-1.2, 1.2])
        m_post = M.flat("mailpost", (0.3, 0.22, 0.14), rough=0.8); m_box = M.flat("mailbox", rng.choice([(0.05, 0.05, 0.05), (0.6, 0.6, 0.6), (0.2, 0.2, 0.5)]), 0.5, 0.6)
        box("mail_post", (STREET_X1 + 0.5, ym, gz + 0.55), (0.1, 0.1, 1.1), col, m_post, SEM["furniture"])
        box("mail_box", (STREET_X1 + 0.5, ym, gz + 1.2), (0.5, 0.2, 0.25), col, m_box, SEM["furniture"])
        box("mail_flag", (STREET_X1 + 0.6, ym + 0.12, gz + 1.25), (0.03, 0.02, 0.15), col, M.flat("flag", (0.8, 0.1, 0.1)), SEM["furniture"])
        # car on the driveway
        if lp["car"] and self.drive_rect and A.available(["covered_car"]):
            gx0, gx1, gy0, gy1 = self.drive_rect
            A.place("covered_car", col, (gx1 - 6.5, (gy0 + gy1) / 2, gz), yaw=rng.choice([0.0, math.pi]) + rng.u(-0.05, 0.05), sem=SEM["vehicle"])
        # bins by the garage / house side
        if lp["bins"] and A.available(["metal_trash_can"]):
            if self.drive_rect:
                gx0, gx1, gy0, gy1 = self.drive_rect; s = 1 if gy1 > hy1 else -1
                for k in range(rng.i(1, 2)):
                    A.place("metal_trash_can", col, (gx1 - 1.0 - k * 0.7, (gy1 + 0.6) if s > 0 else (gy0 - 0.6), gz), yaw=rng.u(0, 6.28), sem=SEM["prop"])
            else:
                for k in range(rng.i(1, 2)):
                    A.place("metal_trash_can", col, (hx0 + 1.0 + k * 0.7, hy0 - 0.6, gz), yaw=rng.u(0, 6.28), sem=SEM["prop"])
            if A.available(["trashbag"]) and rng.p(0.4):
                A.place("trashbag", col, (STREET_X1 + 0.9, self.path_entry_y - 2.5, gz), yaw=rng.u(0, 6.28), sem=SEM["prop"])
        # street lamp + hydrant in the verge
        if lp["lamp"] and A.available(["street_lamp_01"]):
            A.place("street_lamp_01", col, (STREET_X1 + 0.5, self.y1 + rng.u(-1.0, 3.0), gz), yaw=math.pi / 2, target_h=rng.u(4.5, 6.0), sem=SEM["furniture"])
        if lp["hydrant"] and A.available(["fire_hydrant"]):
            A.place("fire_hydrant", col, (STREET_X1 + 0.6, self.y0 + rng.u(0.5, 3.0), gz), yaw=rng.u(0, 6.28), sem=SEM["furniture"])
        # loose yard props
        loose = A.available(["garden_sprinkler_01", "garden_gnome", "watering_can_metal_01", "plastic_crate_01", "wooden_crate_01", "wicker_basket_01", "old_tyre", "wooden_ladder", "cement_bag", "compost_bags", "propane_tank", "wooden_picnic_table", "outdoor_table_chair_set_01", "stone_fire_pit"])
        for aid in rng.shuffle(loose)[:rng.i(2, 5)]:
            back = aid in ("wooden_picnic_table", "outdoor_table_chair_set_01", "stone_fire_pit", "propane_tank", "compost_bags")
            for _ in range(30):
                x = rng.u(hx1 + 1.5, self.lot_x1 - 1.5) if back else rng.u(self.lot_x0 + 1.0, hx0 - 1.0)
                y = rng.u(self.y0 + 1.0, self.y1 - 1.0)
                if self.free(x, y, 0.7):
                    A.place(aid, col, (x, y, gz), yaw=rng.u(0, 6.28), sem=SEM["prop"]); break

    # ----- neighbours and background -----
    def neighbors(self):
        lp, rng = self.lp, self.rng; gz = GROUND_Z
        trees = A.available(TREE_SPECIES)
        Wn = lp["W"]
        # adjacent lots (same side): houses at +-(W/2 + Wn/2), mirrored strata irrelevant -> plain street doors
        for k, side in enumerate((-1, 1)):
            r = rng.sub(f"nb{k}")
            hp = house_params(r, "street", lod="neighbor"); hp["garage"] = r.p(0.6)
            cy = side * (Wn / 2 + Wn / 2 + r.u(0.5, 2.0))
            H = House(r, hp, self.nb_col, front_x=lp["setback"] + r.u(-2.5, 2.5), cy=cy, prefix=f"nb{k}")
            for o in H.objs: o.pass_index = SEM["neighbor"] if o.pass_index in (SEM["wall"], SEM["roof"], SEM["trim"], SEM["window"]) else o.pass_index
            # second-ring neighbours (lower detail): plain masses further out
            cy2 = side * (Wn * 1.5 + Wn / 2 + r.u(1.0, 3.0))
            self.block_house(r, lp["setback"] + r.u(-2, 3), cy2, f"nb{k}b")
            for _ in range(r.i(1, 3)):
                if trees: A.place(r.choice(trees), self.veg_col, (r.u(self.lot_x0 + 2, lp["setback"] - 3), cy + r.u(-Wn / 2 + 1.5, Wn / 2 - 1.5), gz), yaw=r.u(0, 6.28), target_h=r.u(5, 9), sem=SEM["vegetation"])
            for _ in range(r.i(1, 3)):
                if trees: A.place(r.choice(trees), self.veg_col, (r.u(lp["setback"] + 12, self.lot_x1), cy + r.u(-Wn / 2 + 1.5, Wn / 2 - 1.5), gz), yaw=r.u(0, 6.28), target_h=r.u(5, 10), sem=SEM["vegetation"])
        # across the street: three houses facing +X
        for k, cy in enumerate((-Wn * 1.0, 0.0, Wn * 1.0)):
            r = rng.sub(f"across{k}")
            fx = STREET_X0 - VERGE_W - SIDEWALK_W - r.u(7.0, 11.0)
            self.block_house(r, fx, cy + r.u(-2, 2), f"ac{k}", facing_pos_x=True)
            for _ in range(r.i(1, 2)):
                if trees: A.place(r.choice(trees), self.veg_col, (fx + r.u(2.5, 6.0), cy + r.u(-6, 6), gz), yaw=r.u(0, 6.28), target_h=r.u(5, 9), sem=SEM["vegetation"])
        # distant tree belt
        if trees:
            for ang in range(0, 360, 9):
                a = math.radians(ang + rng.u(-3, 3)); R = rng.u(75, 95)
                A.place(rng.choice(trees), self.veg_col, (R * math.cos(a), R * math.sin(a), gz), yaw=rng.u(0, 6.28), target_h=rng.u(9, 14), sem=SEM["vegetation"])

    def block_house(self, r, fx, cy, prefix, facing_pos_x=False):
        """Low-detail house mass with roof and window boxes (background filler)."""
        W, D, st = r.u(10, 14), r.u(8, 11), (2 if r.p(0.5) else 1)
        plinth = 0.4; ztop = plinth + st * 2.85
        walls = M.textures_by_role("wall"); roofs = M.textures_by_role("roof")
        mw = M.pbr(r.choice(walls), tile_m=r.u(1.6, 2.6)) if walls else M.flat("nbwall", (0.7, 0.65, 0.6))
        mr = M.pbr(r.choice(roofs), tile_m=1.2) if roofs else M.flat("nbroof", (0.3, 0.25, 0.22))
        mt = M.flat("nbtrim", (0.9, 0.9, 0.88)); mg = M.glass()
        cx = fx - D / 2 if facing_pos_x else fx + D / 2
        col = self.nb_col
        box(f"{prefix}_mass", (cx, cy, (plinth + ztop) / 2), (D, W, ztop - plinth), col, mw, SEM["neighbor"])
        box(f"{prefix}_found", (cx, cy, plinth / 2), (D + 0.1, W + 0.1, plinth), col, M.flat("nbconc", (0.55, 0.55, 0.52)), SEM["neighbor"])
        ov = 0.5; ph = (D / 2 + ov) * math.tan(math.radians(r.u(22, 32)))
        if r.p(0.5): prism_gable(f"{prefix}_roof", (cx, cy, ztop), (D + 2 * ov, W + 2 * ov, 0), ph, col, mr, SEM["neighbor"], ridge_axis='y')
        else:
            from .common import prism_hip
            prism_hip(f"{prefix}_roof", (cx, cy, ztop), (D + 2 * ov, W + 2 * ov, 0), ph, col, mr, SEM["neighbor"])
        # windows + a door as surface boxes (no boolean)
        f = -1 if not facing_pos_x else 1; wx = fx + f * 0.02
        n = max(1, int((W - 1.2) / 3.0))
        for s_ in range(st):
            for i in range(n):
                yy = cy - W / 2 + W * (i + 0.5) / n
                box(f"{prefix}_wf", (wx, yy, plinth + s_ * 2.85 + 0.9 + 0.7), (0.06, 1.3, 1.5), col, mt, SEM["neighbor"])
                box(f"{prefix}_wg", (wx + f * 0.02, yy, plinth + s_ * 2.85 + 0.9 + 0.7), (0.04, 1.15, 1.35), col, mg, SEM["neighbor"])
        yd = cy + r.u(-0.3, 0.3) * W
        box(f"{prefix}_door", (wx + f * 0.02, yd, plinth + 1.05), (0.08, 1.0, 2.1), col, M.flat("nbdoor", r.choice([(0.5, 0.08, 0.05), (0.05, 0.1, 0.3), (0.05, 0.05, 0.05), (0.8, 0.8, 0.78)])), SEM["neighbor"])
        box(f"{prefix}_stoop", (fx + f * 0.8, yd, plinth / 2), (1.6, 2.0, plinth), col, M.flat("nbconc2", (0.55, 0.55, 0.52)), SEM["neighbor"])
