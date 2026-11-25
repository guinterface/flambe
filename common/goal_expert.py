"""Search for a delivery goal using learned location priors and approach it after sighting."""
from __future__ import annotations

import json, math
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import visual_expert_syn as ve

COMMIT_K = 4             # 0.2 s of continuous sight
MIN_PX = 10.0            # projected extent of the goal (px) before it counts as seen
VIEW_OUT = 5.0           # m in front of the facade for a search viewpoint
COVER_R = 7.0            # m outside the house rectangle for the cover ring
LAND_CLR = 0.5           # m, clearance kept when descending to an object goal
SPIN_LEAD = 0.9          # rad ahead of the current yaw: about 0.6 rad/s with the expert's yaw law
CAM = {"w": 640, "h": 360, "fx": 462.956, "fy": 463.002, "cx": 323.076, "cy": 181.184}
LOOK_M = 0.35            # carrot distance along the planned path
SEARCH_BUDGET_S = 130.0  # no sighting by then -> blind commit (flagged; blind flights are not student data)
V_CORNER = 0.6           # m/s within 1 m of a turn sharper than 30 deg
PHASE_ID = {"cruise": 0, "approach": 1, "descent": 2, "search": 3, "spin": 4, "ccruise": 5, "abort": 6}


class GoalYard(ve.Yard):
    """A Yard whose door attributes point at one face of a goal from goals.json, so GeometricExpert works unchanged."""

    def __init__(self, yard_dir: Path, goal: dict):
        super().__init__(yard_dir)
        self.goal_rec = goal
        t = np.load(yard_dir / "recon" / "truth.npz")
        self.h_top = np.asarray(t["height_top"], float); ex = np.asarray(t["extent_blender"], float)
        self.h_x0, self.h_y0 = float(ex[0]), float(ex[2])
        if goal["kind"] != "door":
            self.set_face(0)

    def set_face(self, i: int):
        f = self.goal_rec["faces"][i] if "faces" in self.goal_rec else self.goal_rec
        self.face = np.asarray(f["face_ned"], float)
        self.n = np.asarray(f["normal_ned"], float); self.n[2] = 0; self.n /= np.linalg.norm(self.n)
        self.hover = np.asarray(f["hover_ned"], float); self.goal = np.asarray(f["land_ned"], float)
        self.standoff = np.asarray(f["standoff_ned"], float)
        self.fly_z = float(self.hover[2]); self.land_z = float(self.goal[2])
        self.cruise_z = float(min(ve.CRUISE_Z_DEFAULT, self.fly_z)); self.step = self.goal.copy()
        self.door_yaw = math.atan2(-self.n[1], -self.n[0])
        if self.goal_rec["kind"] != "window":
            self._safe_land()

    def _safe_land(self):
        """Descend only as low as keeps LAND_CLR to the nearest
        occupied cell; if no height below the hover point does, the flight ends at hover (no descent)."""
        zs = np.arange(self.hover[2], self.goal[2] + 1e-6, 0.05)          # NED: from hover down toward the land point
        P = np.array([[self.hover[0], self.hover[1], z] for z in zs]); c = self.clearance(P)
        ok = np.where(np.minimum.accumulate(c) >= LAND_CLR)[0]
        z = float(zs[ok[-1]]) if len(ok) else float(self.hover[2])
        self.goal = np.array([self.hover[0], self.hover[1], z]); self.land_z = z; self.step = self.goal.copy()

    def slice_mask(self, z, inflate_m=ve.INFLATE_M):
        """A goal flight flies at the cruise height and, near the goal, at the hover height
        (up to 2.3 m for windows), so every planning mask is the union of those slices and the requested one."""
        m = super().slice_mask(z, inflate_m)
        for z2 in {round(float(v), 2) for v in (getattr(self, "cruise_z", z), getattr(self, "fly_z", z), getattr(self, "plan_z", None) or z)}:
            if abs(z2 - z) > 0.2:
                m = m | super().slice_mask(z2, inflate_m)
        return m

    def blocked(self, x) -> bool:
        i = np.floor((x - self.lo) / self.cell).astype(int)
        if np.all(i >= 0) and np.all(i < self.occ3d.shape):
            return bool(self.occ3d[tuple(i)])
        # outside the band: anything the top-down height map says is taller than this point (heights above the street)
        hx = int((x[0] - self.h_x0) / self.cell); hy = int((-x[1] - self.h_y0) / self.cell)
        if 0 <= hx < self.h_top.shape[0] and 0 <= hy < self.h_top.shape[1]:
            return bool(self.h_top[hx, hy] > -x[2] + 0.05)
        return False


def cam_axes(q_xyzw):
    R = Rotation.from_quat(q_xyzw).as_matrix()
    f = R @ np.array([1.0, 0.0, 0.0]); r = R @ np.array([0.0, 1.0, 0.0]); d = np.cross(f, r)
    return f, r, d


def project(eye, q, p):
    f, r, d = cam_axes(q); v = np.asarray(p) - eye; z = v @ f
    if z <= 0.1:
        return None
    u = CAM["fx"] * (v @ r) / z + CAM["cx"]; w = CAM["fy"] * (v @ d) / z + CAM["cy"]
    return (u, w) if (0 <= u < CAM["w"] and 0 <= w < CAM["h"]) else None


def ray_clear(yard: GoalYard, eye, p, end_margin=0.4, step=0.1):
    v = np.asarray(p) - eye; L = float(np.linalg.norm(v))
    for s in np.arange(0.3, max(0.3, L - end_margin), step):
        if yard.blocked(eye + v * (s / L)):
            return False
    return True


def visible(yard, eye, q, pts, need=2):
    uv_all, uv_clear = [], []
    for p in pts:
        uv = project(eye, q, p)
        if uv is None:
            continue
        uv_all.append(uv)
        if ray_clear(yard, eye, p):
            uv_clear.append(uv)
    if len(uv_clear) < need:
        return False, None
    A = np.asarray(uv_all); ext = float(max(np.ptp(A[:, 0]), np.ptp(A[:, 1])))
    return ext >= MIN_PX, np.mean(uv_clear, axis=0).tolist()


def start_yaw(survey: dict, st: dict) -> float:
    """Start heading for goal flights (NED, radians): toward the house centre plus the start's
    own offset (0 or +-60 deg). survey.json's yaw_ned_deg points along the street toward the door's side
    (truth.make_starts), which would tell the student where the door is; this heading is the same for every goal."""
    hx0, hx1, hy0, hy1 = survey["house_rect_ned"][:4]
    p = np.asarray(st["pos_ned"], float)
    b = math.atan2((hy0 + hy1) / 2 - p[1], (hx0 + hx1) / 2 - p[0])
    return float(ve.wrap(b + math.radians(float(st.get("yaw_offset_deg", 0.0)))))


def prior_key(g: dict) -> str:
    if g["kind"] == "door":
        return "door"
    if g["kind"] == "window":
        return "window/" + g.get("side", "front")
    return "object/" + g.get("name", "").split(" at the ")[0]


def load_priors(path: Path, yard_name: str) -> dict:
    if not path.exists():
        return {}
    P = json.load(open(path))
    return P.get("loo", {}).get(yard_name) or P.get("all", {})


def free_near(mask, c, max_r=40):
    """nearest free cell by BFS (cells), larger radius than ve.nearest_free"""
    c = tuple(int(v) for v in np.clip(c, 0, np.array(mask.shape) - 1))
    if not mask[c]:
        return c
    seen = {c}; q = deque([c])
    while q:
        a = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            b = (a[0] + d[0], a[1] + d[1])
            if 0 <= b[0] < mask.shape[0] and 0 <= b[1] < mask.shape[1] and b not in seen and abs(b[0] - c[0]) + abs(b[1] - c[1]) <= max_r:
                if not mask[b]:
                    return b
                seen.add(b); q.append(b)
    return None


class TwoPhaseExpert(ve.GeometricExpert):
    def __init__(self, yard: GoalYard, start_xy, priors: dict, verbose=False):
        self.yard = yard; g = yard.goal_rec
        self.pts = [np.asarray(p, float) for p in g["points_ned"]]
        hx0, hx1, hy0, hy1 = yard.survey["house_rect_ned"][:4]
        self.hc = np.array([(hx0 + hx1) / 2, (hy0 + hy1) / 2]); self.half = np.array([abs(hx1 - hx0) / 2, abs(hy1 - hy0) / 2])
        sx = float(np.mean(yard.survey["street_x_ned"]))
        self.front = np.array([np.sign(sx - self.hc[0]) or -1.0, 0.0]); self.lat = np.array([-self.front[1], self.front[0]])
        self.mask = yard.slice_mask(getattr(yard, "plan_z", None) or yard.cruise_z)
        self.start_xy = np.asarray(start_xy, float)
        modes = priors.get(prior_key(g)) or priors.get("_any") or [{"w": 1.0, "ab": [1.0, 0.0]}]
        self.targets = [self._viewpoint(m["ab"]) for m in sorted(modes, key=lambda m: -m["w"])]   # (viewpoint, look-at)
        self.targets = [t for t in self.targets if t is not None]
        self.n_prior = len(self.targets)          # prior viewpoints get a full turn; cover-ring points do not (camera already on the house)
        self.targets += self._cover_ring()
        self.log = []
        super().__init__(yard, [self.targets[0][0]], yard.cruise_z, verbose)

    # ---- geometry of the search ----
    def _hf_to_xy(self, ab):
        return self.hc + self.front * ab[0] * self.half[0] + self.lat * ab[1] * self.half[1]

    def _viewpoint(self, ab):
        look = self._hf_to_xy(ab); out = look - self.hc
        out = out / (np.linalg.norm(out) + 1e-6)
        c = free_near(self.mask, self.yard.to_cell(look + VIEW_OUT * out))
        return None if c is None else (np.asarray(self.yard.to_xy(c)), look)

    def _cover_ring(self, k=8):
        out = []
        for i in range(k):
            a = 2 * math.pi * i / k; d = np.array([math.cos(a), math.sin(a)])
            # start the ring on the street side and go around
            d = self.front * d[0] + self.lat * d[1]
            p = self.hc + d * (np.abs(d) @ self.half + COVER_R)
            c = free_near(self.mask, self.yard.to_cell(p))
            if c is not None:
                out.append((np.asarray(self.yard.to_xy(c)), self.hc.copy()))
        return out

    def _route_to(self, xy, target_xy):
        c0 = free_near(self.mask, self.yard.to_cell(xy)); c1 = free_near(self.mask, self.yard.to_cell(target_xy))
        if c0 is None or c1 is None:
            return None          # no straight-line fallback through obstacles
        cells = ve.astar(self.mask, c0, c1)
        if cells is None:
            return None
        pts = [tuple(xy)] + [tuple(self.yard.to_xy(c)) for c in cells[1:]]
        return [np.asarray(w, float) for w in (ve.shortcut(pts, self.mask, self.yard.cell, self.yard.lo[:2])[1:] or [target_xy])]

    def reset(self):
        super().reset()
        self.phase = "search"; self.ti = 0; self.seen = 0; self.committed = False; self.blind_commit = False
        self.spin_done = 0.0; self.prev_yaw = None; self.vis = False; self.last_px = None; self.log = []
        self.no_route = False; self.waypoints = None; self.visited = set()
        while self.waypoints is None and self.ti < len(self.targets):
            self.waypoints = self._route_to(self.start_xy, self.targets[self.ti][0])
            if self.waypoints is None:
                self.ti += 1
        if self.waypoints is None:
            self.waypoints = [np.asarray(self.start_xy)]; self.no_route = True; self.phase = "abort"
        self.wp = 0

    def _commit(self, pos, blind=False):
        g = self.yard.goal_rec
        if "faces" in g:     # the reachable face nearest the drone: decided by where the drone is, not by a hidden choice
            i = int(np.argmin([np.linalg.norm(np.asarray(f["standoff_ned"][:2]) - pos[:2]) for f in g["faces"]]))
            self.yard.set_face(i); self.door_ned = self.yard.goal
        wps, _ = ve.plan_route(self.yard, pos[:2], self.yard.cruise_z)
        self.committed = True; self.blind_commit = blind; self.wp = 0
        if not wps:              # no route to the standoff -> hold and end the flight, flagged
            self.waypoints = [np.asarray(pos[:2], float)]; self.no_route = True; self.phase = "abort"; return
        self.waypoints = [np.asarray(w, float) for w in wps]; self.phase = "ccruise"

    def _carrot(self, p):
        """closest point on the planned polyline (monotone progress), the point LOOK_M further on, the path length left,
        and whether a turn sharper than 30 deg is within 1 m"""
        P = self._poly; best = None
        for i in range(self._seg, min(self._seg + 3, len(P) - 1)):
            a, b = P[i], P[i + 1]; ab = b - a; L = float(np.linalg.norm(ab)) + 1e-9
            t = float(np.clip(np.dot(p - a, ab) / L, 0.0, L)); dist = float(np.linalg.norm(a + ab / L * t - p))
            if best is None or dist < best[0] - 1e-6:
                best = (dist, i, t)
        _, i, t = best; self._seg = i
        rem, j, t2 = LOOK_M, i, t
        while True:
            L = float(np.linalg.norm(P[j + 1] - P[j])) + 1e-9
            if t2 + rem <= L or j == len(P) - 2:
                c = P[j] + (P[j + 1] - P[j]) / L * min(t2 + rem, L); break
            rem -= L - t2; j += 1; t2 = 0.0
        seg_len = [float(np.linalg.norm(P[k + 1] - P[k])) for k in range(len(P) - 1)]
        left = seg_len[i] - t + sum(seg_len[i + 1:])
        near = False
        for k in (i, i - 1):          # the next vertex ahead, and the one just passed
            if 0 <= k and k + 2 < len(P):
                u1, u2 = P[k + 1] - P[k], P[k + 2] - P[k + 1]
                cosang = float(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-9))
                dv = (seg_len[i] - t) if k == i else t
                near = near or (cosang < 0.87 and dv < 1.0)
        return c, left, near

    def _next_target(self, pos):
        """prior viewpoints in prior order; after them, the unvisited cover-ring point with the SHORTEST route
        from where the drone is (was: fixed ring order, which crossed the yard back and forth and timed out)."""
        self.visited = getattr(self, "visited", set()) | {self.ti}
        nxt = self.ti + 1
        if nxt < self.n_prior:
            w = self._route_to(pos[:2], self.targets[nxt][0])
            if w is not None:
                self.ti = nxt; self.waypoints = w; self.wp = 0; self.phase = "search"; return
            self.visited.add(nxt); self.ti = nxt
            return self._next_target(pos) if nxt + 1 < self.n_prior else self._next_target_ring(pos)
        return self._next_target_ring(pos)

    def _next_target_ring(self, pos):
        best = None
        for i in range(self.n_prior, len(self.targets)):
            if i in self.visited:
                continue
            w = self._route_to(pos[:2], self.targets[i][0])
            if w is None:
                self.visited.add(i); continue
            L = sum(float(np.linalg.norm(b - a)) for a, b in zip([np.asarray(pos[:2], float)] + w[:-1], w))
            if best is None or L < best[0]:
                best = (L, i, w)
        if best is None:
            self._commit(pos, blind=True); return
        _, self.ti, self.waypoints = best; self.wp = 0; self.phase = "search"

    # ---- control ----
    def _u(self, pos, vel, q, v_des_xy, z_set, yaw_des, w_clip=0.4):
        """Record the guidance targets of every decision (all phases, including GeometricExpert's
        approach / hold / descent), so velocity-level labels come from the flight itself and need no replay with
        priors that may have been refitted since (they were, at 17:19, for the first syn_002 goal_data run)."""
        self._guid = (float(v_des_xy[0]), float(v_des_xy[1]), float(z_set), float(yaw_des))
        return super()._u(pos, vel, q, v_des_xy, z_set, yaw_des, w_clip)

    def decide(self, t, xcr, rgb=None, depth=None):
        pos, vel, q = xcr[0:3], xcr[3:6], xcr[6:10]
        self.vis, self.last_px = visible(self.yard, pos, q, self.pts)
        if not self.committed:
            self.seen = self.seen + 1 if self.vis else 0
            if self.seen >= COMMIT_K:
                self._commit(pos)
            elif t > SEARCH_BUDGET_S and self.phase in ("search", "spin"):
                self._commit(pos, blind=True)
        ph = self.phase
        if ph in ("search", "ccruise"):
            # follow the planned polyline (carrot LOOK_M ahead) instead of jumping to the next waypoint at 1.2 m
            if getattr(self, "_route_id", None) != id(self.waypoints):
                self._route_id = id(self.waypoints); self._seg = 0
                self._poly = [np.asarray(pos[:2], float).copy()] + [np.asarray(w, float) for w in self.waypoints]
            tgt, d, near_turn = self._carrot(np.asarray(pos[:2], float))
            last = self._seg >= len(self._poly) - 2
            self.wp = len(self.waypoints) - 1 if last else min(self._seg, len(self.waypoints) - 1)
            look = (self.targets[self.ti][1] if ph == "search" else self.yard.face[:2]) - pos[:2]
            yaw_des = math.atan2(look[1], look[0])            # the camera stays on the expected place / the goal
            dc = float(np.linalg.norm(tgt - pos[:2])); unit = (tgt - pos[:2]) / (dc + 1e-6)
            speed = min(ve.V_CRUISE if not last else 0.8, 0.5 * d + 0.3)
            if near_turn:
                speed = min(speed, V_CORNER)
            if abs(ve.wrap(yaw_des - ve.yaw_of(q))) > 1.2:
                speed *= 0.5                                  # slow while the camera swings round
            z_set = self.cruise_z     # hold cruise height to the standoff; the approach (checked clear at hover height in goals.py) descends
            u, _ = self._u(pos, vel, q, speed * unit, z_set, yaw_des)
            if last and d < 0.6:
                if ph == "search" and self.ti < self.n_prior:
                    self.phase = "spin"; self.spin_done = 0.0; self.prev_yaw = ve.yaw_of(q)
                elif ph == "search":
                    self._next_target(pos)
                else:
                    self.phase = "approach"
        elif ph == "abort":
            u, _ = self._u(pos, vel, q, np.zeros(2), self.cruise_z, ve.yaw_of(q))
            self._landed = True      # ends the episode; success is False because the drone is not at the goal
        elif ph == "spin":
            y = ve.yaw_of(q); self.spin_done += abs(ve.wrap(y - self.prev_yaw)); self.prev_yaw = y
            u, _ = self._u(pos, vel, q, np.zeros(2), self.cruise_z, y + SPIN_LEAD)
            if self.spin_done >= 2 * math.pi:
                self._next_target(pos)
        else:
            u = super().decide(t, xcr, rgb, depth)
            self.log.append(self.labels(xcr)); return u
        self.counts[self.phase] = self.counts.get(self.phase, 0) + 1
        self.step += 1
        self.log.append(self.labels(xcr))
        return u

    def labels(self, xcr) -> dict:
        pos, q = xcr[0:3], xcr[6:10]
        c = np.asarray(self.yard.goal_rec.get("center_ned", self.yard.goal), float)   # face-independent target
        gb = Rotation.from_quat(q).as_matrix().T @ (c - pos)
        return {"phase": PHASE_ID.get(self.phase, -1), "visible": int(self.vis), "goal_body": gb.tolist(),
                "goal_px": self.last_px or [-1.0, -1.0], "committed": int(self.committed),
                "guid": list(getattr(self, "_guid", (0.0, 0.0, float(pos[2]), 0.0))),
                "land": [float(v) for v in self.yard.goal], "hover": [float(v) for v in self.yard.hover],
                "clr": float(self.yard.clearance(np.asarray(pos, float)[None])[0])}   # per-frame clearance to the true geometry
