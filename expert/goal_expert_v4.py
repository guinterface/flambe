"""Plan and follow collision-aware 3-D routes to delivery goals using yard occupancy."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

import visual_expert_syn as ve
import goal_expert as gx
import grid3d as g3

LOOK_M = 0.45          # carrot distance along the route
V_MAX = 1.5            # m/s on straight segments
V_CORNER = 0.6         # m/s within 1 m of a turn sharper than 30 deg
V_END = 0.15           # m/s floor of the end-of-route ramp
YAW_BLEND = (1.0, 2.5) # m horizontal distance to the goal: below the first, face yaw; above the second, bearing to the goal
HOLD_STEPS = 10        # 0.5 s settled at hover before descent / end
K_VEL = 1.4
K_CT = 1.0             # 1/s cross-track correction toward the closest route point
TAU = 1.0 / K_VEL      # s, velocity-loop time constant (a = K_VEL (v_des - v))
A_DEC = 0.5            # m/s^2 braking used for the speed profile ahead of turns
DEV_K = 0.8            # cross-track overshoot at a turn ~ DEV_K * v * TAU * sin(turn)  (fitted on FiGS / lagged kinematics)
PHASE_ID = {"cruise": 0, "approach": 1, "descent": 2, "abort": 6}


SOFT = 1.5            # expert routes: soft clearance cost (pure-length routes, soft=0, are the reference)


def plan(gr: g3.Grid, g: dict, p0, cache: dict, soft=SOFT):
    """(route polyline, pose, face index, length incl. descent) over the goal's faces, or None"""
    best = None
    for fi, pose in enumerate(g3.goal_poses(gr, g)):
        if pose is None:
            continue
        R = None
        for r in (g3.R_FREE, g3.R_MIN):          # Fall back to R_MIN for this start if needed.
            k = (g["id"], fi, soft, r)
            if k not in cache:
                cache[k] = gr.field(pose["hover"], g3.ceil_for(pose), r, soft)
            if cache[k] is not None:
                R = cache[k](np.asarray(p0, float))
            if R is not None: break
        if R is None:
            continue
        L = g3.plen(R) + float(np.linalg.norm(pose["land"] - pose["hover"]))
        if best is None or L < best[3]:
            best = (R, pose, fi, L)
    return best


class DirectGoalExpert(ve.GeometricExpert):
    def __init__(self, yard: gx.GoalYard, gr: g3.Grid, p0, cache: dict, verbose=False):
        self.yard = yard; self.gr = gr; g = yard.goal_rec
        self.pts = [np.asarray(p, float) for p in g["points_ned"]]
        self.plan = plan(gr, g, p0, cache)
        if self.plan is not None:
            R, pose, fi, L = self.plan
            if "faces" in g:
                yard.set_face(fi)
            yard.hover = np.asarray(pose["hover"], float); yard.goal = np.asarray(pose["land"], float)
            yard.fly_z = float(pose["hover"][2]); yard.land_z = float(pose["land"][2]); yard.door_yaw = float(pose["yaw"])
            self.route = [np.asarray(p, float) for p in R]; self.pose = pose; self.ref_len = L
            self.vv = self._vertex_speeds()
        else:
            self.route = [np.asarray(p0, float)]; self.pose = None; self.ref_len = None; self.vv = [0.0]
        self.log = []
        super().__init__(yard, [self.route[-1][:2]], float(p0[2]), verbose)

    def reset(self):
        super().reset()
        self.phase = "cruise" if self.pose is not None else "abort"
        self.no_route = self.pose is None; self.blind_commit = False; self.committed = True
        self.seg = 0; self._t_on_seg = 0.0; self.hold = 0; self.vis = False; self.last_px = None; self.log = []; self.yaw_cmd = None

    def _vertex_speeds(self):
        """allowed speed at each route vertex: the lag overshoot of a turn must stay inside the clearance there
        (keeping >= 0.42 m to the nearest occupied cell)."""
        P = self.route; vv = [V_MAX] * len(P); vv[-1] = 0.0
        for k in range(1, len(P) - 1):
            u1, u2 = P[k] - P[k - 1], P[k + 1] - P[k]
            cs = float(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-9)); sn = math.sqrt(max(0.0, 1 - min(1.0, cs) ** 2)) if cs > 0 else 1.0
            near = [P[k]] + [P[k] + (P[k + 1] - P[k]) * f for f in (0.2, 0.4)] + [P[k] + (P[k - 1] - P[k]) * f for f in (0.2, 0.4)]
            room = float(self.gr.clearance_plan(np.array(near)).min()) - 0.60
            vv[k] = float(np.clip(room / (DEV_K * TAU * sn + 1e-6), 0.4, V_MAX))
        return vv

    # ---- route following ----
    def _at(self, i, t, s):
        """point s metres further along the route from (segment i, offset t), and its segment index"""
        P = self.route; j, t2, rem = i, t, s
        while True:
            L = float(np.linalg.norm(P[j + 1] - P[j])) + 1e-9
            if t2 + rem <= L or j == len(P) - 2:
                return P[j] + (P[j + 1] - P[j]) / L * min(t2 + rem, L), j
            rem -= L - t2; j += 1; t2 = 0.0

    def _carrot(self, p):
        """carrot LOOK_M ahead, path length left, and a speed cap: V_CORNER if the heading changes by more than 30 deg
        within the next 1.2 m, and a clearance cap from the planned clearance 0..1.5 m ahead."""
        P = self.route; best = None
        for i in range(self.seg, min(self.seg + 3, len(P) - 1)):
            a, b = P[i], P[i + 1]; ab = b - a; L = float(np.linalg.norm(ab)) + 1e-9
            t = float(np.clip(np.dot(p - a, ab) / L, 0.0, L)); d = float(np.linalg.norm(a + ab / L * t - p))
            if best is None or d < best[0] - 1e-6:
                best = (d, i, t)
        if best is None:
            return P[-1], 0.0, V_MAX
        _, i, t = best; self.seg = i; self._t_on_seg = t
        c, _ = self._at(i, t, LOOK_M); self._closest = self._at(i, t, 0.0)[0]
        segL = [float(np.linalg.norm(P[k + 1] - P[k])) for k in range(len(P) - 1)]
        left = segL[i] - t + sum(segL[i + 1:])
        cap = V_MAX; dist = segL[i] - t
        for k in range(i + 1, len(P) - 1):            # braking profile toward every vertex ahead
            cap = min(cap, math.sqrt(self.vv[k] ** 2 + 2 * A_DEC * dist))
            dist += segL[k]
            if dist > 6.0: break
        ahead = np.array([self._at(i, t, s)[0] for s in (0.0, 0.5, 1.0, 1.5)] + [p])
        clr = float(self.gr.clearance_plan(ahead).min())
        cap = min(cap, float(np.clip(0.5 + 2.5 * (clr - g3.R_FREE), 0.5, V_MAX)))
        return c, left, cap

    def _yaw_des(self, pos):
        """camera on the goal while the line of sight to it is clear; otherwise along the route 3 m ahead, so it does not stare
        at a wall while rounding a corner. 5-frame hysteresis.
        Within YAW_BLEND of the goal it turns to face the goal squarely."""
        clear = g3.los(self.gr, np.asarray(pos, float), self.pts)
        self._los_n = getattr(self, "_los_n", 0) + (1 if clear != getattr(self, "_los", clear) else -getattr(self, "_los_n", 0))
        if not hasattr(self, "_los") or self._los_n >= 5:
            self._los = clear; self._los_n = 0
        if self._los:
            look = self.pose["look"]
        else:
            look = self._at(self.seg, self._t_on_seg, 3.0)[0]
        dv = look[:2] - pos[:2]; bearing = math.atan2(dv[1], dv[0]); face = self.pose["yaw"]
        dh = float(np.linalg.norm(self.pose["look"][:2] - pos[:2]))
        w = float(np.clip((dh - YAW_BLEND[0]) / (YAW_BLEND[1] - YAW_BLEND[0]), 0.0, 1.0))
        return ve.wrap(face + w * ve.wrap(bearing - face))

    def decide(self, t, xcr, rgb=None, depth=None):
        pos, vel, q = xcr[0:3], xcr[3:6], xcr[6:10]
        self.vis, self.last_px = g3.visible(self.gr, np.asarray(pos, float), q, self.pts)
        ph = self.phase
        if ph == "cruise":
            tgt, left, cap = self._carrot(np.asarray(pos, float))
            tan = tgt - self._closest; tan = tan / (np.linalg.norm(tan) + 1e-6)   # 3-D tangent: climbs take horizontal speed off
            speed = min(cap, 0.6 * left + V_END)
            dz = abs(float(pos[2] - self._closest[2]))
            speed *= float(np.clip(1.0 - (dz - 0.3) / 0.6, 0.25, 1.0))
            v_des = speed * tan[:2] + K_CT * (self._closest[:2] - pos[:2])     # along the route + back onto it
            nv = float(np.linalg.norm(v_des)); v_des = v_des * min(1.0, V_MAX / (nv + 1e-9))
            u, _ = self._u(pos, vel, q, v_des, float(tgt[2]), self._yaw_des(pos))
            if left < 0.25:
                self.phase = "approach"
        elif ph == "approach":
            hv = self.pose["hover"]; dvec = hv[:2] - pos[:2]; d = float(np.linalg.norm(dvec))
            v_des = min(0.4, 0.8 * d) * dvec / (d + 1e-6)
            u, herr = self._u(pos, vel, q, v_des, float(hv[2]), self.pose["yaw"])
            settled = d < 0.15 and abs(pos[2] - hv[2]) < 0.15 and np.linalg.norm(vel) < 0.15 and abs(herr) < 0.1
            self.hold = self.hold + 1 if settled else 0
            if self.hold >= HOLD_STEPS:
                if np.linalg.norm(self.pose["land"] - hv) < 0.05:
                    self._landed = True                       # windows: the hover pose is the goal
                else:
                    self.phase = "descent"; self.z_set = float(pos[2])
        elif ph == "descent":
            ld = self.pose["land"]
            self.z_set = min(float(ld[2]), self.z_set + ve.V_DESCENT / ve.HZ_CTL)
            v_des = np.clip(ve.K_POS * (ld[:2] - pos[:2]), -0.3, 0.3)
            u, _ = self._u(pos, vel, q, v_des, self.z_set, self.pose["yaw"], w_clip=0.3)
            if pos[2] >= ld[2] - 0.05:
                self._landed = True
        else:  # abort: no route
            u, _ = self._u(pos, vel, q, np.zeros(2), float(pos[2]), ve.yaw_of(q))
            self._landed = True
        self.counts[self.phase] = self.counts.get(self.phase, 0) + 1
        self.step += 1
        self.log.append(self.labels(xcr))
        return u

    def _u(self, pos, vel, q, v_des_xy, z_set, yaw_des, w_clip=0.4):
        """the base controller (visual_expert_syn.GeometricExpert._u) with K_VEL raised to K_VEL"""
        self._guid = (float(v_des_xy[0]), float(v_des_xy[1]), float(z_set), float(yaw_des))
        a = K_VEL * (np.asarray(v_des_xy) - vel[:2])
        z_des = np.array([-a[0], -a[1], ve.G_NED]); z_des /= np.linalg.norm(z_des)
        R = Rotation.from_quat(q).as_matrix(); e = np.cross(R[:, 2], z_des)
        wx = float(np.clip(ve.K_ATT * np.dot(e, R[:, 0]), -w_clip, w_clip)); wy = float(np.clip(ve.K_ATT * np.dot(e, R[:, 1]), -w_clip, w_clip))
        f = ve.F_HOV - ve.K_ALT * (pos[2] - z_set) - ve.K_VZ * vel[2]
        hdg_err = ve.wrap(yaw_des - ve.yaw_of(q))
        oz = float(np.clip(0.7 * hdg_err, -0.8, 0.8)); oz = float(np.clip(oz, self.prev_oz - 0.12, self.prev_oz + 0.12)); self.prev_oz = oz
        u = np.array([float(np.clip(f, ve.THRUST_MIN, ve.THRUST_MAX)), wx, wy, float(np.clip(oz, -ve.OMEGA_CLIP, ve.OMEGA_CLIP))])
        return u, hdg_err

    def labels(self, xcr) -> dict:
        pos, q = xcr[0:3], xcr[6:10]
        c = np.asarray(self.yard.goal_rec.get("center_ned", self.yard.goal), float)
        gb = Rotation.from_quat(q).as_matrix().T @ (c - pos)
        return {"phase": PHASE_ID.get(self.phase, -1), "visible": int(bool(self.vis)), "goal_body": gb.tolist(),
                "goal_px": self.last_px or [-1.0, -1.0], "committed": 1,
                "guid": list(getattr(self, "_guid", (0.0, 0.0, float(pos[2]), 0.0))),
                "land": [float(v) for v in self.yard.goal], "hover": [float(v) for v in self.yard.hover],
                "clr": float(self.gr.clearance(np.asarray(pos, float))[0])}   # 3-D clearance (band + columns above it)
