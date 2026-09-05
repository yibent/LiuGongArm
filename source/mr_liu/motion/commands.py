"""One arm lane: HTTP submits; only the simulation thread touches joints.

Acceptance is not completion. Every finite motion is checked against measured
joint positions. Interrupts bypass the normal lane and cancel pending work.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import ArmKinematics

MOTION_SKILLS = {"home", "move_joint", "move_cartesian", "rotate", "gripper", "set_speed", "resume", "hold", "stop"}


class MotionCommands:
    def __init__(self, urdf, defaults, clock=time.monotonic, config=None):
        self.kinematics = ArmKinematics(urdf)
        self.defaults = defaults
        self.clock = clock
        self.lock = threading.RLock()
        self.records = {}
        self.pending = None
        self.interruption = None
        self.active = None
        self.saved = None
        self.snapshot = {}
        self.speed = 0.5  # rad/s
        self.holding = False
        self.hold_target = None
        cfg = config or {}
        self.speed = np.deg2rad(float(cfg.get("speed_deg_s", 28.65)))
        self.acceleration = np.deg2rad(float(cfg.get("acceleration_deg_s2", 60)))
        self.position_tolerance = np.deg2rad(float(cfg.get("position_tolerance_deg", 1.0)))
        self.velocity_tolerance = np.deg2rad(float(cfg.get("velocity_tolerance_deg_s", 1.0)))
        self.settle_seconds = float(cfg.get("settle_seconds", .35))
        self.settle_timeout = float(cfg.get("settle_timeout_sim_s", 10))
        self.wall_timeout = float(cfg.get("wall_timeout_s", 120))
        self.settle_velocity_source = cfg.get("settle_velocity_source", "reported")
        self.last_sim_time = None
        self.last_joints = None

    def submit(self, skill, params, command_id=None):
        with self.lock:
            cid = command_id or str(uuid.uuid4())
            if cid in self.records:
                return copy.deepcopy(self.records[cid])
            if skill not in MOTION_SKILLS:
                raise NotImplementedError(skill)
            if self.interruption or ((self.pending or self.active) and skill not in {"hold", "stop"}):
                return {"ok": False, "state": "failed", "message": "BUSY：机械臂正在执行，请先暂停或等待完成", "command_id": cid}
            record = {"ok": True, "state": "accepted", "command_id": cid, "skill": skill,
                      "params": copy.deepcopy(params), "message": "控制器已接收，等待仿真线程执行", "accepted_at": self.clock()}
            self.records[cid] = record
            if skill in {"hold", "stop"}:
                self.interruption = cid
            else:
                self.pending = cid
            # Bound status retention without evicting live commands.
            for old in list(self.records):
                if len(self.records) <= 256:
                    break
                if self.records[old]["state"] in {"completed", "failed", "cancelled"}:
                    del self.records[old]
            return copy.deepcopy(record)

    def result(self, cid):
        with self.lock:
            return copy.deepcopy(self.records.get(cid))

    def status(self):
        with self.lock:
            return copy.deepcopy({**self.snapshot, "active_command_id": self.active or self.pending,
                                  "mode": "moving" if self.active else "hold" if self.holding else "idle",
                                  "can_resume": self.saved is not None, "speed_deg_s": np.rad2deg(self.speed),
                                  "drives": getattr(self, "drive_info", {}),
                                  "last_command": next(reversed(self.records.values()), None)})

    def _finish(self, cid, state, message):
        self.records[cid].update(state=state, ok=state == "completed", message=message,
                                 finished_at=self.clock(), measured=copy.deepcopy(self.snapshot))
        print(f"ARM_COMMAND id={cid} skill={self.records[cid]['skill']} state={state} {message}", flush=True)

    def release(self):
        with self.lock:
            if self.active or self.pending or self.interruption:
                raise ValueError("BUSY：请先停止当前动作再启用跟随")
            self.holding = False
            self.saved = None

    def tick(self, joints, base, apply, playing=True, *, sim_time=None, velocities=None):
        """Return True when this lane owns the drives (including HOLD)."""
        with self.lock:
            now = self.clock()
            # Production supplies the physics clock. The fallback keeps standalone
            # pure-controller callers compatible, without conflating audit timestamps.
            sim_time = now if sim_time is None else float(sim_time)
            dt = max(0., sim_time-self.last_sim_time) if self.last_sim_time is not None and playing else 0.
            frame_velocities = ({k: (v-self.last_joints[k])/dt for k, v in joints.items()}
                                if dt > 0 and self.last_joints else {k: float("inf") for k in joints})
            if velocities is None:
                velocities = frame_velocities
            self.last_sim_time, self.last_joints = sim_time, joints.copy()
            pose = self.kinematics.forward(joints, base)
            self.snapshot = {"joint_positions_deg": {k: float(np.rad2deg(v)) for k, v in joints.items()},
                             "joint_velocities_deg_s": {k: float(np.rad2deg(v)) if np.isfinite(v) else None for k, v in velocities.items()},
                             "joint_frame_velocities_deg_s": {k: float(np.rad2deg(v)) if np.isfinite(v) else None for k, v in frame_velocities.items()},
                             "tool_position_world_m": pose[:3, 3].tolist(),
                             "tool_orientation_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
                             "sampled_at": now, "simulation_time": sim_time, "simulation_playing": playing}
            if self.interruption:
                cid, self.interruption = self.interruption, None
                self.saved = copy.deepcopy(self.records[self.active].get("goal")) if self.active else self.saved
                for old in (self.active, self.pending):
                    if old:
                        self._finish(old, "cancelled", "动作已中断，未到达原目标")
                self.active = self.pending = None
                if self.records[cid]["skill"] == "stop":
                    self.saved = None
                self.holding, self.hold_target = True, joints.copy()
                apply(joints)
                self._finish(cid, "completed", "已保持当前关节位置" if self.saved else "已停止当前动作")
            if self.pending:
                cid, self.pending = self.pending, None
                record = self.records[cid]
                try:
                    if not playing:
                        raise ValueError("仿真已暂停，请在 Isaac 中恢复播放后重新下达动作")
                    goal = self._goal(record["skill"], record["params"], joints, base, pose)
                    if goal is None:
                        self._finish(cid, "completed", "运动速度已更新")
                    else:
                        goal = {k: float(v) for k, v in goal.items()}
                        for name, value in goal.items():
                            lo, hi = self.kinematics.limits[name]
                            if not np.isfinite(value) or not lo <= value <= hi:
                                raise ValueError(f"JOINT_LIMIT：{name} 目标超出 [{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}] 度；未执行")
                        distance = max(abs(goal[k]-v) for k, v in joints.items())
                        # Quintic smoothstep peak derivatives: 1.875 and 10/sqrt(3).
                        duration = max(.5, 1.875*distance/self.speed,
                                       np.sqrt((10/np.sqrt(3))*distance/self.acceleration))
                        record.update(state="started", message="轨迹已开始，等待实测到位", goal=goal,
                                      start=joints.copy(), started_at=now, started_sim_time=sim_time,
                                      duration=duration, settled=0, settled_sim_s=0.)
                        self.active = cid
                        self.saved = None
                        self.holding = True
                        print(f"ARM_COMMAND id={cid} skill={record['skill']} state=started goal={goal}", flush=True)
                except (ValueError, KeyError, TypeError) as exc:
                    self._finish(cid, "failed", str(exc))
            if self.active:
                cid = self.active
                record = self.records[cid]
                elapsed = max(0., sim_time-record["started_sim_time"])
                u = min(1., elapsed/record["duration"])
                fraction = u*u*u*(10 + u*(-15 + 6*u))
                target = {k: v+(record["goal"][k]-v)*fraction for k, v in record["start"].items()}
                apply(target)
                error = max(abs(joints[k]-v) for k, v in record["goal"].items())
                record["max_joint_error_deg"] = float(np.rad2deg(error))
                # TGS reports end-of-substep velocities, which can be nonzero at
                # a stationary frame-level equilibrium under gravity. Preserve
                # them in telemetry, but allow measured per-physics-step deltas
                # (not command velocity) for frame-level settling.
                settle_velocities = frame_velocities if self.settle_velocity_source == "position_delta" else velocities
                speed = max(abs(v) for v in settle_velocities.values())
                record["settle_velocity_source"] = self.settle_velocity_source
                record["max_joint_speed_deg_s"] = float(np.rad2deg(speed)) if np.isfinite(speed) else None
                stable = u == 1 and error < self.position_tolerance and speed < self.velocity_tolerance
                record["settled"] = record["settled"]+1 if stable and dt > 0 else 0
                record["settled_sim_s"] = record["settled_sim_s"]+dt if stable else 0.
                record["elapsed_sim_s"] = elapsed
                if playing and record["settled_sim_s"] >= self.settle_seconds:
                    self._finish(cid, "completed", "动作完成，实测关节已到位并停稳")
                    self.hold_target, self.active = record["goal"].copy(), None
                elif (not playing or sim_time < record["started_sim_time"] or elapsed > record["duration"]+self.settle_timeout
                      or now-record["started_at"] > self.wall_timeout):
                    self._finish(cid, "failed", "仿真暂停或运动超时，未确认到位，已保持当前位置")
                    self.hold_target, self.active = joints.copy(), None
                    apply(joints)
            elif self.holding and self.hold_target:
                apply(self.hold_target)
            return self.holding or self.active is not None

    def _goal(self, skill, params, joints, base, pose):
        goal = joints.copy()
        if skill == "home":
            return {k: float(self.defaults.get(k, v)) for k, v in goal.items()}
        if skill == "resume":
            if self.saved is None:
                raise ValueError("没有可继续的暂停动作；如需跟随目标，请说开始跟随")
            return self.saved.copy()
        if skill == "set_speed":
            value = float(params["degrees_per_second"])
            if not np.isfinite(value) or not 1 <= value <= 60:
                raise ValueError("速度需在 1 到 60 度每秒之间")
            self.speed = np.deg2rad(value)
            return None
        if skill == "move_joint":
            name = params["joint"]
            if name not in joints:
                raise ValueError(f"未知关节：{name}")
            value = np.deg2rad(float(params["degrees"]))
            goal[name] = value if params.get("absolute", False) else joints[name]+value
            return goal
        if skill == "gripper":
            ratio = float(params["opening"])
            if not np.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("夹爪开度需在 0 到 1 之间")
            goal["gripper"] = ratio * min(1.4, self.kinematics.limits["gripper"][1])
            return goal
        frame = params.get("frame", "world")
        if frame not in {"world", "base", "tool"}:
            raise ValueError("坐标系应为 world、base 或 tool")
        position, orientation = pose[:3, 3].copy(), None
        basis = np.eye(3) if frame == "world" else base[:3, :3] if frame == "base" else pose[:3, :3]
        if skill == "move_cartesian":
            vector = np.asarray(params["xyz_m"], dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("需要三个有限坐标值，单位为米")
            if params.get("absolute", False):
                if frame == "tool":
                    raise ValueError("工具坐标仅支持相对移动")
                position = vector if frame == "world" else base[:3, 3] + basis @ vector
            else:
                position += basis @ vector
        elif skill == "rotate":
            axis = str(params["axis"]).lower()
            if axis not in {"x", "y", "z"}:
                raise ValueError("旋转轴应为 X/Y/Z")
            degrees = float(params["degrees"])
            if not np.isfinite(degrees) or abs(degrees) > 180:
                raise ValueError("单次末端旋转需在 -180 到 180 度之间")
            turn = Rotation.from_rotvec(basis[:, "xyz".index(axis)] * np.deg2rad(degrees)).as_matrix()
            orientation = turn @ pose[:3, :3]
        else:
            raise ValueError(f"未支持的运动技能：{skill}")
        return self.kinematics.solve(position, orientation, joints, base)
