"""Scene editing and direct manual control through the existing Arena IK action.

Snapshots and reset baselines are host arrays, including when Isaac Lab exposes
Warp arrays. Every simulation write happens on the service's simulation thread.
The editor enumerates live rigid bodies; configured names never gate execution.
"""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from mr_liu.arena.arrays import numpy_data
from mr_liu.arena.teleop import finite_vector

PRESETS = [
    {"id": "initial", "name": "基础抓放", "description": "恢复当前场景的初始物体布局。", "kind": "pick"},
    {"id": "sorting", "name": "桌面整理", "description": "分散排列现有物体，进行桌面整理实验。", "kind": "sort"},
    {"id": "precision", "name": "精确摆放", "description": "紧凑排列现有物体，进行定位与堆叠实验。", "kind": "stack"},
]


def simulation_array(values, device):
    import torch
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)


class Workspace:
    def __init__(self, runtime, teleop=None):
        self.runtime, self.teleop = runtime, teleop
        self.scene_id = "initial"
        self.initial, self.baseline = {}, {}
        self.metadata = {row["name"]: row for key in ("entities", "objects", "destinations")
                         for row in runtime.config.get(key, [])}
        self.snapshot = {}
        self.last_refresh = 0.
        self.refresh()

    def refresh(self):
        runtime = self.runtime
        rows = []
        for name, body in runtime.env.scene.rigid_objects.items():
            if name not in self.initial:
                state = numpy_data(body.data.default_root_state).copy()
                state[:, :3] += numpy_data(runtime.env.scene.env_origins)
                self.initial[name] = state
                self.baseline[name] = state.copy()
            metadata = self.metadata.get(name, {})
            rows.append({"id": name, "name": metadata.get("label", name),
                         "position": numpy_data(body.data.root_pos_w)[0].tolist(),
                         "rotation": Rotation.from_quat(numpy_data(body.data.root_quat_w)[0]).as_euler("xyz", degrees=True).tolist(),
                         "size": metadata.get("size"), "color": metadata.get("color"),
                         "shape": metadata.get("shape"), "editable": True})
        robot = runtime.env.scene["robot"]
        tcp = runtime.tcp_pose()
        self.snapshot = {"available": True, "api_version": 2, "scene_id": self.scene_id,
                         "scenes": [{**p, "active": p["id"] == self.scene_id, "count": len(rows)} for p in PRESETS],
                         "objects": rows, "controller": dict(runtime.config["controller"]),
                         "controls": {"robot_pose": True, "jog": True, "teleop": self.teleop is not None, "gripper": True},
                         "robot": {"position": tcp[:3, 3].tolist(),
                                   "rotation": Rotation.from_matrix(tcp[:3, :3]).as_euler("xyz", degrees=True).tolist(),
                                   "gripper": runtime.gripper,
                                   "joint_positions_deg": np.rad2deg(numpy_data(robot.data.joint_pos)[0]).tolist()},
                         "camera": {k: runtime.config["camera"][k] for k in ("width", "height")}}
        self.last_refresh = time.monotonic()

    def _invalidate_perception(self, *, clear_hold=False):
        runtime = self.runtime
        runtime.tracking = None
        runtime.tracking_stop.set()
        runtime.visual_result = None
        runtime.target_name = None
        runtime.selected_target = None
        for key in ("observed_entities", "prepared_clouds", "observed_clouds"):
            cache = getattr(runtime, key, None)
            if isinstance(cache, dict):
                held_row = cache.get(runtime.held) if not clear_hold and key == "observed_entities" else None
                cache.clear()
                if held_row is not None:
                    cache[runtime.held] = held_row
        if clear_hold:
            runtime.clear_hold()

    def _object_state(self, name):
        body = self.runtime.env.scene.rigid_objects.get(name)
        if body is None:
            raise ValueError("物体已不在当前场景中，请刷新物体列表。")
        state = np.zeros((1, 13))
        state[:, :3] = numpy_data(body.data.root_pos_w)
        state[:, 3:7] = numpy_data(body.data.root_quat_w)
        return state

    def _write_objects(self, states):
        runtime = self.runtime
        for name, state in states.items():
            body = runtime.env.scene.rigid_objects.get(name)
            if body is None:
                continue  # A removed scene body does not block resetting others.
            body.write_root_pose_to_sim_index(root_pose=simulation_array(state[:, :7], runtime.env.device))
            body.write_root_velocity_to_sim_index(root_velocity=simulation_array(np.zeros_like(state[:, 7:13]), runtime.env.device))
            body.reset()

    def _reset_robot(self):
        runtime = self.runtime
        robot = runtime.env.scene["robot"]
        positions = simulation_array(numpy_data(robot.data.default_joint_pos).copy(), runtime.env.device)
        velocities = simulation_array(np.zeros_like(numpy_data(robot.data.default_joint_vel)), runtime.env.device)
        robot.write_joint_state_to_sim_mask(position=positions, velocity=velocities)
        robot.reset()
        robot.set_joint_position_target_index(target=positions)
        runtime.env.action_manager.reset()
        runtime.env.scene["ee_frame"].reset()
        runtime.gripper = 1.

    def _settle(self, ticks=8):
        runtime = self.runtime
        runtime.env.sim.forward()
        runtime.env.scene.update(.001)
        runtime.goal = runtime.tcp_pose().copy()
        for _ in range(ticks):
            runtime.tick()

    @staticmethod
    def _offset(pose, translation, rotation, frame="world"):
        delta = Rotation.from_rotvec(np.deg2rad(rotation)).as_matrix()
        result = pose.copy()
        if frame == "tool":
            result[:3, 3] += pose[:3, :3] @ translation
            result[:3, :3] = pose[:3, :3] @ delta
        elif frame == "world":
            result[:3, 3] += translation
            result[:3, :3] = delta @ pose[:3, :3]
        else:
            raise ValueError("Unknown coordinate frame")
        return result

    def _manual_target(self, params):
        target = params.get("target", "robot")
        if target not in {"robot", "object"}:
            raise ValueError("Unknown manual control target")
        if params.get("frame", "world") not in {"world", "tool"}:
            raise ValueError("Unknown coordinate frame")
        if target == "object":
            self._object_state(params.get("id"))
        return target

    def _jog(self, params):
        target = self._manual_target(params)
        linear = np.asarray(finite_vector(params.get("translation", [0., 0., 0.])))
        angular = np.asarray(finite_vector(params.get("rotation", [0., 0., 0.])))
        self._invalidate_perception(clear_hold=target == "object" and params.get("id") == self.runtime.held)
        if target == "robot":
            goal = self._offset(self.runtime.tcp_pose(), linear, angular, params.get("frame", "world"))
            self.runtime.move(goal, label="manual_jog")
        else:
            name = params.get("id")
            state = self._object_state(name)
            state[0, :3] += linear
            state[0, 3:7] = (Rotation.from_rotvec(np.deg2rad(angular)) * Rotation.from_quat(state[0, 3:7])).as_quat()
            self._write_objects({name: state})
            self._settle(2)

    def _teleoperate(self, command):
        if self.teleop is None:
            raise ValueError("Manual streaming is unavailable")
        runtime = self.runtime
        params, command_id = command["params"], command["command_id"]
        target = self._manual_target(params)
        self._invalidate_perception(clear_hold=target == "object" and params.get("id") == runtime.held)
        goal = runtime.tcp_pose().copy()
        previous = time.monotonic()
        while (values := self.teleop.read(command_id)) is not None:
            if runtime.stop_requested.is_set():
                raise InterruptedError("手动移动已中断")
            now = time.monotonic()
            dt = min(.1, now - previous)
            previous = now
            linear = np.asarray(values["linear"]) * dt
            angular = np.asarray(values["angular"]) * dt
            if target == "robot":
                goal = self._offset(goal, linear, angular, params.get("frame", "world"))
                actual = runtime.tcp_pose()
                # Stop an unreachable or blocked gesture instead of accumulating
                # a distant goal that could move the arm after it becomes free.
                if np.linalg.norm(goal[:3, 3] - actual[:3, 3]) > .05 or Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude() > .35:
                    raise RuntimeError("机械臂未能跟上移动目标，已停止。请换一个方向或降低移动速度。")
                runtime.goal = goal
            else:
                name = params["id"]
                state = self._object_state(name)
                state[0, :3] += linear
                state[0, 3:7] = (Rotation.from_rotvec(np.deg2rad(angular)) * Rotation.from_quat(state[0, 3:7])).as_quat()
                self._write_objects({name: state})
            runtime.tick()
            if now - self.last_refresh > .15:
                self.refresh()
        runtime.goal = runtime.tcp_pose().copy()

    def execute(self, command):
        started = time.monotonic()
        runtime = self.runtime
        params = command.get("params", {})
        action = params.get("action")
        runtime.current = command["command_id"]
        runtime.phase = "manual_control" if action in {"jog", "teleop", "robot", "gripper"} else "workspace_edit"
        runtime.refresh_snapshot()
        try:
            if runtime.stop_requested.is_set():
                raise InterruptedError("操作已中断")
            if action == "controller":
                updates = {k: v for k, v in params.items() if k != "action"}
                if not updates or any(k not in {"position_tolerance_m", "rotation_tolerance_deg", "max_steps"} for k in updates):
                    raise ValueError("Unknown controller setting")
                for key, value in updates.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        raise ValueError("Controller settings must be positive finite numbers")
                    if key == "max_steps" and int(value) != value:
                        raise ValueError("max_steps must be an integer")
                runtime.config["controller"].update(updates)
            elif action == "object":
                name = params.get("id")
                state = self._object_state(name)
                state[0, :3] = finite_vector(params.get("position"))
                state[0, 3:7] = Rotation.from_euler("xyz", finite_vector(params.get("rotation")), degrees=True).as_quat()
                self._invalidate_perception(clear_hold=name == runtime.held)
                self._write_objects({name: state})
                self._settle()
            elif action == "robot":
                goal = np.eye(4)
                goal[:3, 3] = finite_vector(params.get("position"))
                goal[:3, :3] = Rotation.from_euler("xyz", finite_vector(params.get("rotation")), degrees=True).as_matrix()
                self._invalidate_perception()
                runtime.move(goal, label="manual_pose")
            elif action == "jog":
                self._jog(params)
            elif action == "teleop":
                self._teleoperate(command)
            elif action == "gripper":
                if params.get("state") not in {"open", "close"}:
                    raise ValueError("Unknown gripper state")
                self._invalidate_perception(clear_hold=params["state"] == "open")
                runtime.gripper = 1. if params["state"] == "open" else -1.
                for _ in range(24):
                    runtime.tick()
            elif action in {"reset", "scene"}:
                scope = "all" if action == "scene" else params.get("scope", "all")
                scene_id = params.get("scene_id", self.scene_id)
                if scope not in {"all", "objects", "robot"}:
                    raise ValueError("Unknown reset scope")
                if scene_id not in {p["id"] for p in PRESETS}:
                    raise ValueError("Unknown scene preset")
                if action == "scene":
                    self.scene_id = scene_id
                    self.initial = {name: state.copy() for name, state in self.baseline.items()}
                    if scene_id != "initial":
                        index = 0
                        for name, state in self.initial.items():
                            body = runtime.env.scene.rigid_objects.get(name)
                            if body is None:
                                continue
                            rigid = getattr(getattr(body.cfg, "spawn", None), "rigid_props", None)
                            if getattr(rigid, "kinematic_enabled", False):
                                continue
                            spacing = .13 if scene_id == "sorting" else .085
                            state[0, 0] = .43 + (index % 3) * spacing
                            state[0, 1] = -.16 + (index // 3) * spacing
                            index += 1
                self._invalidate_perception(clear_hold=True)
                if scope in {"all", "objects"}:
                    self._write_objects(self.initial)
                if scope in {"all", "robot"}:
                    self._reset_robot()
                self._settle()
            else:
                raise ValueError("Unknown workspace action")
            result = {"ok": True, "state": "completed", "message": "操作已完成"}
        except Exception as error:
            runtime.goal = runtime.tcp_pose().copy()
            result = {"ok": False, "state": "cancelled" if isinstance(error, InterruptedError) else "failed", "message": str(error)}
        finally:
            if self.teleop is not None:
                self.teleop.finish(command["command_id"])
        result.update(command_id=command["command_id"], skill="workspace", action=action,
                      elapsed_s=time.monotonic() - started, tcp_pose_world=runtime.tcp_pose().tolist())
        runtime.last_result = result
        runtime.current = None
        runtime.phase = "idle"
        runtime.refresh_snapshot()
        self.refresh()
        return result
