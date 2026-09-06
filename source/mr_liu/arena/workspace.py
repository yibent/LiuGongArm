"""Workbench editing on the simulation thread, separate from task grounding.

Scene presets rearrange existing rigid bodies. Object IDs identify editor handles
only; this module never restricts the objects perception or manipulation can use.
"""
import math
import time

PRESETS = [
    {"id": "initial", "name": "基础抓放", "description": "开放工作台上的拿取、转移与放置。", "kind": "pick"},
    {"id": "sorting", "name": "桌面整理", "description": "分散排列现有物体，进行桌面整理实验。", "kind": "sort"},
    {"id": "precision", "name": "精确摆放", "description": "紧凑排列现有物体，进行定位与堆叠实验。", "kind": "stack"},
]


def finite_vector(value, count=3):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"Expected {count} numeric coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Coordinates must be finite numbers")
    return value


class Workspace:
    def __init__(self, runtime):
        self.runtime = runtime
        self.scene_id = "initial"
        self.initial = {}
        self.metadata = {}
        for key in ("entities", "objects", "destinations"):
            for row in runtime.config.get(key, []):
                self.metadata[row["name"]] = row
        # Use the real scene registry, including bodies not in a config list.
        for name, body in runtime.env.scene.rigid_objects.items():
            state = body.data.default_root_state.clone()
            state[:, :3] += runtime.env.scene.env_origins
            self.initial[name] = state
        self.baseline = {name: state.clone() for name, state in self.initial.items()}
        self.snapshot = {}
        self.refresh()

    def refresh(self):
        from mr_liu.arena.arrays import numpy_data
        from scipy.spatial.transform import Rotation
        rows = []
        for name in self.initial:
            body = self.runtime.env.scene.rigid_objects[name]
            metadata = self.metadata.get(name, {})
            rows.append({"id": name, "name": metadata.get("label", name),
                         "position": numpy_data(body.data.root_pos_w)[0].tolist(),
                         "rotation": Rotation.from_quat(numpy_data(body.data.root_quat_w)[0]).as_euler("xyz", degrees=True).tolist(),
                         "size": metadata.get("size"), "color": metadata.get("color"),
                         "shape": metadata.get("shape"), "editable": True})
        self.snapshot = {"available": True, "scene_id": self.scene_id,
                         "scenes": [{**p, "active": p["id"] == self.scene_id, "count": len(rows)} for p in PRESETS],
                         "objects": rows, "controller": dict(self.runtime.config["controller"]),
                         "camera": {k: self.runtime.config["camera"][k] for k in ("width", "height")}}

    def _invalidate_perception(self, *, clear_hold=True):
        runtime = self.runtime
        runtime.tracking = None
        runtime.tracking_stop.set()
        runtime.visual_result = None
        runtime.target_name = None
        # Generalized runtime maintains frame-derived handles; layout edits make
        # them stale. Invalidate only these optional caches, never model memory.
        for key in ("observed_entities", "observed_clouds"):
            cache = getattr(runtime, key, None)
            if isinstance(cache, dict):
                held_row = cache.get(runtime.held) if not clear_hold else None
                cache.clear()
                if held_row is not None: cache[runtime.held] = held_row
        if clear_hold:
            if hasattr(runtime, "clear_hold"):
                runtime.clear_hold()
            else:
                runtime.held = None
        runtime.last_result = None

    def _write_objects(self, states):
        for name, state in states.items():
            body = self.runtime.env.scene.rigid_objects[name]
            body.write_root_pose_to_sim(state[:, :7])
            body.write_root_velocity_to_sim(state[:, 7:13] * 0)
            body.reset()

    def _reset_robot(self):
        runtime = self.runtime
        robot = runtime.env.scene["robot"]
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        robot.reset()
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        runtime.env.action_manager.reset()
        runtime.gripper = 1.

    def _settle(self):
        runtime = self.runtime
        runtime.env.sim.forward()
        runtime.env.scene.update(0.)
        runtime.goal = runtime.tcp_pose()
        runtime.stop_requested.clear()
        for _ in range(16): runtime.tick(check_stop=False)
        runtime.phase = "idle"
        runtime.refresh_snapshot()
        self.refresh()

    def execute(self, command):
        started = time.monotonic()
        runtime = self.runtime
        params = command.get("params", {})
        action = params.get("action")
        try:
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
                self.refresh()
            elif action == "object":
                name = params.get("id")
                if name not in self.initial: raise ValueError("Object is no longer in the scene")
                # Validate all fields before the first physical mutation.
                position = finite_vector(params.get("position"))
                rotation = finite_vector(params.get("rotation"))
                from scipy.spatial.transform import Rotation
                state = self.initial[name].clone()
                state[0, :3] = state.new_tensor(position)
                state[0, 3:7] = state.new_tensor(Rotation.from_euler("xyz", rotation, degrees=True).as_quat())
                self._invalidate_perception(clear_hold=name == runtime.held)
                self._write_objects({name: state})
                self._settle()
            elif action in {"reset", "scene"}:
                scope = params.get("scope", "all")
                scene_id = params.get("scene_id", self.scene_id)
                if scope not in {"all", "objects", "robot"}: raise ValueError("Unknown reset scope")
                if scene_id not in {p["id"] for p in PRESETS}: raise ValueError("Unknown scene preset")
                if action == "scene":
                    self.scene_id = scene_id
                    self.initial = {name: state.clone() for name, state in self.baseline.items()}
                    if scene_id != "initial":
                        # Keep kinematic supports where they are; rearrange the
                        # remaining bodies geometrically, independent of names.
                        index = 0
                        for name, state in self.initial.items():
                            cfg = runtime.env.scene.rigid_objects[name].cfg
                            rigid = getattr(getattr(cfg, "spawn", None), "rigid_props", None)
                            if getattr(rigid, "kinematic_enabled", False): continue
                            spacing = .13 if scene_id == "sorting" else .085
                            state[0, 0] = .43 + (index % 3) * spacing
                            state[0, 1] = -.16 + (index // 3) * spacing
                            index += 1
                self._invalidate_perception()
                if scope in {"all", "objects"}: self._write_objects(self.initial)
                if scope in {"all", "robot"}: self._reset_robot()
                self._settle()
            else:
                raise ValueError("Unknown workspace action")
            return {"ok": True, "state": "completed", "command_id": command["command_id"], "message": "Workspace updated", "elapsed_s": time.monotonic() - started}
        except Exception as error:
            return {"ok": False, "state": "failed", "command_id": command["command_id"], "message": str(error)}
