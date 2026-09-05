"""Live-scene FineGrasp assembly; never creates a demo scene or teleports joints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import uuid

import numpy as np


PHASE_MESSAGES = {
    "observe": "正在用腕部相机观察目标。",
    "generate": "正在计算可行抓取姿态。",
    "pregrasp": "正在接近抓取位置。",
    "servo": "正在根据腕部图像微调位置。",
    "close": "正在闭合夹爪，尚未确认夹持成功。",
    "verify_close": "正在验证是否夹住目标。",
    "lift": "正在抬升目标，等待验证。",
    "verify_lift": "正在验证目标是否随夹爪抬升。",
}


def jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


class GuardedAdapter:
    """Interrupt before and after blocking operations; stop itself is unguarded."""
    def __init__(self, delegate, guard):
        self.delegate, self.guard = delegate, guard

    def __getattr__(self, name):
        value = getattr(self.delegate, name)
        if not callable(value) or name == "stop":
            return value
        def call(*args, **kwargs):
            self.guard()
            result = value(*args, **kwargs)
            self.guard()
            return result
        return call


def observation_tool_pose(T_world_tool, T_world_camera, point, height=.16):
    """Place the optical centre above measured surface, looking down.

    Use the configured hand-eye transform, not the demo's hard-coded joints.
    The five-axis planner still has to accept this pose and collision path.
    """
    T_tool_camera = np.linalg.inv(T_world_tool) @ T_world_camera
    x = T_world_camera[:2, 0]
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1., 0.])
    camera = np.eye(4)
    camera[:3, :3] = [[x[0], x[1], 0], [x[1], -x[0], 0], [0, 0, -1]]
    camera[:3, 3] = np.asarray(point) + [0, 0, height]
    return camera @ np.linalg.inv(T_tool_camera)


def observation_tool_candidates(T_world_tool, T_world_camera, point, height=.16):
    """Same measured optical centre, alternative viewing yaws for a 5-DOF arm.

    This searches observation poses only, not grasp poses or geometric grasps.
    The actual collision-aware planner must approve a candidate before motion.
    """
    original = observation_tool_pose(T_world_tool, T_world_camera, point, height)
    T_tool_camera = np.linalg.inv(T_world_tool) @ T_world_camera
    camera = original @ T_tool_camera
    candidates = []
    for angle in np.linspace(0., 2*np.pi, 12, endpoint=False):
        c, s = np.cos(angle), np.sin(angle)
        rotated = camera.copy()
        rotated[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ camera[:3, :3]
        candidates.append(rotated @ np.linalg.inv(T_tool_camera))
    return sorted(candidates, key=lambda pose: np.linalg.norm(pose[:3, 3]-T_world_tool[:3, 3]))


class IsaacGraspSession:
    def __init__(self, arm, wrist, app, physics_dt, grounded, target, guard, progress,
                 *, backend="graspgenx", output_root=Path("output/busagent_grasp")):
        self.arm, self.wrist, self.app, self.physics_dt = arm, wrist, app, physics_dt
        self.grounded, self.target = grounded, target
        self.guard, self.progress = guard, progress
        self.backend, self.output_root = backend, output_root
        self.executor = None
        self.old_efforts = None
        self.backend_instance = None

    def check(self):
        from isaacsim.core.experimental.utils import app as app_utils
        self.guard()
        if not self.app.is_running() or not app_utils.is_playing():
            raise RuntimeError("仿真已暂停或退出，抓取已停止，未确认成功。")

    def advance(self):
        self.check()
        self.app.update()
        self.check()

    def execute(self, command_id):
        from isaacsim.core.experimental.utils import stage as stage_utils
        from pxr import UsdPhysics
        from mr_liu.config import fine_grasp_config
        from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera
        from mr_liu.grasp.adapters.isaac_gripper import IsaacGripperController
        from mr_liu.grasp.adapters.isaac_motion import IsaacCumotionExecutor
        from mr_liu.grasp.backends.factory import create_grasp_backend
        from mr_liu.grasp.contracts import FineGraspRequest, TargetSpec
        from mr_liu.grasp.node import GeneralGraspNode
        from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
        from mr_liu.grasp.settings import FineGraspSettings
        from mr_liu.grasp.verification import VisionGripperVerifier
        from mr_liu.grasp.debug import FineGraspDebugRecorder, DebugSegmenter

        self.check()
        object_id = self.grounded.get("object_id")
        if not object_id:
            raise ValueError("顶部相机检测到了目标，但实例分割尚未关联实体；未执行抓取。")
        if not isinstance(object_id, str) or not object_id.startswith("/World/TabletopProps/"):
            raise ValueError("目标未关联到可抓取的桌面实体；控制用目标标记不能抓取。")
        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(object_id)
        from pxr import Usd
        if not prim.IsValid() or not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
            raise ValueError("目标没有动态刚体，不能执行物理抓取。")
        config = fine_grasp_config()
        config["backend"] = self.backend
        if self.backend == "graspgenx":
            # Never silently degrade the live BusAgent path to geometric grasps.
            config["backends"]["graspgenx"]["fallback_backend"] = "none"
            config["backends"]["graspgenx"]["fallback_on_empty"] = False
        config["fusion"] = {"enabled": False}
        config["active_views"] = {"enabled": False}
        recorder = FineGraspDebugRecorder(self.output_root / uuid.uuid4().hex)
        self.old_efforts = self.arm.articulation.get_dof_max_efforts()
        # Reuse live calibrated gains; never apply the standalone demo's gains.
        self.executor = IsaacCumotionExecutor(
            self.arm, advance_frame=self.advance, physics_dt_s=self.physics_dt,
            track_orientation=True, target_object_paths=[object_id],
            max_move_steps=max(480, int(8/self.physics_dt)),
            max_servo_steps=max(240, int(4/self.physics_dt)),
        )
        motion = GuardedAdapter(self.executor, self.check)
        gripper = GuardedAdapter(IsaacGripperController(
            self.arm.articulation, advance_frame=self.advance, physics_dt_s=self.physics_dt,
        ), self.check)
        point = self.grounded["object_position_world_m"]
        self.progress("approach", "正在规划到目标上方的接近路径。")
        sample = self.wrist.rgbd_frame()
        if sample is None:
            raise ValueError("腕部相机尚无同步 RGB-D 与渲染位姿，未执行接近。")
        camera_pose = sample.T_world_camera
        entry = next((pose for pose in observation_tool_candidates(self.arm.T_base_ee(), camera_pose, point)
                      if motion.is_reachable(pose)), None)
        if entry is None:
            raise ValueError("NO_PATH：当前目标上方没有可达的腕部观察姿态，未执行闭爪。")
        if not motion.move_to(entry, speed_scale=.4):
            raise ValueError("NO_PATH：无法到达目标上方的腕部观察位姿，未执行闭爪。")
        if not gripper.open(.075, speed_mps=.025):
            raise ValueError("夹爪未到达观察开度，抓取已停止。")
        camera = GuardedAdapter(IsaacWristCamera(
            self.wrist, ee_pose_provider=self.arm.T_base_ee, advance_frame=self.advance,
        ), self.check)

        def trace(event):
            recorder.trace(event)
            phase = event.get("phase")
            if phase in PHASE_MESSAGES:
                self.progress(phase, PHASE_MESSAGES[phase])

        self.backend_instance = create_grasp_backend(config)
        node = GeneralGraspNode(
            camera=camera, motion=motion, gripper=gripper,
            segmenter=DebugSegmenter(AppearanceDepthTrackerSegmenter(SeededDepthSegmenter()), recorder),
            backend=GuardedAdapter(self.backend_instance, self.check),
            verifier=VisionGripperVerifier(), settings=FineGraspSettings.from_mapping(config), trace=trace,
        )
        result = node.execute(FineGraspRequest(
            target=TargetSpec(object_id=object_id, semantic_label=self.grounded.get("prompt"),
                              coarse_position_base_m=tuple(point)),
            request_id=command_id, timeout_s=45., dry_run=False,
        ))
        return jsonable(result)

    def close(self):
        # Main thread only; no automatic opening or retry after a failed grasp.
        if self.executor:
            self.executor.stop()
        self.arm.articulation.set_dof_velocity_targets([[0.] * len(self.arm.dof_names)])
        if self.old_efforts is not None:
            self.arm.articulation.set_dof_max_efforts(self.old_efforts)
        backend = getattr(self.backend_instance, "primary", self.backend_instance)
        transport = getattr(backend, "transport", None)
        if transport is not None:
            transport.close()
