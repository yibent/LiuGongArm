"""Live-scene FineGrasp assembly; never creates a demo scene or teleports joints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import json
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

RECOVERY_MESSAGES = {
    "label_request": "正在重新观察并定位目标。",
    "semantic_find": "已获得视觉候选，正在核对新画面。",
    "coarse_start": "正在接近目标，途中持续跟踪位置。",
    "coarse_guard_stop": "目标观测发生变化，已暂停接近并重新定位。",
    "coarse_handoff": "已接近目标，正在核对腕部视野。",
    "attempt_start": "正在开始第{attempt}次抓取尝试。",
    "safe_retreat": "正在退让，准备重新观察目标。",
    "target_reacquired": "已重新定位目标，准备再次尝试。",
    "active_view_move": "正在从侧向观察目标。",
    "recovery_blocked": "持物状态不确定，已停止恢复并保持夹爪。",
}


class GraspProgressReporter:
    """Forward factual transitions, not every servo tick or predicted success."""
    def __init__(self, progress):
        self.progress, self.last = progress, None
        self.attempt = 1

    def __call__(self, event):
        self.attempt = int(event.get("attempt", self.attempt))
        phase, name = event.get("phase"), event.get("event")
        message = RECOVERY_MESSAGES.get(name)
        if message is None and name is None:
            message = PHASE_MESSAGES.get(phase)
        if not message:
            return
        key = (phase, name, self.attempt)
        if key == self.last:
            return
        self.last = key
        self.progress(phase, message.format(attempt=self.attempt), attempt=self.attempt,
                      event=name, recovery_stop_reason=event.get("reason"))


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

    @property
    def execution_guard(self):
        return self.delegate.execution_guard

    @execution_guard.setter
    def execution_guard(self, value):
        # CoarseApproach assigns this callback; it must reach the real executor.
        self.delegate.execution_guard = value

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
                 *, backend="graspgenx", output_root=Path("output/busagent_grasp"),
                 scene=None):
        self.arm, self.wrist, self.app, self.physics_dt = arm, wrist, app, physics_dt
        self.grounded, self.target = grounded, target
        self.guard, self.progress = guard, progress
        self.backend, self.output_root = backend, output_root
        self.executor = None
        self.old_efforts = None
        self.backend_instance = None
        self.scene = scene
        self.node = None
        self.recorder = None

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
        from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera, IsaacSceneCamera
        from mr_liu.grasp.adapters.isaac_gripper import IsaacGripperController
        from mr_liu.grasp.adapters.isaac_motion import IsaacCumotionExecutor
        from mr_liu.grasp.backends.factory import create_grasp_backend
        from mr_liu.grasp.contracts import FineGraspRequest, TargetSpec
        from mr_liu.grasp.live_assembly import assemble_live_grasp
        from mr_liu.grasp.debug import FineGraspDebugRecorder

        self.check()
        if self.scene is None:
            raise ValueError("双相机抓取需要固定 RGB-D 相机，未执行运动。")
        object_id = self.grounded.get("object_id")
        if not object_id:
            association = self.grounded.get("association") or {}
            import json
            print("GRASP_ASSOCIATION_FAILED " + json.dumps(association, ensure_ascii=False), flush=True)
            reason = association.get("reason", "unresolved")
            raise ValueError(f"目标实体关联失败（{reason}）；尚未调用抓取模型或执行闭爪。")
        if isinstance(object_id, str) and (object_id == "/World/TargetCube" or object_id.startswith("/World/TargetCube/")):
            raise ValueError("绿色方块是跟随控制标记，不是可抓取刚体；请改选桌面上的红色方块等物体。未执行抓取。")
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
        recorder = FineGraspDebugRecorder(self.output_root / uuid.uuid4().hex)
        self.recorder = recorder
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
        base_position, _ = self.executor.controller.world_binding.get_world_interface().get_world_to_robot_base_transform()
        if hasattr(base_position, "numpy"):
            base_position = base_position.numpy()
        camera = GuardedAdapter(IsaacWristCamera(
            self.wrist, ee_pose_provider=self.arm.T_base_ee, advance_frame=self.advance,
            robot_mask_provider=lambda: self.wrist.instance_mask("/World/SO101", leaf_paths=True),
            approach_context={
                "robot_position": np.asarray(base_position).reshape(-1, 3)[0].tolist(),
                "table_height": float(config["selection"]["table_height_m"]),
            },
        ), self.check)
        scene_camera = GuardedAdapter(IsaacSceneCamera(
            self.scene, T_base_world=np.eye(4), advance_frame=self.advance,
            robot_mask_provider=lambda: self.scene.instance_mask("/World/SO101", leaf_paths=True),
        ), self.check) if self.scene is not None else None
        report_progress = GraspProgressReporter(self.progress)

        def trace(event):
            recorder.trace(event)
            report_progress(event)

        self.backend_instance = create_grasp_backend(config)
        target = TargetSpec(object_id=object_id, semantic_label=self.grounded.get("prompt"),
                            coarse_position_base_m=None)
        from mr_liu.grasp.live_coarse import ResponsiveLocator, approach_for_live_grasp
        from mr_liu.perception.semantic_target import LocalSemanticLocator, TargetObservationError
        locator = LocalSemanticLocator()
        locator.memory_id = self.target.get("memory_id")
        try:
            self.progress("find", "正在定位目标，随后接近并核对腕部视野。")
            target = approach_for_live_grasp(
                scene_camera=scene_camera, wrist_camera=camera, motion=motion, gripper=gripper,
                locator=ResponsiveLocator(locator, self.advance, self.target.get("spatial_ref")),
                target=target, table_height_m=config["selection"]["table_height_m"],
                trace=trace, recorder=FineGraspDebugRecorder(recorder.output_dir / "coarse"),
            )
        except TargetObservationError as exc:
            # Keep the colleague's algorithm intact; surface its failure and only
            # offer a checked preparation move for reach/view failures.
            reason = str(exc)
            if reason.startswith(("coarse_path", "coarse_motion", "coarse_step", "wrist_handoff")):
                if reason.startswith("coarse_motion"):
                    diagnostic = getattr(self.executor, "last_motion_diagnostic", None)
                    return self.observation_blocked(
                        [{"reason": reason, "motion": diagnostic}],
                        "目标已定位，但接近动作未通过到位检查，已停止接近",
                        failure="coarse_motion_not_converged")
                if reason.startswith(("coarse_path", "coarse_step")):
                    return self.observation_blocked(
                        [{"reason": reason}], "接近路径不可行或接近步数已耗尽",
                        failure=reason)
                return self.observation_blocked([{"reason": reason}], "接近或腕部交接未通过：" + reason)
            return dict(success=False, failure=reason, message="目标观测未确认，已停止，尚未闭爪。", retry_available=False)
        finally:
            locator.session.close()
        self.node = assemble_live_grasp(
            config=config, camera=camera, scene_camera=scene_camera, motion=motion, gripper=gripper,
            backend=GuardedAdapter(self.backend_instance, self.check),
            recorder=recorder, trace=trace, target=target,
        )
        result = self.node.execute(FineGraspRequest(
            target=target, request_id=command_id,
            timeout_s=110., dry_run=False,
        ))
        return self.result_output(result)

    def observation_blocked(self, diagnostics, reason="当前姿态下未找到可达且无碰撞的腕部观察位置", *, failure="observation_unavailable"):
        proposal = self.executor.initial_pose_proposal()
        if proposal is not None:
            proposal.update(id=uuid.uuid4().hex, requires_confirmation=True,
                            description="回到配置的初始准备姿态，保持夹爪开度；到位后重新评估抓取条件，不自动抓取，也不保证可抓取。")
        message = reason + "，尚未闭爪。"
        message += ("可先回初始准备姿态，保持夹爪开度，再评估能否抓取。是否同意先移动？"
                    if proposal else "当前没有已验证的初始准备路径；请先调整目标位置或相机安装，不会自动移动。")
        output = dict(success=False, failure=failure, message=message,
                      diagnostics=diagnostics, preparation=proposal, retry_available=False)
        (self.recorder.output_dir / "observation-blocked.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print("GRASP_OBSERVATION_BLOCKED " + json.dumps(output, ensure_ascii=False), flush=True)
        return output

    def prepare(self, command_id):
        self.check()
        self.progress("preparation", "已确认，正在移动到初始准备姿态，保持夹爪开度。", event="preparation_start")
        reached = self.executor.move_to_initial_pose()
        self.check()
        return dict(success=bool(reached), preparation_only=True,
                    message=("已到初始准备姿态，夹爪开度保持；尚未抓取，可重新下达抓取指令进行评估。" if reached
                             else "初始准备姿态未到位或路径不再可用，已保持当前位置；尚未抓取。"))

    def result_output(self, result):
        output = {**jsonable(result), "retry_available": self.node.pending_retry is not None,
                  "attempt_results": jsonable(self.node.attempt_results)}
        (self.recorder.output_dir / f"report-{uuid.uuid4().hex}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def retry(self, command_id):
        self.check()
        return self.result_output(self.node.retry(command_id))

    def hold_for_retry(self):
        self.executor.stop()
        self.arm.articulation.set_dof_velocity_targets([[0.] * len(self.arm.dof_names)])
        if self.old_efforts is not None:
            self.arm.articulation.set_dof_max_efforts(self.old_efforts)

    def hold_after_grasp(self):
        """Transfer loaded drive targets, not the measured contact opening.

        A stalled jaw has a nonzero position error that supplies clamping
        torque. Replacing its target with the measured angle unloads the jaw.
        Keep the force-limited close target and arm gravity compensation.
        """
        articulation = self.arm.articulation
        values = articulation.get_dof_position_targets()
        if hasattr(values, "numpy"):
            values = values.numpy()
        values = np.asarray(values, dtype=float).reshape(-1)
        names = self.arm.dof_names
        if len(values) != len(names) or not np.isfinite(values).all():
            raise ValueError("抓取保持目标无效，未自动松爪。")
        targets = dict(zip(names, values.tolist()))
        articulation.set_dof_velocity_targets([[0.] * len(names)])
        # Do not restore old max efforts: retain the close policy's force cap.
        print("GRASP_HOLD_HANDOFF " + json.dumps({
            "joint_targets_rad": targets, "preserve_clamping_force": True}), flush=True)
        return targets

    def close(self, *, stop_motion=True):
        # Main thread only; no automatic opening or retry after a failed grasp.
        if stop_motion and self.executor:
            self.executor.stop()
        if stop_motion:
            self.arm.articulation.set_dof_velocity_targets([[0.] * len(self.arm.dof_names)])
        if stop_motion and self.old_efforts is not None:
            self.arm.articulation.set_dof_max_efforts(self.old_efforts)
        backend = getattr(self.backend_instance, "primary", self.backend_instance)
        transport = getattr(backend, "transport", None)
        if transport is not None:
            transport.close()
