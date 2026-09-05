"""Verified transport and placement state machine for a held object."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from mr_liu.grasp.contracts import (
    FailureCode,
    GripperController,
    MotionExecutor,
    PickPlacePhase,
    PlaceRequest,
    PlaceResult,
    RGBDObservation,
    TargetSpec,
    VerificationResult,
)
from mr_liu.grasp.transforms import invert_transform, transform_points


PlacementPlanner = Callable[
    [RGBDObservation, np.ndarray, np.ndarray, PlaceRequest],
    Sequence[tuple[np.ndarray, float]],
]
PlacementObserver = Callable[[TargetSpec, PlaceRequest], np.ndarray | None]


class PickPlaceController:
    """Move a verified payload, release it, and require measured placement evidence.

    The controller never treats an ``open`` command as success.  A placement
    only succeeds when the observer reports the target centre inside the
    requested region after release and the gripper is no longer loaded.
    """

    def __init__(
        self,
        *,
        motion: MotionExecutor,
        gripper: GripperController,
        planner: PlacementPlanner,
        camera,
        target: TargetSpec,
        observer: PlacementObserver,
        max_width_m: float = 0.075,
        speed_scale: float = 0.35,
        release_speed_mps: float = 0.025,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.motion, self.gripper, self.planner = motion, gripper, planner
        self.camera, self.target, self.observer = camera, target, observer
        self.max_width_m, self.speed_scale = float(max_width_m), float(speed_scale)
        self.release_speed_mps, self.clock = float(release_speed_mps), clock

    def execute(
        self,
        request: PlaceRequest,
        *,
        held_points_base: np.ndarray,
        before_observation: RGBDObservation,
    ) -> PlaceResult:
        started = self.clock()
        metrics: dict[str, float | int | str] = {
            "held_points": int(len(held_points_base)),
            "place_candidates": 0,
        }
        if len(held_points_base) < 6:
            return self._failure(request, FailureCode.PLACE_VERIFICATION_FAILED,
                                 "持物点云不足，拒绝执行放置", metrics, started)
        state = self.gripper.state()
        loaded = bool((state.contact is True or state.stalled is True) and
                      state.width_m is not None and state.width_m > 0.001)
        if not loaded:
            return self._failure(request, FailureCode.GRIPPER_FAILURE,
                                 "放置前未确认夹爪仍持有目标", metrics, started)
        if request.dry_run:
            current = self.motion.robot_state().T_base_ee
            held_camera = transform_points(invert_transform(before_observation.T_base_camera), held_points_base)
            candidates = list(self.planner(before_observation, held_camera, current, request))
            metrics["place_candidates"] = len(candidates)
            if not candidates:
                return self._failure(request, FailureCode.PLACE_UNREACHABLE,
                                     "M2T2 未返回满足放置区域的候选姿态", metrics, started)
            return PlaceResult(True, PickPlacePhase.SUCCEEDED, message="放置姿态可达（dry-run）",
                               selected_pose=candidates[0][0], metrics=metrics)

        current = self.motion.robot_state().T_base_ee
        # M2T2's live adapter accepts points in the observation camera frame.
        held_camera = transform_points(invert_transform(before_observation.T_base_camera), held_points_base)
        candidates = list(self.planner(before_observation, held_camera, current, request))
        metrics["place_candidates"] = len(candidates)
        for pose, score in candidates:
            if not self.motion.is_reachable(pose):
                continue
            if not self.motion.is_collision_free(pose, margin_m=0.006, allow_target_contact=False):
                continue
            metrics["selected_place_score"] = float(score)
            if not self.motion.move_to(pose, speed_scale=self.speed_scale):
                continue
            metrics["transport_completed"] = 1
            if not self.gripper.open(self.max_width_m, speed_mps=self.release_speed_mps):
                return self._failure(request, FailureCode.GRIPPER_FAILURE,
                                     "到达放置区域后夹爪释放失败", metrics, started, pose)
            metrics["release_commanded"] = 1
            for _ in range(3):
                observation = self.camera.capture(self.target)
                if observation is not None:
                    break
            center = self.observer(self.target, request)
            if center is None:
                return self._failure(request, FailureCode.PLACE_VERIFICATION_FAILED,
                                     "释放后未观测到目标，放置状态未知", metrics, started, pose)
            center = np.asarray(center, dtype=float).reshape(3)
            region_center = np.asarray(request.center_base_m, dtype=float)
            half = np.asarray(request.size_xy_m, dtype=float) * 0.5
            inside = bool(np.all(np.abs(center[:2] - region_center[:2]) <= half))
            metrics["placed_center_x_m"] = float(center[0])
            metrics["placed_center_y_m"] = float(center[1])
            metrics["placed_center_z_m"] = float(center[2])
            metrics["inside_place_region"] = int(inside)
            post_state = self.gripper.state()
            released = bool(post_state.contact is not True and post_state.stalled is not True)
            metrics["gripper_released"] = int(released)
            if inside and released:
                metrics["elapsed_s"] = self.clock() - started
                return PlaceResult(True, PickPlacePhase.SUCCEEDED,
                                   message="抓取、搬运、放置及落点验证成功",
                                   selected_pose=pose, metrics=metrics)
            return self._failure(request, FailureCode.PLACE_VERIFICATION_FAILED,
                                 "目标未稳定落入指定放置区域", metrics, started, pose)
        return self._failure(request, FailureCode.PLACE_UNREACHABLE,
                             "没有通过可达性与碰撞检查的放置姿态", metrics, started)

    def _failure(self, request, failure, message, metrics, started, pose=None):
        metrics["elapsed_s"] = self.clock() - started
        return PlaceResult(False, PickPlacePhase.FAILED, failure=failure, message=message,
                           selected_pose=pose, metrics=metrics)
