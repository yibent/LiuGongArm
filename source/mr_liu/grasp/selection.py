"""Model-independent grasp feasibility filtering and ranking."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Sequence, Callable

import numpy as np

from mr_liu.grasp.contracts import GraspCandidate, MotionExecutor, RGBDObservation, SelectedGrasp
from mr_liu.grasp.geometry import ObjectGeometry, support_completed_center
from mr_liu.grasp.policy import ExecutionPolicy
from mr_liu.grasp.transforms import invert_transform, make_transform


@dataclass(frozen=True)
class SelectionConfig:
    min_score: float
    min_width_m: float
    max_width_m: float
    pregrasp_distance_m: float
    collision_margin_m: float
    table_height_m: float
    max_candidates: int = 128
    max_elongated_axis_error_deg: float = 18.0
    max_feasible_candidates: int = 0


@dataclass(frozen=True)
class SelectionResult:
    grasp: SelectedGrasp | None
    rejected: dict[str, int]
    considered: int
    diagnostics: tuple[dict[str, Any], ...] = ()


def select_grasp(
    candidates: Sequence[GraspCandidate],
    observation: RGBDObservation,
    geometry: ObjectGeometry,
    motion: MotionExecutor,
    policy: ExecutionPolicy,
    config: SelectionConfig,
    *,
    local_path_check: Callable[[np.ndarray, np.ndarray, float], bool] | None = None,
    failed_grasps: Sequence[SelectedGrasp] = (),
) -> SelectionResult:
    configure_solver = getattr(motion, "set_grasp_constraints", None)
    if callable(configure_solver):
        configure_solver(True)
    rejected: Counter[str] = Counter()
    feasible: list[SelectedGrasp] = []
    diagnostics: list[dict[str, Any]] = []
    elongated_minor_axis: np.ndarray | None = None
    xy = np.asarray(geometry.points_base[:, :2], dtype=np.float64)
    if len(xy) >= 32:
        covariance = np.cov(xy - np.median(xy, axis=0), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[0] > 1e-12 and np.sqrt(eigenvalues[1] / eigenvalues[0]) >= 2.0:
            elongated_minor_axis = eigenvectors[:, 0]

    def closing_axis_error_deg(pose: np.ndarray) -> float | None:
        if elongated_minor_axis is None:
            return None
        axis = np.asarray(pose[:2, 0], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return 90.0
        cosine = float(np.clip(abs(np.dot(axis / norm, elongated_minor_axis)), 0.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)[: config.max_candidates]
    for candidate_index, candidate in enumerate(ordered):
        diagnostic: dict[str, Any] | None = None
        if candidate_index < config.max_candidates:
            diagnostic = {
                "rank": candidate_index,
                "backend": candidate.backend,
                "branch": candidate.metadata.get("branch"),
                "score": candidate.score,
                "width_m": candidate.width_m,
            }

        def reject(reason: str) -> None:
            rejected[reason] += 1
            if diagnostic is not None:
                diagnostic["rejection"] = reason
                diagnostics.append(diagnostic)

        def check(stage: str, pose: np.ndarray, operation: Callable[[], bool]) -> bool:
            started = perf_counter()
            passed = bool(operation())
            # Snapshot immediately: the next pose check overwrites adapter state.
            detail = deepcopy(getattr(motion, "last_plan_diagnostic", {}))
            if diagnostic is not None:
                diagnostic.setdefault("checks", []).append({
                    "stage": stage, "passed": passed,
                    "elapsed_ms": (perf_counter() - started) * 1000.0,
                    "target_world_matrix": pose.tolist(),
                    "planning": detail,
                })
                if not passed:
                    diagnostic["failed_stage"] = stage
            return passed

        if candidate.observation_sequence != observation.sequence:
            reject("stale_candidate")
            continue
        if candidate.score < config.min_score:
            reject("low_score")
            continue
        if candidate.width_m < config.min_width_m or candidate.width_m > config.max_width_m:
            reject("gripper_width")
            continue
        T_base_grasp_center = observation.T_base_camera @ candidate.T_camera_grasp
        desired_axis_error = closing_axis_error_deg(T_base_grasp_center)
        if desired_axis_error is not None:
            if diagnostic is not None:
                diagnostic["elongated_axis_error_deg"] = desired_axis_error
            if desired_axis_error > config.max_elongated_axis_error_deg:
                reject("closing_axis_misaligned")
                continue
        support_height_constrained = False
        if (float(T_base_grasp_center[2, 2]) < -0.6
                and not candidate.metadata.get("contact_height_scored", False)):
            # Eye-in-hand depth mainly observes the top surface during the
            # final vertical approach.  A learned proposal may consequently
            # place its pinch centre at (or just above) that surface.  Complete
            # the occluded lower half from the known support plane while
            # retaining the model's lateral contact location and orientation.
            completed_center = support_completed_center(
                geometry, table_height_m=config.table_height_m
            )
            if completed_center[2] != geometry.T_base_object[2, 3]:
                safe_contact_z = (
                    config.table_height_m + policy.required_table_clearance_m + 0.00025
                )
                observed_top_z = float(np.percentile(geometry.points_base[:, 2], 95.0))
                if safe_contact_z <= observed_top_z - 0.001:
                    completed_center[2] = max(completed_center[2], safe_contact_z)
                T_base_grasp_center = T_base_grasp_center.copy()
                T_base_grasp_center[2, 3] = completed_center[2]
                support_height_constrained = True
        T_base_grasp = T_base_grasp_center
        ee_pose_for_grasp = getattr(motion, "ee_pose_for_grasp", None)
        if callable(ee_pose_for_grasp):
            # Learned grasp networks use a symmetric two-finger centre frame.
            # Some physical grippers (including SO-101) expose an EE frame near
            # the fixed finger, so convert through the adapter's calibrated,
            # width-dependent tool geometry before IK and execution.
            T_base_grasp = np.asarray(
                ee_pose_for_grasp(T_base_grasp_center,
                    float(candidate.metadata.get("contact_width_m", candidate.width_m))), dtype=np.float64
            )
        if diagnostic is not None:
            diagnostic.update(
                {
                    "pinch_center_xyz_m": T_base_grasp_center[:3, 3].tolist(),
                    "approach_axis_base": T_base_grasp_center[:3, 2].tolist(),
                    "ee_xyz_m": T_base_grasp[:3, 3].tolist(),
                }
            )
        if T_base_grasp[2, 3] < config.table_height_m + policy.required_table_clearance_m:
            if diagnostic is not None:
                diagnostic["clearance"] = {
                    "ee_z_m": float(T_base_grasp[2, 3]),
                    "table_z_m": config.table_height_m,
                    "required_m": policy.required_table_clearance_m,
                }
            reject("table_clearance")
            continue
        T_grasp_pregrasp = make_transform(translation=np.asarray([0.0, 0.0, -config.pregrasp_distance_m]))
        T_base_pregrasp = T_base_grasp @ T_grasp_pregrasp
        if any(np.linalg.norm(T_base_grasp[:3, 3] - old.T_base_grasp[:3, 3]) < 0.012
               and abs(np.dot(T_base_grasp[:3, 0], old.T_base_grasp[:3, 0])) > np.cos(np.deg2rad(15))
               and np.dot(T_base_grasp[:3, 2], old.T_base_grasp[:3, 2]) > np.cos(np.deg2rad(15))
               for old in failed_grasps):
            reject("previous_failed_grasp")
            continue
        if local_path_check is not None and not local_path_check(T_base_pregrasp, T_base_grasp, candidate.width_m):
            if diagnostic is not None:
                diagnostic["failed_stage"] = "observed_finger_path"
            reject("observed_finger_collision")
            continue
        if not check("pregrasp_reachability", T_base_pregrasp,
                     lambda: motion.is_reachable(T_base_pregrasp)) or not check(
                         "grasp_reachability", T_base_grasp,
                         lambda: motion.is_reachable(T_base_grasp)):
            reject("ik_unreachable")
            continue
        if not check("pregrasp_collision", T_base_pregrasp, lambda: motion.is_collision_free(
            T_base_pregrasp, margin_m=config.collision_margin_m, allow_target_contact=False
        )):
            reject("pregrasp_collision")
            continue
        if not check("grasp_collision", T_base_grasp, lambda: motion.is_collision_free(
            T_base_grasp, margin_m=config.collision_margin_m, allow_target_contact=True
        )):
            reject("grasp_collision")
            continue
        transition_check = getattr(motion, "is_grasp_transition_reachable", None)
        if callable(transition_check) and not transition_check(T_base_pregrasp, T_base_grasp):
            if diagnostic is not None:
                diagnostic["failed_stage"] = "pregrasp_to_grasp_transition"
                diagnostic["transition"] = deepcopy(
                    getattr(motion, "last_transition_diagnostic", {}))
            reject(
                str(
                    getattr(
                        motion,
                        "last_transition_rejection_reason",
                        "approach_disconnected",
                    )
                )
            )
            continue
        realized_pose = getattr(motion, "realized_grasp_pose", None)
        if elongated_minor_axis is not None and callable(realized_pose):
            realized = realized_pose(T_base_grasp)
            realized_axis_error = (
                None if realized is None else closing_axis_error_deg(np.asarray(realized))
            )
            if diagnostic is not None:
                diagnostic["realized_elongated_axis_error_deg"] = realized_axis_error
            if (
                realized_axis_error is None
                or realized_axis_error > config.max_elongated_axis_error_deg
            ):
                reject("realized_closing_axis_misaligned")
                continue
        T_object_grasp = invert_transform(geometry.T_base_object) @ T_base_grasp
        distance_from_center = float(np.linalg.norm(T_object_grasp[:3, 3]))
        extent_scale = max(float(np.linalg.norm(geometry.extents_m)), 0.02)
        adjusted_score = float(candidate.score - 0.1 * distance_from_center / extent_scale)
        feasible.append(
            SelectedGrasp(
                T_base_grasp=T_base_grasp,
                T_base_pregrasp=T_base_pregrasp,
                T_object_grasp=T_object_grasp,
                width_m=candidate.width_m,
                score=adjusted_score,
                observation_sequence=observation.sequence,
                backend=candidate.backend,
                metadata={
                    **dict(candidate.metadata),
                    "raw_score": candidate.score,
                    "support_height_constrained": support_height_constrained,
                },
            )
        )
        if diagnostic is not None:
            diagnostic["rejection"] = None
            diagnostics.append(diagnostic)
        # Live motion needs a validated model-ranked grasp, not an exhaustive
        # ranking after a feasible one is already available. Zero retains the
        # exhaustive policy for callers that need full benchmark comparisons.
        if config.max_feasible_candidates > 0 and len(feasible) >= config.max_feasible_candidates:
            break
    feasible.sort(key=lambda item: item.score, reverse=True)
    return SelectionResult(
        feasible[0] if feasible else None,
        dict(rejected),
        len(diagnostics),
        tuple(diagnostics),
    )
