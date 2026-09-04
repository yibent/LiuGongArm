"""Model-independent grasp feasibility filtering and ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mr_liu.grasp.contracts import GraspCandidate, MotionExecutor, RGBDObservation, SelectedGrasp
from mr_liu.grasp.geometry import ObjectGeometry
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


@dataclass(frozen=True)
class SelectionResult:
    grasp: SelectedGrasp | None
    rejected: dict[str, int]
    considered: int


def select_grasp(
    candidates: Sequence[GraspCandidate],
    observation: RGBDObservation,
    geometry: ObjectGeometry,
    motion: MotionExecutor,
    policy: ExecutionPolicy,
    config: SelectionConfig,
) -> SelectionResult:
    rejected: Counter[str] = Counter()
    feasible: list[SelectedGrasp] = []
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)[: config.max_candidates]
    for candidate in ordered:
        if candidate.observation_sequence != observation.sequence:
            rejected["stale_candidate"] += 1
            continue
        if candidate.score < config.min_score:
            rejected["low_score"] += 1
            continue
        if candidate.width_m < config.min_width_m or candidate.width_m > config.max_width_m:
            rejected["gripper_width"] += 1
            continue
        T_base_grasp = observation.T_base_camera @ candidate.T_camera_grasp
        if T_base_grasp[2, 3] < config.table_height_m + policy.required_table_clearance_m:
            rejected["table_clearance"] += 1
            continue
        T_grasp_pregrasp = make_transform(translation=np.asarray([0.0, 0.0, -config.pregrasp_distance_m]))
        T_base_pregrasp = T_base_grasp @ T_grasp_pregrasp
        if not motion.is_reachable(T_base_pregrasp) or not motion.is_reachable(T_base_grasp):
            rejected["ik_unreachable"] += 1
            continue
        if not motion.is_collision_free(
            T_base_pregrasp, margin_m=config.collision_margin_m, allow_target_contact=False
        ):
            rejected["pregrasp_collision"] += 1
            continue
        if not motion.is_collision_free(
            T_base_grasp, margin_m=config.collision_margin_m, allow_target_contact=True
        ):
            rejected["grasp_collision"] += 1
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
                metadata={**dict(candidate.metadata), "raw_score": candidate.score},
            )
        )
    feasible.sort(key=lambda item: item.score, reverse=True)
    return SelectionResult(feasible[0] if feasible else None, dict(rejected), len(ordered))
