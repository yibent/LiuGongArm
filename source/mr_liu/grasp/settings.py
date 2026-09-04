"""Validated settings for the fine-grasp loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from mr_liu.grasp.selection import SelectionConfig
from mr_liu.grasp.servo import ServoConfig


@dataclass(frozen=True)
class ObservationSettings:
    max_age_s: float
    min_mask_pixels: int
    min_valid_depth_pixels: int
    min_valid_depth_ratio: float
    depth_min_m: float
    depth_max_m: float
    depth_outlier_mad_scale: float
    self_mask_bottom_fraction: float
    max_centroid_jump_m: float


@dataclass(frozen=True)
class GripperSettings:
    min_width_m: float
    max_width_m: float
    default_force_n: float
    default_speed_mps: float


@dataclass(frozen=True)
class LoopSettings:
    max_total_s: float
    max_observation_failures: int
    max_replans: int
    model_period_s: float
    model_replan_translation_m: float
    model_replan_rotation_rad: float
    servo_max_steps: int
    lift_distance_m: float
    verify_settle_s: float


@dataclass(frozen=True)
class FineGraspSettings:
    observation: ObservationSettings
    gripper: GripperSettings
    selection: SelectionConfig
    servo: ServoConfig
    loop: LoopSettings
    table_clearance_m: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "FineGraspSettings":
        obs = config["observation"]
        grip = config["gripper"]
        select = config["selection"]
        loop = config["loop"]
        return cls(
            observation=ObservationSettings(
                max_age_s=float(obs["max_age_s"]),
                min_mask_pixels=int(obs["min_mask_pixels"]),
                min_valid_depth_pixels=int(obs["min_valid_depth_pixels"]),
                min_valid_depth_ratio=float(obs["min_valid_depth_ratio"]),
                depth_min_m=float(obs["depth_min_m"]),
                depth_max_m=float(obs["depth_max_m"]),
                depth_outlier_mad_scale=float(obs["depth_outlier_mad_scale"]),
                self_mask_bottom_fraction=float(obs["self_mask_bottom_fraction"]),
                max_centroid_jump_m=float(obs["max_centroid_jump_m"]),
            ),
            gripper=GripperSettings(
                min_width_m=float(grip["min_width_m"]),
                max_width_m=float(grip["max_width_m"]),
                default_force_n=float(grip["default_force_n"]),
                default_speed_mps=float(grip["default_speed_mps"]),
            ),
            selection=SelectionConfig(
                min_score=float(select["min_model_score"]),
                min_width_m=float(grip["min_width_m"]),
                max_width_m=float(grip["max_width_m"]),
                pregrasp_distance_m=float(select["pregrasp_distance_m"]),
                collision_margin_m=float(select["collision_margin_m"]),
                table_height_m=float(select.get("table_height_m", 0.0)),
                max_candidates=int(select["max_candidates"]),
            ),
            servo=ServoConfig(
                translation_tolerance_m=float(loop["servo_translation_tolerance_m"]),
                rotation_tolerance_rad=math.radians(float(loop["servo_rotation_tolerance_deg"])),
                max_translation_step_m=float(loop["servo_max_translation_step_m"]),
                max_rotation_step_rad=math.radians(float(loop["servo_max_rotation_step_deg"])),
                stable_frames=int(loop["servo_stable_frames"]),
            ),
            loop=LoopSettings(
                max_total_s=float(loop["max_total_s"]),
                max_observation_failures=int(loop["max_observation_failures"]),
                max_replans=int(loop["max_replans"]),
                model_period_s=float(loop["model_period_s"]),
                model_replan_translation_m=float(loop["model_replan_translation_m"]),
                model_replan_rotation_rad=math.radians(float(loop["model_replan_rotation_deg"])),
                servo_max_steps=int(loop["servo_max_steps"]),
                lift_distance_m=float(loop["lift_distance_m"]),
                verify_settle_s=float(loop["verify_settle_s"]),
            ),
            table_clearance_m=float(select["table_clearance_m"]),
        )
