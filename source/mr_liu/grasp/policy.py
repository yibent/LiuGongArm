"""Object-aware gripper and motion policy without task-specific demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

from mr_liu.grasp.contracts import ObjectProperties, TargetSpec


@dataclass(frozen=True)
class ExecutionPolicy:
    force_n: float
    close_speed_mps: float
    motion_speed_scale: float
    required_table_clearance_m: float
    allow_depth_hole_recovery: bool
    require_thin_object_clearance: bool


_FRUIT_WORDS = {"apple", "banana", "orange", "pear", "peach", "fruit", "苹果", "香蕉", "橙", "梨", "水果"}
_THIN_WORDS = {"key", "card", "washer", "钥匙", "卡片", "垫片", "薄片"}
_REFLECTIVE_WORDS = {"metal", "chrome", "steel", "aluminum", "金属", "钢", "铝", "反光"}


def _contains_any(label: str, words: set[str]) -> bool:
    value = label.casefold()
    return any(word.casefold() in value for word in words)


def policy_for_target(
    target: TargetSpec,
    *,
    default_force_n: float,
    default_speed_mps: float,
    table_clearance_m: float,
) -> ExecutionPolicy:
    label = target.semantic_label or ""
    properties: ObjectProperties = target.properties
    fragile = properties.fragile or _contains_any(label, _FRUIT_WORDS)
    thin = properties.thin or _contains_any(label, _THIN_WORDS)
    reflective = properties.reflective or _contains_any(label, _REFLECTIVE_WORDS)

    force = min(default_force_n, 6.0) if fragile else default_force_n
    if properties.max_grip_force_n is not None:
        force = min(force, max(0.5, float(properties.max_grip_force_n)))
    close_speed = min(default_speed_mps, 0.010) if fragile else default_speed_mps
    return ExecutionPolicy(
        force_n=force,
        close_speed_mps=close_speed,
        motion_speed_scale=0.35 if fragile or thin else 0.55,
        required_table_clearance_m=max(table_clearance_m, 0.012 if thin else table_clearance_m),
        allow_depth_hole_recovery=reflective,
        require_thin_object_clearance=thin,
    )
