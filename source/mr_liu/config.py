from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from mr_liu.paths import repo_root

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(relative_path: str | Path) -> dict[str, Any]:
    """Load a YAML file relative to the repository root."""
    path = repo_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Use Isaac Sim's python.bat.")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return data


def robot_config() -> dict[str, Any]:
    name = os.environ.get("MR_LIU_ROBOT", "so101").lower()
    if name not in {"so101", "franka"}:
        raise ValueError(f"Unknown robot profile: {name}")
    return load_yaml(f"configs/robot_{name}.yaml")


def _profile_config(name: str) -> dict[str, Any]:
    data = load_yaml(f"configs/{name}.yaml")
    def merge(destination, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = value
    merge(data, robot_config().get("overrides", {}).get(name, {}))
    # Legacy configs contain SO-101 exclusion/mount paths.  Keep the same
    # files readable while making every active profile use its own USD root.
    active_root = str(robot_config().get("usd_prim_path", "/World/SO101"))
    def rewrite(value):
        if isinstance(value, str):
            return value.replace("/World/SO101", active_root)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value
    return rewrite(data)


def scene_config() -> dict[str, Any]:
    return _profile_config("scene")


def motion_config() -> dict[str, Any]:
    return _profile_config("motion")


def cameras_config() -> dict[str, Any]:
    return _profile_config("cameras")


def fine_grasp_config() -> dict[str, Any]:
    """Configuration for the pluggable eye-in-hand fine-grasp node."""
    return _profile_config("fine_grasp")
