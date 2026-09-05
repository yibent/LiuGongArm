from __future__ import annotations

from pathlib import Path
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
    return load_yaml("configs/robot_so101.yaml")


def scene_config() -> dict[str, Any]:
    return load_yaml("configs/scene.yaml")


def motion_config() -> dict[str, Any]:
    return load_yaml("configs/motion.yaml")


def cameras_config() -> dict[str, Any]:
    return load_yaml("configs/cameras.yaml")


def fine_grasp_config() -> dict[str, Any]:
    """Configuration for the pluggable eye-in-hand fine-grasp node."""
    return load_yaml("configs/fine_grasp.yaml")
