"""Joint-name mapping between USD, URDF, and cuMotion XRDF."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from mr_liu.config import robot_config
from mr_liu.paths import repo_root

_XRDF_JOINT_BLOCK = re.compile(
    r"cspace:\s*\n\s*joint_names:\s*\n((?:\s*-\s*\"[^\"]+\"\s*\n)+)",
    re.MULTILINE,
)


def robot_dir() -> Path:
    cfg = robot_config()
    return repo_root() / cfg["robot_dir"]


def parse_urdf_joints(urdf_path: Path | None = None) -> list[str]:
    path = urdf_path or (robot_dir() / robot_config()["urdf"])
    root = ET.parse(path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.get("type") in ("revolute", "prismatic", "continuous"):
            name = joint.get("name")
            if name:
                names.append(name)
    return names


def parse_xrdf_cspace(xrdf_path: Path | None = None) -> list[str]:
    path = xrdf_path or (robot_dir() / robot_config()["xrdf"])
    text = path.read_text(encoding="utf-8")
    match = _XRDF_JOINT_BLOCK.search(text)
    if not match:
        raise ValueError(f"No cspace.joint_names block in {path}")
    return re.findall(r"-\s*\"([^\"]+)\"", match.group(1))


def expected_usd_joints() -> list[str]:
    return list(robot_config()["all_joints"])


def assert_joint_mapping(*, usd_dof_names: list[str] | None = None) -> None:
    """Raise if URDF / XRDF / config (and optional live USD DOFs) disagree."""
    cfg = robot_config()
    urdf_joints = parse_urdf_joints()
    xrdf_joints = parse_xrdf_cspace()
    configured = list(cfg["all_joints"])
    controlled = list(cfg["controlled_joints"])

    if urdf_joints != configured:
        raise ValueError(f"URDF joints {urdf_joints} != config all_joints {configured}")
    if xrdf_joints != controlled:
        raise ValueError(f"XRDF cspace {xrdf_joints} != config controlled_joints {controlled}")
    if not set(controlled).issubset(set(urdf_joints)):
        raise ValueError(f"controlled_joints {controlled} not a subset of URDF {urdf_joints}")
    if usd_dof_names is not None:
        missing = [name for name in configured if name not in usd_dof_names]
        if missing:
            raise ValueError(f"USD articulation is missing joints {missing}; have {usd_dof_names}")
