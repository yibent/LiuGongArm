"""Capture SO-101 wrist views at both gripper joint limits for calibration."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import json
import sys
import traceback
from pathlib import Path

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import XformPrim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import scene_config
from mr_liu.grasp.adapters.isaac_camera import T_world_opencv_from_ros_pose
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.robot.so101 import So101Arm
from mr_liu.sim.spawn import spawn_table_and_so101


def main() -> None:
    stage_utils.create_new_stage()
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene_config()["distant_intensity"]))
    spawn_table_and_so101()
    simulation_app.update()
    cameras = spawn_configured_cameras()
    arm = So101Arm()
    app_utils.play()
    for _ in range(30):
        simulation_app.update()
    arm.apply_configured_pose()
    for _ in range(30):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    related_prims = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if "gripper" in str(prim.GetPath()).casefold() or "jaw" in str(prim.GetPath()).casefold()
    ]
    limits_raw = arm.articulation.get_dof_limits()
    limits = limits_raw.numpy() if hasattr(limits_raw, "numpy") else np.asarray(limits_raw)
    limits = np.asarray(limits, dtype=np.float64)
    gripper_index = arm.dof_names.index("gripper")
    # Isaac Sim 6's experimental Articulation API returns [lower/upper, batch, dof],
    # while older releases returned [batch, dof, lower/upper].  Normalize both.
    if limits.ndim == 3 and limits.shape[0] == 2:
        low = float(limits[0, 0, gripper_index])
        high = float(limits[1, 0, gripper_index])
    elif limits.shape[-1] == 2:
        low, high = [float(v) for v in limits.reshape(-1, 2)[gripper_index]]
    else:
        raise RuntimeError(f"Unexpected DOF-limit shape: {limits.shape}")
    output = ROOT / "output" / "gripper_calibration"
    output.mkdir(parents=True, exist_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    captures: list[dict[str, object]] = []
    for label, joint_value in (("lower", low), ("mid", (low + high) * 0.5), ("upper", high)):
        positions_raw = arm.articulation.get_dof_positions()
        positions = positions_raw.numpy() if hasattr(positions_raw, "numpy") else np.asarray(positions_raw)
        positions = np.asarray(positions, dtype=np.float32).reshape(-1).copy()
        positions[gripper_index] = joint_value
        arm.articulation.set_dof_positions(positions)
        arm.articulation.set_dof_position_targets(positions)
        for _ in range(30):
            simulation_app.update()
        rgb = cameras["wrist"].rgb_rgba()
        assert rgb is not None
        camera_position, camera_orientation = cameras["wrist"].world_pose(camera_axes="ros")
        jaw_positions, jaw_orientations = XformPrim(
            "/World/SO101/moving_jaw_so101_v1"
        ).get_world_poses()
        path = output / f"wrist_{label}.png"
        cv2.imwrite(str(path), rgb[:, :, :3][:, :, ::-1])
        captures.append(
            {
                "label": label,
                "joint_rad": joint_value,
                "path": str(path),
                "T_base_camera": T_world_opencv_from_ros_pose(
                    np.asarray(camera_position), np.asarray(camera_orientation)
                ).round(8).tolist(),
                "jaw_position_base_m": np.asarray(jaw_positions).reshape(-1, 3)[0].round(8).tolist(),
                "jaw_orientation_wxyz": np.asarray(jaw_orientations).reshape(-1, 4)[0].round(8).tolist(),
            }
        )
    report = {
        "dof_names": arm.dof_names,
        "gripper_limit_rad": [low, high],
        "related_prims": related_prims,
        "captures": captures,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


try:
    main()
except Exception:  # pragma: no cover - diagnostic script must persist Kit failures
    output = ROOT / "output" / "gripper_calibration"
    output.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (output / "error.txt").write_text(error, encoding="utf-8")
    print(error, file=sys.stderr)
    raise
finally:
    simulation_app.close()
