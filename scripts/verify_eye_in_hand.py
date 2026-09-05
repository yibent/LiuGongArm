"""Headless runtime proof that the wrist RGB-D pose follows the moving tool.

This checks the complete identity T_base_camera = T_base_ee @ T_ee_camera
before and after an arm joint motion and rejects a drifting hand-eye extrinsic.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import json
import sys
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import scene_config
from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera
from mr_liu.grasp.contracts import TargetSpec
from mr_liu.grasp.transforms import pose_error
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
    for _ in range(20):
        simulation_app.update()
    arm.apply_configured_pose()
    for _ in range(20):
        simulation_app.update()

    source = IsaacWristCamera(
        cameras["wrist"],
        ee_pose_provider=arm.T_base_ee,
        advance_frame=simulation_app.update,
    )
    first = source.capture(TargetSpec("calibration-probe"))
    assert first is not None

    names = arm.dof_names
    pan_index = names.index("shoulder_pan")
    positions = np.asarray(arm.articulation.get_dof_positions(), dtype=np.float32)
    positions[pan_index] += 0.16
    arm.articulation.set_dof_position_targets(positions)
    for _ in range(60):
        simulation_app.update()
    second = source.capture(TargetSpec("calibration-probe"))
    assert second is not None

    camera_translation, camera_rotation = pose_error(first.T_base_camera, second.T_base_camera)
    extrinsic_translation, extrinsic_rotation = pose_error(first.T_ee_camera, second.T_ee_camera)
    camera_motion = float(np.linalg.norm(camera_translation))
    extrinsic_drift = float(np.linalg.norm(extrinsic_translation))
    assert second.sequence > first.sequence
    assert camera_motion > 0.005, f"Wrist camera did not move with the arm: {camera_motion}"
    assert extrinsic_drift < 0.001, extrinsic_drift
    assert extrinsic_rotation < 0.002, extrinsic_rotation
    np.testing.assert_allclose(
        second.T_base_ee @ second.T_ee_camera, second.T_base_camera, atol=2e-4
    )

    report = {
        "status": "EYE_IN_HAND_OK",
        "camera_translation_m": camera_motion,
        "camera_rotation_rad": camera_rotation,
        "extrinsic_translation_drift_m": extrinsic_drift,
        "extrinsic_rotation_drift_rad": extrinsic_rotation,
        "T_ee_camera": second.T_ee_camera.round(8).tolist(),
        "intrinsics": second.intrinsics.matrix.round(5).tolist(),
    }
    output = ROOT / "output" / "eye_in_hand_verify"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


try:
    main()
finally:
    simulation_app.close()
