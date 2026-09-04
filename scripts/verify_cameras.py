"""Headless runtime check for the tabletop RGB and wrist RGBD cameras."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import sys
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import cv2
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import cameras_config, scene_config
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.sim.spawn import spawn_table_and_so101


def main() -> None:
    stage_utils.create_new_stage()
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene_config()["distant_intensity"]))
    spawn_table_and_so101()
    simulation_app.update()

    cameras = spawn_configured_cameras()
    assert set(cameras) == {"scene", "wrist"}, f"Unexpected cameras: {sorted(cameras)}"
    app_utils.play()
    for _ in range(30):
        simulation_app.update()

    cfg = cameras_config()
    output_dir = ROOT / "output" / "camera_verify"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, camera in cameras.items():
        rgb = camera.rgb_rgba()
        assert rgb is not None, f"{name}: no RGB frame"
        expected_width, expected_height = cfg[name]["resolution"]
        assert rgb.shape[:2] == (expected_height, expected_width), (name, rgb.shape)
        cv2.imwrite(str(output_dir / f"{name}_rgb.png"), rgb[:, :, :3][:, :, ::-1])
        position, orientation = camera.world_pose()
        print(
            f"CAMERA_OK name={name} rgb_shape={rgb.shape} "
            f"position={np.asarray(position).round(4).tolist()} "
            f"orientation_wxyz={np.asarray(orientation).round(4).tolist()}"
        )

        if name == "wrist":
            depth = camera.depth_m()
            assert depth is not None, "wrist: no depth frame"
            assert depth.shape == (expected_height, expected_width), depth.shape
            valid = np.isfinite(depth) & (depth > 0)
            assert valid.any(), "wrist: depth contains no finite positive samples"
            depth_vis = np.zeros(depth.shape, dtype=np.uint8)
            lo, hi = np.percentile(depth[valid], [2, 98])
            if hi > lo:
                depth_vis[valid] = np.clip((depth[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "wrist_depth.png"), depth_vis)
            print(
                f"DEPTH_OK name=wrist shape={depth.shape} valid={int(valid.sum())} "
                f"min_m={float(depth[valid].min()):.4f} max_m={float(depth[valid].max()):.4f}"
            )

    wrist_parent = cfg["wrist"]["parent_prim_path"]
    assert cfg["wrist"]["prim_path"].rsplit("/", 1)[0] == wrist_parent
    print(f"ATTACHMENT_OK wrist_parent={wrist_parent}")
    print(f"CAPTURES_WRITTEN path={output_dir}")


try:
    main()
finally:
    simulation_app.close()
