"""Headless runtime check and synchronized captures for both RGB-D cameras."""

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="New directory for captures and report.json")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
output_dir = args.output or ROOT / "output" / "camera_verify" / datetime.now().strftime("%Y%m%d_%H%M%S")
# Do not overwrite evidence from an earlier run.
output_dir.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import sys

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import cv2
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane

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
    report = {"status": "running", "cameras": {}}
    for name, camera in cameras.items():
        assert camera.has_depth, f"{name}: depth is not configured"
        sample = camera.rgbd_frame()
        for _ in range(120):
            if sample is not None:
                break
            simulation_app.update()
            sample = camera.rgbd_frame()
        assert sample is not None, f"{name}: no complete RGB-D/render-pose acquisition"
        rgb, depth = sample.rgba, sample.depth_m
        expected_width, expected_height = cfg[name]["resolution"]
        assert rgb.shape[:2] == (expected_height, expected_width), (name, rgb.shape)
        assert depth.shape == (expected_height, expected_width), (name, depth.shape)
        valid = np.isfinite(depth) & (depth > 0)
        assert valid.any(), f"{name}: depth contains no finite positive samples"
        assert sample.intrinsics[0, 0] > 0 and sample.intrinsics[1, 1] > 0
        np.testing.assert_allclose(np.linalg.det(sample.T_world_camera[:3, :3]), 1., atol=1e-5)
        assert cv2.imwrite(str(output_dir / f"{name}_rgb.png"), rgb[:, :, :3][:, :, ::-1])
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi > lo:
            depth_vis[valid] = np.clip((depth[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        assert cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_vis)
        np.savez_compressed(
            output_dir / f"{name}_rgbd.npz", rgb=rgb[:, :, :3], depth_m=depth,
            K=sample.intrinsics, T_world_camera=sample.T_world_camera,
            rendering_time_s=sample.rendering_time_s,
            rendering_frame=-1 if sample.rendering_frame is None else sample.rendering_frame,
            render_reference=np.asarray(sample.render_reference or (-1, -1), dtype=np.int64),
        )
        # A complete packet must also advance when simulation renders again.
        newer = None
        for _ in range(120):
            simulation_app.update()
            newer = camera.rgbd_frame()
            if (newer is not None and newer.acquisition_id != sample.acquisition_id
                    and newer.rendering_time_s > sample.rendering_time_s):
                break
        else:
            raise AssertionError(f"{name}: RGB-D acquisition did not advance")
        report["cameras"][name] = {
            "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
            "valid_depth_pixels": int(valid.sum()), "valid_depth_ratio": float(valid.mean()),
            "depth_min_m": float(depth[valid].min()), "depth_max_m": float(depth[valid].max()),
            "intrinsics": sample.intrinsics.tolist(),
            "T_world_camera": sample.T_world_camera.tolist(),
            "rendering_time_s": sample.rendering_time_s,
            "acquisition_id": sample.acquisition_id, "next_acquisition_id": newer.acquisition_id,
        }
        print(
            f"RGBD_OK name={name} rgb_shape={rgb.shape} depth_shape={depth.shape} "
            f"valid={int(valid.sum())} range_m=[{float(depth[valid].min()):.4f}, "
            f"{float(depth[valid].max()):.4f}] acquisition={sample.acquisition_id}->{newer.acquisition_id}"
        )

    wrist_parent = cfg["wrist"]["parent_prim_path"]
    assert cfg["wrist"]["prim_path"].rsplit("/", 1)[0] == wrist_parent
    print(f"ATTACHMENT_OK wrist_parent={wrist_parent}")
    report["status"] = "passed"
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CAPTURES_WRITTEN path={output_dir}")


exit_code = 0
try:
    main()
except Exception as exc:
    exit_code = 1
    error = traceback.format_exc()
    (output_dir / "failure.json").write_text(
        json.dumps({"status": "failed", "error": repr(exc), "traceback": error}, indent=2), encoding="utf-8"
    )
    print(error, file=sys.stderr, flush=True)
finally:
    # Isaac 6 fast shutdown exits the process from close(); a later raise is
    # never reached. Preserve the actual validation result in the exit code.
    simulation_app.close(exit_code=exit_code)

raise SystemExit(exit_code)
