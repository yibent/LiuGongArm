"""Headless physics and camera verification for configured tabletop props."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import sys
import traceback
from pathlib import Path

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import robot_config, scene_config
from mr_liu.perception.camera import SceneCamera
from mr_liu.sim.spawn import spawn_table_and_so101


def _subtree_counts(root) -> tuple[int, int]:
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    rigid = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims)
    colliders = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims)
    return rigid, colliders


def main() -> None:
    stage_utils.create_new_stage()
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(float(scene_config()["distant_intensity"]))
    spawn_table_and_so101()
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_config()["usd_prim_path"])
    robot_world = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    robot_quat = robot_world.ExtractRotationQuat()
    actual = np.asarray([robot_quat.GetReal(), *robot_quat.GetImaginary()], dtype=np.float64)
    expected = np.asarray(scene_config()["robot_orient_wxyz"], dtype=np.float64)
    alignment = abs(float(np.dot(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected))))
    assert alignment > 0.99999, f"Robot orientation mismatch: actual={actual}, expected={expected}"
    print(f"ROBOT_POSE_OK orientation_wxyz={actual.round(6).tolist()} alignment={alignment:.6f}")

    initial_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    initial_min_z = {
        prop["name"]: float(
            initial_bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prop["prim_path"]))
            .ComputeAlignedBox()
            .GetMin()[2]
        )
        for prop in scene_config()["tabletop_props"]
    }

    camera = SceneCamera("scene")
    camera.spawn()
    app_utils.play()
    for _ in range(180):
        simulation_app.update()

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    table_z = float(scene_config()["table_pos"][2])
    for prop in scene_config()["tabletop_props"]:
        prim = stage.GetPrimAtPath(prop["prim_path"])
        assert prim.IsValid(), f"Missing prop: {prop['prim_path']}"
        rigid, colliders = _subtree_counts(prim)
        assert rigid >= 1, f"{prop['name']}: no rigid body"
        assert colliders >= 1, f"{prop['name']}: no collider"
        world_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = world_box.GetMin()
        maximum = world_box.GetMax()
        size = maximum - minimum
        assert minimum[2] > table_z - 0.08, f"{prop['name']}: fell through table: z={minimum[2]}"
        assert max(size) < 0.40, f"{prop['name']}: too large for manipulation: size={size}"
        drop = initial_min_z[prop["name"]] - float(minimum[2])
        assert drop > 0.005, f"{prop['name']}: did not respond to gravity: drop={drop}"
        print(
            f"PROP_OK name={prop['name']} label={prop['semantic_label']} "
            f"rigid_bodies={rigid} colliders={colliders} "
            f"gravity_drop_m={drop:.4f} "
            f"bounds_min={[round(float(v), 4) for v in minimum]} "
            f"bounds_max={[round(float(v), 4) for v in maximum]}"
        )

    rgb = camera.rgb_rgba()
    assert rgb is not None, "Tabletop camera did not produce an RGB frame"
    output_dir = ROOT / "output" / "tabletop_props_verify"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "tabletop_rgb.png"
    cv2.imwrite(str(image_path), rgb[:, :, :3][:, :, ::-1])
    print(f"TABLETOP_PROPS_OK count={len(scene_config()['tabletop_props'])} image={image_path}")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
