"""SDK-boundary regressions for merging grasp support with motion stability."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.config import robot_config
from mr_liu.perception.camera import SceneCamera
from mr_liu.sim import spawn


def load_arm_module():
    sdk = ModuleType("isaacsim.core.experimental.prims")
    sdk.Articulation = Mock()
    sdk.XformPrim = Mock()
    backend = ModuleType("isaacsim.core.experimental.utils.backend")
    backend.use_backend = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    spec = importlib.util.spec_from_file_location(
        "merge_test_so101", ROOT / "source/mr_liu/robot/so101.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {sdk.__name__: sdk, backend.__name__: backend}):
        spec.loader.exec_module(module)
    return module


class MergeIntegrationTests(unittest.TestCase):
    def test_basic_arm_import_does_not_load_optional_grasp_stack(self):
        with patch.dict(sys.modules, {"mr_liu.grasp.transforms": None}):
            module = load_arm_module()
            self.assertTrue(callable(module.So101Arm.configure_drives))
            self.assertTrue(callable(module.So101Arm.T_base_ee))

    def test_camera_spawn_keeps_render_pose_and_scene_semantics(self):
        sdk = ModuleType("isaacsim.sensors.camera")
        sdk.Camera = Mock()
        with patch.dict(sys.modules, {sdk.__name__: sdk}):
            for name in ("scene", "wrist"):
                with self.subTest(camera=name):
                    sdk.Camera.reset_mock()
                    camera = SceneCamera(name)
                    with patch.object(camera, "_require_parent"):
                        camera.spawn()
                    sensor = sdk.Camera.return_value
                    sensor.add_distance_to_image_plane_to_frame.assert_called_once()
                    sensor.attach_annotator.assert_called_once_with("camera_params")
                    if name == "scene":
                        sensor.add_semantic_segmentation_to_frame.assert_called_once_with(
                            {"colorize": False}
                        )
                    else:
                        sensor.add_semantic_segmentation_to_frame.assert_not_called()

    def test_arm_keeps_damping_and_recovers_tool_pose(self):
        module = load_arm_module()
        arm = module.So101Arm()
        cfg = robot_config()
        names = list(cfg["drive_damping"])
        arm.articulation.dof_names = names
        damping = [cfg["drive_damping"][name] for name in names]
        stiffness_tensor, damping_tensor = Mock(), Mock()
        stiffness_tensor.numpy.return_value = np.full((1, len(names)), 100.)
        damping_tensor.numpy.return_value = np.array([damping])
        arm.articulation.get_dof_gains.return_value = stiffness_tensor, damping_tensor
        gains = arm.configure_drives()
        arm.articulation.set_dof_gains.assert_called_once_with(dampings=[damping])
        arm.articulation.set_dof_velocity_targets.assert_called_once_with([[0.] * len(names)])
        self.assertEqual(gains[names[0]]["stiffness_nm_rad"], 100.)

        # A rotated world mount must rotate the fixed URDF tool translation too.
        s = np.sqrt(0.5)
        module.XformPrim.return_value.get_world_poses.return_value = (
            np.array([[1., 2., 3.]]), np.array([[s, 0., 0., s]])
        )
        result = arm.T_base_ee()
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(
            result[:3, 3], [1., 2., 3.] + rotation @ np.array(cfg["tool_from_parent_translation"])
        )
        np.testing.assert_allclose(result[:3, :3], rotation @ np.diag([-1., 1., -1.]), atol=1e-12)
        module.XformPrim.assert_called_once_with(cfg["tool_parent_prim_path"], reset_xform_op_properties=False)
        module.use_backend.assert_called_once_with("fabric", raise_on_fallback=True)

    def test_mount_reuses_root_instead_of_adding_second_constraint(self):
        # Mock only SDK boundaries; exercise the production mount function.
        stage, base, root, duplicate, joint = (Mock() for _ in range(5))
        cfg = robot_config()
        stage.GetPrimAtPath.side_effect = lambda path: {
            cfg["articulation_prim_path"]: root, spawn.MOUNT_JOINT_PATH: duplicate
        }[path]
        stage.Traverse.return_value = []
        root.IsValid.return_value = root.IsA.return_value = True
        duplicate.IsValid.return_value = True
        base.GetPath.return_value = "/World/SO101/base"
        usd = ModuleType("omni.usd")
        usd.get_context = Mock()
        usd.get_context.return_value.get_stage.return_value = stage
        omni = ModuleType("omni")
        omni.usd = usd
        pxr = ModuleType("pxr")
        pxr.Gf, pxr.Usd, pxr.UsdGeom, pxr.UsdPhysics, pxr.PhysxSchema = (Mock() for _ in range(5))
        pxr.UsdPhysics.FixedJoint.return_value = joint
        with patch.dict(sys.modules, {"omni": omni, "omni.usd": usd, "pxr": pxr}), \
                patch.object(spawn, "_find_so101_base", return_value=base):
            spawn.mount_so101_to_table()
        pxr.UsdPhysics.FixedJoint.Define.assert_not_called()
        duplicate.SetActive.assert_called_once_with(False)
        joint.CreateBody0Rel.return_value.ClearTargets.assert_called_once_with(True)
        joint.CreateBody1Rel.return_value.SetTargets.assert_called_once_with([base.GetPath()])
        articulation = pxr.PhysxSchema.PhysxArticulationAPI.Apply.return_value
        articulation.CreateSolverPositionIterationCountAttr.return_value.Set.assert_called_once_with(
            cfg["solver_position_iterations"]
        )
        articulation.CreateSolverVelocityIterationCountAttr.return_value.Set.assert_called_once_with(
            cfg["solver_velocity_iterations"]
        )


if __name__ == "__main__":
    unittest.main()
