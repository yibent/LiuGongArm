"""Camera configuration checks that do not require Isaac Sim."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import cameras_config  # noqa: E402
from mr_liu.grasp.transforms import quaternion_wxyz_to_matrix  # noqa: E402


class CameraConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = cameras_config()

    def test_rig_is_enabled(self) -> None:
        self.assertTrue(self.cfg["enabled"])

    def test_successful_wrist_mount_is_default_with_explicit_wide_comparison(self) -> None:
        for name in ("scene", "wrist"):
            self.assertEqual(self.cfg[name]["frequency"], 15)
        wrist = self.cfg["wrist"]
        self.assertEqual(wrist["translation"], [0.060, 0.0, -0.030])
        self.assertEqual(wrist["orientation_wxyz"], [0.53895, 0.0, 0.84234, 0.0])
        self.assertEqual(wrist["focal_length"], 1.8)
        for key, value in wrist["profiles"]["fine_grasp"].items():
            self.assertEqual(wrist[key], value)
        self.assertEqual(wrist["profiles"]["tabletop_wide"]["translation"], [0.150, 0.0, -0.100])
        oblique = self.cfg["scene"]["recovery_oblique"]
        self.assertEqual(oblique["position"], [0.65, -0.65, 1.75])
        self.assertEqual(oblique["target"], [0.35, -0.15, 1.08])

    def test_tabletop_camera_provides_rgbd_semantics_and_is_above_table(self) -> None:
        scene = self.cfg["scene"]
        self.assertEqual(set(scene["annotators"]), {"rgb", "distance_to_image_plane", "semantic_segmentation"})
        self.assertEqual(scene["prim_path"], "/World/Cameras/TableTopRGB")
        self.assertGreater(float(scene["position"][2]), float(scene["target"][2]))
        self.assertEqual(scene["position"][:2], scene["target"][:2])

    def test_placement_camera_is_dedicated_fixed_rgbd_observer(self) -> None:
        placement = self.cfg["placement"]
        self.assertEqual(placement["prim_path"], "/World/Cameras/PlacementRGB")
        self.assertEqual(set(placement["annotators"]), {"rgb", "distance_to_image_plane", "semantic_segmentation"})
        self.assertGreater(float(placement["position"][2]), float(placement["target"][2]))
        self.assertNotEqual(placement["prim_path"], self.cfg["scene"]["prim_path"])

    def test_wrist_camera_is_attached_rgbd(self) -> None:
        wrist = self.cfg["wrist"]
        self.assertEqual(wrist["parent_prim_path"], "/World/SO101/gripper")
        self.assertEqual(wrist["prim_path"].rsplit("/", 1)[0], wrist["parent_prim_path"])
        self.assertEqual(set(wrist["annotators"]), {"rgb", "distance_to_image_plane"})

    def test_default_mount_aims_at_the_tool_region_in_gripper_frame(self) -> None:
        wrist = self.cfg["wrist"]
        # Isaac's authored camera pose uses +X forward. Runtime RGB-D poses
        # instead use OpenCV +Z forward; do not mix these two conventions.
        forward = quaternion_wxyz_to_matrix(wrist["orientation_wxyz"])[:, 0]
        toward_tool = np.array([0., 0., -.16]) - np.asarray(wrist["translation"])
        toward_tool /= np.linalg.norm(toward_tool)
        self.assertGreater(float(forward @ toward_tool), .9999)

    def test_resolutions_are_width_height_pairs(self) -> None:
        for name in ("scene", "wrist"):
            resolution = self.cfg[name]["resolution"]
            self.assertEqual(len(resolution), 2)
            self.assertTrue(all(int(v) > 0 for v in resolution))


if __name__ == "__main__":
    unittest.main()
