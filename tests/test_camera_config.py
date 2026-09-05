"""Camera configuration checks that do not require Isaac Sim."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.config import cameras_config  # noqa: E402


class CameraConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = cameras_config()

    def test_rig_is_enabled(self) -> None:
        self.assertTrue(self.cfg["enabled"])

    def test_tabletop_camera_is_rgbd_and_above_table(self) -> None:
        scene = self.cfg["scene"]
        self.assertEqual(set(scene["annotators"]), {"rgb", "distance_to_image_plane"})
        self.assertEqual(scene["prim_path"], "/World/Cameras/TableTopRGB")
        self.assertGreater(float(scene["position"][2]), float(scene["target"][2]))
        self.assertEqual(scene["position"][:2], scene["target"][:2])

    def test_wrist_camera_is_attached_rgbd(self) -> None:
        wrist = self.cfg["wrist"]
        self.assertEqual(wrist["parent_prim_path"], "/World/SO101/gripper")
        self.assertEqual(wrist["prim_path"].rsplit("/", 1)[0], wrist["parent_prim_path"])
        self.assertEqual(set(wrist["annotators"]), {"rgb", "distance_to_image_plane"})

    def test_resolutions_are_width_height_pairs(self) -> None:
        for name in ("scene", "wrist"):
            resolution = self.cfg[name]["resolution"]
            self.assertEqual(len(resolution), 2)
            self.assertTrue(all(int(v) > 0 for v in resolution))


if __name__ == "__main__":
    unittest.main()
