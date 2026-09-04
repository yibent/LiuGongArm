"""Frame algebra and bounded-servo tests for eye-in-hand grasping."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.transforms import (  # noqa: E402
    compose,
    interpolate_pose_step,
    invert_transform,
    make_transform,
    pose_error,
    quaternion_wxyz_to_matrix,
    transform_points,
)


class GraspTransformTests(unittest.TestCase):
    def test_eye_in_hand_chain_uses_current_camera_pose(self) -> None:
        T_base_ee = make_transform(translation=np.asarray([0.2, -0.1, 1.0]))
        T_ee_camera = make_transform(translation=np.asarray([0.05, 0.0, 0.02]))
        T_camera_object = make_transform(translation=np.asarray([0.0, 0.0, 0.08]))
        T_base_object = compose(T_base_ee, T_ee_camera, T_camera_object)
        np.testing.assert_allclose(T_base_object[:3, 3], [0.25, -0.1, 1.1])

        moved = make_transform(translation=np.asarray([0.21, -0.08, 1.01]))
        updated = compose(moved, T_ee_camera, T_camera_object)
        self.assertFalse(np.allclose(updated[:3, 3], T_base_object[:3, 3]))

    def test_inverse_and_point_transform_roundtrip(self) -> None:
        rotation = quaternion_wxyz_to_matrix(np.asarray([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]))
        T = make_transform(rotation, np.asarray([1.0, 2.0, 3.0]))
        points = np.asarray([[0.1, 0.2, 0.3], [-1.0, 0.0, 2.0]])
        recovered = transform_points(invert_transform(T), transform_points(T, points))
        np.testing.assert_allclose(recovered, points, atol=1e-8)

    def test_servo_pose_step_is_bounded(self) -> None:
        current = make_transform()
        target = make_transform(
            quaternion_wxyz_to_matrix(np.asarray([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])),
            np.asarray([0.1, 0.0, 0.0]),
        )
        command = interpolate_pose_step(
            current, target, max_translation_m=0.006, max_rotation_rad=math.radians(3.0)
        )
        translation, rotation = pose_error(current, command)
        self.assertAlmostEqual(float(np.linalg.norm(translation)), 0.006, places=7)
        self.assertAlmostEqual(rotation, math.radians(3.0), places=7)


if __name__ == "__main__":
    unittest.main()
