from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from mr_liu.grasp.contracts import CameraIntrinsics, GripperState, PlaceRequest, PickPlacePhase, RGBDObservation, RobotState, TargetSpec
from mr_liu.grasp.place import PickPlaceController
from mr_liu.grasp.transforms import make_transform


class Motion:
    def __init__(self):
        self.pose = make_transform(translation=np.array([0.2, 0.0, 1.2]))

    def robot_state(self): return RobotState(1.0, self.pose)
    def is_reachable(self, pose): return True
    def is_collision_free(self, pose, *, margin_m, allow_target_contact=False): return True
    def move_to(self, pose, *, speed_scale): self.pose = pose; return True
    def servo_to(self, pose, *, speed_scale): self.pose = pose; return True
    def stop(self): pass


class Gripper:
    def __init__(self): self.opened = False
    def state(self): return GripperState(1.0, width_m=0.02 if not self.opened else 0.075, contact=not self.opened, stalled=not self.opened)
    def open(self, width_m, *, speed_mps): self.opened = True; return True
    def close(self, width_m, *, force_n, speed_mps): self.opened = False; return True


class Camera:
    def capture(self, target): return None


def obs():
    return RGBDObservation(1, 1.0, np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), CameraIntrinsics(4, 4, 10, 10, 2, 2), np.eye(4))


class PickPlaceTests(unittest.TestCase):
    def test_release_requires_region_evidence(self):
        motion, gripper = Motion(), Gripper()
        planner = lambda observation, points, ee, request: [(make_transform(translation=np.array([0.5, 0, 1.2])), 0.8)]
        controller = PickPlaceController(motion=motion, gripper=gripper, planner=planner, camera=Camera(), target=TargetSpec("obj"), observer=lambda target, request: np.array([0.5, 0, 1.05]))
        result = controller.execute(PlaceRequest((0.5, 0, 1.05), (0.2, 0.2)), held_points_base=np.ones((20, 3)), before_observation=obs())
        self.assertTrue(result.success)
        self.assertEqual(result.phase, PickPlacePhase.SUCCEEDED)
        self.assertEqual(result.metrics["inside_place_region"], 1)

    def test_missing_evidence_cannot_report_success(self):
        motion, gripper = Motion(), Gripper()
        planner = lambda observation, points, ee, request: [(np.eye(4), 0.8)]
        controller = PickPlaceController(motion=motion, gripper=gripper, planner=planner, camera=Camera(), target=TargetSpec("obj"), observer=lambda target, request: None)
        result = controller.execute(PlaceRequest((0.5, 0, 1.05)), held_points_base=np.ones((20, 3)), before_observation=obs())
        self.assertFalse(result.success)
        self.assertEqual(result.failure.value, "place_verification_failed")


if __name__ == "__main__":
    unittest.main()
