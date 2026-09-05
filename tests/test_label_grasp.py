from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.control.coarse_approach import CoarseApproach
from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, TargetSpec
from mr_liu.perception.optical_flow import OpticalFlowTracker
from mr_liu.perception.semantic_target import MetricTarget, SemanticFlowTarget, TargetObservationError, metric_component
from mr_liu.sim.grasp_faults import OneShotCoarseShift
from mr_liu.perception.camera import SceneCamera


def sample(sequence=1, timestamp=10., second=False):
    rgb = np.zeros((100, 120, 3), np.uint8) + 45
    rgb[20:40, 25:45] = np.random.default_rng(1).integers(60, 250, (20, 20, 3), dtype=np.uint8)
    depth = np.ones((100, 120), np.float32)
    depth[20:40, 25:45] = 0.95
    if second:
        rgb[60:80, 75:95] = rgb[20:40, 25:45]
        depth[60:80, 75:95] = 0.95
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 2.
    return RGBDObservation(sequence, timestamp, rgb, depth, CameraIntrinsics(120, 100, 100, 100, 60, 50), pose)


class LabelPerceptionTests(unittest.TestCase):
    def test_camera_profiles_do_not_mutate_the_selected_default(self):
        live = SceneCamera("wrist")
        wide = SceneCamera("wrist", profile="tabletop_wide")
        close = SceneCamera("wrist", profile="fine_grasp")
        self.assertEqual(wide.config["translation"], [.150, 0., -.100])
        self.assertEqual(live.config["translation"], [.060, 0., -.030])
        self.assertEqual(close.config["translation"], [.060, 0., -.030])
        self.assertEqual(SceneCamera("wrist").config, live.config)
        with self.assertRaises(ValueError):
            SceneCamera("wrist", profile="missing")

    def test_lk_tracks_real_translation(self):
        obs = sample()
        mask = obs.depth_m < 1
        tracker = OpticalFlowTracker()
        self.assertTrue(tracker.initialize(obs.rgb, mask))
        current = cv2.warpAffine(obs.rgb, np.float32([[1, 0, 3], [0, 1, 2]]), (120, 100))
        moved = tracker.update(current)
        self.assertIsNotNone(moved)
        self.assertEqual(tracker.diagnostic["flow_reason"], "lk_forward_backward")
        np.testing.assert_allclose(tracker.diagnostic["flow_shift_px"], [3, 2], atol=0.2)

    def test_textureless_target_cannot_claim_flow(self):
        tracker = OpticalFlowTracker()
        self.assertFalse(tracker.initialize(np.zeros((50, 50, 3), np.uint8), np.ones((50, 50), bool)))
        self.assertIsNone(tracker.update(np.zeros((50, 50, 3), np.uint8)))

    def test_metric_projection_uses_camera_transform_not_pixel_heuristic(self):
        obs = sample()
        evidence = metric_component(obs, obs.depth_m < 1, 1.)
        pose = obs.T_base_camera.copy()
        pose[:2, 3] += [0.4, -0.2]
        shifted = metric_component(replace(obs, T_base_camera=pose), obs.depth_m < 1, 1.)
        np.testing.assert_allclose(shifted.center_base_m - evidence.center_base_m, [0.4, -0.2, 0], atol=1e-6)
        self.assertAlmostEqual(evidence.top_z_m, 1.05, places=5)

    def test_missing_depth_is_not_geometry(self):
        obs = sample()
        with self.assertRaisesRegex(TargetObservationError, "insufficient_target_depth"):
            metric_component(replace(obs, depth_m=np.full_like(obs.depth_m, np.nan)), np.ones_like(obs.depth_m, bool), 1.)

    def test_robot_mask_cannot_be_selected_as_object(self):
        obs = sample()
        with self.assertRaises(TargetObservationError):
            metric_component(replace(obs, metadata={"robot_self_mask": obs.depth_m < 1}), np.ones_like(obs.depth_m, bool), 1.)

    def test_find_requires_fresh_observation_after_model_latency(self):
        now = [10.]
        camera = Mock()
        camera.capture.side_effect = lambda target: sample(camera.capture.call_count, now[0])
        locator = Mock()
        def detect(obs, label):
            now[0] += 12
            return {"boxes": [{"xyxy": [20, 15, 50, 45]}], "sequence": obs.sequence}
        locator.locate.side_effect = detect
        tracker = SemanticFlowTarget(camera, locator, TargetSpec("target", "part"), 1., clock=lambda: now[0])
        evidence = tracker.find()
        self.assertEqual(evidence.observation.sequence, 2)
        self.assertEqual(evidence.observation.timestamp_s, 22.)
        self.assertEqual(tracker.flow_count, 1)

    def test_multiple_label_instances_are_ambiguous(self):
        camera, locator = Mock(), Mock()
        camera.capture.return_value = sample(second=True)
        locator.locate.return_value = {"boxes": [{"xyxy": [20, 15, 50, 45]}, {"xyxy": [70, 55, 100, 85]}]}
        tracker = SemanticFlowTarget(camera, locator, TargetSpec("target", "part"), 1., clock=lambda: 10.)
        with self.assertRaisesRegex(TargetObservationError, "ambiguous_label_instances"):
            tracker.find()

    def test_repeated_rgbd_frame_is_rejected(self):
        camera = Mock()
        camera.capture.return_value = sample()
        tracker = SemanticFlowTarget(camera, Mock(), TargetSpec("target", "part"), 1., clock=lambda: 10.)
        tracker.capture()
        with self.assertRaisesRegex(TargetObservationError, "stale_rgbd"):
            tracker.capture()

    def test_wrist_handoff_requires_identity_and_freshness(self):
        obs = sample()
        tracker = SemanticFlowTarget(Mock(), Mock(), TargetSpec("target", "part"), 1., clock=lambda: 10.)
        tracker.latest = metric_component(obs, obs.depth_m < 1, 1.)
        tracker.anchor_color = tracker.latest.color
        tracker.handoff_after_s = 9.99
        verified = tracker.validate_wrist_handoff(obs, obs.depth_m < 1)
        self.assertEqual(verified["mask_pixels"], 400)
        with self.assertRaisesRegex(TargetObservationError, "stale_wrist_handoff"):
            tracker.validate_wrist_handoff(replace(obs, timestamp_s=9.), obs.depth_m < 1)
        pose = obs.T_base_camera.copy()
        pose[0, 3] += .10
        with self.assertRaisesRegex(TargetObservationError, "identity_mismatch"):
            tracker.validate_wrist_handoff(replace(obs, sequence=2, T_base_camera=pose), obs.depth_m < 1)
        with self.assertRaisesRegex(TargetObservationError, "stale_wrist_handoff"):
            tracker.validate_wrist_handoff(obs, obs.depth_m < 1)

    def test_wrist_table_mask_cannot_authorize_handoff(self):
        obs = sample()
        tracker = SemanticFlowTarget(Mock(), Mock(), TargetSpec("target", "part"), 1., clock=lambda: 10.)
        tracker.latest = metric_component(obs, obs.depth_m < 1, 1.)
        tracker.anchor_color = tracker.latest.color
        tracker.handoff_after_s = 9.99
        with self.assertRaisesRegex(TargetObservationError, "insufficient_target_depth"):
            tracker.validate_wrist_handoff(obs, obs.depth_m == 1)


class FakeMotion:
    def __init__(self, reachable=True):
        self.pose = np.diag([1., -1., -1., 1.])
        self.pose[:3, 3] = [0.38, 0., 1.25]
        self.execution_guard = None
        self.stops = self.moves = 0
        self.reachable = reachable
    def robot_state(self): return Mock(T_base_ee=self.pose.copy())
    def stop(self): self.stops += 1
    def clear_grasp_plan(self): pass
    def ee_pose_for_grasp(self, pose, width): return pose.copy()
    def is_observation_path_safe(self, *args): return self.reachable
    def move_to(self, pose, speed_scale):
        self.moves += 1
        if not self.execution_guard(): return False
        self.pose = pose.copy()
        return True


class CoarseHandoffTests(unittest.TestCase):
    def tracker(self):
        obs = sample()
        evidence = MetricTarget(obs, obs.depth_m < 1, np.array([[.2, 0, 1.05]]), np.array([.2, 0, 1.025]), 1.05, np.ones(3))
        tracker = Mock()
        tracker.target = TargetSpec("instance", "red part")
        tracker.find.return_value = tracker.observe.return_value = evidence
        tracker.find_count = 1
        return tracker

    def test_success_stops_coarse_owner_and_hands_measured_target_to_fine(self):
        motion, events = FakeMotion(), []
        coarse = CoarseApproach(self.tracker(), motion, trace=events.append)
        target = coarse.run()
        self.assertEqual(target.coarse_position_base_m, (.2, 0., 1.025))
        self.assertGreater(motion.moves, 1)
        self.assertGreater(coarse.guard_count, 1)
        self.assertIsNone(motion.execution_guard)
        self.assertGreater(motion.stops, 1)
        self.assertEqual(events[-1]["event"], "coarse_handoff")

    def test_infeasible_path_does_not_move_or_handoff(self):
        motion = FakeMotion(reachable=False)
        with self.assertRaisesRegex(TargetObservationError, "coarse_path_infeasible"):
            CoarseApproach(self.tracker(), motion).run()
        self.assertEqual(motion.moves, 0)
        self.assertIsNone(motion.execution_guard)

    def test_in_motion_target_change_vetoes_command(self):
        tracker, motion = self.tracker(), FakeMotion()
        coarse = CoarseApproach(tracker, motion)
        coarse.started = coarse.clock()
        coarse.command_center = np.array([.3, 0., 1.025])
        self.assertFalse(coarse._guard())
        self.assertEqual(coarse.guard_reason, "target_moved_during_transit")

    def test_generic_alternate_clearance_is_still_safety_checked(self):
        motion, events = FakeMotion(), []
        original = motion.is_observation_path_safe
        checks = []
        def safety(pose, points, finger):
            checks.append(pose.copy())
            return len(checks) > 1 and original(pose, points, finger)
        motion.is_observation_path_safe = safety
        CoarseApproach(self.tracker(), motion, trace=events.append).run()
        self.assertEqual(events[-1]["requested_clearance_m"], .05)
        self.assertTrue(any(e["event"] == "coarse_alternate_clearance" for e in events))
        self.assertGreater(len(checks), motion.moves)

    def test_invalid_coarse_safety_parameters_rejected(self):
        for kwargs in ({"clearance_m": .01}, {"step_m": .2}, {"speed_scale": 2}):
            with self.assertRaises(ValueError):
                CoarseApproach(self.tracker(), FakeMotion(), **kwargs)

    def test_coarse_fault_only_changes_physics_once(self):
        apply = Mock()
        fault = OneShotCoarseShift(.025, apply)
        self.assertIsNone(fault.on_event({"event": "coarse_move", "iteration": 0}))
        event = fault.on_event({"event": "coarse_move", "iteration": 1})
        self.assertTrue(event["simulation_only"])
        fault.on_event({"event": "coarse_move", "iteration": 1})
        apply.assert_called_once_with((.025, 0., 0.))

    def test_ik_view_proposal_requires_a_second_complete_safety_check(self):
        motion, events, checks = FakeMotion(), [], []
        def safety(pose, points, finger):
            checks.append(pose.copy())
            return len(checks) > 1
        def propose(pose):
            from scipy.spatial.transform import Rotation
            result = pose.copy()
            result[:3, :3] = Rotation.from_euler("y", 10, degrees=True).as_matrix() @ pose[:3, :3]
            return result
        motion.is_observation_path_safe = safety
        motion.propose_observation_pose = propose
        CoarseApproach(self.tracker(), motion, trace=events.append).run()
        self.assertTrue(any(e["event"] == "coarse_ik_view_adjustment" for e in events))
        self.assertGreater(len(checks), motion.moves)
        np.testing.assert_allclose(motion.pose[:3, :3], checks[1][:3, :3], atol=1e-6)

    def test_unsafe_ik_view_suggestion_cannot_bypass_path_checks(self):
        motion = FakeMotion(reachable=False)
        motion.propose_observation_pose = lambda pose: pose.copy()
        with self.assertRaisesRegex(TargetObservationError, "coarse_path_infeasible"):
            CoarseApproach(self.tracker(), motion).run()
        self.assertEqual(motion.moves, 0)


if __name__ == "__main__":
    unittest.main()
