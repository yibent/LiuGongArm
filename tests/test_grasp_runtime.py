"""Composite grasp lifecycle without an Isaac SDK or model connection."""
from concurrent.futures import Future
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from find_and_track.types import RuntimeConfig
from mr_liu.grasp.runtime import GraspRuntime
from mr_liu.grasp.isaac_session import GuardedAdapter, observation_tool_pose, observation_tool_candidates
from mr_liu.motion.commands import MotionCommands
from mr_liu.vision.control import VisionRuntimeControl, execute_control_command
from mr_liu.vision.grounding import SceneGrounding, instance_object_path


class GraspRuntimeTests(unittest.TestCase):
    def test_observation_yaw_search_preserves_optical_center_and_clearance(self):
        tool = np.eye(4)
        camera = np.eye(4)
        camera[:3, 3] = [.15, 0, -.10]
        point = np.array([.4, -.2, 1.07])
        poses = observation_tool_candidates(tool, camera, point)
        self.assertEqual(len(poses), 12)
        for pose in poses:
            observed = pose @ camera
            np.testing.assert_allclose(observed[:3, 3], point + [0, 0, .16], atol=1e-10)
            np.testing.assert_allclose(observed[:3, 2], [0, 0, -1], atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)

    def setUp(self):
        self.now = 1.
        self.q = dict(shoulder_pan=0., shoulder_lift=-1.2, elbow_flex=1.3,
                      wrist_flex=1.2, wrist_roll=0., gripper=0.)
        self.control = VisionRuntimeControl(RuntimeConfig())
        self.control.motion = MotionCommands(ROOT / "assets/robots/so101/robot.urdf", self.q, lambda: self.now)
        self.control.grounding = Mock(cancel_epoch=0)
        self.control.grounding.valid.return_value = True
        self.session = Mock()
        self.session.execute.return_value = {"success": True, "phase": "succeeded"}
        self.factory = Mock(return_value=self.session)
        self.runtime = GraspRuntime(self.control, self.factory, clock=lambda: self.now)
        self.runtime.pool.shutdown()
        self.future = Future()
        self.runtime.pool = Mock()
        self.runtime.pool.submit.return_value = self.future
        self.control.grasp = self.runtime
        self.target = {"category": "block", "attributes": {"color": "green"}, "quantity": 1}

    def ready(self):
        self.future.set_result(dict(ok=True, object_id="/World/TabletopProps/Block",
                                    object_position_world_m=[.2, 0, 1.1], prompt="green block"))

    def submit(self):
        return execute_control_command(self.control, "grasp", {"target": self.target}, "pick")

    def test_acceptance_waits_for_fresh_grounding_and_main_thread_execution(self):
        self.assertIn("grasp", self.control.capabilities()["skills"])
        self.assertEqual(self.submit()["state"], "accepted")
        self.runtime.poll()
        self.factory.assert_not_called()
        self.assertFalse(self.control.follow_enabled())
        self.assertFalse(self.control.motion.submit("home", {}, "other")["ok"])
        self.assertEqual(self.submit()["state"], "accepted")
        self.runtime.pool.submit.assert_called_once()
        self.ready()
        owner = threading.get_ident()
        self.session.execute.side_effect = lambda _: {"success": threading.get_ident() == owner}
        self.runtime.poll()
        self.assertEqual(self.control.motion.result("pick")["state"], "completed")
        self.session.close.assert_called_once()
        self.assertIsNone(self.control.motion.external_owner)
        self.assertEqual(self.submit()["state"], "completed")
        self.factory.assert_called_once()

    def test_external_owner_does_not_write_old_hold_targets(self):
        self.control.motion.holding = True
        self.control.motion.hold_target = {k: 5. for k in self.q}
        self.submit()
        apply = Mock()
        self.assertTrue(self.control.motion.tick(self.q, np.eye(4), apply))
        apply.assert_not_called()
        self.runtime.cancel()
        self.runtime.poll()
        self.control.motion.tick(self.q, np.eye(4), apply)
        apply.assert_called_once_with(self.q)

    def test_interrupt_before_grounding_completes_never_moves(self):
        self.submit()
        stop = execute_control_command(self.control, "stop", command_id="stop")
        self.assertEqual(stop["state"], "accepted")
        self.runtime.poll()
        self.assertEqual(self.control.motion.result("pick")["state"], "cancelled")
        self.ready()
        self.runtime.poll()
        self.factory.assert_not_called()
        self.control.motion.tick(self.q, np.eye(4), Mock())
        self.assertEqual(self.control.motion.result("stop")["state"], "completed")

    def test_interrupt_during_execution_cannot_report_success(self):
        self.submit()
        self.ready()
        def execute(_):
            execute_control_command(self.control, "hold", command_id="hold")
            return {"success": True}
        self.session.execute.side_effect = execute
        self.runtime.poll()
        self.assertEqual(self.control.motion.result("pick")["state"], "cancelled")
        self.session.close.assert_called_once()

    def test_verification_failure_is_terminal_and_not_retried(self):
        self.submit()
        self.ready()
        self.session.execute.return_value = {"success": False, "failure": "lift_verification_failed"}
        self.runtime.poll()
        result = self.control.motion.result("pick")
        self.assertFalse(result["ok"])
        self.assertIn("lift_verification_failed", result["message"])
        self.runtime.poll()
        self.session.execute.assert_called_once()

    def test_stale_grounding_and_deadline_do_not_start_session(self):
        self.submit()
        self.ready()
        self.control.grounding.valid.return_value = False
        self.runtime.poll()
        self.assertEqual(self.control.motion.result("pick")["state"], "failed")
        self.factory.assert_not_called()

    def test_pending_grounding_timeout_releases_lane(self):
        self.submit()
        self.now += self.runtime.timeout_s + 1
        self.runtime.poll()
        self.assertIsNone(self.control.motion.external_owner)
        self.assertEqual(self.control.motion.result("pick")["state"], "failed")
        self.factory.assert_not_called()

    def test_instance_mapping_is_sensor_based_and_rejects_ambiguity(self):
        mask = np.ones((2, 2), dtype=bool)
        payload = {"data": np.array([[1, 1], [2, 2]]), "info": {"idToLabels": {
            "1": "/World/TabletopProps/Block/mesh1", "2": "/World/TabletopProps/Block/mesh2"}}}
        self.assertEqual(instance_object_path(payload, mask), "/World/TabletopProps/Block")
        payload["info"]["idToLabels"]["2"] = "/World/TabletopProps/Other/mesh"
        self.assertIsNone(instance_object_path(payload, mask))

    def test_cancel_epoch_rejects_a_late_grounding_worker(self):
        grounding = SceneGrounding()
        grounding.cancel()
        control = Mock()
        result = grounding.resolve(control, {"category": "block", "cancel_epoch": 0})
        self.assertFalse(result["ok"])
        control.set_prompt.assert_not_called()

    def test_optical_approach_respects_hand_eye_translation(self):
        tool, camera = np.eye(4), np.eye(4)
        camera[:3, 3] = [.1, .02, .03]
        goal = observation_tool_pose(tool, camera, [.2, .1, 1.1])
        projected_camera = goal @ camera
        np.testing.assert_allclose(projected_camera[:3, 3], [.2, .1, 1.26])
        np.testing.assert_allclose(projected_camera[:3, 2], [0, 0, -1])

    def test_guard_prevents_action_but_allows_stop(self):
        delegate = Mock()
        guard = Mock(side_effect=RuntimeError("cancelled"))
        wrapped = GuardedAdapter(delegate, guard)
        with self.assertRaises(RuntimeError):
            wrapped.move_to(np.eye(4))
        delegate.move_to.assert_not_called()
        wrapped.stop()
        delegate.stop.assert_called_once()

    def test_failed_admission_does_not_leave_arm_reserved(self):
        self.target["attributes"] = "invalid"
        self.assertFalse(self.submit()["ok"])
        self.assertIsNone(self.control.motion.external_owner)
        self.target["attributes"] = {}
        self.runtime.pool.submit.side_effect = RuntimeError("worker unavailable")
        self.assertFalse(self.submit()["ok"])
        self.assertIsNone(self.control.motion.external_owner)
        self.assertIsNone(self.runtime.job)

    def awaiting_retry(self):
        self.submit()
        self.ready()
        self.session.execute.return_value = {"success": False, "retry_available": True,
            "failure": "target_not_visible", "metrics": {"recovery_stop_reason": "awaiting_retry_authorization"}}
        self.runtime.poll()

    def test_failure_waits_without_an_automatic_retry_and_accepts_one_spoken_retry(self):
        self.awaiting_retry()
        self.assertTrue(self.runtime.status()["retry_available"])
        self.runtime.poll()
        self.session.retry.assert_not_called()
        self.session.hold_for_retry.assert_called_once()
        self.session.close.assert_not_called()
        self.assertIsNone(self.control.motion.external_owner)
        result = self.runtime.submit({"retry_last": True}, "retry")
        self.assertEqual(result["state"], "accepted")
        self.session.retry.return_value = {"success": True, "retry_available": False}
        self.runtime.poll()
        self.session.retry.assert_called_once_with("retry")
        self.assertEqual(self.control.motion.result("retry")["state"], "completed")
        self.session.close.assert_called_once()
        self.assertFalse(self.runtime.submit({"retry_last": True}, "third")["ok"])
        self.runtime.pool.submit.assert_called_once()  # no new blind target transaction

    def test_stop_expires_recovery_without_moving_and_cleans_on_main_thread(self):
        self.awaiting_retry()
        self.runtime.cancel()
        self.assertFalse(self.runtime.status()["retry_available"])
        self.runtime.poll()
        self.session.close.assert_called_once_with(stop_motion=False)
        self.session.retry.assert_not_called()

    def test_other_motion_or_timeout_invalidates_previous_grasp_context(self):
        self.awaiting_retry()
        self.control.motion.submit("home", {}, "move-after-failure")
        self.assertFalse(self.runtime.status()["retry_available"])
        self.runtime.poll()
        self.session.close.assert_called_once_with(stop_motion=False)
        self.assertFalse(self.runtime.submit({"retry_last": True}, "retry")["ok"])

    def test_no_modes_and_dual_camera_default(self):
        self.assertTrue(self.runtime.capabilities()["dual_camera_default"])
        self.assertFalse(self.runtime.submit({"target": self.target, "recovery_mode": "active"})["ok"])
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
