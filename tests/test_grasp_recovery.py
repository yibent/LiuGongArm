"""Recovery must earn authority to move; a failed result is not an empty hand."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.contracts import (FailureCode, FineGraspPhase, FineGraspRequest,
    FineGraspResult, GripperState, TargetSpec)
from mr_liu.grasp.recovery import RecoveringGraspNode, RecoveryConfig


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        def clock():
            self.now += .001
            return self.now
        self.clock = clock
        self.pose = np.eye(4)
        self.pose[:3, 3] = [0., 0., 1.15]
        self.motion = Mock()
        self.motion.robot_state.side_effect = lambda: SimpleNamespace(T_base_ee=self.pose.copy())
        self.motion.is_observation_path_safe.return_value = True
        self.motion.move_to.return_value = True
        self.gripper = Mock()
        self.gripper.state.return_value = GripperState(10., width_m=.075, contact=None, stalled=False)
        self.evidence = SimpleNamespace(center_base_m=np.array([0., 0., 1.10]),
                                        points_base=np.array([[0., 0., 1.10]]))
        self.observer = Mock()
        self.observer.latest = self.evidence
        self.observer.observe.return_value = self.evidence
        self.request = FineGraspRequest(TargetSpec("object", coarse_position_base_m=(0., 0., 1.1)))
        self.factory = Mock()
        self.failed = FineGraspResult(False, FineGraspPhase.FAILED, None,
                                      FailureCode.TARGET_NOT_VISIBLE, metrics={"close_commanded": 0})
        self.succeeded = FineGraspResult(True, FineGraspPhase.SUCCEEDED, None,
                                         metrics={"close_commanded": 1})

    def node(self, results):
        self.factory.side_effect = [SimpleNamespace(execute=Mock(return_value=r), cancel=Mock()) for r in results]
        return RecoveringGraspNode(attempt_factory=self.factory, observer=self.observer,
            motion=self.motion, gripper=self.gripper, clock=self.clock)

    def test_empty_precontact_failure_reacquires_retracts_and_enables_second_attempt(self):
        result = self.node([self.failed, self.succeeded]).execute(self.request)
        self.assertTrue(result.success)
        self.assertEqual(result.metrics["recovery_attempts"], 2)
        self.assertEqual(result.metrics["grasp_attempts"], 1)
        self.assertEqual(self.factory.call_args_list[1].args[0], 1)
        self.motion.is_observation_path_safe.assert_called_once()
        self.motion.move_to.assert_called_once()
        self.gripper.open.assert_not_called()

    def test_possibly_held_payload_blocks_open_move_and_retry(self):
        from dataclasses import replace
        failure = replace(self.failed, failure=FailureCode.LIFT_VERIFICATION_FAILED,
                          metrics={"close_commanded": 1})
        for width in (None, .016):
            with self.subTest(width=width):
                self.gripper.state.return_value = GripperState(10., width_m=width, stalled=True)
                result = self.node([failure]).execute(self.request)
                self.assertFalse(result.success)
                self.assertEqual(result.metrics["recovery_stop_reason"], "payload_uncertain_do_not_open")
        self.motion.move_to.assert_not_called()
        self.gripper.open.assert_not_called()

    def test_no_external_identity_means_no_recovery_motion(self):
        self.observer.observe.return_value = None
        result = self.node([self.failed]).execute(self.request)
        self.assertFalse(result.success)
        self.assertEqual(result.metrics["recovery_stop_reason"], "target_identity_uncertain")
        self.motion.move_to.assert_not_called()

    def test_rejected_actual_joint_path_is_not_executed(self):
        self.motion.is_observation_path_safe.return_value = False
        result = self.node([self.failed]).execute(self.request)
        self.assertEqual(result.metrics["recovery_stop_reason"], "retreat_path_unverified")
        self.motion.move_to.assert_not_called()

    def test_dry_run_does_not_start_observation_motion_or_retry(self):
        from dataclasses import replace
        result = self.node([self.failed]).execute(replace(self.request, dry_run=True))
        self.assertEqual(result.metrics["recovery_attempts"], 1)
        self.observer.observe.assert_not_called()
        self.motion.move_to.assert_not_called()

    def test_calibration_failure_is_not_retried(self):
        from dataclasses import replace
        result = self.node([replace(self.failed, failure=FailureCode.CALIBRATION_INVALID)]).execute(self.request)
        self.assertEqual(result.metrics["recovery_stop_reason"], "non_retryable_failure")
        self.motion.move_to.assert_not_called()

    def test_retry_count_is_bounded(self):
        result = self.node([self.failed, self.failed]).execute(self.request)
        self.assertEqual(result.metrics["recovery_attempts"], 2)
        self.assertFalse(result.success)

    def test_cancel_during_slow_path_check_never_starts_motion(self):
        node = self.node([self.failed])
        def check(*args):
            node.cancel()
            return True
        self.motion.is_observation_path_safe.side_effect = check
        result = node.execute(self.request)
        self.assertEqual(result.failure, FailureCode.ABORTED)
        self.motion.move_to.assert_not_called()

    def test_configuration_rejects_unbounded_retry_or_fast_recovery(self):
        for kwargs in ({"max_attempts": 4}, {"speed_scale": .9}, {"retreat_m": .5}):
            with self.assertRaises(ValueError):
                RecoveryConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
