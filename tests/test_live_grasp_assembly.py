"""Web uses two cameras by default and never starts its own retry."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.config import fine_grasp_config
from mr_liu.grasp.contracts import TargetSpec
from mr_liu.grasp.debug import FineGraspDebugRecorder
from mr_liu.grasp.live_assembly import assemble_live_grasp
from mr_liu.grasp.identity import RobotMaskedSegmenter, GeometricIdentitySegmenter
from mr_liu.grasp.scene_observer import SceneAssistedVerifier
from mr_liu.grasp.isaac_session import GraspProgressReporter


class LiveAssemblyTests(unittest.TestCase):
    def test_default_cameras_and_fresh_retry_node(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Mock()
            node = assemble_live_grasp(
                config=fine_grasp_config(), camera=Mock(), scene_camera=Mock(), motion=Mock(),
                gripper=Mock(), backend=backend, recorder=FineGraspDebugRecorder(directory),
                trace=Mock(), target=TargetSpec("block"),
            )
            self.assertTrue(node.require_retry_authorization)
            self.assertEqual(node.config.max_attempts, 2)
            first = node.attempt_factory(0, (), TargetSpec("block"))
            second = node.attempt_factory(1, (), TargetSpec("block"))
            self.assertIsInstance(first.verifier, SceneAssistedVerifier)
            self.assertIsInstance(first.segmenter.segmenter, RobotMaskedSegmenter)
            self.assertTrue(first.segmenter.segmenter.require_mask)
            self.assertFalse(first.settings.active_views.enabled)
            self.assertTrue(second.settings.active_views.enabled)
            self.assertTrue(second.settings.fusion.enabled)
            self.assertIsInstance(second.segmenter.segmenter.segmenter, GeometricIdentitySegmenter)
            self.assertIsNot(first.segmenter, second.segmenter)
            self.assertIsNot(first.verifier, second.verifier)
            self.assertIs(first.backend, backend)
            self.assertIs(second.backend, backend)

    def test_progress_deduplicates_servo_ticks_but_not_new_attempts(self):
        progress = Mock()
        reporter = GraspProgressReporter(progress)
        reporter({"phase": "servo", "attempt": 1})
        reporter({"phase": "servo", "attempt": 1})
        reporter({"phase": "recovery", "event": "attempt_start", "attempt": 2})
        reporter({"phase": "servo", "attempt": 2})
        self.assertEqual(progress.call_count, 3)
        self.assertEqual(progress.call_args.kwargs["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
