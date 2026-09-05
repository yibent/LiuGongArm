from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.grasp.isaac_session import GuardedAdapter, GraspProgressReporter, IsaacGraspSession
from mr_liu.grasp.live_coarse import ResponsiveLocator, approach_for_live_grasp
from mr_liu.perception.semantic_target import TargetObservationError
from mr_liu.vision.inventory import visible_inventory
from mr_liu.vision.control import VisionRuntimeControl, execute_control_command
from find_and_track import RuntimeConfig, ObjectMemoryStore, Detection


class LiveIntegrationTests(unittest.TestCase):
    def test_motion_failure_keeps_its_reason_and_confirmation(self):
        session = object.__new__(IsaacGraspSession)
        session.executor = Mock()
        session.executor.initial_pose_proposal.return_value = {'path_checked': True}
        with tempfile.TemporaryDirectory() as root:
            session.recorder = Mock(output_dir=Path(root))
            result = session.observation_blocked(
                [{'reason': 'coarse_motion_not_converged', 'motion': {'position_error_mm': 4.2}}],
                '目标已定位，但接近动作未通过到位检查，已停止接近',
                failure='coarse_motion_not_converged')
        self.assertEqual(result['failure'], 'coarse_motion_not_converged')
        self.assertIn('尚未闭爪', result['message'])
        self.assertNotIn('视觉不可用', result['message'])
        self.assertEqual(result['diagnostics'][0]['motion']['position_error_mm'], 4.2)
        self.assertTrue(result['preparation']['requires_confirmation'])

    def test_coarse_callback_reaches_real_executor(self):
        delegate = Mock(execution_guard=None)
        wrapped = GuardedAdapter(delegate, Mock())
        guard = lambda: False
        wrapped.execution_guard = guard
        self.assertIs(delegate.execution_guard, guard)
        wrapped.execution_guard = None
        self.assertIsNone(delegate.execution_guard)

    def test_locator_keeps_kit_responsive_and_filters_selector(self):
        locator = Mock()
        def locate(*args):
            time.sleep(.02)
            return {'boxes': [{'xyxy': [1, 0, 3, 4]}, {'xyxy': [20, 0, 23, 4]}]}
        locator.locate.side_effect = locate
        advance = Mock(side_effect=lambda: time.sleep(.001))
        result = ResponsiveLocator(locator, advance, 'rightmost').locate(Mock(), 'red block')
        self.assertGreater(advance.call_count, 0)
        self.assertEqual(result['boxes'][0]['xyxy'][0], 20)

    def test_locator_cancellation_does_not_wait_for_http(self):
        locator = Mock()
        locator.locate.side_effect = lambda *args: (time.sleep(.1), {})[1]
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            ResponsiveLocator(locator, Mock(side_effect=RuntimeError('cancelled'))).locate(Mock(), 'part')
        self.assertLess(time.monotonic() - started, .08)

    def test_handoff_must_pass_before_fine_grasp(self):
        tracker = Mock()
        tracker.clock.return_value = 10.
        tracker.handoff_after_s = 9.
        wrist = Mock()
        wrist.capture.return_value.timestamp_s = 9.9
        tracker.validate_wrist_handoff.side_effect = TargetObservationError('wrist_handoff_identity_mismatch')
        with patch('mr_liu.grasp.live_coarse.SemanticFlowTarget', return_value=tracker), \
             patch('mr_liu.grasp.live_coarse.CoarseApproach'), \
             patch('mr_liu.grasp.live_coarse.SeededDepthSegmenter'):
            with self.assertRaises(TargetObservationError):
                approach_for_live_grasp(scene_camera=Mock(), wrist_camera=wrist, motion=Mock(),
                    gripper=Mock(), locator=Mock(), target=Mock(), table_height_m=1., trace=Mock(), recorder=Mock())

    def test_coarse_progress_is_factual_and_deduplicated(self):
        progress = Mock()
        report = GraspProgressReporter(progress)
        report({'phase': 'coarse', 'event': 'coarse_start'})
        report({'phase': 'coarse', 'event': 'coarse_start'})
        progress.assert_called_once()
        self.assertIn('接近', progress.call_args.args[1])

    def test_inventory_uses_visible_pixels_not_spawn_list(self):
        image = np.zeros((30, 30, 3), np.uint8)
        image[4:12, 4:12] = [0, 0, 255]
        ids = np.zeros((30, 30), np.uint32)
        ids[4:12, 4:12] = 1
        report = visible_inventory(image, {'data': ids, 'info': {'idToLabels': {'1': {'class': 'block'}, '2': {'class': 'nut'}}}}, 3.)
        self.assertEqual(len(report['objects']), 1)
        self.assertEqual(report['objects'][0]['color'], 'red')

    def test_named_memory_retains_physical_category(self):
        with tempfile.TemporaryDirectory() as root:
            store = ObjectMemoryStore(root)
            memory = store.remember('零件A', {'scene': np.zeros((30, 30, 3), np.uint8)},
                {'scene': Detection(np.array([2, 2, 20, 20]), 'red block')}, metadata={'prompt': 'red block'})
            control = VisionRuntimeControl(RuntimeConfig())
            control.memory_store = store
            self.assertEqual(control.recall(memory.memory_id).prompt, 'red block')
            result = execute_control_command(control, 'forget', {'memory_id': memory.memory_id})
            self.assertTrue(result['ok'])
            self.assertIsNone(control.snapshot().memory_id)

    def test_memory_cannot_capture_stale_or_ambiguous_frames(self):
        with tempfile.TemporaryDirectory() as root:
            control = VisionRuntimeControl(RuntimeConfig())
            control.memory_store = ObjectMemoryStore(root)
            control._raw_frames['scene'] = np.zeros((30, 30, 3), np.uint8)
            control._detections['scene'] = [{'xyxy': [2, 2, 20, 20]}]
            with self.assertRaises(ValueError):
                control.remember('part', fresh=True)
            control._capture_meta['scene'] = (0, time.monotonic())
            control._detections['scene'] *= 2
            with self.assertRaises(ValueError):
                control.remember('part', fresh=True)


if __name__ == '__main__':
    unittest.main()
