import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.arena.observation_request import capture_observation, observation_result
from mr_liu.arena.service import CommandQueue


class ReadOnlyObservationTests(unittest.TestCase):
    def test_observe_while_arm_busy_preserves_motion_and_returns_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'frame').mkdir()
            packet = {'request_id': 'frame', 'command_id': 'motion', 'task_id': 'motion-task', 'scope': 'collection'}
            measured = {'ok': True, 'request_id': 'frame', 'scope': 'collection', 'collection': {'instances': [{'ref': 'object'}]}}
            bridge = SimpleNamespace(root=root, capture=Mock(return_value=packet), request=Mock(return_value=measured))
            runtime = SimpleNamespace(current='motion', phase='transport', held='part',
                                      target_value=lambda _: ('block', None), perception=bridge)
            queue = CommandQueue()
            queue.submit({'command_id': 'motion', 'skill': 'pick_place'})
            queue.claim('motion')
            packet = capture_observation(runtime, {'command_id': 'query', 'params': {'category': 'block'}, 'correlation_id': 'user'})
            result = observation_result(runtime, packet)
            self.assertEqual(queue.result('motion')['state'], 'running')
            self.assertEqual((runtime.current, runtime.phase, runtime.held), ('motion', 'transport', 'part'))
            self.assertEqual(result['collection'], measured['collection'])
            self.assertNotIn('task_id', packet)
            self.assertEqual(json.loads((root/'frame/request.json').read_text())['command_id'], 'query')

    def test_tracking_not_silently_treated_as_a_readonly_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'tracking'):
            capture_observation(None, {'params': {'tracking': True}})
