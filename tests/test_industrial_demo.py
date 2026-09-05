"""Industrial demo admission, completion, interruption and waypoint reachability."""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from mr_liu.config import robot_config, scene_config
from mr_liu.motion.commands import MotionCommands
from mr_liu.sim.industrial_demo import IndustrialDemo, layout, solve_waypoint


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        cfg = robot_config()
        self.motion = MotionCommands(ROOT/cfg['robot_dir']/cfg['urdf'], cfg['default_joint_positions'])
        self.motion.snapshot = {'simulation_playing': True}
        self.events = []
        drive = SimpleNamespace(begin=lambda item, phase: self.events.append((item['index'], phase)),
                                step=lambda t: None)
        control = SimpleNamespace(motion=self.motion, set_follow_enabled=lambda _: None)
        self.demo = IndustrialDemo(control, layout(2), drive, lambda: self.now)

    def tick(self):
        self.demo.poll()
        self.now += 1.

    def test_complete_only_after_all_releases_and_retreats(self):
        request = {'target': {'category': '金属柱'}, 'demo_stack': True}
        self.demo.submit(request, 'run')
        self.demo.submit(request, 'run')
        self.assertFalse(self.demo.submit(request, 'busy')['ok'])
        for _ in range(16):
            self.tick()
        self.assertEqual(self.motion.result('run')['state'], 'started')
        self.tick()
        self.assertEqual(self.motion.result('run')['state'], 'completed')
        self.assertEqual(self.demo.placed, 2)
        self.assertEqual(len(self.events), 16)
        self.assertIsNone(self.motion.external_owner)
        self.assertFalse(self.demo.submit(request, 'empty')['ok'])

    def test_stop_releases_lane_without_claiming_placement(self):
        self.demo.submit({'target': {'category': 'metal cylinder'}}, 'run')
        self.tick()
        self.demo.cancel()
        self.tick()
        self.assertEqual(self.motion.result('run')['state'], 'cancelled')
        self.assertEqual(self.demo.placed, 0)
        self.assertIsNone(self.motion.external_owner)
        self.assertTrue(self.motion.holding)

    def test_wrong_target_and_pause(self):
        self.assertFalse(self.demo.submit({'target': {'category': 'wrench'}})['ok'])
        self.demo.submit({'target': {'category': '圆柱'}}, 'run')
        self.motion.snapshot['simulation_playing'] = False
        self.demo.poll()
        self.assertEqual(self.events, [])

    def test_all_waypoints_reachable_and_basket_clearance(self):
        scene = scene_config()
        base = np.eye(4)
        base[:3, 3] = scene['robot_pos']
        w, x, y, z = scene['robot_orient_wxyz']
        base[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        q = robot_config()['default_joint_positions'].copy()
        for item in layout(96):
            for key in ['source', 'destination']:
                for dz in [0., .12]:
                    p = np.array(item[key])+[0, 0, dz]
                    q = solve_waypoint(self.motion.kinematics, p, q, base, self.motion.defaults)
                    np.testing.assert_allclose(self.motion.kinematics.forward(q, base)[:3, 3], p, atol=.004)
            x, y, z = item['destination']
            self.assertGreater(x-.02, .304)
            self.assertLess(x+.02, .504)
            self.assertGreater(y-.008, .035)
            self.assertLess(y+.008, .175)
            self.assertLess(z+.008, 1.14)


if __name__ == '__main__':
    unittest.main()
