"""Motion lifecycle tests with a measured-joint fake drive, independent of Kit."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from mr_liu.motion.commands import MotionCommands


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.joints = dict(shoulder_pan=0., shoulder_lift=-1.2, elbow_flex=1.3, wrist_flex=1.2, wrist_roll=0., gripper=0.)
        self.original = self.joints.copy()
        self.motion = MotionCommands(ROOT/'assets/robots/so101/robot.urdf', self.joints.copy(), lambda:self.now)
        self.base = np.eye(4)

    def tick(self, frozen=False):
        self.now += .1
        self.motion.tick(self.joints.copy(), self.base, lambda q: None if frozen else self.joints.update(q))

    def complete(self, cid):
        for _ in range(300):
            self.tick()
            result = self.motion.result(cid)
            if result['state'] in {'completed','failed','cancelled'}:
                return result
        self.fail('motion never settled')

    def test_relative_absolute_home_and_idempotency(self):
        record = self.motion.submit('move_joint', {'joint':'wrist_roll','degrees':30}, 'turn')
        self.assertEqual(record['state'], 'accepted')
        self.assertEqual(self.joints, self.original)
        self.assertEqual(self.complete('turn')['state'], 'completed')
        self.assertAlmostEqual(self.joints['wrist_roll'], np.pi/6)
        self.motion.submit('move_joint', {'joint':'wrist_roll','degrees':30}, 'turn')
        self.tick()
        self.assertAlmostEqual(self.joints['wrist_roll'], np.pi/6)
        self.motion.submit('move_joint', {'joint':'wrist_roll','degrees':-10,'absolute':True}, 'absolute')
        self.assertEqual(self.complete('absolute')['state'], 'completed')
        self.assertAlmostEqual(self.joints['wrist_roll'], np.deg2rad(-10))
        self.motion.submit('home', {}, 'home')
        self.assertEqual(self.complete('home')['state'], 'completed')
        np.testing.assert_allclose(list(self.joints.values()), list(self.original.values()))

    def test_hold_bypasses_busy_lane_and_resume(self):
        self.motion.submit('move_joint', {'joint':'shoulder_pan','degrees':90}, 'move')
        for _ in range(5): self.tick()
        self.assertFalse(self.motion.submit('home', {}, 'busy')['ok'])
        self.motion.submit('hold', {}, 'pause')
        self.tick()
        held = self.joints.copy()
        self.assertEqual(self.motion.result('move')['state'], 'cancelled')
        self.assertEqual(self.motion.result('pause')['state'], 'completed')
        for _ in range(5): self.tick()
        self.assertEqual(self.joints, held)
        self.motion.submit('resume', {}, 'resume')
        self.assertEqual(self.complete('resume')['state'], 'completed')
        self.assertAlmostEqual(self.joints['shoulder_pan'], np.pi/2)

    def test_stop_discards_resume_and_pending_command(self):
        self.motion.submit('home', {}, 'queued')
        self.motion.submit('stop', {}, 'stop')
        self.tick()
        self.assertEqual(self.motion.result('queued')['state'], 'cancelled')
        self.motion.submit('resume', {}, 'resume')
        self.assertEqual(self.complete('resume')['state'], 'failed')

    def test_rejects_limits_and_unreachable_without_moving(self):
        for cid, skill, params in [('limit','move_joint',{'joint':'shoulder_pan','degrees':500}),
                                   ('ik','move_cartesian',{'xyz_m':[10,0,0]}),
                                   ('nan','gripper',{'opening':float('nan')})]:
            self.motion.submit(skill, params, cid)
            self.assertEqual(self.complete(cid)['state'], 'failed')
            self.assertEqual(self.joints, self.original)

    def test_no_completion_without_measured_arrival(self):
        self.motion.submit('move_joint', {'joint':'wrist_roll','degrees':30}, 'stalled')
        for _ in range(150): self.tick(frozen=True)
        self.assertEqual(self.motion.result('stalled')['state'], 'failed')
        self.assertIn('超时', self.motion.result('stalled')['message'])

    def test_cartesian_translation_and_tool_rotation(self):
        before = self.motion.kinematics.forward(self.joints)
        self.motion.submit('move_cartesian', {'xyz_m':[0,0,.005]}, 'up')
        self.assertEqual(self.complete('up')['state'], 'completed')
        after = self.motion.kinematics.forward(self.joints)
        self.assertLess(np.linalg.norm(after[:3,3]-before[:3,3]-[0,0,.005]), .004)
        self.motion.submit('rotate', {'axis':'z','degrees':10,'frame':'tool'}, 'rotate')
        self.assertEqual(self.complete('rotate')['state'], 'completed')
        final = self.motion.kinematics.forward(self.joints)
        expected = after[:3,:3] @ Rotation.from_euler('z',10,degrees=True).as_matrix()
        self.assertLess(Rotation.from_matrix(expected @ final[:3,:3].T).magnitude(), np.deg2rad(3))

    def test_gripper_and_speed(self):
        self.motion.submit('set_speed', {'degrees_per_second':20}, 'speed')
        self.assertEqual(self.complete('speed')['state'], 'completed')
        for opening in (1,0.5,0):
            cid=str(opening)
            self.motion.submit('gripper', {'opening':opening}, cid)
            self.assertEqual(self.complete(cid)['state'], 'completed')
            self.assertAlmostEqual(self.joints['gripper'],1.4*opening)

if __name__ == '__main__': unittest.main()
