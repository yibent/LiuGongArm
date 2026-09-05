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

    def test_wall_clock_stall_does_not_skip_simulation_trajectory(self):
        self.motion.submit('move_joint', {'joint': 'shoulder_pan', 'degrees': 30}, 'slow')
        targets = []
        apply = lambda q: targets.append(q.copy())
        zero = {k: 0. for k in self.joints}
        self.motion.tick(self.joints, self.base, apply, sim_time=0, velocities=zero)
        self.now = 20.  # inference/render stall, not twenty seconds of physics
        self.motion.tick(self.joints, self.base, apply, sim_time=.01, velocities=zero)
        self.assertLess(abs(targets[-1]['shoulder_pan']), .0001)
        self.assertEqual(self.motion.result('slow')['state'], 'started')

    def test_position_alone_is_not_settled_and_repeated_timestamp_does_not_count(self):
        self.motion.submit('home', {}, 'settle')
        fast = {k: .1 for k in self.joints}
        zero = {k: 0. for k in self.joints}
        for t in np.arange(0, 1., .01):
            self.motion.tick(self.joints, self.base, lambda q: None, sim_time=t, velocities=fast)
        self.assertEqual(self.motion.result('settle')['state'], 'started')
        for _ in range(100):
            self.motion.tick(self.joints, self.base, lambda q: None, sim_time=1., velocities=zero)
        self.assertEqual(self.motion.result('settle')['state'], 'started')
        for t in np.arange(1.01, 1.5, .01):
            self.motion.tick(self.joints, self.base, lambda q: None, sim_time=t, velocities=zero)
        self.assertEqual(self.motion.result('settle')['state'], 'completed')

    def test_quintic_targets_obey_velocity_and_acceleration_limits(self):
        self.motion.submit('move_joint', {'joint': 'shoulder_pan', 'degrees': 30}, 'smooth')
        samples = []
        for t in np.arange(0, 5, .002):
            self.now = t
            self.motion.tick(self.joints.copy(), self.base,
                             lambda q: (samples.append(q['shoulder_pan']), self.joints.update(q)),
                             sim_time=t)
        velocity = np.diff(samples)/.002
        acceleration = np.diff(velocity)/.002
        self.assertLessEqual(abs(velocity).max(), self.motion.speed*1.001)
        self.assertLessEqual(abs(acceleration).max(), self.motion.acceleration*1.001)
        self.assertAlmostEqual(velocity[-1], 0.)

    def test_pause_fails_active_motion_and_holds_measured_position(self):
        self.motion.submit('move_joint', {'joint': 'shoulder_pan', 'degrees': 30}, 'pause')
        self.tick()
        self.motion.tick(self.joints, self.base, lambda q: None, playing=False)
        self.assertEqual(self.motion.result('pause')['state'], 'failed')
        self.assertEqual(self.motion.hold_target, self.joints)

    def test_tgs_frame_level_settling_keeps_raw_velocities_visible(self):
        self.motion.settle_velocity_source = 'position_delta'
        self.motion.submit('home', {}, 'bias')
        raw = {k: .1 for k in self.joints}
        for t in np.arange(0, 1.5, .01):
            self.motion.tick(self.joints, self.base, lambda q: None, sim_time=t, velocities=raw)
        self.assertEqual(self.motion.result('bias')['state'], 'completed')
        self.assertGreater(self.motion.status()['joint_velocities_deg_s']['shoulder_pan'], 5)
        self.assertEqual(self.motion.status()['joint_frame_velocities_deg_s']['shoulder_pan'], 0)

    def test_frame_level_settling_rejects_measured_oscillation_near_goal(self):
        self.motion.settle_velocity_source = 'position_delta'
        self.motion.submit('home', {}, 'oscillating')
        for t in np.arange(0, 2, .01):
            self.joints['shoulder_pan'] = np.deg2rad(.5)*np.sin(t*40)
            self.motion.tick(self.joints.copy(), self.base, lambda q: None, sim_time=t,
                             velocities={k: 0. for k in self.joints})
        self.assertEqual(self.motion.result('oscillating')['state'], 'started')

if __name__ == '__main__': unittest.main()
