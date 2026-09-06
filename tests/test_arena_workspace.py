"""Workbench contracts using NumPy-only doubles for Isaac Lab 3's Warp data."""
import sys
import unittest
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.arena.workspace import Workspace
from mr_liu.arena.teleop import TeleopInput

class WarpArray:
    def __init__(self, value): self.value = np.asarray(value, dtype=float)
    def numpy(self): return self.value.copy()

class Body:
    def __init__(self, position, fixed=False):
        self.data = NS(default_root_state=WarpArray([[*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]), root_pos_w=WarpArray([position]), root_quat_w=WarpArray([[0, 0, 0, 1]]))
        self.cfg = NS(spawn=NS(rigid_props=NS(kinematic_enabled=fixed)))
        self.writes = 0
    def write_root_pose_to_sim_index(self, *, root_pose):
        self.writes += 1
        self.data.root_pos_w = WarpArray(root_pose[:, :3]); self.data.root_quat_w = WarpArray(root_pose[:, 3:7])
    def write_root_velocity_to_sim_index(self, *, root_velocity): self.velocity = root_velocity.copy()
    def reset(self): pass

class Robot:
    def __init__(self):
        self.data = NS(default_joint_pos=WarpArray([[1, 2, 3]]), default_joint_vel=WarpArray([[0, 0, 0]]), joint_pos=WarpArray([[0, 0, 0]]))
        self.reset_count = 0
    def write_joint_state_to_sim_mask(self, *, position, velocity): self.pos = position; self.vel = velocity
    def reset(self): self.reset_count += 1
    def set_joint_position_target_index(self, *, target): self.target = target

class Scene:
    def __init__(self):
        self.rigid_objects = {'unconfigured': Body([.5, 0, .04]), 'support': Body([.5, .2, .01], True)}
        self.env_origins = WarpArray([[0, 0, 0]])
        self.robot = Robot()
    def __getitem__(self, name):
        if name == 'robot': return self.robot
        if name == 'ee_frame': return NS(reset=lambda: None)
        return self.rigid_objects[name]
    def update(self, dt): pass

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.array_patch = patch('mr_liu.arena.workspace.simulation_array', lambda x, device: np.asarray(x))
        self.array_patch.start(); self.addCleanup(self.array_patch.stop)
        scene = Scene()
        self.pose = np.eye(4)
        self.runtime = NS(env=NS(scene=scene, device='cpu', sim=NS(forward=lambda: None), action_manager=NS(reset=lambda: None)), config={'camera': {'width': 800, 'height': 600}, 'controller': {'position_tolerance_m': .006, 'rotation_tolerance_deg': 5, 'max_steps': 360}}, tracking=None, tracking_stop=threading.Event(), stop_requested=threading.Event(), held=None, observed_entities={}, prepared_clouds={}, visual_result=None, last_result=None, gripper=1., tcp_pose=lambda: self.pose.copy(), tick=lambda **kw: None, refresh_snapshot=lambda: None)
        self.runtime.clear_hold = lambda: setattr(self.runtime, 'held', None)
        self.moves = []
        def move(goal, **kw): self.moves.append(goal.copy()); self.pose = goal.copy()
        self.runtime.move = move
        self.workspace = Workspace(self.runtime)
    def command(self, **params): return self.workspace.execute({'command_id': 'editor', 'params': params})
    def test_registry_and_warp_snapshots_do_not_require_config_or_tensor_clone(self):
        self.assertEqual([o['id'] for o in self.workspace.snapshot['objects']], ['unconfigured', 'support'])
        self.runtime.env.scene.rigid_objects['new_body'] = Body([.4, .1, .1])
        self.workspace.refresh()
        self.assertIn('new_body', [o['id'] for o in self.workspace.snapshot['objects']])
    def test_object_edit_and_reset_restore_initial_pose_and_zero_velocity(self):
        self.assertTrue(self.command(action='object', id='unconfigured', position=[.6, .1, .08], rotation=[0, 0, 45])['ok'])
        self.assertTrue(self.command(action='reset', scope='objects')['ok'])
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w.numpy(), [[.5, 0, .04]])
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].velocity, 0)
        self.assertEqual(self.runtime.env.scene.robot.reset_count, 0)
    def test_preset_repeatable_keeps_supports_and_reset_baseline(self):
        self.assertTrue(self.command(action='scene', scene_id='sorting')['ok'])
        first = self.runtime.env.scene['unconfigured'].data.root_pos_w.numpy()
        self.command(action='object', id='unconfigured', position=[.2, .3, .4], rotation=[0, 0, 0])
        self.command(action='reset', scope='all')
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w.numpy(), first)
        np.testing.assert_allclose(self.runtime.env.scene['support'].data.root_pos_w.numpy(), [[.5, .2, .01]])
        self.assertEqual(self.runtime.env.scene.robot.reset_count, 2)
    def test_invalid_values_do_not_partially_mutate_scene_or_controller(self):
        self.assertFalse(self.command(action='object', id='unconfigured', position=[.7, 0, 0], rotation=[float('nan'), 0, 0])['ok'])
        self.assertEqual(self.runtime.env.scene['unconfigured'].writes, 0)
        self.assertFalse(self.command(action='controller', position_tolerance_m=.002, rotation_tolerance_deg=-1)['ok'])
        self.assertEqual(self.runtime.config['controller']['position_tolerance_m'], .006)
    def test_editing_unrelated_body_preserves_hold_but_invalidates_stale_clouds(self):
        self.runtime.held = 'unconfigured'
        self.runtime.observed_entities = {'unconfigured': {'label': 'new object'}, 'support': {}}
        self.runtime.prepared_clouds = {'unconfigured': 'stale'}
        self.command(action='object', id='support', position=[.6, .2, .01], rotation=[0, 0, 0])
        self.assertEqual(self.runtime.held, 'unconfigured')
        self.assertEqual(self.runtime.observed_entities, {'unconfigured': {'label': 'new object'}})
        self.assertEqual(self.runtime.prepared_clouds, {})
    def test_robot_reset_uses_lab3_writers_without_moving_objects(self):
        self.command(action='object', id='unconfigured', position=[.6, .1, .08], rotation=[0, 0, 0])
        self.assertTrue(self.command(action='reset', scope='robot')['ok'])
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w.numpy(), [[.6, .1, .08]])
        np.testing.assert_allclose(self.runtime.env.scene.robot.target, [[1, 2, 3]])
        self.assertEqual(self.runtime.gripper, 1.)
    def test_robot_jog_uses_existing_move_and_tool_frame(self):
        self.pose[:3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]
        result = self.command(action='jog', target='robot', translation=[.01,0,0], frame='tool')
        self.assertTrue(result['ok'], result)
        np.testing.assert_allclose(self.moves[-1][:3,3], [0,.01,0], atol=1e-12)
        self.assertIsNone(self.runtime.current)
        self.assertEqual(self.runtime.phase, 'idle')
    def test_object_relative_jog_uses_measured_pose_not_reset_baseline(self):
        self.command(action='object', id='unconfigured', position=[.6,.1,.08], rotation=[0,0,0])
        result = self.command(action='jog', target='object', id='unconfigured', translation=[.01,0,0])
        self.assertTrue(result['ok'], result)
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w.numpy(), [[.61,.1,.08]])
    def test_stop_prevents_any_write_and_reports_cancelled(self):
        self.runtime.stop_requested.set()
        self.assertEqual(self.command(action='reset', scope='all')['state'], 'cancelled')
        self.assertEqual(self.runtime.env.scene['unconfigured'].writes, 0)
    def test_expired_teleop_never_moves_or_leaves_mailbox_owned(self):
        now = [0.]
        teleop = TeleopInput(clock=lambda: now[0])
        self.workspace.teleop = teleop
        teleop.start('editor', {'linear': [.1,0,0]})
        now[0] = 1.
        result = self.command(action='teleop', target='robot')
        self.assertTrue(result['ok'], result)
        self.assertEqual(self.moves, [])
        self.assertIsNone(teleop.session)
    def test_open_gripper_clears_hold_but_close_does_not_invent_a_grasp(self):
        self.runtime.held = 'unconfigured'
        self.command(action='gripper', state='open')
        self.assertIsNone(self.runtime.held)
        self.command(action='gripper', state='close')
        self.assertIsNone(self.runtime.held)
        self.assertEqual(self.runtime.gripper, -1.)

class TeleopTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.mailbox = TeleopInput(clock=lambda: self.now)
    def test_lease_expiry_and_late_packets_cannot_restart_motion(self):
        self.mailbox.start('one', {'linear': [.01,0,0]})
        self.now = .7
        self.assertIsNone(self.mailbox.read('one'))
        self.assertFalse(self.mailbox.update('one', {'linear': [.01,0,0]}))
    def test_stop_is_scoped_and_latest_input_replaces_instead_of_queuing(self):
        self.mailbox.start('one', {})
        self.assertFalse(self.mailbox.update('other', {'stop': True}))
        self.assertTrue(self.mailbox.update('one', {'linear': [.01,0,0]}))
        self.assertTrue(self.mailbox.update('one', {'linear': [0,.02,0]}))
        self.assertEqual(self.mailbox.read('one')['linear'], [0,.02,0])
        self.assertTrue(self.mailbox.update('one', {'stop': True}))
        self.assertFalse(self.mailbox.update('one', {'linear': [1,0,0]}))
        self.mailbox.finish('other')
        self.assertIsNotNone(self.mailbox.session)
        self.mailbox.finish('one')
        self.assertIsNone(self.mailbox.session)
    def test_invalid_velocity_is_not_accepted(self):
        with self.assertRaises(ValueError): self.mailbox.start('one', {'linear': [float('nan'),0,0]})
        self.assertIsNone(self.mailbox.session)

if __name__ == '__main__': unittest.main()
