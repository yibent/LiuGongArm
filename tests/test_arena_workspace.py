"""Editor contract tests without starting Isaac or changing a live scene."""
import sys
import unittest
import threading
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.arena.workspace import Workspace

class Tensor(np.ndarray):
    def clone(self): return self.copy().view(Tensor)
    def new_tensor(self, value): return np.asarray(value).view(Tensor)

def tensor(value): return np.asarray(value, dtype=float).view(Tensor)
class Body:
    def __init__(self, position, fixed=False):
        self.data = NS(default_root_state=tensor([[*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]), root_pos_w=tensor([position]), root_quat_w=tensor([[0, 0, 0, 1]]))
        self.cfg = NS(spawn=NS(rigid_props=NS(kinematic_enabled=fixed)))
    def write_root_pose_to_sim(self, pose):
        self.data.root_pos_w=pose[:, :3].copy(); self.data.root_quat_w=pose[:, 3:7].copy()
    def write_root_velocity_to_sim(self, velocity): self.velocity = velocity.copy()
    def reset(self): pass
class Robot:
    def __init__(self): self.data = NS(default_joint_pos=tensor([[1, 2, 3]]), default_joint_vel=tensor([[0, 0, 0]])); self.reset_count=0
    def write_joint_state_to_sim(self, pos, vel): self.pos=pos; self.vel=vel
    def reset(self): self.reset_count+=1
    def set_joint_position_target(self, pos): self.target=pos
class Scene:
    def __init__(self): self.rigid_objects={'unconfigured':Body([.5,0,.04]),'support':Body([.5,.2,.01],True)}; self.env_origins=tensor([[0,0,0]]); self.robot=Robot()
    def __getitem__(self,name): return self.robot if name=='robot' else self.rigid_objects[name]
    def update(self,dt): pass
class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        scene=Scene()
        self.runtime=NS(env=NS(scene=scene,sim=NS(forward=lambda:None),action_manager=NS(reset=lambda:None)),config={'camera':{'width':800,'height':600},'controller':{'position_tolerance_m':.006,'rotation_tolerance_deg':5,'max_steps':360}},tracking=None,tracking_stop=threading.Event(),stop_requested=threading.Event(),held=None,observed_entities={},visual_result=None,last_result=None,tcp_pose=lambda:np.eye(4),tick=lambda **kw:None,refresh_snapshot=lambda:None)
        self.workspace=Workspace(self.runtime)
    def command(self, **params): return self.workspace.execute({'command_id':'editor','params':params})
    def test_scene_registry_drives_editor_even_without_configured_assets(self):
        self.assertEqual([o['id'] for o in self.workspace.snapshot['objects']], ['unconfigured','support'])
    def test_object_edit_and_reset_restore_true_initial_pose(self):
        self.assertTrue(self.command(action='object',id='unconfigured',position=[.6,.1,.08],rotation=[0,0,45])['ok'])
        self.assertTrue(self.command(action='reset',scope='objects')['ok'])
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w,[[.5,0,.04]])
        self.assertEqual(self.runtime.env.scene.robot.reset_count,0)
    def test_preset_is_repeatable_keeps_supports_and_sets_reset_baseline(self):
        self.assertTrue(self.command(action='scene',scene_id='sorting')['ok'])
        first=self.runtime.env.scene['unconfigured'].data.root_pos_w.copy()
        self.command(action='object',id='unconfigured',position=[.2,.3,.4],rotation=[0,0,0])
        self.command(action='reset',scope='all')
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w,first)
        np.testing.assert_allclose(self.runtime.env.scene['support'].data.root_pos_w,[[.5,.2,.01]])
        self.assertEqual(self.runtime.env.scene.robot.reset_count,2)
    def test_invalid_values_do_not_partially_mutate_scene_or_controller(self):
        self.assertFalse(self.command(action='object',id='unconfigured',position=[float('nan'),0,0],rotation=[0,0,0])['ok'])
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w,[[.5,0,.04]])
        self.assertFalse(self.command(action='controller',position_tolerance_m=.002,rotation_tolerance_deg=-1)['ok'])
        self.assertEqual(self.runtime.config['controller']['position_tolerance_m'],.006)
    def test_editing_unrelated_body_preserves_held_object_context(self):
        self.runtime.held='unconfigured';self.runtime.observed_entities={'unconfigured':{'label':'new object'},'support':{}}
        self.command(action='object',id='support',position=[.6,.2,.01],rotation=[0,0,0])
        self.assertEqual(self.runtime.held,'unconfigured')
        self.assertEqual(self.runtime.observed_entities,{'unconfigured':{'label':'new object'}})
    def test_robot_only_reset_does_not_move_objects(self):
        self.command(action='object',id='unconfigured',position=[.6,.1,.08],rotation=[0,0,0])
        self.command(action='reset',scope='robot')
        np.testing.assert_allclose(self.runtime.env.scene['unconfigured'].data.root_pos_w,[[.6,.1,.08]])
        self.assertEqual(self.runtime.gripper,1.)
if __name__=='__main__':unittest.main()
