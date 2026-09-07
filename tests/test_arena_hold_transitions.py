"""Regression for a failed lift leaving an unresolvable held-object marker."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from mr_liu.arena.contracts import ManipulationRequest
from mr_liu.arena.cascade import FastPathFailure
from mr_liu.grasp.transforms import invert_transform, transform_points


def test_grasp_records_closure_reference_before_lift_can_fail():
    path=Path(__file__).resolve().parents[1]/'source/mr_liu/arena/fast.py'
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fast_pick_place')
    class Controller:
        phase=0
        def __init__(self,*args,**kwargs):pass
        def is_done(self):return self.phase>=10
        def get_current_event(self):return self.phase
        def forward(self,*args,**kwargs):self.phase+=1
    namespace={'np':np,'PickPlaceController':Controller,'ArenaCartesian':lambda r:r,'ArenaGripper':lambda r:r,
        'PHASES':[str(i) for i in range(10)],'FastPathFailure':FastPathFailure,
        'invert_transform':invert_transform,'transform_points':transform_points}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    remember=Mock()
    r=SimpleNamespace(prepare_task=lambda req:({'name':'observed-body'},None),cloud=lambda name:np.array([[0,0,0],[.02,.02,.06]]),
        visual_result={'request_id':'before'},config={'fast':{'phase_steps':[1]*10}},event=Mock(),tick=Mock(),
        tcp_pose=lambda:np.eye(4),max_lift=0.,remember_hold=remember,bind_orientation=lambda *args:None)
    with pytest.raises(FastPathFailure,match='did not lift'):
        namespace['fast_pick_place'](r,ManipulationRequest('part',mode='basic'))
    remember.assert_called_once()
    assert remember.call_args.args[1]['observation_ref']=='before'


def test_release_waits_for_actual_ik_and_keeps_failures_visible():
    from scipy.spatial.transform import Rotation
    path=Path(__file__).resolve().parents[1]/'source/mr_liu/arena/fast.py'
    function=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_release_pose')
    namespace={'np':np,'Rotation':Rotation,'FastPathFailure':FastPathFailure}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    r=SimpleNamespace(holding_status=lambda:{'verified':True},move=Mock(side_effect=RuntimeError('Arena IK did not reach')))
    with pytest.raises(RuntimeError,match='Arena IK'):
        namespace['verify_release_pose'](r,[.4,.2,.06],[0,1,0,0])
    pose=r.move.call_args.args[0]
    np.testing.assert_allclose(pose[:3,3],[.4,.2,.06])
    np.testing.assert_allclose(pose[:3,:3],np.diag([1,-1,-1]))
