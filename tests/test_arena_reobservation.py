"""Exercise the runtime method without importing Isaac Sim on a CPU host."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mr_liu.arena.instances import InstanceConflict
from mr_liu.grasp.transforms import transform_points


def runtime_method(name='cloud'):
    path = Path(__file__).resolve().parents[1] / 'source/mr_liu/arena/runtime.py'
    runtime = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ArenaRuntime')
    method = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'InstanceConflict': InstanceConflict, 'transform_points': transform_points}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def runtime():
    observation = {'ok': True, 'request_id': 'fresh', 'views': [], 'perception_source': 'rgbd'}
    perception = Mock()
    perception.capture.return_value = {'request_id': 'fresh'}
    perception.request.return_value = observation
    perception.witness.return_value = ('body_87', {'body_87': 12})
    perception.cloud.return_value = (np.ones((5, 3)), {})
    return SimpleNamespace(prepared_clouds={}, observed_entities={'body_87': {'label': 'unconfigured cyan ring'}},
        observing=False, vision_worker=SimpleNamespace(available=True), config={'camera': {'render_interval': 1}},
        tick=Mock(), perception=perception, event=Mock(), infer=lambda f, *args: f(*args),
        held='body_87', body_entities={'body_87': {}}, sequence=3)


def test_later_place_uses_visual_label_not_internal_instance_id():
    r = runtime()
    points = runtime_method()(r, 'body_87')
    assert points.shape == (5, 3)
    assert r.perception.capture.call_args.args[1] == 'unconfigured cyan ring'
    assert r.visual_result['physical_witness']['instance_id'] == 'body_87'
    assert not r.observing


def test_held_reobservation_can_change_cameras_without_duplicate_keyword():
    r = runtime()
    r.perception.request.side_effect = [{'ok': False}, {'ok': True, 'request_id': 'fresh', 'views': [], 'perception_source': 'rgbd'}]
    runtime_method()(r, 'body_87', cameras=['wrist_camera'])
    assert r.perception.capture.call_args.kwargs['cameras'] == ['scene_camera', 'side_camera']
    assert r.perception.capture.call_args.args[1] == 'unconfigured cyan ring'


def test_label_mapping_does_not_accept_another_physical_object():
    r = runtime()
    r.perception.witness.return_value = ('different_body', {'different_body': 12})
    with pytest.raises(InstanceConflict):
        runtime_method()(r, 'body_87')


def test_multiview_instance_conflict_retries_sam3_on_primary_view_only():
    r = runtime(); r.held = None
    observed = {'ok': True, 'request_id': 'fresh', 'views': [], 'perception_source': 'rgbd'}
    r.perception.request.side_effect = [observed, observed]
    r.perception.witness.side_effect = [
        InstanceConflict('side camera selected the robot base'),
        ('body_87', {'body_87': 12}),
    ]
    points = runtime_method()(r, 'body_87', associate=True, manipulation_target=True)
    assert points.shape == (5, 3)
    assert r.perception.capture.call_count == 2
    assert r.perception.capture.call_args.kwargs['vision_mode'] == 'slow'
    assert r.perception.capture.call_args.kwargs['slow_provider'] == 'sam3'
    assert r.perception.capture.call_args.kwargs['cameras'] == ['scene_camera']


def test_held_geometry_moves_with_gripper_without_resegmenting_occluded_object():
    points = np.array([[.01, .02, -.04], [-.01, -.02, -.02]])
    pose = np.eye(4); pose[:3, 3] = [.4, -.2, .3]
    r = SimpleNamespace(holding_status=lambda: {'verified': True},
        held_geometry={'points_tcp': points, 'observation_ref': 'before-jaw-occlusion'},
        event=Mock(), tcp_pose=lambda: pose, cloud=Mock(side_effect=AssertionError('should not invoke vision')))
    np.testing.assert_allclose(runtime_method('held_cloud')(r), points + pose[:3, 3])
    assert not r.cloud.called
    r.holding_status = lambda: {'verified': False}
    with pytest.raises(RuntimeError, match='夹持状态已改变'):
        runtime_method('held_cloud')(r)


def test_normal_post_action_captures_new_frame_without_another_model_call():
    r=runtime();r.perception.capture.return_value={'request_id':'after','observed_at':4.,'views':[{'camera':'scene_camera'}]}
    result={'ok':True}
    runtime_method('post_action_observation')(r,result,SimpleNamespace(cell_ref=None))
    assert result['post_action_snapshot']['observation_ref']=='after'
    r.perception.request.assert_not_called()


def test_uncertain_grid_check_preserves_completed_motion_and_requests_review():
    r=runtime();r.perception.capture.return_value={'request_id':'after','observed_at':4.,'views':[{'camera':'scene_camera'}]}
    r.perception.resolve_reference.return_value={'container_ref':'bin-ref','row':2,'column':3}
    r.perception.request.return_value={'ok':True,'geometry':{'cells':[{'row':2,'column':3,'occupancy':'unknown'}]}}
    result={'ok':True,'evaluation':{'physical_success':True}}
    runtime_method('post_action_observation')(r,result,SimpleNamespace(cell_ref='cell-ref',destination='bin'))
    assert result['ok'] and result['review_required']
    assert not result['postconditions']['cell_visually_occupied']['satisfied']
