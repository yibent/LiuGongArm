import numpy as np
from mr_liu.arena.observed_scene import ObservedScene
from mr_liu.arena.placement_geometry import inspect_grid
from test_arena_placement_geometry import grid_cloud


def observation(frame, camera='scene_camera', label='part', geometry=None):
    ref = f'obs:{frame * 32}:{camera}:0'
    return {'request_id': frame * 32, 'label': label, 'observed_at': ord(frame),
        'references': [{'ref': ref, 'kind': 'object', 'camera': camera, 'label': label}],
        **({'geometry': geometry} if geometry else {})}


def cloud():
    rng = np.random.default_rng(42)
    return rng.uniform([.48, -.02, .0], [.52, .02, .06], (100, 3))


def test_same_object_survives_new_view_and_grasp_release_without_config_names():
    state = ObservedScene()
    points = cloud()
    first = observation('a')
    identity = state.observe_target(first, points)
    next_view = observation('b', 'side_camera')
    assert state.observe_target(next_view, points + [.001, 0, 0]) == identity
    assert set(state.view_references(first['references'][0]['ref'])) == {'scene_camera', 'side_camera'}
    state.action_result(identity, 'grasp', {'command_id': 'pick', 'holding': {'verified': True}})
    moved = observation('c', 'wrist_camera')
    assert state.observe_target(moved, points + [.1, .2, .3], first['references'][0]['ref']) == identity
    state.action_result(identity, 'place_held', {'command_id': 'place', 'ok': True,
        'holding': {'verified': False}, 'review_required': True}, {'ref': 'observed-container'})
    row = state.snapshot()['objects'][0]
    assert row['track_id'] == identity and row['state'] == 'released_unverified'
    assert row['placement']['ref'] == 'observed-container'
    assert len(state.snapshot()['objects']) == 1


def test_held_object_cannot_be_associated_to_another_part_at_its_old_location():
    state = ObservedScene()
    first = observation('a'); identity = state.observe_target(first, cloud())
    state.action_result(identity, 'grasp', {'holding': {'verified': True}})
    another = state.observe_target(observation('b'), cloud())
    assert another != identity


def test_reboxing_same_container_retains_grid_identity_only_with_current_depth_confirmation():
    points, _ = grid_cloud()
    state = ObservedScene()
    first = observation('a', label='tray', geometry={'kind': 'grid', 'camera': 'scene_camera', **inspect_grid(points)})
    identity = state.observe_target(first, points, scene=points)
    cells = first['geometry']['cells']
    new = observation('b', 'side_camera', 'tray', {'kind': 'grid', 'status': 'unknown', 'camera': 'side_camera'})
    assert state.observe_target(new, points + [.0001, 0, 0], scene=points) == identity
    assert new['geometry']['status'] == 'observed'
    assert [c['cell_id'] for c in new['geometry']['cells']] == [c['cell_id'] for c in cells]
    assert all(':side_camera:' in c['ref'] for c in new['geometry']['cells'])
    assert new['geometry']['basis_xy'] == first['geometry']['basis_xy']
    moved = observation('c', label='tray', geometry={'kind': 'grid', 'camera': 'scene_camera', **inspect_grid(points + [.08, 0, 0])})
    state.observe_target(moved, points + [.08, 0, 0], first['references'][0]['ref'], scene=points + [.08, 0, 0])
    assert moved['geometry']['grid_frame_id'] != first['geometry']['grid_frame_id']


def test_regrasp_requirement_never_replays_the_same_oriented_placement_as_model_fallback():
    from mr_liu.arena.contracts import ManipulationRequest
    from mr_liu.arena.cascade import run_cascade
    from mr_liu.arena.failure import RegraspRequired, failure_feedback
    from unittest.mock import Mock
    import pytest
    fast = Mock(side_effect=RegraspRequired('当前抓法无法满足目标'))
    enhanced, recover = Mock(), Mock()
    with pytest.raises(RegraspRequired):
        run_cascade(ManipulationRequest('part', 'tray'), fast, enhanced, recover, Mock())
    enhanced.assert_not_called(); recover.assert_not_called()
    assert failure_feedback(RegraspRequired('换抓'), 'transport', {'verified': True})['code'] == 'REGRASP_REQUIRED'
