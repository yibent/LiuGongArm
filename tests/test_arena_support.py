import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from mr_liu.arena.evaluation import support_metrics


def test_scene_construction_needs_no_semantic_target_catalogue():
    config = json.loads((ROOT/'configs/arena_panda.json').read_text())
    assert 'objects' not in config and 'destinations' not in config
    assert all('label' not in row and 'aliases' not in row for row in config['entities'])
    assert {row['name'] for row in config['entities']} == {'part_01', 'part_02', 'surface_01'}


def test_overhanging_cube_supported_by_smaller_cylinder_is_not_rejected():
    # 45 mm cube on a 40 mm diameter cylinder: containment is impossible,
    # but real upward contact can support the cube's centre of mass.
    result = support_metrics([.5, .2, .0725], [.5, .2, .025], [0, 0, .5886], [0, 0, 0])
    assert result['supported_region']
    assert result['destination_xy_error_m'] == 0.


def test_nearby_or_side_contact_is_not_a_successful_stack():
    for force in ([0, 0, 0], [.5, 0, 0], [0, 0, -.5]):
        assert not support_metrics([.5, .2, .07], [.5, .2, .025], force, [0, 0, 0])['supported_region']


def test_support_measurements_follow_the_current_destination_pose():
    base = np.array([.46, .2, .041])
    result = support_metrics(base + [0, 0, .0475], base, [0, 0, .6], [.01, 0, 0])
    assert result['destination_xy_error_m'] == 0.
    assert result['destination_position_world_m'] == base.tolist()
    assert result['support_linear_speed_mps'] == .01
