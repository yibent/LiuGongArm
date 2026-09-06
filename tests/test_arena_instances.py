from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.arena.instances import instance_votes, physical_witness, consistent_witness, InstanceConflict
from mr_liu.arena.perception import PerceptionBridge
from mr_liu.arena.holding import holding_measurement


def test_unknown_model_label_associates_mask_to_nested_physical_instance(tmp_path):
    directory = tmp_path / 'observation'; directory.mkdir()
    mask = np.zeros((5, 5), bool); mask[1:4, 1:4] = True
    ids = np.full((5, 5), 90); ids[1:4, 1:4] = 123
    bodies = {'body_7': {'prim_path': '/World/envs/env_0/body_7'}}
    np.savez(directory / 'masks.npz', camera=mask)
    np.savez(directory / 'physical_instances.npz', camera=ids)
    (directory/'physical_instances.json').write_text(json.dumps({'camera': {
        '123': '/World/envs/env_0/body_7/geometry/mesh', '90': '/World/table'}}))
    name, votes = PerceptionBridge(tmp_path, 'unused').witness(
        {'request_id': 'observation', 'label': 'never configured industrial object'}, bodies)
    assert name == 'body_7' and votes == {'body_7': 1}


def test_instance_ids_do_not_invent_detection_outside_the_model_mask():
    votes = instance_votes(np.zeros((3, 3), bool), np.ones((3, 3)), {'1': '/Body'},
                           {'body': {'prim_path': '/Body'}})
    assert not votes
    with pytest.raises(RuntimeError): physical_witness(votes)


def test_large_wrong_view_cannot_outvote_a_small_correct_object_in_another_view():
    with pytest.raises(InstanceConflict):
        consistent_witness({'front': Counter({'object': 600}), 'side': Counter({'pad': 10500})})
    assert consistent_witness({'front': Counter({'object': 600}),
                               'side': Counter({'object': 900, 'floor': 5})}) == 'object'


def test_similar_prim_prefix_is_not_the_same_object_and_mixed_views_do_not_choose_randomly():
    votes = instance_votes(np.ones((2, 2), bool), np.ones((2, 2)), {'1': '/BodyOther/mesh'},
                           {'body': {'prim_path': '/Body'}})
    assert not votes
    with pytest.raises(RuntimeError): physical_witness(Counter({'a': 10, 'b': 10}))


def test_holding_continuity_accepts_transport_but_detects_open_jaws_or_dropped_object():
    reference = np.eye(4); reference[:3, 3] = [.001, 0, .002]
    assert holding_measurement(reference, reference.copy(), .044)['verified']
    assert not holding_measurement(reference, reference, .08)['verified']
    dropped = reference.copy(); dropped[2, 3] -= .12
    assert not holding_measurement(reference, dropped, .044)['verified']
