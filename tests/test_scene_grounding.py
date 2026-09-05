import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.vision.grounding import SceneGrounding, surface_point, confirmed_mask, instance_association


class GroundingTests(unittest.TestCase):
    def test_leaf_meshes_resolve_when_grouped_annotator_has_no_paths(self):
        mask = np.ones((2, 3), dtype=bool)
        grouped = {'data': np.ones((2, 3)), 'info': {'idToLabels': {'1': {'class': 'block'}}}}
        leaf = {'data': np.array([[5, 5, 6], [6, 6, 5]]), 'info': {'idToLabels': {
            '5': '/World/TabletopProps/Block/meshA', '6': {'primPath': '/World/TabletopProps/Block/meshB'}}}}
        result = instance_association(grouped, mask, leaf)
        self.assertEqual(result['object_id'], '/World/TabletopProps/Block')
        self.assertEqual(result['grouped']['reason'], 'no_instance_paths')
        self.assertEqual(result['leaf']['coverage'], 1)

    def test_conflicting_instances_and_low_coverage_are_not_guessed(self):
        mask = np.ones((2, 3), dtype=bool)
        def payload(path):
            return {'data': np.ones((2, 3)), 'info': {'idToLabels': {'1': path}}}
        result = instance_association(payload('/World/TabletopProps/A'), mask, payload('/World/TabletopProps/B'))
        self.assertIsNone(result['object_id'])
        self.assertEqual(result['reason'], 'conflicting_instances')
        sparse = payload('/World/TabletopProps/A')
        sparse['data'][1] = 0
        result = instance_association(sparse, mask)
        self.assertIsNone(result['object_id'])
        self.assertEqual(result['grouped']['reason'], 'insufficient_coverage')

    def test_control_marker_is_not_relabelled_as_a_tabletop_prop(self):
        mask = np.ones((2, 3), dtype=bool)
        # Actual Isaac 6 runtime: the grouped annotator returns an empty
        # path for TargetCube, while the leaf annotator retains its identity.
        grouped = {'data': np.full((2, 3), 2), 'info': {'idToLabels': {'2': ''}}}
        leaf = {'data': np.ones((2, 3)), 'info': {'idToLabels': {'1': '/World/TargetCube'}}}
        result = instance_association(grouped, mask, leaf)
        self.assertEqual(result['object_id'], '/World/TargetCube')
        self.assertEqual(result['grouped']['reason'], 'no_instance_paths')
        self.assertEqual(result['leaf']['coverage'], 1.0)

    def test_missing_and_misaligned_segmentation_have_diagnostics(self):
        result = instance_association({'data': np.ones((3, 3))}, np.ones((2, 3), dtype=bool))
        self.assertIsNone(result['object_id'])
        self.assertEqual(result['grouped']['reason'], 'shape_mismatch_or_empty_mask')
        self.assertEqual(result['leaf']['reason'], 'missing_annotator')

    def test_wrong_class_or_color_cannot_become_a_target(self):
        image = np.full((20,20,3), [0,255,255], dtype=np.uint8)
        semantic = {'data': np.ones((20,20), dtype=np.uint32), 'info': {'idToLabels': {'1': {'class': 'nut'}}}}
        self.assertIsNone(confirmed_mask(image, semantic, [0,0,20,20], 'yellow bolt')[0])
        self.assertIsNone(confirmed_mask(image, semantic, [0,0,20,20], 'purple nut')[0])
        self.assertIsNotNone(confirmed_mask(image, semantic, [0,0,20,20], 'yellow nut')[0])
    def test_depth_uses_real_unprojection_and_rejects_invalid_depth(self):
        depth = np.ones((30, 30))
        point = surface_point([5,5,25,25], depth, lambda uv,z: np.c_[uv*.01,z])
        np.testing.assert_allclose(point, [.15,.15,1])
        self.assertIsNone(surface_point([5,5,25,25], depth*np.nan, lambda *_: None))

    def resolve(self, candidates, request):
        ground = SceneGrounding()
        class Control:
            def set_prompt(self, prompt):
                ground.publish(2, prompt, candidates, time.monotonic())
                return SimpleNamespace(prompt_version=2)
        return ground.resolve(Control(), request, timeout=.1)

    def test_object_surface_plus_offset_not_tool_relative(self):
        result = self.resolve([dict(xyxy=[0,0,10,10], world_position_m=[.3,.1,1.05], score=.8)], dict(category='bolt', color='yellow', offset_m=[0,0,.1]))
        np.testing.assert_allclose(result['target_world_m'], [.3,.1,1.15])
        self.assertEqual(result['source'], 'scene_camera_rgbd+isaac_semantics')

    def test_ambiguous_and_missing_depth_do_not_create_motion(self):
        candidates = [dict(xyxy=[x,0,x+10,10], world_position_m=None) for x in [0,20]]
        self.assertFalse(self.resolve(candidates, dict(category='bolt', offset_m=[0,0,.1]))['ok'])
        self.assertFalse(self.resolve(candidates[:1], dict(category='bolt', offset_m=[0,0,.1]))['ok'])
        self.assertTrue(self.resolve(candidates, dict(category='bolt'))['ok'])

    def test_missing_and_stale_results_are_not_reused(self):
        ground = SceneGrounding()
        ground.publish(1, 'old', [dict(world_position_m=[1,2,3])], time.monotonic()-10)
        control = SimpleNamespace(set_prompt=lambda _: SimpleNamespace(prompt_version=2))
        self.assertFalse(ground.resolve(control, dict(category='star'), timeout=.01)['ok'])
        self.assertFalse(ground.valid(dict(observed_at=time.monotonic()-4,cancel_epoch=0)))

    def test_pause_invalidates_pending_grounding_and_tokens(self):
        ground = SceneGrounding()
        token = dict(observed_at=time.monotonic(), cancel_epoch=0)
        self.assertTrue(ground.valid(token))
        control = SimpleNamespace(set_prompt=lambda _: SimpleNamespace(prompt_version=1))
        timer = threading.Timer(.02, ground.cancel)
        timer.start()
        result = ground.resolve(control, dict(category='bolt'), timeout=.2)
        timer.join()
        self.assertFalse(result['ok'])
        self.assertFalse(ground.valid(token))


if __name__ == '__main__': unittest.main()
