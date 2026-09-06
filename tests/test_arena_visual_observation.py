import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.perception.arena_vision import ImagePipeline
from find_and_track.types import Detection
from mr_liu.arena.perception import PerceptionBridge
from find_and_track.memory import ObjectMemoryStore


def test_image_detection_then_lk_reuses_actual_mask_without_models():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0,255,(64,64,3),dtype=np.uint8)
    mask = np.zeros((64,64),bool); mask[15:45,15:45]=True
    finder = Mock(); finder.find.return_value=[Detection(np.array([15,15,45,45]),'part')]
    yolo = Mock(); yolo.detect.return_value=finder.find.return_value
    sam = Mock(); sam.predict.return_value=(mask[None],np.array([.9]),None)
    pipe = ImagePipeline(finder,yolo,sam)
    first, detail = pipe.observe(rgb,scene_id='s',camera='scene',label='part',sequence=1)
    second, tracked = pipe.observe(rgb.copy(),scene_id='s',camera='scene',label='part',sequence=2)
    assert np.array_equal(first,second)
    assert [x['model'] for x in detail['stages']] == ['yoloe','sam2_tiny']
    assert detail['loop'] == 'fast' and detail['semantic_status'] == 'detected'
    assert tracked['stages'][0]['model']=='lk'
    finder.find.assert_not_called(); sam.predict.assert_called_once()


def test_missing_target_returns_no_mask_and_never_calls_sam():
    finder,yolo,sam=Mock(),Mock(),Mock();finder.find.return_value=[];yolo.detect.return_value=[]
    mask,result=ImagePipeline(finder,yolo,sam).observe(np.zeros((32,32,3),np.uint8),
        scene_id='s',camera='scene',label='absent',sequence=1)
    assert mask is None and result['status']=='not_found'
    sam.predict.assert_not_called()


def test_point_cloud_uses_saved_frame_pose_and_model_mask(tmp_path):
    bridge=PerceptionBridge(tmp_path,'unused')
    directory=tmp_path/'frame';directory.mkdir()
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    np.savez(directory/'frames.npz',scene_depth=np.array([[1.,2.],[3.,4.]]),scene_K=np.eye(3),scene_T=pose)
    np.savez(directory/'masks.npz',scene=np.array([[True,False],[False,False]]))
    points,diagnostics=bridge.cloud({'request_id':'frame'})
    assert np.allclose(points,[[1,2,4]])
    assert diagnostics['scene']['points']==1


def test_point_cloud_removes_mask_boundary_depth_bleed(tmp_path):
    bridge = PerceptionBridge(tmp_path, 'unused')
    directory = tmp_path/'edge'; directory.mkdir()
    depth = np.full((5,5), 2.); depth[2,2] = 1.
    mask = np.zeros((5,5), bool); mask[1:4,1:4] = True
    np.savez(directory/'frames.npz', scene_depth=depth, scene_K=np.eye(3), scene_T=np.eye(4))
    np.savez(directory/'masks.npz', scene=mask)
    points, _ = bridge.cloud({'request_id':'edge'})
    assert np.allclose(points, [[2.,2.,1.]])


def test_placement_obstacles_use_unmasked_depth_from_the_same_frame(tmp_path):
    bridge = PerceptionBridge(tmp_path, 'unused')
    directory = tmp_path/'scene'; directory.mkdir()
    depth = np.ones((4,4)); depth[0,2] = np.nan
    pose = np.eye(4); pose[:3,3] = [1,2,3]
    np.savez(directory/'frames.npz', scene_depth=depth, scene_K=np.eye(3), scene_T=pose)
    # No target masks or simulator instance sidecars are needed for obstacles.
    points = bridge.scene_cloud({'request_id': 'scene'})
    assert np.allclose(points, [[1,2,4], [1,4,4], [3,4,4]])


def test_scene_queries_require_current_image_detections():
    finder = Mock()
    finder.describe.return_value = {'<DENSE_REGION_CAPTION>': {'labels': ['toy'], 'bboxes': [[1, 2, 4, 5]]}}
    yolo, sam = Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array([1, 2, 4, 5]), 'yellow cylinder', .8)]
    pipe = ImagePipeline(finder, yolo, sam)
    result = pipe.describe(np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=20,
                           scene_id='s', queries=['yellow cylinder', 'absent part'])
    assert [o['label'] for o in result['objects']] == ['yellow cylinder']
    assert result['unconfirmed_queries'][0]['label'] == 'absent part'
    assert result['regions'][0]['description'] == 'toy'
    yolo.detect.assert_called_once()
    finder.find.assert_not_called()
    sam.predict.assert_not_called()


def test_capture_scene_and_unknown_target_without_catalog_binding(tmp_path):
    data = SimpleNamespace(output={'rgb': np.zeros((1, 8, 8, 4), np.uint8),
                                  'distance_to_image_plane': np.ones((1, 8, 8, 1))},
                           intrinsic_matrices=np.eye(3)[None], pos_w=np.zeros((1, 3)),
                           quat_w_ros=np.array([[1., 0, 0, 0]]))
    runtime = SimpleNamespace(held=None, current='command', sequence=20, bus_context={},
        config={'objects': [{'name': 'yellow_cylinder', 'label': 'yellow cylinder'}], 'destinations': [],
                'vision': {'vocabulary': ['yellow object']}},
        env=SimpleNamespace(scene={name: SimpleNamespace(data=data) for name in ['scene_camera', 'side_camera']}))
    bridge = PerceptionBridge(tmp_path, 'unused')
    scene = bridge.capture(runtime, None)
    assert scene['scope'] == 'scene'
    assert [v['camera'] for v in scene['views']] == ['scene_camera', 'side_camera']
    assert scene['queries'] == ['yellow object']
    target = bridge.capture(runtime, 'unregistered part')
    assert target['scope'] == 'target' and target['label'] == 'unregistered part'
    assert target['queries'] == []
    tracking = bridge.capture(runtime, 'unregistered part', transient=True, cameras=['scene_camera'])
    with np.load(bridge.root/tracking['request_id']/'frames.npz') as frames:
        assert frames.files == ['scene_camera_rgb']


def test_low_confidence_slow_candidate_is_recalled_after_restart_without_semantic_promotion(tmp_path):
    rgb = np.random.default_rng(3).integers(0, 255, (64,64,3), np.uint8)
    mask = np.zeros((64,64), bool); mask[15:45,15:45] = True
    detection = Detection(np.array([15,15,45,45]), 'new part', .9)
    finder, yolo, sam = Mock(), Mock(), Mock()
    finder.find.return_value = [detection]
    yolo.detect.return_value = [Detection(detection.xyxy, 'new part', .2)]
    sam.predict.return_value = (mask[None], np.array([.99]), None)
    memory = ObjectMemoryStore(tmp_path)
    pipe = ImagePipeline(finder, yolo, sam, memory_store=memory)
    _, slow = pipe.observe(rgb, camera='scene', sequence=1, scene_id='s', label='new part')
    assert slow['loop'] == 'slow' and slow['fallback_reason'] == 'low_confidence'
    assert slow['status'] == 'candidate' and slow['score'] is None
    saved = memory.find('new part')[0]
    assert saved.views[0].crop_bbox == saved.views[0].bbox  # Full-frame reference keeps box coordinates.
    assert np.array_equal(cv2.imread(saved.views[0].image), rgb[:, :, ::-1])
    # New process / camera epoch: only persistent image references survive.
    restarted = ImagePipeline(finder, yolo, sam, memory_store=ObjectMemoryStore(tmp_path))
    yolo.detect.return_value = [detection]
    _, recall = restarted.observe(rgb, camera='scene', sequence=1, scene_id='new-scene', label='new part')
    assert recall['loop'] == 'fast'
    assert recall['semantic_status'] == 'candidate'
    assert recall['memory_id'] == slow['memory_id']
    assert recall['stages'][0]['operation'] == 'memory_recall'
    finder.find.assert_called_once()
    _, tracked = restarted.observe(rgb.copy(), camera='scene', sequence=2, scene_id='new-scene', label='new part')
    assert tracked['stages'][0]['model'] == 'lk'
    assert restarted.forget('new part') == 1
    assert memory.find('new part') == [] and not restarted.tracks


def test_fast_only_failure_and_unavailable_slow_provider_do_not_call_other_models():
    finder, yolo, sam = Mock(), Mock(), Mock(); yolo.detect.return_value = []
    pipe = ImagePipeline(finder, yolo, sam)
    kwargs = dict(camera='scene', sequence=1, scene_id='s', label='part')
    image = np.zeros((32,32,3), np.uint8)
    mask, detail = pipe.observe(image, **kwargs, mode='fast')
    assert mask is None and detail['status'] == 'not_found'
    mask, detail = pipe.observe(image, **kwargs, mode='slow', slow_provider='sam3')
    assert mask is None and detail['status'] == 'provider_unavailable'
    finder.find.assert_not_called(); sam.predict.assert_not_called()


def test_frame_gap_relocalizes_instead_of_reusing_stale_flow():
    finder, yolo, sam = Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array([5,5,25,25]), 'part', .8)]
    sam.predict.return_value = (np.ones((1,32,32), bool), np.array([.9]), None)
    pipe = ImagePipeline(finder, yolo, sam, max_frame_gap=10)
    kwargs = dict(camera='scene', scene_id='s', label='part')
    image = np.zeros((32,32,3), np.uint8)
    pipe.observe(image, sequence=1, **kwargs)
    _, current = pipe.observe(image, sequence=40, **kwargs)
    assert current['stages'][0]['operation'] == 'text_detect'
    finder.find.assert_not_called()


def test_fast_only_multiple_candidates_do_not_call_slow_models():
    finder, yolo, sam = Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array(box), 'part', .9) for box in [[1,1,5,5],[10,10,20,20]]]
    mask, detail = ImagePipeline(finder, yolo, sam).observe(np.zeros((32,32,3), np.uint8),
        camera='scene', sequence=1, scene_id='s', label='part', mode='fast')
    assert mask is None and detail['status'] == 'ambiguous'
    finder.find.assert_not_called(); sam.predict.assert_not_called()


def test_fast_multiple_candidates_are_resolved_by_sam3_then_tracked(tmp_path):
    image = np.random.default_rng(71).integers(0, 255, (64,64,3), np.uint8)
    correct = np.zeros((64,64), bool); correct[15:45,15:45] = True
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array(box), 'green block', .9)
                               for box in [[15,15,45,45], [48,48,60,60]]]
    sam3.locate.return_value = [(yolo.detect.return_value[0], correct)]
    memory = ObjectMemoryStore(tmp_path)
    pipe = ImagePipeline(finder, yolo, sam2, memory_store=memory,
                         slow_localizer='sam3', localizers={'sam3': sam3})
    kwargs = dict(scene_id='s', camera='scene', label='green block')
    mask, detail = pipe.observe(image, sequence=1, **kwargs)
    assert np.array_equal(mask, correct)
    assert detail['fallback_reason'] == 'multiple_fast_candidates'
    assert detail['origin'] == 'sam3' and detail['status'] == 'observed'
    assert len(detail['stages'][0]['accepted_boxes']) == 2
    assert memory.find('green block')[0].metadata['origin'] == 'sam3'
    _, tracked = pipe.observe(image.copy(), sequence=2, **kwargs)
    assert tracked['loop'] == 'fast' and tracked['stages'][0]['model'] == 'lk'
    sam3.locate.assert_called_once(); finder.find.assert_not_called(); sam2.predict.assert_not_called()


def test_slow_confirmed_multiple_instances_still_return_no_target():
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    found = [Detection(np.array(box), 'part', .9) for box in [[1,1,5,5], [10,10,20,20]]]
    yolo.detect.return_value = found
    sam3.locate.return_value = [(d, np.ones((32,32), bool)) for d in found]
    pipe = ImagePipeline(finder, yolo, sam2, slow_localizer='sam3', localizers={'sam3': sam3})
    mask, detail = pipe.observe(np.zeros((32,32,3), np.uint8),
        scene_id='s', camera='scene', label='part', sequence=1)
    assert mask is None and detail['status'] == 'ambiguous' and detail['loop'] == 'slow'
    assert not pipe.tracks
    sam3.locate.assert_called_once(); finder.find.assert_not_called(); sam2.predict.assert_not_called()


def test_sam3_concept_mask_hands_back_to_fast_tracking_without_florence_or_sam2(tmp_path):
    image = np.random.default_rng(19).integers(0, 255, (64,64,3), np.uint8)
    mask = np.zeros((64,64), bool); mask[15:45,15:45] = True
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = []
    sam3.locate.return_value = [(Detection(np.array([15,15,45,45]), 'part', .85), mask)]
    pipe = ImagePipeline(finder, yolo, sam2, memory_store=ObjectMemoryStore(tmp_path),
                         slow_localizer='sam3', localizers={'sam3': sam3})
    kwargs = dict(scene_id='s', camera='scene', label='part')
    first, detail = pipe.observe(image, sequence=1, **kwargs)
    assert np.array_equal(first, mask)
    assert detail['origin'] == 'sam3' and detail['semantic_status'] == 'detected'
    assert detail['score'] == .85 and detail['loop'] == 'slow'
    _, next_frame = pipe.observe(image.copy(), sequence=2, **kwargs)
    assert next_frame['loop'] == 'fast' and next_frame['stages'][0]['model'] == 'lk'
    sam3.locate.assert_called_once(); finder.find.assert_not_called(); sam2.predict.assert_not_called()


def test_sam3_absence_does_not_force_florence_to_invent_a_box():
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = []; sam3.locate.return_value = []
    pipe = ImagePipeline(finder, yolo, sam2, slow_localizer='sam3', localizers={'sam3': sam3})
    mask, detail = pipe.observe(np.zeros((32,32,3), np.uint8), scene_id='s', camera='scene', label='absent', sequence=1)
    assert mask is None and detail['status'] == 'not_found'
    finder.find.assert_not_called(); sam2.predict.assert_not_called()
