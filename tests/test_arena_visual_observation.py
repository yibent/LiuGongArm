import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock
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


def test_scene_caption_is_one_local_florence_request():
    finder = Mock()
    finder.describe.return_value = {'<DETAILED_CAPTION>': 'A red block beside a tray.'}
    yolo, sam = Mock(), Mock()
    yolo.detect.return_value = []
    result = ImagePipeline(finder, yolo, sam).describe(
        np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=21,
        scene_id='s', queries=['block'], mode='caption')
    assert result['caption'] == 'A red block beside a tray.'
    assert result['loop'] == 'slow'
    finder.describe.assert_called_once_with(ANY, detail='detailed', beams=1)
    sam.predict.assert_not_called()


def test_scene_auto_uses_one_low_threshold_yolo_batch_without_florence():
    finder, yolo, sam = Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array([1, 2, 4, 5]), 'cylinder', .36)]
    result = ImagePipeline(finder, yolo, sam).describe(
        np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=22,
        scene_id='s', queries=['block', 'cylinder'], mode='auto')
    assert [item['label'] for item in result['objects']] == ['cylinder']
    assert result['loop'] == 'fast'
    assert result['stages'][0]['accept_conf'] == .3
    finder.describe.assert_not_called()
    yolo.detect.assert_called_once()


def test_scene_auto_does_not_treat_the_table_alone_as_an_object_inventory():
    finder, yolo, sam = Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array([0, 0, 8, 8]), 'table', .36)]
    finder.describe.return_value = {'<DETAILED_CAPTION>': 'Parts on a work surface.'}
    result = ImagePipeline(finder, yolo, sam).describe(
        np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=23,
        scene_id='s', queries=['table', 'cylinder'], mode='auto')
    assert result['caption'] == 'Parts on a work surface.'
    assert result['loop'] == 'slow'
    finder.describe.assert_called_once()


def test_scene_auto_uses_one_batched_sam3_fallback_before_florence():
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = []
    mask = np.ones((8, 8), bool)
    sam3.locate.return_value = [
        (Detection(np.array([1, 1, 4, 6]), 'washer', .83), mask),
        (Detection(np.array([5, 1, 7, 7]), 'cylinder', .79), mask),
    ]
    result = ImagePipeline(
        finder, yolo, sam2, localizers={'sam3': sam3}, slow_localizer='sam3'
    ).describe(
        np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=24,
        scene_id='s', queries=['washer', 'cylinder', 'table'], mode='auto')
    assert {item['label'] for item in result['objects']} == {'washer', 'cylinder'}
    assert result['loop'] == 'slow'
    sam3.locate.assert_called_once_with(ANY, ('washer', 'cylinder', 'table'))
    finder.describe.assert_not_called()


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


def test_geometry_reinspection_retains_both_views_of_same_observed_instance(tmp_path):
    import json
    bridge = PerceptionBridge(tmp_path,'unused')
    rid='a'*32; origin=tmp_path/rid;origin.mkdir()
    refs=[{'ref':f'obs:{rid}:{camera}:0','camera':camera,'kind':'object','label':'unregistered bin'}
          for camera in ['scene_camera','side_camera']]
    (origin/'request.json').write_text(json.dumps({'scene_id':bridge.scene_id}))
    (origin/'result.json').write_text(json.dumps({'scope':'target','references':refs}))
    data=SimpleNamespace(output={'rgb':np.zeros((1,8,8,3),np.uint8),'distance_to_image_plane':np.ones((1,8,8,1))},
        intrinsic_matrices=np.eye(3)[None],pos_w=np.zeros((1,3)),quat_w_ros=np.array([[0,0,0,1.]]))
    runtime=SimpleNamespace(held=None,current='inspect',sequence=20,bus_context={},config={'vision':{}},
        env=SimpleNamespace(scene={r['camera']:SimpleNamespace(data=data) for r in refs}))
    request=bridge.capture(runtime,'actual query label',visual_ref=refs[0]['ref'],inspect='grid')
    assert request['label']=='actual query label'
    assert request['visual_refs']=={r['camera']:r['ref'] for r in refs}
    assert len(request['views'])==2


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
    finder.find.return_value = []
    found = [Detection(np.array(box), 'part', .9) for box in [[1,1,5,5], [10,10,20,20]]]
    yolo.detect.return_value = found
    sam3.locate.return_value = [(d, np.ones((32,32), bool)) for d in found]
    pipe = ImagePipeline(finder, yolo, sam2, slow_localizer='sam3', localizers={'sam3': sam3})
    mask, detail = pipe.observe(np.zeros((32,32,3), np.uint8),
        scene_id='s', camera='scene', label='part', sequence=1)
    assert mask is None and detail['status'] == 'ambiguous' and detail['loop'] == 'slow'
    assert not pipe.tracks
    sam3.locate.assert_called_once(); finder.find.assert_called_once(); sam2.predict.assert_not_called()


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


def test_sam3_absence_uses_florence_final_fallback_but_rejects_full_frame_box():
    finder, yolo, sam2, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = []; sam3.locate.return_value = []
    finder.find.return_value = [Detection(np.array([0,0,32,32]), 'absent', .1)]
    pipe = ImagePipeline(finder, yolo, sam2, slow_localizer='sam3', localizers={'sam3': sam3})
    mask, detail = pipe.observe(np.zeros((32,32,3), np.uint8), scene_id='s', camera='scene', label='absent', sequence=1)
    assert mask is None and detail['status'] == 'not_found'
    finder.find.assert_called_once(); sam2.predict.assert_not_called()


def test_visual_reference_is_reacquired_and_lost_reference_never_reuses_old_box():
    rng=np.random.default_rng(8)
    rgb=rng.integers(0,255,(64,64,3),dtype=np.uint8)
    mask=np.zeros((64,64),bool);mask[15:45,15:45]=True
    sam,yolo=Mock(),Mock();sam.predict.return_value=(mask[None],np.array([.9]),None)
    yolo.detect.return_value=[]
    pipe=ImagePipeline(Mock(),yolo,sam)
    ref={'ref':'obs:example','label':'part','box':[15,15,45,45],'semantic_status':'detected'}
    result,detail=pipe.from_reference(rgb.copy(),rgb,ref,camera='scene',sequence=10,scene_id='s')
    assert result.any() and detail['selected_ref']==ref['ref']
    yolo.detect.assert_not_called()
    empty=np.zeros_like(rgb)
    result,detail=pipe.from_reference(empty,empty,ref,camera='scene',sequence=11,scene_id='s')
    assert result is None and detail['status']=='reference_lost'
    yolo.set_visual_prompt.assert_called_once()
