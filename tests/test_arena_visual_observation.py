import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.perception.arena_vision import ImagePipeline
from find_and_track.types import Detection
from mr_liu.arena.perception import PerceptionBridge


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
    assert [x['model'] for x in detail['stages']] == ['florence2','yoloe','sam2_tiny']
    assert tracked['stages'][0]['model']=='lk'
    finder.find.assert_called_once(); sam.predict.assert_called_once()


def test_missing_target_returns_no_mask_and_never_calls_sam():
    finder,yolo,sam=Mock(),Mock(),Mock();finder.find.return_value=[]
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


def test_scene_queries_require_current_image_detections():
    finder = Mock()
    finder.describe.return_value = {'<DENSE_REGION_CAPTION>': {'labels': ['toy'], 'bboxes': [[1, 2, 4, 5]]}}
    pipe = ImagePipeline(finder, Mock(), Mock())
    pipe.observe = Mock(side_effect=[
        (np.ones((8, 8), bool), {'label': 'yellow cylinder', 'status': 'observed'}),
        (None, {'label': 'absent part', 'status': 'not_found'}),
    ])
    result = pipe.describe(np.zeros((8, 8, 3), np.uint8), camera='scene', sequence=20,
                           scene_id='s', queries=['yellow cylinder', 'absent part'])
    assert [o['label'] for o in result['objects']] == ['yellow cylinder']
    assert result['unconfirmed_queries'] == [{'label': 'absent part', 'status': 'not_found'}]
    assert result['regions'][0]['description'] == 'toy'
    assert all(c.kwargs['reset'] for c in pipe.observe.call_args_list)


def test_capture_scene_and_unknown_target_without_catalog_binding(tmp_path):
    data = SimpleNamespace(output={'rgb': np.zeros((1, 8, 8, 4), np.uint8),
                                  'distance_to_image_plane': np.ones((1, 8, 8, 1))},
                           intrinsic_matrices=np.eye(3)[None], pos_w=np.zeros((1, 3)),
                           quat_w_ros=np.array([[1., 0, 0, 0]]))
    runtime = SimpleNamespace(held=None, current='command', sequence=20, bus_context={},
        config={'objects': [{'name': 'yellow_cylinder', 'label': 'yellow cylinder'}], 'destinations': []},
        env=SimpleNamespace(scene={name: SimpleNamespace(data=data) for name in ['scene_camera', 'side_camera']}))
    bridge = PerceptionBridge(tmp_path, 'unused')
    scene = bridge.capture(runtime, None)
    assert scene['scope'] == 'scene'
    assert [v['camera'] for v in scene['views']] == ['scene_camera', 'side_camera']
    assert scene['queries'] == ['yellow cylinder']
    target = bridge.capture(runtime, 'unregistered part')
    assert target['scope'] == 'target' and target['label'] == 'unregistered part'
    assert target['queries'] == []
