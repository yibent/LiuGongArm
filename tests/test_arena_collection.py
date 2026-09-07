from unittest.mock import Mock
import numpy as np
from find_and_track.types import Detection
from mr_liu.perception.arena_vision import ImagePipeline
from mr_liu.arena.visual_refs import annotate_references
from mr_liu.arena.observed_scene import collection_geometry, spatial_groups, ObservedScene


def test_multiple_fast_objects_succeed_without_slow_or_class_memory():
    finder, yolo, sam, memory = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = [Detection(np.array(box), 'part', .9) for box in [[2,2,8,8], [15,2,22,8]]]
    masks = []
    for d in yolo.detect.return_value:
        mask = np.zeros((32,32), bool)
        x1,y1,x2,y2 = d.xyxy.astype(int)
        mask[y1:y2,x1:x2] = True
        masks.append((mask[None], np.array([.9]), None))
    sam.predict.side_effect = masks
    pipe = ImagePipeline(finder, yolo, sam, memory_store=memory)
    result_masks, result = pipe.collect(np.zeros((32,32,3), np.uint8), camera='scene_camera', sequence=1,
                                       scene_id='s', label='part', mode='auto')
    assert result['status'] == 'observed' and len(result['objects']) == 2
    assert len(result_masks) == 2 and not pipe.tracks
    sam.set_image.assert_called_once()
    finder.find.assert_not_called()
    memory.remember.assert_not_called()


def test_collection_uses_sam3_masks_without_resegmenting_or_florence():
    finder, yolo, sam, sam3 = Mock(), Mock(), Mock(), Mock()
    yolo.detect.return_value = []
    masks = []
    for x in [2,15]:
        mask = np.zeros((32,32), bool); mask[3:10,x:x+5] = True
        masks.append((Detection(np.array([x,3,x+5,10]), 'part', .9), mask))
    sam3.locate.return_value = masks
    pipe = ImagePipeline(finder, yolo, sam, slow_localizer='sam3', localizers={'sam3': sam3})
    output, result = pipe.collect(np.zeros((32,32,3), np.uint8), camera='scene_camera', sequence=1, scene_id='s', label='part')
    assert result['loop'] == 'slow' and len(output) == 2
    assert result['fallback_reason'] == 'no_text_detection'
    sam.predict.assert_not_called(); finder.find.assert_not_called()


def test_multiview_collection_deduplicates_parts_but_keeps_adjacent_instances():
    frames, masks, views = {}, {}, []
    for camera, shift in [('scene_camera', 0), ('side_camera', 5)]:
        frames[camera+'_rgb'] = np.zeros((64,64,3), np.uint8)
        frames[camera+'_depth'] = np.ones((64,64))
        frames[camera+'_K'] = np.array([[500.,0.,32.+shift],[0.,500.,32.],[0.,0.,1.]])
        frames[camera+'_T'] = np.eye(4); frames[camera+'_T'][0,3] = shift/500.
        objects = []
        for i,x in enumerate([16,36]):
            key = f'{camera}__{i}'
            masks[key] = np.zeros((64,64), bool); masks[key][20:40,x:x+8] = True
            objects.append({'label':'part','box':[x,20,x+8,40],'score':.9,'mask_key':key,'semantic_status':'detected'})
        views.append({'camera':camera,'sequence':5,'objects':objects})
    result = {'request_id':'a'*32,'label':'part','scope':'collection','observed_at':1.,'views':views}
    annotate_references(result,frames)
    collection_geometry(result,frames,masks)
    collection = result['collection']
    assert collection['count'] == 2
    assert all(len(x['references']) == 2 for x in collection['instances'])
    assert len({r['ref'] for r in result['references']}) == len(result['references'])
    assert not collection['complete']
    scene = ObservedScene(); scene.update(result)
    track_ids = [item['track_id'] for item in collection['instances']]
    scene.update(result)
    assert track_ids == [item['track_id'] for item in collection['instances']]


def test_groups_follow_observed_geometry_and_do_not_require_fixed_counts():
    parts = [{'position_m':[x,y,.03], 'extent_m':[.025,.025,.06]} for x,y in
        [(.31,-.14),(.35,-.14),(.47,-.22),(.514,-.22),(.47,-.176),(.514,-.176),(.562,-.132),
         (.695,-.16),(.735,-.16),(.715,-.112)]]
    assert [len(group) for group in spatial_groups(parts)] == [5,3,2]
    assert [len(group) for group in spatial_groups(parts[:4])] == [2,2]


def test_missing_geometry_does_not_manufacture_an_instance():
    camera = 'scene_camera'
    frames = {camera+'_rgb': np.zeros((5,5,3),np.uint8), camera+'_depth':np.full((5,5),np.nan),
              camera+'_K':np.eye(3),camera+'_T':np.eye(4)}
    result = {'request_id':'a'*32,'label':'part','views':[{'camera':camera,'sequence':1,'objects':[
        {'box':[0,0,5,5],'label':'part','mask_key':'m','score':.9,'semantic_status':'detected'}]}]}
    annotate_references(result,frames)
    collection_geometry(result,frames,{'m':np.ones((5,5),bool)})
    assert result['collection']['count'] == 0
    assert result['collection']['unlocalized_count'] == 1
    assert not result['collection']['complete']


def test_image_box_uses_original_frame_and_rejects_another_scene(tmp_path):
    import json
    import pytest
    from mr_liu.arena.visual_refs import load_snapshot
    directory = tmp_path/('b'*32); directory.mkdir()
    (directory/'request.json').write_text(json.dumps({'scene_id':'current','views':[{'camera':'scene_camera'}]}))
    np.savez(directory/'frames.npz', scene_camera_rgb=np.zeros((100,200,3),np.uint8))
    _, box = load_snapshot(tmp_path,'b'*32,'current','scene_camera',[.1,.2,.3,.4])
    assert box == [20.,20.,60.,40.]
    with pytest.raises(RuntimeError):
        load_snapshot(tmp_path,'b'*32,'different','scene_camera',[.1,.2,.3,.4])
    with pytest.raises(ValueError):
        load_snapshot(tmp_path,'b'*32,'current','scene_camera',[.3,.2,.1,.4])
