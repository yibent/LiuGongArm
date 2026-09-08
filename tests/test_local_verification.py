import json
from types import SimpleNamespace
import numpy as np
import pytest
from mr_liu.perception.local_verification import verify_region


def test_verification_uses_exact_snapshot_roi_and_bgr_without_physical_claim(tmp_path):
    ref='a'*32
    path=tmp_path/ref; path.mkdir()
    (path/'request.json').write_text(json.dumps({'observed_at':123}))
    rgb=np.zeros((100,200,3),dtype=np.uint8);rgb[:]=[10,20,30]
    np.savez(path/'frames.npz',scene_camera_rgb=rgb)
    class Finder:
        def find(self, image, labels, beams):
            assert image.shape==(20,60,3)
            assert image[0,0].tolist()==[30,20,10]
            assert labels==['metal part'] and beams==1
            return [SimpleNamespace(label='metal part',xyxy=[1,2,8,12])]
    args=dict(snapshot_ref=ref,camera='scene',box_2d=[100,200,300,500],target_label='metal part')
    result=verify_region(tmp_path,Finder(),args)
    assert result['verdict']=='passed' and result['observed_at']==123
    assert 'physical_success' not in result
    result=verify_region(tmp_path,Finder(),{**args,'predicate':'upright'})
    assert result['verdict']=='uncertain'
    with pytest.raises(ValueError):verify_region(tmp_path,Finder(),{**args,'snapshot_ref':'../outside'})
    class Missing:
        def find(self,*args,**kwargs):return []
    assert verify_region(tmp_path,Missing(),args)['verdict']=='uncertain'


def test_verification_can_ask_florence_to_locate_the_region_before_checking_target(tmp_path):
    ref='b'*32
    path=tmp_path/ref;path.mkdir()
    (path/'request.json').write_text(json.dumps({'observed_at':321}))
    np.savez(path/'frames.npz',scene_camera_rgb=np.zeros((100,200,3),dtype=np.uint8))
    class Finder:
        def find(self,image,labels,beams):
            if labels==['blue tray']:
                return [SimpleNamespace(label='blue tray',xyxy=[40,20,160,90])]
            assert labels==['metal part'] and image.shape[0] < 100
            return [SimpleNamespace(label='metal part',xyxy=[10,10,30,40])]
    result=verify_region(tmp_path,Finder(),dict(snapshot_ref=ref,camera='scene',
        region_label='blue tray',target_label='metal part',predicate='inside'))
    assert result['verdict']=='passed' and result['region_label']=='blue tray'
    assert result['predicate']=='inside'
    class WholeImage:
        def find(self,image,*args,**kwargs):
            h,w=image.shape[:2]
            return [SimpleNamespace(label='metal part',xyxy=[0,0,w-1,h-1])]
    # Actual Florence negative control: it grounded a cylinder to an empty
    # tabletop crop's entire extent. This must never pass a local check.
    negative=verify_region(tmp_path,WholeImage(),dict(snapshot_ref=ref,camera='scene',
        box_2d=[100,200,300,500],target_label='metal part'))
    assert negative['verdict']=='uncertain'
    assert not negative['objects'][0]['localized_in_roi']
