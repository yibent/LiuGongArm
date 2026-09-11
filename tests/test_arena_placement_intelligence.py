import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from mr_liu.arena.free_space import choose_free_support, free_support_candidates
from mr_liu.arena.visual_refs import annotate_references, load_reference
from mr_liu.arena.cascade import run_cascade, FastPathFailure
from mr_liu.arena.contracts import ManipulationRequest
from mr_liu.arena.failure import failure_feedback
from test_arena_free_space import plane
from test_arena_reobservation import runtime_method


def test_rotates_long_part_into_narrow_space():
    support = plane(-.07,.07,-.18,.18)
    child = np.array([[-.10,-.015,.1],[-.10,.015,.12],[.10,-.015,.12],[.10,.015,.1]])
    with pytest.raises(RuntimeError):
        choose_free_support(support,support,child,[0,0,.3],allow_rotation=False)
    _, selected = choose_free_support(support,support,child,[0,0,.3])
    assert abs(abs(selected['yaw_delta_rad'])-np.pi/2) < .001
    assert selected['footprint_size_m'][0] < .14


def test_left_right_preferences_and_region_are_honored():
    support = plane()
    child = np.array([[-.02,-.02,.1],[.02,.02,.14],[-.02,.02,.14],[.02,-.02,.1]])
    left,_ = choose_free_support(support,support,child,[0,0,.3],preference='left')
    right,_ = choose_free_support(support,support,child,[0,0,.3],preference='right')
    assert left[0]<-.15 and right[0]>.15
    rows = free_support_candidates(support,support,child,[0,0,.3],region={'box':[.1,-.2,.25,.2]})
    assert all(.1 <= r['position_world_m'][0] <= .25 for r in rows)


def test_explicit_basic_never_calls_enhanced_or_recovery():
    enhanced,recover = Mock(),Mock()
    with pytest.raises(FastPathFailure):
        run_cascade(ManipulationRequest('part',mode='basic'),Mock(side_effect=FastPathFailure('missed')),enhanced,recover,Mock())
    enhanced.assert_not_called(); recover.assert_not_called()


def test_references_bind_scene_frame_not_configured_names(tmp_path):
    rid='a'*32;directory=tmp_path/rid;directory.mkdir()
    frames={'scene_camera_rgb':np.zeros((100,200,3))}
    result={'request_id':rid,'label':'scene','views':[{'camera':'scene_camera','sequence':17,
        'objects':[{'label':'unknown metal part','box':[20,10,60,40]}]}]}
    annotate_references(result,frames)
    ref=result['references'][0]
    assert ref['box_normalized']==[.1,.1,.3,.4]
    (directory/'request.json').write_text(json.dumps({'scene_id':'current'}))
    (directory/'result.json').write_text(json.dumps(result))
    assert load_reference(tmp_path,ref['ref'],'current')[0]['label']=='unknown metal part'
    with pytest.raises(RuntimeError,match='另一个场景'): load_reference(tmp_path,ref['ref'],'reset')
    with pytest.raises(ValueError):load_reference(tmp_path,'../../.env','current')


def test_manipulation_target_rejects_floor_witness_before_motion():
    method = runtime_method('cloud')
    result = {'ok':True,'request_id':'a'*32,'perception_source':'test','views':[]}
    perception = SimpleNamespace(
        capture=lambda *args, **kwargs:{'request_id':'a'*32},
        request=lambda packet:result,
        witness=lambda observed, entities:('floor',{'floor':99}),
        cloud=lambda observed:(np.zeros((64,3)),{}),
    )
    runtime = SimpleNamespace(
        prepared_clouds={}, observed_entities={}, perception=perception,
        execution_policy={'loop':'fast_then_slow'}, observing=False,
        vision_worker=SimpleNamespace(available=True), tick=lambda *args, **kwargs:None,
        event=Mock(), infer=lambda fn,*args,**kwargs:fn(*args,**kwargs),
        body_entities={'floor':{}}, held=None, sequence=1, last_track=0,
        visual_result=None, config={'camera':{'render_interval':1}},
    )
    with pytest.raises(Exception, match='地面'):
        method(runtime,'metal cylinder',associate=True,manipulation_target=True)
    assert any(call.args[0] == 'visual_relocalization' for call in runtime.event.call_args_list)


def test_manipulation_target_rejects_a_physical_fixed_fixture():
    method = runtime_method('cloud')
    result = {'ok':True,'request_id':'a'*32,'perception_source':'test','views':[]}
    perception = SimpleNamespace(
        capture=lambda *args, **kwargs:{'request_id':'a'*32},
        request=lambda packet:result,
        witness=lambda observed, entities:('fixture',{'fixture':99}),
        cloud=lambda observed:(np.zeros((64,3)),{}),
    )
    runtime = SimpleNamespace(
        prepared_clouds={}, observed_entities={}, perception=perception,
        execution_policy={'loop':'slow'}, observing=False,
        vision_worker=SimpleNamespace(available=True), tick=lambda *args, **kwargs:None,
        event=Mock(), infer=lambda fn,*args,**kwargs:fn(*args,**kwargs),
        body_entities={'fixture':{'manipulable':False}}, held=None, sequence=1,
        last_track=0, visual_result=None, config={'camera':{'render_interval':1}},
    )
    with pytest.raises(ValueError, match='固定工装'):
        method(runtime,'grey block',associate=True,manipulation_target=True)


def test_precontact_ik_failure_reuses_next_candidate_and_stop_never_retries():
    r=SimpleNamespace(event=Mock(),holding_status=lambda:{'verified':True},
                      move=Mock(side_effect=[RuntimeError('Arena IK did not reach pregrasp'),None]))
    selected=runtime_method('try_precontact_candidates')(r,[1,2,3],lambda x:x,phase='pregrasp')
    assert selected==2 and r.move.call_count==2
    assert any(c.kwargs.get('inference_reused') for c in r.event.call_args_list)
    r.move=Mock(side_effect=InterruptedError('stop'))
    with pytest.raises(InterruptedError):runtime_method('try_precontact_candidates')(r,[1,2],lambda x:x,phase='transport',holding=True)
    assert r.move.call_count==1
    r.move=Mock(side_effect=RuntimeError('approach collision'))
    with pytest.raises(RuntimeError):runtime_method('try_precontact_candidates')(r,[1,2],lambda x:x,phase='pregrasp')
    assert r.move.call_count==1


def test_failure_feedback_preserves_successful_hold():
    feedback=failure_feedback('Arena IK did not reach transport','transport',{'verified':True})
    assert feedback['code']=='NO_IK' and feedback['suggested_recovery'][:2]==['preserve_grasp','place_held']


def test_failed_cell_metrics_survive_the_fast_path_and_inform_recovery():
    evaluation={'physical_success':False,'postconditions':{'requested_cell':{
        'satisfied':False,'minimum_wall_margin_m':-.014,'bottom_gap_m':.03,'row':2,'column':3}}}
    with pytest.raises(FastPathFailure) as error:
        run_cascade(ManipulationRequest('part','bin',mode='basic'),lambda request:evaluation,Mock(),Mock(),Mock())
    assert error.value.evaluation is evaluation
    feedback=failure_feedback(str(error.value),'finish',{'verified':False},error.value.evaluation)
    assert feedback['code']=='WRONG_CELL'
    assert feedback['measured_postcondition']['row']==2
    assert 'preserve_requested_cell' in feedback['suggested_recovery']


def test_hold_monitor_ignores_release_and_requires_persistent_loss():
    from mr_liu.arena.holding import HoldMonitor
    monitor=HoldMonitor()
    assert not monitor.update(False,'transport',True)
    assert not monitor.update(True,'transport',True)
    assert not monitor.update(False,'transport',True)
    assert not monitor.update(False,'transport',True)
    assert monitor.update(False,'transport',True)
    assert not monitor.update(False,'release',False)


def test_jaws_can_overhang_support_but_not_visible_obstacles():
    support = plane(-.04,.04,-.04,.04)
    surround = plane(-.18,.18,-.18,.18); surround[:,2] -= .02
    child=np.array([[-.015,-.015,.1],[.015,.015,.13],[-.015,.015,.13],[.015,-.015,.1]])
    scene=np.r_[surround,support]
    centre,_=choose_free_support(support,scene,child,[0,0,.3],allow_rotation=False)
    assert np.linalg.norm(centre[:2]) < .02
    wall=plane(-.05,.05,.038,.065);wall[:,2]+=.04
    with pytest.raises(RuntimeError):
        choose_free_support(support,np.r_[scene,wall],child,[0,0,.3],allow_rotation=False,
                            region={'box':[-.003,-.003,.003,.003]})
