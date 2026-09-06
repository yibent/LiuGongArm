"""Replay real recorded RGB-D through M2T2; never reports a physical success."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from mr_liu.grasp.backends.m2t2 import M2T2Client,M2T2Backend
from mr_liu.grasp.contracts import RGBDObservation,CameraIntrinsics,TargetSpec
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import transform_points,invert_transform


def observation(path, sequence=1):
    data=np.load(path)
    depth=data['depth_m'];h,w=depth.shape
    k=data['intrinsics'] if 'intrinsics' in data else data['K']
    has_ee='T_base_ee' in data and data['T_base_ee'].shape==(4,4) and data['T_ee_camera'].shape==(4,4)
    return RGBDObservation(sequence,0.,data['rgb'],depth,CameraIntrinsics(w,h,k[0,0],k[1,1],k[0,2],k[1,2]),
        data['T_base_camera'],data['T_base_ee'] if has_ee else None,
        data['T_ee_camera'] if has_ee else None,
        metadata={'robot_self_mask':data['robot_self_mask']} if 'robot_self_mask' in data else {})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'output/fine_place/20_sequential_models')
    parser.add_argument('--output',type=Path,default=ROOT/'output/m2t2/replay.json')
    parser.add_argument('--place-only',action='store_true')
    parser.add_argument('--pick-only',action='store_true')
    parser.add_argument('--pick-surface-range-m',type=float,default=0.)
    parser.add_argument('--scene-radius-m',type=float,default=.4)
    parser.add_argument('--input-surface-range-m',type=float,default=0.)
    args=parser.parse_args()
    client=M2T2Client(timeout_s=120.)
    obs=observation(args.run/'initial_rgbd.npz')
    mask=cv2.imread(str(args.run/'initial_mask.png'),0)>0
    points=deproject_depth(obs.depth_m,obs.intrinsics,mask & (obs.depth_m>0))
    candidates=[] if args.place_only else M2T2Backend(client,input_surface_range_m=args.pick_surface_range_m).generate(obs,points,TargetSpec('recorded_target'))
    result={'kind':'offline_model_inference_only','physical_success':None,
            'target_grasp_candidates':len(candidates),
            'grasps':[{'pose':c.T_camera_grasp.tolist(),'score':c.score,'width_m':c.width_m,'metadata':dict(c.metadata)} for c in candidates]}
    print('pick target candidates:',len(candidates),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.pick_only:return
    trace=json.loads((args.run/'place/trace.json').read_text(encoding='utf-8'))
    bootstrap=next(e for e in trace if e['phase']=='place_bootstrap')
    detection=next(e for e in trace if e.get('event')=='florence_destination')['response']
    observed=next(e for e in trace if e['phase']=='place_observe')
    scene_obs=observation(args.run/f"place/overhead_camera_{observed['sequence']:04d}.npz",observed['sequence'])
    from mr_liu.place.geometry import scene_cloud,polygon_mask,support_evidence
    scene,_,_=scene_cloud(scene_obs)
    pose=np.asarray(bootstrap['expected_pose'])
    near=np.linalg.norm(scene-pose[:3,3],axis=1)<.40
    scene=scene[near][::2]
    old_center=np.asarray(bootstrap['source_center_m'])
    obj=transform_points(pose,transform_points(obs.T_base_camera,points)-old_center)
    handoff=next(e for e in trace if e['phase']=='place_handoff')
    ee=pose@invert_transform(np.asarray(handoff['T_ee_object']))
    placement=client.infer(task='place',scene_points=scene.tolist(),object_points=obj.tolist(),ee_pose=ee.tolist(),sequence=2)
    result['place']=placement
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('place candidates:',len(placement['placements']),'latency_ms:',placement['inference_ms'],flush=True)
    from mr_liu.place.contracts import HeldObject,PlaceRequest
    from mr_liu.place.m2t2 import M2T2PlaceBackend
    held=HeldObject('recorded_payload',np.asarray(handoff['T_ee_object']),np.asarray(bootstrap['half_extents_m']),
        transform_points(obs.T_base_camera,points)-old_center,np.array([.7,.15,.15]),0.,True)
    evidence=support_evidence(scene_obs,polygon_mask(detection['boxes'][0],scene_obs.depth_m.shape))
    backend=M2T2PlaceBackend(client,scene_radius_m=args.scene_radius_m,input_surface_range_m=args.input_surface_range_m)
    try:
        xy=backend.select(PlaceRequest('recorded_destination'),held,evidence,ee,.013)
        result['guarded_place']={'xy':xy.tolist(),**backend.last_metrics}
    except Exception as exc:
        result['guarded_place']={'failure':str(exc),**backend.last_metrics}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('guarded_place:',result['guarded_place'],flush=True)


if __name__=='__main__':main()
