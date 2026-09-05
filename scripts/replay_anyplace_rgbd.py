"""Replay a recorded AnyPlace request, optionally using destination segmentation.

Uses captured RGB-D only, never USD truth. Output remains inference-only.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from mr_liu.place.anyplace_runtime import AnyPlaceRuntime
from mr_liu.place.geometry import polygon_mask,scene_cloud
from mr_liu.grasp.contracts import RGBDObservation,CameraIntrinsics
from mr_liu.place.appearance import appearance_matches
from mr_liu.grasp.geometry import register_object_translation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parent-region',action='store_true')
    parser.add_argument('--init-current-orientation',action='store_true')
    parser.add_argument('--fuse-observed-child',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=sorted((args.run/'model_requests').glob('*.json'))
    if len(paths)!=1:raise ValueError('Select a run with exactly one recorded request')
    request=json.loads(paths[0].read_text())['request']
    parent=np.array(request['parent']);child=np.array(request['child'])
    fusion={}
    if args.parent_region or args.fuse_observed_child:
        trace=json.loads((args.run/'place/trace.json').read_text(encoding='utf-8'))
        locator=next(e['response'] for e in trace if e.get('event')=='florence_destination')
        path=args.run/'place'/f"overhead_camera_{request['sequence']:04d}.npz"
        with np.load(path,allow_pickle=False) as data:
            depth=data['depth_m'];K=data['K'];h,w=depth.shape
            obs=RGBDObservation(int(data['sequence']),float(data['timestamp_s']),data['rgb'],depth,
                CameraIntrinsics(w,h,K[0,0],K[1,1],K[0,2],K[1,2]),data['T_base_camera'],
                metadata={'robot_self_mask':data['robot_self_mask']})
        mask=polygon_mask(locator['boxes'][0],depth.shape)
        points,rows,cols=scene_cloud(obs)
        if args.parent_region:parent=points[mask[rows,cols]]
        if args.fuse_observed_child:
            bootstrap=next(e for e in trace if e['phase']=='place_bootstrap')
            center=np.asarray(bootstrap['expected_pose'])[:3,3]
            half=np.asarray(bootstrap['half_extents_m'])
            with np.load(args.run/'initial_rgbd.npz',allow_pickle=False) as initial:
                original_rgb=initial['rgb']
            mask0=cv2.imread(str(args.run/'initial_mask.png'),0)>0
            color=original_rgb[mask0].astype(float);color/=np.maximum(color.sum(1,keepdims=True),12)
            chosen=(np.linalg.norm(points-center,axis=1)<np.linalg.norm(half)+.025)
            chosen &= appearance_matches(obs.rgb[rows,cols],np.median(color,axis=0))
            visible=points[chosen]
            registration=register_object_translation(child,visible,min_inliers=24,max_correspondence_m=.02,
                                                     initial_translation_m=np.zeros(3))
            if not registration.reliable or registration.residual_m>.004:
                raise ValueError('Cannot register partial wrist and external observed object clouds')
            # Offline comparison: explicitly record alignment of the old cloud
            # to the new observation. No robot command or synthetic completion.
            fusion={'translation_m':registration.translation_base_m.tolist(),
                    'residual_m':registration.residual_m,'new_observed_points':len(visible)}
            child=np.vstack([child+registration.translation_base_m,visible])
            keys=np.floor(child/.0008).astype(np.int64)
            _,indices=np.unique(keys,axis=0,return_index=True);child=child[np.sort(indices)]
    runtime=AnyPlaceRuntime(ROOT/'_models/anyplace/anyplace_ckpts/anyplace_multitask/model.pth')
    result=runtime.infer(parent,child,init_current_orientation=args.init_current_orientation)
    transformed=child[None]@np.array(result['transforms'])[:,:3,:3].transpose(0,2,1)+np.array(result['transforms'])[:,None,:3,3]
    result.update(parent_mode='destination_segmentation' if args.parent_region else 'recorded_context',
        observed_child_fusion=fusion,
        source_request=str(paths[0].resolve()),observed_child_final_z_range=np.stack([
            transformed[:,:,2].min(1),transformed[:,:,2].max(1)],1).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='transforms'},indent=2))


if __name__=='__main__':main()
