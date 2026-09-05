"""Offline SAM2 contour probe against saved real RGB-D; never moves a robot."""
from pathlib import Path
import argparse,json,sys,time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from mr_liu.place.appearance import appearance_matches
from mr_liu.place.geometry import scene_cloud
from mr_liu.grasp.contracts import RGBDObservation,CameraIntrinsics
from mr_liu.grasp.transforms import transform_points,invert_transform

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    trace=json.loads((args.run/'place/trace.json').read_text(encoding='utf-8'))
    seq=max(e['sequence'] for e in trace if e['phase']=='place_observe')
    a=np.load(args.run/f'place/overhead_camera_{seq:04d}.npz');h,w=a['depth_m'].shape;k=a['K']
    obs=RGBDObservation(seq,1,a['rgb'],a['depth_m'],CameraIntrinsics(w,h,k[0,0],k[1,1],k[0,2],k[1,2]),
        a['T_base_camera'],metadata={'robot_self_mask':a['robot_self_mask']})
    report=json.loads((args.run/'place_report.json').read_text(encoding='utf-8'))
    held=next(e for e in trace if e['phase']=='place_handoff')
    pose=np.array(report['actual_final_ee'])@held['T_ee_object'];half=np.array(held['half_extents_m'])
    initial=np.load(args.run/'initial_rgbd.npz');imask=cv2.imread(str(args.run/'initial_mask.png'),0)>0
    colors=initial['rgb'][imask].astype(float);colors/=np.maximum(colors.sum(axis=1,keepdims=True),12)
    anchor=np.median(colors,axis=0)
    pts,rows,cols=scene_cloud(obs);local=transform_points(invert_transform(pose),pts)
    choose=np.all(np.abs(local)<half+.012,axis=1)&appearance_matches(obs.rgb[rows,cols],anchor)
    yy,xx=rows[choose],cols[choose]
    box=np.array([xx.min()-2,yy.min()-2,xx.max()+2,yy.max()+2])
    pick=np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
        str(ROOT/'_models/sam2/sam2.1_hiera_tiny.pt'),device='cuda'))
    started=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(obs.rgb)
        masks,scores,_=predictor.predict(box=box,point_coords=np.array([[xx[pick],yy[pick]]]),
                                        point_labels=np.array([1]),multimask_output=True)
    selected=masks[int(np.argmax(scores))].astype(bool)
    output=args.run/f'place/sam2_probe_{seq:04d}.npz'
    np.savez_compressed(output,mask=selected,scores=scores,seed_box=box,sequence=seq)
    print(json.dumps({'output':str(output),'scores':scores.tolist(),'seed_pixels':int(choose.sum()),
        'mask_pixels':int(selected.sum()),'inference_s':time.perf_counter()-started}))

if __name__=='__main__':main()
