"""Inspect recorded release geometry without simulation or robot commands."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from mr_liu.grasp.contracts import RGBDObservation, CameraIntrinsics
from mr_liu.grasp.transforms import invert_transform, transform_points
from mr_liu.place.contracts import HeldObject
from mr_liu.place.geometry import scene_cloud
from mr_liu.place.isaac_motion import IsaacPlaceMotion, HeldSweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'place_report.json').read_text())
    trace = json.loads((args.run / 'place/trace.json').read_text(encoding='utf-8'))
    handoff = next(e for e in trace if e['phase'] == 'place_handoff')
    pose = np.array(report['actual_final_ee'])
    path = sorted((args.run / 'place').glob('overhead_camera_*.npz'))[-1]
    with np.load(path) as data:
        depth = data['depth_m']; k = data['K']; h, w = depth.shape
        obs = RGBDObservation(int(data['sequence']), float(data['timestamp_s']), data['rgb'], depth,
            CameraIntrinsics(w,h,k[0,0],k[1,1],k[0,2],k[1,2]), data['T_base_camera'],
            metadata={'robot_self_mask': data['robot_self_mask']})
    with np.load(args.run / 'initial_rgbd.npz') as initial:
        color=initial['rgb'][cv2.imread(str(args.run/'initial_mask.png'),0)>0].astype(float)
    color/=np.maximum(color.sum(1,keepdims=True),12)
    held = HeldObject('recorded',np.array(handoff['T_ee_object']),np.array(handoff['half_extents_m']),
        np.zeros((32,3)), np.median(color,axis=0), 0, True)
    executor = SimpleNamespace(clear_grasp_plan=lambda:None,
                              robot_state=lambda:SimpleNamespace(T_base_ee=pose))
    motion = IsaacPlaceMotion(executor,None)
    support=next(e['center_base_m'][2] for e in reversed(trace) if e['phase']=='place_observe')
    evidence = SimpleNamespace(observation=obs,support_z_m=support)
    points = motion._points(evidence,None)
    outside = motion._points(evidence,held)
    from scipy.spatial import cKDTree
    owned = cKDTree(outside).query(points,k=1)[0]>1e-8
    sweep = HeldSweep(None,.075,reference_pose=pose,max_distance_m=.1,contact_mask=owned)
    depth = sweep._contact_depth(points,pose)
    hits = depth>-.001
    local = transform_points(invert_transform(pose),points[hits])
    print(json.dumps({'frame':str(path),'hits':int(hits.sum()),'owned_hits':int(owned[hits].sum()),
        'depth_mm':(depth[hits]*1000).tolist(),'local_points_mm':(local*1000).tolist(),
        'held_in_ee':held.T_ee_object.tolist(),'remaining_collision_count':sweep.collision_count(points,pose)},indent=2))


if __name__=='__main__':main()
