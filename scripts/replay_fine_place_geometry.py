"""CPU-only replay of recorded Florence polygons and same-frame RGB-D."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import cv2
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source"))
from mr_liu.grasp.contracts import RGBDObservation,CameraIntrinsics
from mr_liu.grasp.transforms import transform_points,invert_transform
from mr_liu.place.geometry import polygon_mask,support_evidence,select_site,box_corners,scene_cloud
from mr_liu.place.isaac_motion import IsaacPlaceMotion,HeldSweep


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--collision",action="store_true",help="Recheck recorded start/goal local volumes, not full joint-path IK")
    parser.add_argument("--sam2-probe",action="store_true",help="Use the separately generated local SAM2 probe mask")
    args=parser.parse_args()
    directory=args.run/"place"
    trace=json.loads((directory/"trace.json").read_text(encoding="utf-8"))
    event=next(e for e in trace if e.get("event")=="florence_destination")
    response=event["response"]
    data=np.load(directory/f"overhead_camera_{response['sequence']:04d}.npz")
    depth=data["depth_m"];h,w=depth.shape;k=data["K"]
    obs=RGBDObservation(int(data["sequence"]),float(data["timestamp_s"]),data["rgb"],depth,
        CameraIntrinsics(w,h,k[0,0],k[1,1],k[0,2],k[1,2]),data["T_base_camera"],camera_frame="overhead_camera")
    mask=polygon_mask(response["boxes"][0],depth.shape)
    evidence=support_evidence(obs,mask)
    bootstrap=next(e for e in trace if e["phase"]=="place_bootstrap")
    pose=np.array(bootstrap["expected_pose"]);half=np.array(bootstrap["half_extents_m"])
    corners=transform_points(pose,box_corners(half))
    radius=np.max(np.abs(corners[:,:2]-pose[:2,3]),axis=0)
    result={"mask_pixels":int(mask.sum()),"support_points":len(evidence.surface_points),
            "support_z_m":evidence.support_z_m,"radius_xy_m":radius.tolist(),
            "center_m":evidence.center_base_m.tolist(),"description":response.get("description")}
    try: result["site_xy_m"]=select_site(evidence,np.r_[radius,half[2]],.013).tolist()
    except Exception as exc: result["failure"]=str(exc)
    if args.collision:
        report=json.loads((args.run/"place_report.json").read_text(encoding="utf-8"))
        handoff=next(e for e in trace if e["phase"]=="place_handoff")
        seq=max(e["sequence"] for e in trace if e["phase"]=="place_observe")
        data=np.load(directory/f"overhead_camera_{seq:04d}.npz")
        obs=RGBDObservation(seq,1,data["rgb"],data["depth_m"],obs.intrinsics,data["T_base_camera"],
                            metadata={"robot_self_mask":data["robot_self_mask"]})
        initial=np.load(args.run/"initial_rgbd.npz")
        mask=cv2.imread(str(args.run/"initial_mask.png"),0)>0
        colors=initial["rgb"][mask].astype(float)
        colors/=np.maximum(colors.sum(axis=1,keepdims=True),12)
        held=SimpleNamespace(T_ee_object=np.array(handoff["T_ee_object"]),
            half_extents_m=np.array(handoff["half_extents_m"]),chromaticity=np.median(colors,axis=0))
        ee=np.array(report["actual_final_ee"])
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=ee))
        refiner=None
        if args.sam2_probe:
            probe=np.load(directory/f'sam2_probe_{seq:04d}.npz')
            refiner=lambda observation,seed:probe['mask']
        motion=IsaacPlaceMotion(executor,None,mask_refiner=refiner)
        points=motion._points(SimpleNamespace(observation=obs,support_z_m=evidence.support_z_m),held)
        sweep=HeldSweep(held,report["gripper"]["width_m"])
        goal=np.array(next(e for e in reversed(trace) if e["phase"]=="place_path_rejected")["goal"])
        result["collision_replay"]={"external_points":len(points),
            "start_count":sweep.collision_count(points,ee),"goal_count":sweep.collision_count(points,goal),
            "note":"Recorded measured final pose approximates stopped start; not a fresh motion authorization"}
        local=transform_points(invert_transform(goal@held.T_ee_object),points)
        hit=np.all(np.abs(local)<held.half_extents_m+.001,axis=1)
        from mr_liu.grasp.contact import FingerGeometry
        fingers=transform_points(invert_transform(goal),points)
        for center,half in FingerGeometry(open_width_m=report['gripper']['width_m']).boxes():
            hit |= np.all(np.abs(fingers-center)<half+.001,axis=1)
        cam=transform_points(invert_transform(obs.T_base_camera),points[hit])
        pix=cam@obs.intrinsics.matrix.T
        pix=np.rint(pix[:,:2]/pix[:,2:]).astype(int)
        result["collision_replay"]["goal_hits_rgb"]=obs.rgb[pix[:,1],pix[:,0]][:50].tolist()
        result["collision_replay"]["goal_hits_pixel"]=pix.tolist()
        result["collision_replay"]["goal_hits_base_m"]=points[hit].tolist()
        from scipy.spatial import cKDTree
        allpts,_,_=scene_cloud(obs)
        near=np.linalg.norm(allpts-ee[:3,3],axis=1)<.18
        allpts=allpts[near]
        removed=allpts[cKDTree(points).query(allpts)[0]>1e-6]
        result["collision_replay"]["hit_nearest_excluded_m"]=cKDTree(removed).query(points[hit])[0].tolist()
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
