"""Isolated Isaac development adapter. Runtime perception receives no USD truth."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import time
import cv2
import numpy as np
from mr_liu.grasp.geometry import deproject_depth, register_object_translation
from mr_liu.grasp.transforms import invert_transform, transform_points
from .contracts import HeldObject, PlaceRequest, PlaceError
from .geometry import scene_cloud
from .perception import FlorencePlaceLocator, RGBDPlacePerception
from .isaac_motion import IsaacPlaceMotion
from .node import GeneralPlaceNode


def spawn_place_fixture(kind="region", *, x=.25, y=-.198):
    """Scene construction only; these coordinates never enter PlaceRequest."""
    from pxr import UsdGeom, UsdPhysics, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = "/World/PlaceDestination"
    paths=[]
    def box(name, position, scale, color, collision=True):
        paths.append(root+"/"+name)
        from pxr import Vt
        import trimesh
        mesh = trimesh.creation.box(extents=scale)
        prim = UsdGeom.Mesh.Define(stage,root+"/"+name)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,dtype=np.float32)))
        prim.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
        prim.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist())
        prim.CreateSubdivisionSchemeAttr("none")
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddOrientOp().Set(Gf.Quatf(1.))
        # cuMotion WorldBinding requires explicit standard transform properties,
        # even when all scale is baked into mesh vertices.
        xf.AddScaleOp().Set(Gf.Vec3f(1.,1.,1.))
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    box("Floor",(x,y,1.051),(.10,.10,.002),(.05,.12,.85))
    if kind == "tray":
        for axis in (0,1):
            for sign in (-1,1):
                pos=[x,y,1.068];pos[axis]+=sign*.05
                size=[.106,.106,.036];size[axis]=.006
                box(f"Wall{axis}{'p' if sign>0 else 'n'}",pos,size,(.04,.10,.80))
    return paths


def run_place_after_grasp(*, result, initial_observation, initial_mask, target, camera, scene,
                          executor, gripper, port, label, relation, output, trace,
                          stable_base_asserted=False):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]
    def event(data):
        records.append(data)
        trace(data)
        print("[FinePlace] "+json.dumps(data,ensure_ascii=False),flush=True)
        (output/"trace.json").write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding="utf-8")
    def capture(obs):
        tag=f"{obs.camera_frame}_{obs.sequence:04d}"
        cv2.imwrite(str(output/(tag+".png")),obs.rgb[:,:,::-1])
        np.savez_compressed(output/(tag+".npz"),rgb=obs.rgb,depth_m=obs.depth_m,
            K=obs.intrinsics.matrix,T_base_camera=obs.T_base_camera,
            T_base_ee=obs.T_base_ee,T_ee_camera=obs.T_ee_camera,sequence=obs.sequence,timestamp_s=obs.timestamp_s)
    if not result.success or result.selected_grasp is None:
        raise PlaceError("grasp_handoff_unverified")
    if initial_mask is None or initial_mask.sum()<80:
        raise PlaceError("initial_payload_geometry_unavailable")
    mask=initial_mask & np.isfinite(initial_observation.depth_m) & (initial_observation.depth_m>0)
    original=transform_points(initial_observation.T_base_camera,
        deproject_depth(initial_observation.depth_m,initial_observation.intrinsics,mask))
    # Estimate the actual support from surrounding RGB-D, not CASE dimensions/Z.
    all_points,_,_=scene_cloud(initial_observation)
    near=np.linalg.norm(all_points[:,:2]-np.median(original[:,:2],axis=0),axis=1)<.09
    heights=all_points[near,2]
    levels,counts=np.unique(np.rint(heights/.003).astype(int),return_counts=True)
    support_candidates=levels[counts>=30]*.003
    top=float(np.percentile(original[:,2],98))
    support_candidates=support_candidates[(support_candidates<top-.012)&(support_candidates>top-.065)]
    if len(support_candidates)==0:
        raise PlaceError("payload_support_geometry_unavailable")
    support=float(np.min(support_candidates))
    lo,hi=np.percentile(original,[1,99],axis=0)
    lo[2]=support
    center=(lo+hi)*.5
    half=(hi-lo)*.5
    color=initial_observation.rgb[mask].astype(float)
    color/=np.maximum(color.sum(axis=1,keepdims=True),12)
    anchor=np.median(color,axis=0)
    ref=original-center
    pose=np.eye(4)
    ee=executor.robot_state().T_base_ee
    grasp_ee=executor.ee_pose_for_grasp(result.selected_grasp.T_base_grasp,result.selected_grasp.width_m)
    pose[:3,3]=center+ee[:3,3]-grasp_ee[:3,3]
    held=HeldObject(target.object_id,invert_transform(ee)@pose,half,ref,anchor,time.monotonic(),stable_base_asserted)
    perception=RGBDPlacePerception(scene,camera,FlorencePlaceLocator(port),trace=event,recorder=capture)
    measured=perception.payload(held,pose)
    held=replace(held,T_ee_object=invert_transform(executor.robot_state().T_base_ee)@measured.T_base_object,
                 verified_at_s=time.monotonic())
    request=PlaceRequest(label,relation,request_id="isaac-fine-place")
    motion=IsaacPlaceMotion(executor,gripper,refresh_destination=lambda:perception.destination(request),trace=event)
    event({"phase":"place_handoff","half_extents_m":half.tolist(),"T_ee_object":held.T_ee_object.tolist(),
           "support_z_from_rgbd_m":support,"stable_base_asserted_by_fixture":stable_base_asserted,
           "limitations":"compact stable-base fixture, translation registration; not arbitrary object rotation tracking"})
    node=GeneralPlaceNode(perception=perception,motion=motion,gripper=gripper,trace=event)
    return node.execute(request,held)
