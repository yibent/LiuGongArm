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
from .payload_mask import Sam2PayloadRefiner
from .recording import AsyncRGBDRecorder


def spawn_place_fixture(kind="region", *, x=.25, y=-.198, clutter=0):
    """Scene construction only; these coordinates never enter PlaceRequest."""
    from .fixtures import fixture_boxes
    parts = fixture_boxes(kind, x=x, y=y, clutter=clutter)
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
    for part in parts:
        box(part.name, part.position, part.size, part.color)
    return paths


def run_place_after_grasp(**kwargs):
    writer=AsyncRGBDRecorder(kwargs['output'])
    result=None
    try:
        result=_run_place_after_grasp(**kwargs,observation_recorder=writer)
        return result
    finally:
        errors=writer.close()
        if errors:
            print('[FinePlace] recording errors: '+repr(errors),flush=True)
            # Debug I/O errors do not rewrite the physical released state.
            if result is not None:result.metrics['recording_errors']=errors


def _run_place_after_grasp(*, result, initial_observation, initial_mask, target, camera, scene,
                          executor, gripper, port, label, relation, output, trace,
                          stable_base_asserted=False, grasp_attachment_ee=None, graspgenx_port=5556,
                          observation_recorder, place_backend='geometric',anyplace_url='http://127.0.0.1:5590',
                          settings=None, m2t2_url='http://127.0.0.1:5580',
                          m2t2_surface_range_m=.02, m2t2_scene_radius_m=.2,
                          anyplace_init_current_orientation=False):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]
    def event(data):
        records.append(data)
        trace(data)
        print("[FinePlace] "+json.dumps(data,ensure_ascii=False),flush=True)
        # Per-event evidence is already durably appended by trace() to the
        # shared debug/trace.jsonl. Rewriting a growing JSON array on every
        # safety check can itself expire the just-validated observation.
        if data.get('phase') in {'place_bootstrap','place_handoff','place_failed','place_succeeded'}:
            (output/"trace.json").write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding="utf-8")
    if not result.success or result.selected_grasp is None:
        raise PlaceError("grasp_handoff_unverified")
    if initial_mask is None or initial_mask.sum()<80:
        raise PlaceError("initial_payload_geometry_unavailable")
    from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport
    if place_backend == 'm2t2':
        from mr_liu.grasp.backends.m2t2 import M2T2Client
        mask_transport = M2T2Client(m2t2_url, timeout_s=60.)
    else:
        mask_transport = ZmqGraspGenXTransport('127.0.0.1', graspgenx_port, 60000)
    refiner=Sam2PayloadRefiner(mask_transport,trace=event,track_between_inference=True)
    # Defer SAM2 until the first path preflight, after Florence has evicted its
    # weights. On a 16 GB host simultaneous loading can exhaust Windows commit.
    # A cold first check never authorizes motion: move_checked reacquires and
    # rechecks new RGB-D afterwards, with the normal 350 ms age gate unchanged.
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
    pose[:3,3]=center
    ee=executor.robot_state().T_base_ee
    # Use measured achieved attachment pose, never the network's requested EE
    # pose. A five-axis IK solution can have a different unconstrained yaw.
    if grasp_attachment_ee is None:
        raise PlaceError("measured_attachment_pose_missing")
    pose=ee @ invert_transform(grasp_attachment_ee) @ pose
    held=HeldObject(target.object_id,invert_transform(ee)@pose,half,ref,anchor,time.monotonic(),stable_base_asserted)
    # The first handoff uses translation ICP because the object is still in the
    # measured pickup pose and the wrist view can expose only one face.  Once
    # the attachment/reference cloud is established, AnyPlace enables bounded
    # 6D registration for every subsequent view and release check.
    perception=RGBDPlacePerception(
        scene,camera,FlorencePlaceLocator(port),trace=event,
        recorder=observation_recorder,track_6d=False
    )
    event({"phase":"place_bootstrap","expected_pose":pose.tolist(),"half_extents_m":half.tolist(),
           "source_center_m":center.tolist(),"support_z_m":support,
           "measured_attachment_T_base_ee":np.asarray(grasp_attachment_ee).tolist()})
    measured=perception.payload(held,pose)
    # Keep an observed wrist cloud in the object frame.  During transport a
    # Panda hand can hide the top face from the wrist camera while the fixed
    # placement camera sees a side face; registering against both measured
    # subsets avoids treating that viewpoint change as slip.  Only depth/
    # appearance-associated points within the measured envelope are retained.
    measured_object = transform_points(invert_transform(measured.T_base_object), measured.points_base)
    measured_object = measured_object[
        np.all(np.abs(measured_object) <= held.half_extents_m + .004, axis=1)
    ]
    held=replace(held,T_ee_object=invert_transform(executor.robot_state().T_base_ee)@measured.T_base_object,
                 reference_points_object=np.vstack((held.reference_points_object, measured_object)),
                 verified_at_s=time.monotonic())
    # AnyPlace's diffusion input requires >=1024 real child points. Acquire
    # fresh wrist and fixed-placement views after handoff and register each
    # measured cloud into the same object frame. No interpolation or geometry
    # padding is used; insufficient multi-view evidence remains a hard refusal.
    if place_backend == 'anyplace':
        perception.track_6d = True
        fused=[held.reference_points_object]
        for view_camera in (perception.wrist, perception.scene):
            try:
                view_obs=perception.capture(view_camera)
                view_expected=executor.robot_state().T_base_ee @ held.T_ee_object
                view_payload=perception.payload(held,view_expected,observation=view_obs)
                if np.linalg.norm(view_payload.T_base_object[:3,3]-view_expected[:3,3]) > .008:
                    raise PlaceError('payload_slipped')
            except PlaceError as exc:
                event({'phase':'place_multiview_payload_rejected',
                       'camera':getattr(getattr(view_camera,'camera',None),'which','unknown'),
                       'failure':exc.code})
                continue
            points_object=transform_points(invert_transform(view_payload.T_base_object),view_payload.points_base)
            points_object=points_object[np.all(np.abs(points_object)<=held.half_extents_m+.004,axis=1)]
            if len(points_object):
                fused.append(points_object)
                event({'phase':'place_multiview_payload',
                       'camera':view_payload.observation.camera_frame,
                       'points':int(len(points_object))})
        merged=np.vstack(fused)
        # Deterministic voxel thinning removes duplicate pixels from repeated
        # views without inventing samples or changing the measured envelope.
        keys=np.floor(merged/.0008).astype(np.int64)
        _,unique=np.unique(keys,axis=0,return_index=True)
        merged=merged[np.sort(unique)]
        held=replace(held,reference_points_object=merged,verified_at_s=time.monotonic())
        event({'phase':'place_multiview_fusion','points':int(len(merged)),
               'minimum_required':1024,'sufficient':bool(len(merged)>=1024)})
    request=PlaceRequest(label,relation,request_id="isaac-fine-place")
    motion=IsaacPlaceMotion(executor,gripper,refresh_destination=lambda:perception.destination(request),trace=event,
                           mask_refiner=refiner)
    event({"phase":"place_handoff","half_extents_m":half.tolist(),"T_ee_object":held.T_ee_object.tolist(),
           "support_z_from_rgbd_m":support,"stable_base_asserted_by_fixture":stable_base_asserted,
           "payload_tracking":"local_6d_rigid_registration" if place_backend == 'anyplace' else "translation_registration",
           "orientation_control":getattr(executor,'orientation_mode','unknown')})
    backend=None
    if place_backend=='anyplace':
        from .anyplace import AnyPlaceBackend,AnyPlaceClient
        backend=AnyPlaceBackend(AnyPlaceClient(anyplace_url),
                              init_current_orientation=anyplace_init_current_orientation,
                              supports_full_pose=(
                                  getattr(executor,'orientation_mode',None) == 'full'
                                  and bool(getattr(executor,'tracks_orientation',False))
                              ))
    elif place_backend=='m2t2':
        from .m2t2 import M2T2PlaceBackend
        from mr_liu.grasp.backends.m2t2 import M2T2Client
        backend=M2T2PlaceBackend(M2T2Client(m2t2_url),trace=event,
                                 input_surface_range_m=m2t2_surface_range_m,
                                 scene_radius_m=m2t2_scene_radius_m)
    elif place_backend!='geometric':raise PlaceError('unknown_place_backend')
    node=GeneralPlaceNode(perception=perception,motion=motion,gripper=gripper,trace=event,backend=backend,
                          settings=settings)
    return node.execute(request,held)
