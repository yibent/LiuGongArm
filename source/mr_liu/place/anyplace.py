"""Torch-free AnyPlace backend, constrained by the current placement controller.

The V1 controller cannot execute arbitrary object rotations. Such predictions
are explicitly rejected, not projected onto the table or replaced geometrically.
"""
import json
import urllib.request
from collections import Counter
import numpy as np
from mr_liu.grasp.transforms import assert_transform, invert_transform, transform_points
from .contracts import PlaceError
from .geometry import box_corners, footprint_supported, scene_cloud


class AnyPlaceClient:
    def __init__(self,url='http://127.0.0.1:5590',timeout_s=120):
        self.url=url.rstrip('/');self.timeout_s=timeout_s
    def infer(self,**request):
        data=json.dumps({'protocol':1,**request},allow_nan=False).encode()
        message=urllib.request.Request(self.url+'/infer',data,{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(message,timeout=self.timeout_s) as response:return json.load(response)
        except Exception as exc:
            detail=exc.read().decode(errors='replace') if hasattr(exc,'read') else str(exc)
            raise PlaceError('placement_model_unavailable',detail) from exc


class AnyPlaceBackend:
    name='anyplace'
    def __init__(self,client=None):
        self.client=client or AnyPlaceClient();self.last_metrics={}

    def parent_cloud(self,evidence):
        points,rows,cols=scene_cloud(evidence.observation)
        return points[evidence.mask[rows,cols]]

    def select(self,request,held,evidence,ee_pose,margin):
        pose=ee_pose@held.T_ee_object
        parent=self.parent_cloud(evidence)
        # AnyPlace consumes an object pair, not the surrounding grasp scene.
        # Segmentation affects only model features; full collision data remains.
        keep=np.ones(len(parent),bool)
        local=transform_points(invert_transform(pose),parent)
        keep &= ~np.all(np.abs(local)<held.half_extents_m+.008,axis=1)
        parent=parent[keep]
        child=transform_points(pose,held.reference_points_object)
        if min(len(parent),len(child))<1024:
            raise PlaceError('anyplace_insufficient_observed_geometry',f'parent={len(parent)},child={len(child)}; need multiview acquisition')
        parent=parent[np.linspace(0,len(parent)-1,min(len(parent),32768),dtype=int)]
        child=child[np.linspace(0,len(child)-1,min(len(child),8192),dtype=int)]
        self.last_metrics={'place_backend':'anyplace','input_geometry':'partial_rgbd_not_certified_complete',
                           'parent_points':len(parent),'child_points':len(child),
                           'parent_context':'destination_segmentation'}
        result=self.client.infer(sequence=evidence.observation.sequence,parent=parent.tolist(),child=child.tolist(),
            input_geometry='partial_rgbd_not_certified_complete')
        if result.get('protocol')!=1 or result.get('sequence')!=evidence.observation.sequence:
            raise PlaceError('placement_observation_sequence_mismatch')
        self.last_metrics.update(inference_s=result.get('inference_s'),checkpoint_sha256=result.get('checkpoint_sha256'))
        rejected=Counter();options=[];diagnostics=[]
        for transform in result.get('transforms',[]):
            try:
                relative=np.asarray(transform,float);assert_transform(relative)
            except (ValueError,TypeError):
                rejected['invalid_transform']+=1;continue
            final=relative@pose
            angle=np.arccos(np.clip((np.trace(final[:3,:3]@pose[:3,:3].T)-1)/2,-1,1))
            violations=[]
            if angle>np.deg2rad(1):
                violations.append('rotation_requires_pose_controller')
            corners=transform_points(final,box_corners(held.half_extents_m))
            gap=corners[:,2].min()-evidence.support_z_m
            if not -.003<=gap<=.012:
                violations.append('model_height_not_supported')
            xy=final[:2,3]
            radius=np.max(np.abs(corners[:,:2]-xy),axis=0)
            if not footprint_supported(evidence,xy,radius,margin):
                violations.append('unsupported_footprint')
            if request.relation=='relative' and np.linalg.norm(xy-evidence.center_base_m[:2])>.003:
                violations.append('requested_relative_offset')
            diagnostics.append({'rotation_change_deg':float(np.degrees(angle)),
                                'predicted_bottom_gap_m':float(gap),'violations':violations})
            if violations:
                rejected.update(violations);continue
            options.append((np.linalg.norm(xy-evidence.center_base_m[:2]),xy,final))
        self.last_metrics.update(raw_candidates=len(result.get('transforms',[])),rejected=dict(rejected),
                                 feasible_candidates=len(options),candidate_diagnostics=diagnostics,
                                 rejection_counts_are_nonexclusive=True)
        if not options:raise PlaceError('no_executable_anyplace_candidate',str(dict(rejected)))
        _,xy,chosen=min(options,key=lambda row:row[0])
        self.last_metrics['selected_object_pose']=chosen.tolist()
        self.last_metrics['orientation_tolerance_deg']=1.
        return xy.copy()
