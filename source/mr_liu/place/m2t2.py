"""M2T2 placement proposals within the existing verified upright envelope.

V1 deliberately rejects rotations the translation-only placement servo cannot
execute. The model is never treated as a collision/stability certificate.
"""
import numpy as np
from mr_liu.grasp.backends.m2t2 import M2T2Client
from mr_liu.grasp.transforms import assert_transform,transform_points
from .contracts import PlaceError
from .geometry import box_corners,footprint_supported


class M2T2PlaceBackend:
    name='m2t2'

    def __init__(self,client=None,trace=lambda e:None,*,scene_radius_m=.4,input_surface_range_m=0.):
        self.client=client or M2T2Client()
        self.trace=trace
        self.last_metrics={}
        if not .15 <= scene_radius_m <= .5 or not 0 <= input_surface_range_m <= .02:
            raise ValueError('Invalid placement preprocessing envelope')
        self.scene_radius_m=scene_radius_m
        self.input_surface_range_m=input_surface_range_m

    def select(self,request,held,evidence,ee_pose,margin):
        pose=ee_pose@held.T_ee_object
        obj=transform_points(pose,held.reference_points_object)
        scene=evidence.scene_points
        near=np.linalg.norm(scene-evidence.center_base_m,axis=1)<self.scene_radius_m
        scene=scene[near]
        if len(scene)>32768:scene=scene[np.linspace(0,len(scene)-1,32768).astype(int)]
        corners=transform_points(pose,box_corners(held.half_extents_m))
        bottom=pose[:3,3].copy();bottom[2]=corners[:,2].min()
        extent=np.full(2,np.linalg.norm(held.half_extents_m[:2])
                       +held.half_extents_m[2]*np.sin(np.deg2rad(5.)))
        center_bounds=np.stack((evidence.surface_points.min(0),evidence.surface_points.max(0)))
        # Conservative broad phase must contain footprint_supported's 6 mm
        # sampling-neighbour tolerance. The exact dense check below is final.
        center_bounds[0,:2]+=extent+margin-.006
        center_bounds[1,:2]-=extent+margin-.006
        if np.any(center_bounds[0]>center_bounds[1]):
            raise PlaceError('no_supported_placement_region','Payload yaw envelope exceeds observed destination')
        try:
            response=self.client.infer(task='place',scene_points=scene.tolist(),object_points=obj.tolist(),
                ee_pose=ee_pose.tolist(),bottom_center=bottom.tolist(),sequence=evidence.observation.sequence,
                input_surface_range_m=self.input_surface_range_m,support_z_m=evidence.support_z_m,
                placement_bounds=center_bounds.tolist())
        except Exception as exc:
            raise PlaceError('placement_model_unavailable',str(exc)) from exc
        if response.get('sequence')!=evidence.observation.sequence:
            raise PlaceError('placement_model_sequence_mismatch')
        rejected={'rotation':0,'support':0,'relative':0,'invalid':0}
        self.last_metrics={'place_backend':'m2t2','model_candidates':len(response.get('placements',[])),
            'model_seed':response.get('seed'),'model_provenance':response.get('model_provenance'),
            'raw_model_candidates':response.get('raw_candidates'),
            'region_model_candidates':response.get('region_candidates'),
            'destination_sampled_points':response.get('destination_sampled_points'),
            'destination_max_confidence_by_yaw':response.get('destination_max_confidence_by_yaw'),
            'object_fps_valid_points':response.get('object_fps_valid_points'),
            'input_surface_range_m':self.input_surface_range_m,'scene_radius_m':self.scene_radius_m}
        self.last_metrics['proposal_center_bounds_m']=center_bounds.tolist()
        for item in sorted(response.get('placements',[]),key=lambda x:x['score'],reverse=True):
            if not np.isfinite(item['score']) or not 0 <= item['score'] <= 1:
                rejected['invalid']+=1;continue
            goal=np.asarray(item['pose'],float)
            assert_transform(goal)
            cosine=(np.trace(goal[:3,:3]@ee_pose[:3,:3].T)-1)/2
            if np.arccos(np.clip(cosine,-1,1))>np.deg2rad(1.):
                rejected['rotation']+=1;continue
            placed=goal@held.T_ee_object
            xy=placed[:2,3]
            if request.relation=='relative' and np.linalg.norm(xy-evidence.center_base_m[:2])>.003:
                rejected['relative']+=1;continue
            if not footprint_supported(evidence,xy,extent,margin):
                rejected['support']+=1;continue
            self.last_metrics={**self.last_metrics,'model_score':float(item['score']),
                'model_latency_ms':response.get('inference_ms'),'model_ops':response.get('ops'),
                'yaw_bin':item['yaw_bin'],'model_candidates':len(response['placements']),
                'raw_model_candidates':response.get('raw_candidates'),
                'region_model_candidates':response.get('region_candidates'),
                'rejected':rejected,'rotation_mode':'preserve_verified_upright_pose'}
            self.trace({'phase':'place_model_proposal',**self.last_metrics,'xy':xy.tolist()})
            return xy.copy()
        self.trace({'phase':'place_model_rejected','rejected':rejected,**self.last_metrics})
        raise PlaceError('no_model_placement_in_envelope',str(rejected))
