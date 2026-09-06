"""Torch-free client and calibrated M2T2 grasp candidate adapter."""
import json
import base64
import urllib.request
import os
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.contracts import GraspCandidate, GraspBackendError, FailureCode
from mr_liu.grasp.geometry import deproject_depth
from mr_liu.grasp.transforms import invert_transform, transform_points, assert_transform


class M2T2Client:
    def __init__(self, url='http://127.0.0.1:5580', timeout_s=45.):
        self.url, self.timeout_s = url.rstrip('/'), timeout_s

    def infer(self, **payload):
        record_path=None
        record_dir=os.environ.get('BUSAGENT_M2T2_RECORD_DIR')
        if record_dir:
            directory=Path(record_dir);directory.mkdir(parents=True,exist_ok=True)
            record_path=directory/f"{payload.get('task','unknown')}_{payload.get('sequence',0)}_{time.time_ns()}.json"
            record_path.write_text(json.dumps({'request':payload},allow_nan=False),encoding='utf-8')
        request = urllib.request.Request(self.url+'/infer',
            data=json.dumps(payload, allow_nan=False).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.load(response)
        if result.get('backend') != 'm2t2' or result.get('protocol') != 1:
            raise ValueError('Invalid M2T2 response')
        if result.get('error'):
            raise RuntimeError(result['error'])
        if record_path is not None:
            record_path.write_text(json.dumps({'request':payload,'response':result},allow_nan=False),encoding='utf-8')
        return result

    def _request(self,payload):
        """SAM2 transport adapter; no GraspGenX model needs to be loaded."""
        import cv2
        if payload.get('action')!='segment':
            raise ValueError('Only segmentation is supported by this transport')
        ok,encoded=cv2.imencode('.png',cv2.cvtColor(np.asarray(payload['rgb'],np.uint8),cv2.COLOR_RGB2BGR))
        if not ok:raise ValueError('Image encoding failed')
        body={k:v for k,v in payload.items() if k!='rgb'}
        body['rgb_png']=base64.b64encode(encoded).decode()
        request=urllib.request.Request(self.url+'/segment',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=self.timeout_s) as response:result=json.load(response)
        if result.get('mask_png'):
            result['mask']=cv2.imdecode(np.frombuffer(base64.b64decode(result.pop('mask_png')),np.uint8),0)>0
        return result


class M2T2Backend:
    name = 'm2t2'

    def __init__(self, client=None, max_candidates=96, width_margin_m=.010,
                 input_surface_range_m=0.):
        self.client = client or M2T2Client()
        self.max_candidates = max_candidates
        if not 0 <= width_margin_m <= .020:raise ValueError('Invalid width margin')
        self.width_margin_m=width_margin_m
        if not 0 <= input_surface_range_m <= .02:raise ValueError('Invalid input surface range')
        self.input_surface_range_m=input_surface_range_m

    @staticmethod
    def support_height(scene,target):
        """Observed dominant horizontal mode below target, never scene truth."""
        center=np.median(target,axis=0)
        heights=scene[np.linalg.norm(scene[:,:2]-center[:2],axis=1)<.09,2]
        top=np.percentile(target[:,2],98)
        heights=heights[(heights<top-.012)&(heights>top-.065)]
        if len(heights)<80:return None
        bins,counts=np.unique(np.rint(heights/.003).astype(int),return_counts=True)
        level=bins[np.argmax(counts)]*.003
        plane=heights[np.abs(heights-level)<.003]
        if len(plane)<80:return None
        return float(np.median(plane))

    def generate(self, observation, object_points_camera, target):
        points = np.asarray(object_points_camera, float)
        if len(points) < 32:
            return []
        valid = np.isfinite(observation.depth_m) & (observation.depth_m > .02) & (observation.depth_m < 2.)
        robot = observation.metadata.get('robot_self_mask')
        if robot is not None:
            valid &= ~np.asarray(robot, bool)
        scene = deproject_depth(observation.depth_m, observation.intrinsics, valid)
        center = np.median(points, axis=0)
        # Include surrounding support/obstacles, not only a segmented object.
        scene = scene[np.linalg.norm(scene-center, axis=1) < .35]
        if len(scene) > 32768:
            scene = scene[np.linspace(0, len(scene)-1, 32768).astype(int)]
        base_scene = transform_points(observation.T_base_camera, scene)
        base_target = transform_points(observation.T_base_camera, points)
        preprocessing={}
        if self.input_surface_range_m:
            support=self.support_height(base_scene,base_target)
            if support is not None:
                preprocessing={'support_z_m':support,'input_surface_range_m':self.input_surface_range_m}
        try:
            result = self.client.infer(task='pick', scene_points=base_scene.tolist(),
                object_points=base_target.tolist(), ee_pose=observation.T_base_ee.tolist(),
                sequence=observation.sequence,seed=int(observation.metadata.get('model_seed',0)),**preprocessing)
            if result.get('sequence') != observation.sequence:
                raise ValueError('M2T2 observation sequence mismatch')
            return self.decode(result, observation, points)
        except Exception as exc:
            raise GraspBackendError(FailureCode.MODEL_INFERENCE_FAILED, str(exc), retryable=True) from exc

    def decode(self, result, observation, object_points_camera):
        tree = cKDTree(object_points_camera)
        camera_base = invert_transform(observation.T_base_camera)
        candidates = []
        for item in result.get('grasps', []):
            pose = np.asarray(item['pose'], float)
            assert_transform(pose)
            contact = np.asarray(item['contact'], float)
            # Published model origin is 103.4 mm behind the symmetric pinch.
            pinch = pose.copy()
            pinch[:3, 3] += pose[:3, 2] * .1034
            width = 2 * float(np.dot(pinch[:3, 3]-contact, pose[:3, 0]))
            local_contact = transform_points(camera_base, contact[None])[0]
            if tree.query(local_contact)[0] > .012 or not .002 < width <= .080:
                continue
            opposite = 2*pinch[:3, 3]-contact
            score = float(item['score'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid M2T2 confidence')
            local_pinch=camera_base@pinch
            local=(object_points_camera-local_pinch[:3,3])@local_pinch[:3,:3]
            aperture=width+self.width_margin_m
            section=local[np.abs(local[:,1])<=.009]
            if len(section)>=24:
                low,high=np.percentile(section[:,0],[3,97])
                # A network width is not permission to descend with jaws
                # narrower than the observed material in the finger corridor.
                aperture=max(aperture,2*max(abs(low),abs(high))+self.width_margin_m)
            candidates.append(GraspCandidate(local_pinch, aperture, score,
                observation.sequence, self.name,
                contacts_camera=transform_points(camera_base, np.stack((contact, opposite))),
                metadata={'branch':'m2t2', 'model_gripper_depth_m':.1034,
                          'model_width_m':width,'aperture_guard_m':aperture,
                          'model_seed':result.get('seed'),'model_provenance':result.get('model_provenance'),
                          'input_surface_range_m':result.get('input_surface_range_m',0.),
                          'pose_compatibility':result.get('model_provenance',{}).get('pose_compatibility'),
                          'ops':result.get('ops'), 'inference_ms':result.get('inference_ms')}))
        # Retain distinct hypotheses rather than filling the IK budget with
        # hundreds of adjacent contact pixels. Never rotate/translate a pose.
        diverse=[]
        for candidate in sorted(candidates,key=lambda c:c.score,reverse=True):
            pose=candidate.T_camera_grasp
            if diverse:
                previous=np.stack([c.T_camera_grasp for c in diverse])
                same=(np.linalg.norm(previous[:,:3,3]-pose[:3,3],axis=1)<.005)
                same &= (previous[:,:3,2]@pose[:3,2])>np.cos(np.deg2rad(10))
                # Opposite finger assignments are not interchangeable on the
                # asymmetric SO-101 tool: they map to different EE poses/IK.
                same &= (previous[:,:3,0]@pose[:3,0])>np.cos(np.deg2rad(10))
                if same.any():continue
            diverse.append(candidate)
            if len(diverse)>=self.max_candidates:break
        return diverse
