"""Florence slow proposals, fresh RGB-D tracking and measured payload evidence."""
from __future__ import annotations
import base64
import time
import cv2
import numpy as np
import requests
from mr_liu.grasp.geometry import register_object_translation
from mr_liu.grasp.transforms import transform_points
from mr_liu.perception.optical_flow import OpticalFlowTracker
from .contracts import PlaceError, PayloadEvidence
from .geometry import polygon_mask, support_evidence, scene_cloud


class FlorencePlaceLocator:
    def __init__(self, port=5570, timeout_s=90., describe=True):
        self.url = f"http://127.0.0.1:{int(port)}/locate"
        self.timeout_s, self.describe = timeout_s, describe
        self.session = requests.Session()
        self.session.trust_env = False

    def locate(self, obs, label):
        rgb = obs.rgb.copy()
        robot = obs.metadata.get("robot_self_mask")
        if robot is not None:
            rgb[np.asarray(robot,bool)] = 0
        ok, jpeg = cv2.imencode(".jpg", rgb[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise PlaceError("image_encode_failed")
        response = self.session.post(self.url, json={"sequence":obs.sequence,"label":label,
            "mode":"florence", "segment":True, "describe":self.describe,
            "jpeg":base64.b64encode(jpeg).decode()}, timeout=self.timeout_s)
        response.raise_for_status()
        result = response.json()
        if result.get("protocol") != 1 or result.get("sequence") != obs.sequence:
            raise PlaceError("model_frame_mismatch")
        return result


class RGBDPlacePerception:
    def __init__(self, scene_camera, wrist_camera, locator, *, clock=time.monotonic,
                 trace=lambda e: None, recorder=None):
        self.scene, self.wrist, self.locator = scene_camera, wrist_camera, locator
        self.clock, self.trace, self.recorder = clock, trace, recorder
        self.last = {}
        self.flow = OpticalFlowTracker()
        self.anchor = None
        self.reference_surface = None
        self.request_key = None
        self.latest_destination = None

    def capture(self, camera):
        obs = camera.capture(None)
        if obs is None:
            raise PlaceError("camera_unavailable")
        key = id(camera)
        if obs.sequence <= self.last.get(key,-1) or not -.02 <= self.clock()-obs.timestamp_s <= .35:
            raise PlaceError("stale_observation")
        self.last[key] = obs.sequence
        if self.recorder:
            self.recorder(obs)
        return obs

    def destination(self, request):
        key = (request.request_id,request.destination_label,request.relation,tuple(request.offset_base_m))
        if self.request_key != key:
            obs = self.capture(self.scene)
            result = self.locator.locate(obs, request.destination_label)
            self.trace({"phase":"place_find", "event":"florence_destination", "response":result})
            boxes = result.get("boxes", [])
            # Ambiguity is a decision, not a reason to silently choose the largest.
            if len(boxes) != 1:
                raise PlaceError("destination_not_found" if not boxes else "ambiguous_destination")
            mask = polygon_mask(boxes[0], obs.depth_m.shape)
            evidence = support_evidence(obs, mask, relation=request.relation, offset=request.offset_base_m)
            if not self.flow.initialize(obs.rgb, mask):
                raise PlaceError("destination_tracking_unavailable")
            self.anchor = evidence.center_base_m.copy()
            self.reference_surface = evidence.surface_points.copy()
            self.description = result.get("description")
            self.request_key = key
            # Always reobserve after slow inference; its input is not a move token.
        obs = self.capture(self.scene)
        mask = self.flow.update(obs.rgb)
        if mask is None:
            raise PlaceError("destination_tracking_lost")
        evidence = support_evidence(obs, mask, relation=request.relation, offset=request.offset_base_m)
        shift = np.linalg.norm(evidence.center_base_m - self.anchor)
        self.trace({"phase":"place_observe", "sequence":obs.sequence,
                    "center_base_m":evidence.center_base_m.tolist(), "shift_m":float(shift)})
        if shift > .012:
            # Stop and require a new request to generate a plan, not old trajectory.
            raise PlaceError("destination_moved")
        # Previously measured support is usable only with the same fixed-camera
        # identity and current visible support consistency. Never fill new holes.
        if abs(evidence.support_z_m-self.anchor[2]) > .003:
            raise PlaceError("support_changed")
        evidence.surface_points = np.vstack((evidence.surface_points,self.reference_surface))
        evidence.center_base_m = self.anchor.copy()
        evidence.description = self.description
        self.latest_destination = evidence
        return evidence

    def payload(self, held, expected, *, released=False):
        # Wrist first during alignment. After release, external vision is needed
        # to establish independence from the moving wrist, not a fixed image box.
        cameras = [self.scene] if released else [self.wrist,self.scene]
        failures = []
        for camera in cameras:
            if camera is None:
                continue
            try:
                obs = self.capture(camera)
                points, rows, cols = scene_cloud(obs)
                color = obs.rgb[rows,cols].astype(float)
                color /= np.maximum(color.sum(axis=1,keepdims=True),12)
                radius = np.linalg.norm(held.half_extents_m) + .025
                chosen = ((np.linalg.norm(points-expected[:3,3],axis=1)<radius)
                          & (np.linalg.norm(color-held.chromaticity,axis=1)<.13))
                cloud = points[chosen]
                if len(cloud) < 32:
                    raise PlaceError("payload_not_visible")
                reference = transform_points(expected,held.reference_points_object)
                registration = register_object_translation(reference, cloud, min_inliers=24,
                    max_correspondence_m=.020, initial_translation_m=np.zeros(3))
                if not registration.reliable or registration.residual_m > .004:
                    raise PlaceError("payload_registration_uncertain")
                pose = expected.copy()
                pose[:3,3] += registration.translation_base_m
                # Translation tracking is deliberately restricted to stable,
                # approximately upright payloads. It does not estimate 6D rotation.
                return PayloadEvidence(obs,pose,cloud,registration.residual_m)
            except PlaceError as exc:
                failures.append(exc.code)
        raise PlaceError("payload_unverified", ",".join(failures))
