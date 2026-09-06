"""Florence slow proposals, fresh RGB-D tracking and measured payload evidence."""
from __future__ import annotations
import base64
import time
import cv2
import numpy as np
import requests
from mr_liu.grasp.geometry import register_object_rigid, register_object_translation
from mr_liu.grasp.transforms import transform_points
from mr_liu.perception.optical_flow import OpticalFlowTracker
from .contracts import PlaceError, PayloadEvidence
from .geometry import polygon_mask, support_evidence, scene_cloud
from .appearance import appearance_matches


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
        if not response.ok:
            try:detail=response.json().get('message',response.reason)
            except ValueError:detail=response.reason
            raise PlaceError('destination_model_unavailable',str(detail)[:500])
        result = response.json()
        if result.get("protocol") != 1 or result.get("sequence") != obs.sequence:
            raise PlaceError("model_frame_mismatch")
        return result


class RGBDPlacePerception:
    def __init__(self, scene_camera, wrist_camera, locator, *, clock=time.monotonic,
                 trace=lambda e: None, recorder=None, track_6d=False):
        self.scene, self.wrist, self.locator = scene_camera, wrist_camera, locator
        self.clock, self.trace, self.recorder = clock, trace, recorder
        self.last = {}
        self.flow = OpticalFlowTracker()
        self.anchor = None
        self.reference_surface = None
        self.request_key = None
        self.latest_destination = None
        self.track_6d = bool(track_6d)

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

    def payload_at_destination(self,held,expected,destination):
        """Release-time object/support evidence from one current RGB-D frame."""
        obs=destination.observation
        if not -.02<=self.clock()-obs.timestamp_s<=.35:raise PlaceError('stale_observation')
        try:
            return self.payload(held,expected,observation=obs)
        except PlaceError as primary:
            # The fixed placement camera is authoritative for the destination,
            # but the held object can be partly hidden by the fingers during a
            # low descent.  Acquire a *new* wrist frame as bounded payload
            # evidence rather than treating a single failed registration as a
            # dropped object or reusing an old frame.  The caller still applies
            # its normal camera-sequence, age, slip, collision and release
            # checks to the returned observation.
            self.trace({'phase':'place_payload_destination_fallback',
                        'primary_failure':primary.code,
                        'destination_camera':getattr(obs,'camera_frame','unknown')})
            return self.payload(held,expected)

    def payload(self, held, expected, *, released=False, observation=None):
        # Wrist first during alignment. After release, external vision is needed
        # to establish independence from the moving wrist, not a fixed image box.
        cameras = [self.scene] if released or observation is not None else [self.wrist,self.scene]
        failures = []
        for camera in cameras:
            if camera is None:
                continue
            try:
                obs = observation if observation is not None else self.capture(camera)
                points, rows, cols = scene_cloud(obs)
                radius = np.linalg.norm(held.half_extents_m) + .025
                chosen = ((np.linalg.norm(points-expected[:3,3],axis=1)<radius)
                          & appearance_matches(obs.rgb[rows,cols],held.chromaticity))
                cloud = points[chosen]
                self.trace({"phase":"place_payload_attempt", "camera":getattr(obs,"camera_frame","unknown"),
                            "expected_center_m":expected[:3,3].tolist(), "points":int(len(cloud))})
                if len(cloud) < 32:
                    raise PlaceError("payload_not_visible")
                reference = transform_points(expected,held.reference_points_object)
                if self.track_6d:
                    registration = register_object_rigid(
                        reference, cloud, min_inliers=24,
                        max_correspondence_m=.020,
                        max_translation_m=.012,
                        max_rotation_rad=np.deg2rad(12.),
                    )
                    translation_m = registration.translation_m
                    orientation_residual_rad = registration.rotation_rad
                else:
                    registration = register_object_translation(reference, cloud, min_inliers=24,
                        max_correspondence_m=.020, initial_translation_m=np.zeros(3))
                    translation_m = float(np.linalg.norm(registration.translation_base_m))
                    orientation_residual_rad = 0.
                if not registration.reliable or registration.residual_m > .004:
                    self.trace({"phase":"place_payload_rejected", "camera":getattr(obs,"camera_frame","unknown"),
                                "points":int(len(cloud)), "residual_m":float(registration.residual_m),
                                "inlier_ratio":float(registration.inlier_ratio), "reliable":bool(registration.reliable),
                                "track_6d":self.track_6d,
                                "translation_correction_m":translation_m,
                                "rotation_correction_deg":float(np.degrees(orientation_residual_rad))})
                    raise PlaceError("payload_registration_uncertain")
                if self.track_6d:
                    # register_object_rigid returns the bounded base-frame
                    # correction that maps the expected cloud onto this fresh
                    # observation. Applying it on the left preserves the full
                    # measured object orientation rather than projecting the
                    # model target back onto the pickup orientation.
                    pose = registration.T_reference_current @ expected
                else:
                    pose = expected.copy()
                    pose[:3,3] += registration.translation_base_m
                self.trace({"phase":"place_payload_tracked", "camera":getattr(obs,"camera_frame","unknown"),
                            "track_6d":self.track_6d,
                            "translation_correction_m":translation_m,
                            "rotation_correction_deg":float(np.degrees(orientation_residual_rad)),
                            "residual_m":float(registration.residual_m),
                            "inlier_ratio":float(registration.inlier_ratio)})
                return PayloadEvidence(obs,pose,cloud,registration.residual_m,
                                       orientation_residual_rad)
            except PlaceError as exc:
                self.trace({"phase":"place_payload_camera_failed", "camera":getattr(camera,"which","unknown"),
                            "failure":exc.code})
                failures.append(exc.code)
        raise PlaceError("payload_unverified", ",".join(failures))
