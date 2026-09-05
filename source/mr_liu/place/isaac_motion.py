"""Isaac placement adapter: actual joint path + payload/jaw swept volume.

V1 is restricted to compact payloads in the calibrated fingertip corridor.
Long tools/full attached-body self-collision are deliberately not advertised.
"""
import time
from dataclasses import replace
import cv2
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.contact import FingerGeometry
from mr_liu.grasp.transforms import invert_transform, transform_points
from .geometry import box_corners, scene_cloud
from .appearance import appearance_matches


class HeldSweep:
    def __init__(self, held, width, *, opening=False, reference_pose=None,max_distance_m=None,contact_mask=None,
                 contact_uncertainty_m=.001):
        self.held = held
        self.width = width
        self.opening = opening
        self.reference_pose,self.max_distance_m=reference_pose,max_distance_m
        self._tree_points=None;self._last_pose=None;self._last_count=0
        self.contact_mask=contact_mask
        if not 0 < contact_uncertainty_m <= .003:
            raise ValueError('Contact uncertainty must be in (0,3 mm]')
        self.contact_uncertainty_m=contact_uncertainty_m

    def _contact_depth(self,points,pose):
        # pose is validated FK; direct rigid inverse avoids repeatedly running
        # transform validation for every point subset at every path sample.
        local=(points-pose[:3,3])@pose[:3,:3]
        depths=[np.min(half-np.abs(local-center),axis=1)
                for center,half in FingerGeometry(open_width_m=self.width).boxes()]
        return np.maximum.reduce(depths)

    def collision_count(self, points, pose):
        if self.reference_pose is not None:
            from .contracts import PlaceError
            if np.linalg.norm(pose[:3,3]-self.reference_pose[:3,3])>self.max_distance_m:
                raise PlaceError('nonlocal_ik_path')
            cosine=(np.trace(pose[:3,:3]@self.reference_pose[:3,:3].T)-1)/2
            if np.arccos(np.clip(cosine,-1,1))>np.deg2rad(10):
                raise PlaceError('nonlocal_ik_rotation')
        if self._tree_points is not points:
            self._tree=cKDTree(points);self._tree_points=points;self._last_pose=None
            self._contact_initial=None
            if self.contact_mask is not None:
                initial=self._contact_depth(points,self.reference_pose)
                # Only previously measured payload contact within the declared
                # geometry uncertainty. No whole-object or new-contact exemption.
                self._contact_initial=initial
                self._contact_allowed=(np.asarray(self.contact_mask,bool)&(initial>=-.001)
                                       &(initial<=self.contact_uncertainty_m))
        if self._last_pose is not None and np.array_equal(pose,self._last_pose):
            return self._last_count
        widths = np.unique(np.linspace(self.width,.075,12)) if self.opening else [self.width]
        # Exact conservative broad phase: each sphere contains its entire box,
        # including the original margin. Keep the unchanged narrow-phase test.
        centers,radii=[],[]
        for width in widths:
            finger=FingerGeometry(open_width_m=float(width))
            for center,half in finger.boxes():
                centers.append(pose[:3,:3]@center+pose[:3,3])
                radii.append(np.linalg.norm(half+finger.margin_m)+1e-9)
        if self.held is not None:
            centers.append((pose@self.held.T_ee_object)[:3,3])
            radii.append(np.linalg.norm(self.held.half_extents_m+.001)+1e-9)
        hits=self._tree.query_ball_point(centers,radii)
        indices=np.unique(np.concatenate(hits)).astype(int) if any(len(v) for v in hits) else np.empty(0,int)
        if self._contact_initial is not None and len(indices):
            depth=self._contact_depth(points[indices],pose)
            escaping=(self._contact_allowed[indices]
                      & (depth<=self._contact_initial[indices]+.0002)&(depth<=self.contact_uncertainty_m))
            indices=indices[~escaping]
        nearby=points[indices]
        local=(nearby-pose[:3,3])@pose[:3,:3]
        count=0
        for width in widths:
            finger=FingerGeometry(open_width_m=float(width))
            occupied=np.zeros(len(local),bool)
            for center,half in finger.boxes():
                occupied |= np.all(np.abs(local-center)<half+finger.margin_m,axis=1)
            count=max(count,int(occupied.sum()))
        if self.held is not None:
            object_pose=pose@self.held.T_ee_object
            local = (nearby-object_pose[:3,3])@object_pose[:3,:3]
            count += int(np.all(np.abs(local)<self.held.half_extents_m+.001,axis=1).sum())
        self._last_pose=pose.copy();self._last_count=count
        return count

    def path_collision_count(self, points, start, end):
        # Caller samples the actual FK path densely (<= .005 rad); do not
        # substitute a straight Cartesian path for a curved five-axis path.
        return max(self.collision_count(points,start),self.collision_count(points,end))

    def contact_diagnostic(self, points, pose):
        depth = self._contact_depth(points,pose)
        indices = np.flatnonzero(depth > -.001)
        owned = np.zeros(len(points),bool) if self.contact_mask is None else self.contact_mask
        chosen = indices[:32]
        return {'raw_hits':len(indices),'owned_hits':int(np.asarray(owned)[indices].sum()),
                'points_base':points[chosen].tolist(),'depth_m':depth[chosen].tolist(),
                'initial_depth_m':self._contact_depth(points[chosen],self.reference_pose).tolist(),
                'pose':pose.tolist(),'reference_pose':self.reference_pose.tolist(),
                'contact_uncertainty_m':self.contact_uncertainty_m}


class IsaacPlaceMotion:
    def __init__(self, executor, gripper, *, refresh_destination=None, trace=lambda e:None, mask_refiner=None):
        self.executor, self.gripper = executor, gripper
        self.refresh_destination, self.trace = refresh_destination,trace
        self.last_failure = None
        self.mask_refiner=mask_refiner
        self.release_contact=None
        self._release_retreat=None
        self._points_observation=None
        self._points_cache={}
        self.executor.clear_grasp_plan()

    def robot_state(self):
        return self.executor.robot_state()

    def stop(self):
        self.executor.stop()

    def _points(self, evidence, held):
        current=self.robot_state().T_base_ee
        obs=evidence.observation
        if self._points_observation is not obs:
            self._points_observation=obs
            self._points_cache={}
        held_key=None if held is None else (np.asarray(held.T_ee_object).tobytes(),
            np.asarray(held.half_extents_m).tobytes(),np.asarray(held.chromaticity).tobytes())
        key=(current.tobytes(),held_key,float(evidence.support_z_m),id(self.mask_refiner))
        if key in self._points_cache:return self._points_cache[key]
        def remember(points):
            self._points_cache[key]=points
            return points
        points, rows, cols = scene_cloud(evidence.observation)
        # Full arm/world checks stay with cuMotion. The explicit payload/jaw
        # volume only needs the local 18 cm neighbourhood of this <=6 mm step.
        near=np.linalg.norm(points-current[:3,3],axis=1)<.18
        points,rows,cols=points[near],rows[near],cols[near]
        robot=getattr(evidence.observation,"metadata",{}).get("robot_self_mask")
        outside_robot_guard=np.ones(len(points),bool)
        if robot is not None:
            # One-pixel registration guard around the known robot silhouette.
            # Full calibrated robot collision remains checked by cuMotion; do
            # not interpret these uncertain self-boundary samples as external
            # objects or apply this guard to the destination support geometry.
            guard=cv2.dilate(np.asarray(robot,np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            outside_robot_guard=~guard[rows,cols]
        if held is None:
            return remember(points[outside_robot_guard])
        pose = current @ held.T_ee_object
        # A partial reference cloud cannot label newly visible object faces as
        # obstacles. Associate a unique measured appearance component in the
        # bounded held-object ROI; never exclude an entire geometric box.
        local=transform_points(invert_transform(pose),points)
        candidate=(np.all(np.abs(local)<held.half_extents_m+.012,axis=1)
                   & appearance_matches(evidence.observation.rgb[rows,cols],held.chromaticity))
        mask=np.zeros(evidence.observation.depth_m.shape,np.uint8)
        mask[rows[candidate],cols[candidate]]=1
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=12]
        if len(components)!=1:
            from .contracts import PlaceError
            raise PlaceError("held_collision_mask_uncertain")
        own=labels[rows,cols]==components[0]
        owned_mask=labels==components[0]
        if self.mask_refiner is not None:
            # Refinement adds contour evidence; it must not discard the unique
            # depth/appearance seed and turn the held object's own surface into
            # a new obstacle merely because SAM missed a few edge pixels.
            owned_mask=owned_mask | self.mask_refiner(evidence.observation,owned_mask)
            proposed=owned_mask[rows,cols]
            roi=np.all(np.abs(local)<held.half_extents_m+.012,axis=1)
            # RGB mask boundaries can project onto background depth. They must
            # remain world obstacles, not invalidate all well-associated object
            # pixels or be erased by expanding the 3-D attachment ROI.
            outside=int(np.sum(proposed&~roi))
            if outside:
                self.trace({'phase':'place_mask_depth_rejected','sequence':evidence.observation.sequence,
                            'outside_attachment_points_retained':outside})
            # A model mask never erases the observed support plane.
            own=proposed&roi&(np.abs(points[:,2]-evidence.support_z_m)>.003)
        # Registered RGB/depth still has mixed-color edge pixels (anti-aliasing
        # in Isaac, finite color/depth registration on hardware). Admit only a
        # two-pixel ring with <=5 mm measured 3D continuity, never missing depth
        # or the observed support plane. This is bounded association uncertainty.
        ring=cv2.dilate(owned_mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2).astype(bool)
        nearby=cKDTree(points[own]).query(points,k=1)[0]<.005
        edge=(ring[rows,cols]&nearby
              & np.all(np.abs(local)<held.half_extents_m+.012,axis=1)
              & (points[:,2]>evidence.support_z_m+.006))
        own |= edge
        # Form object identity before applying the robot registration guard.
        # Otherwise a valid contact-edge seed is removed first, leaving its
        # depth-continuous fringe incorrectly classified as an external object.
        # The exact robot mask was already removed by scene_cloud; the final
        # one-pixel guard is unchanged and never grows with the object mask.
        return remember(points[~own & outside_robot_guard])

    def _safe(self, goal, held, evidence, opening=False,width_override=None,contact_payload=None):
        self.last_failure = None
        started=time.monotonic()
        if held is not None:
            corners = transform_points(held.T_ee_object,box_corners(held.half_extents_m))
            # Explicit calibrated V1 envelope: reject long/back-projecting tools.
            if np.any(held.half_extents_m>.030) or corners[:,2].min()<-.050:
                self.last_failure = "payload_outside_calibrated_tool_corridor"
                return False
        width = self.gripper.state().width_m if width_override is None else width_override
        if width is None:
            self.last_failure = "gripper_width_unknown"
            return False
        points = self._points(evidence,held)
        contact_mask=None
        if contact_payload is not None:
            external=self._points(evidence,contact_payload)
            contact_mask=cKDTree(external).query(points,k=1)[0]>1e-8
        points_s=time.monotonic()-started
        current=self.robot_state().T_base_ee
        limit=max(.010,1.5*np.linalg.norm(goal[:3,3]-current[:3,3])+.002)
        sweep = HeldSweep(held,width,opening=opening,reference_pose=current,max_distance_m=limit,
                          contact_mask=contact_mask,
                          contact_uncertainty_m=(min(.003,contact_payload.uncertainty_m)
                                                 if contact_payload is not None else .001))
        if not self.executor.is_observation_path_safe(goal,points,sweep):
            self.last_failure = self.executor.last_observation_path_rejection
            self.trace({"phase":"place_path_rejected","reason":self.last_failure,
                        "ik":getattr(self.executor,"last_plan_diagnostic",{}),"goal":goal.tolist(),
                        "profile":getattr(self.executor,"last_observation_path_profile",{})})
            return False
        self.trace({'phase':'place_path_checked','points_s':points_s,'elapsed_s':time.monotonic()-started,
                    'observation_age_s':time.monotonic()-evidence.observation.timestamp_s,
                    'profile':getattr(self.executor,'last_observation_path_profile',{})})
        return True

    def opening_safe(self, held, evidence):
        pose = self.robot_state().T_base_ee
        sweep = HeldSweep(None,self.gripper.state().width_m,opening=True)
        if sweep.collision_count(self._points(evidence,held),pose):
            self.last_failure = "opening_sweep_collision"
            return False
        # Pull back along the calibrated finger axis. World-vertical withdrawal
        # can scrape a held face when the achieved wrist is slightly tilted.
        direction=-pose[:3,2]
        if direction[2]<.9:
            self.last_failure='retreat_axis_not_upward'
            return False
        # The payload can touch the side of a fixed finger, not just its inner
        # X face. Search outward relative to the measured attachment, including
        # Y clearance, and validate the actual FK path for every hypothesis.
        away=-np.asarray(held.T_ee_object[:2,3],float)
        away/=max(np.linalg.norm(away),1e-12)
        directions=[np.array([1.,0.]),away,np.array([0.,1.]),np.array([0.,-1.])]
        offsets=[np.zeros(2)]
        offsets += [v*distance for distance in (.003,.006,.010,.015,.020)
                    for v in directions if np.dot(v,away)>=0]
        candidates=[]
        if self._release_retreat is not None:
            # Preserve the exact target pose, not just its position. Rebuilding
            # it from tiny measured wrist jitter invalidates every FK sample.
            # Fresh depth/world tests and the actual-q -> cached-path join are
            # still checked; cached safety is never reused.
            candidates.append((self._release_retreat.copy(),None))
        for offset in offsets:
            retreat = pose.copy()
            retreat[:3,3] += direction*.045 + pose[:3,:2]@offset
            candidates.append((retreat,offset))
        for retreat,offset in candidates:
            ok=self._safe(retreat,None,evidence,width_override=.075,contact_payload=held)
            self.trace({'phase':'place_retreat_candidate','lateral_escape_m':None if offset is None else float(np.linalg.norm(offset)),
                        'tool_xy_escape_m':None if offset is None else offset.tolist(),
                        'cached_pose_proposal':offset is None,
                        'safe':ok,'reason':self.last_failure})
            if ok:
                self.release_contact=(held,pose.copy(),pose@held.T_ee_object)
                self._release_retreat=retreat.copy()
                return True
        return False

    def retreat_target(self):
        if self._release_retreat is None:
            from .contracts import PlaceError
            raise PlaceError('retreat_not_preflighted')
        return self._release_retreat.copy()

    def move_checked(self, goal, held, evidence, *, opening=False):
        self.executor.clear_grasp_plan()
        contact_payload=None
        if held is None and opening and self.release_contact is not None:
            original,release_ee,object_pose=self.release_contact
            current=self.robot_state().T_base_ee
            if np.linalg.norm(current[:3,3]-release_ee[:3,3])<.015:
                # The released object stays in the world; it must not be moved
                # along with the retreating wrist's attachment transform.
                contact_payload=replace(original,T_ee_object=invert_transform(current)@object_pose)
        refreshed_mask=False
        def check(observation):
            nonlocal refreshed_mask
            if self._safe(goal,held,observation,opening,contact_payload=contact_payload):return True
            refresh=getattr(self.mask_refiner,'request_model_refresh',None)
            if (held is not None and not refreshed_mask
                    and self.last_failure=='observed_surface_finger_collision'
                    and refresh is not None and refresh()):
                refreshed_mask=True;self._points_cache={}
                self.trace({'phase':'place_reobserve_after_tracked_contour_collision'})
                return None  # No motion: require new model evidence and full check.
            return False
        if check(evidence) is False:return False
        # A slow IK check cannot authorize motion against an old observation.
        if self.refresh_destination is None:
            self.last_failure = "missing_fresh_observation_provider"
            return False
        for attempt in range(3):
            fresh = self.refresh_destination()
            if np.linalg.norm(fresh.center_base_m-evidence.center_base_m)>.008:
                self.last_failure = "destination_changed_during_plan"
                return False
            checked=check(fresh)
            if checked is False:return False
            if checked is None:continue
            if time.monotonic()-fresh.observation.timestamp_s <= .35:
                break
            # Timing failure alone permits bounded reobservation, never motion
            # on stale depth or a retry of a rejected collision/changed target.
            self.trace({'phase':'place_reobserve_after_slow_check','attempt':attempt+1})
        else:
            self.last_failure = "scene_stale_after_plan"
            return False
        # A bounded step only. Scene/world/self collision checks remain active.
        ok=self.executor.move_to(goal,speed_scale=.20)
        if not ok:
            self.last_failure="checked_motion_endpoint_not_reached"
        return ok
