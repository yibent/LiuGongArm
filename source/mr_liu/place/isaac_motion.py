"""Isaac placement adapter: actual joint path + payload/jaw swept volume.

V1 is restricted to compact payloads in the calibrated fingertip corridor.
Long tools/full attached-body self-collision are deliberately not advertised.
"""
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.contact import FingerGeometry
from mr_liu.grasp.transforms import invert_transform, transform_points
from .geometry import box_corners, scene_cloud
from .appearance import appearance_matches


class HeldSweep:
    def __init__(self, held, width, *, opening=False):
        self.held = held
        self.width = width
        self.opening = opening

    def collision_count(self, points, pose):
        widths = np.linspace(self.width,.075,12) if self.opening else [self.width]
        count = max(FingerGeometry(open_width_m=float(w)).collision_count(points,pose) for w in widths)
        if self.held is not None:
            local = transform_points(invert_transform(pose @ self.held.T_ee_object),points)
            count += int(np.all(np.abs(local)<self.held.half_extents_m+.001,axis=1).sum())
        return count

    def path_collision_count(self, points, start, end):
        # Caller samples the actual FK path densely (<= .005 rad); do not
        # substitute a straight Cartesian path for a curved five-axis path.
        return max(self.collision_count(points,start),self.collision_count(points,end))


class IsaacPlaceMotion:
    def __init__(self, executor, gripper, *, refresh_destination=None, trace=lambda e:None):
        self.executor, self.gripper = executor, gripper
        self.refresh_destination, self.trace = refresh_destination,trace
        self.last_failure = None
        self.executor.clear_grasp_plan()

    def robot_state(self):
        return self.executor.robot_state()

    def stop(self):
        self.executor.stop()

    def _points(self, evidence, held):
        points, rows, cols = scene_cloud(evidence.observation)
        # Full arm/world checks stay with cuMotion. The explicit payload/jaw
        # volume only needs the local 18 cm neighbourhood of this <=6 mm step.
        near=np.linalg.norm(points-self.robot_state().T_base_ee[:3,3],axis=1)<.18
        points,rows,cols=points[near],rows[near],cols[near]
        if held is None:
            return points
        pose = self.robot_state().T_base_ee @ held.T_ee_object
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
        # Registered RGB/depth still has mixed-color edge pixels (anti-aliasing
        # in Isaac, finite color/depth registration on hardware). Admit only a
        # two-pixel ring with <=5 mm measured 3D continuity, never missing depth
        # or the observed support plane. This is bounded association uncertainty.
        ring=cv2.dilate((labels==components[0]).astype(np.uint8),np.ones((3,3),np.uint8),iterations=2).astype(bool)
        nearby=cKDTree(points[own]).query(points,k=1)[0]<.005
        edge=(ring[rows,cols]&nearby
              & np.all(np.abs(local)<held.half_extents_m+.012,axis=1)
              & (points[:,2]>evidence.support_z_m+.006))
        own |= edge
        return points[~own]

    def _safe(self, goal, held, evidence, opening=False):
        self.last_failure = None
        if held is not None:
            corners = transform_points(held.T_ee_object,box_corners(held.half_extents_m))
            # Explicit calibrated V1 envelope: reject long/back-projecting tools.
            if np.any(held.half_extents_m>.030) or corners[:,2].min()<-.050:
                self.last_failure = "payload_outside_calibrated_tool_corridor"
                return False
        width = self.gripper.state().width_m
        if width is None:
            self.last_failure = "gripper_width_unknown"
            return False
        points = self._points(evidence,held)
        sweep = HeldSweep(held,width,opening=opening)
        if not self.executor.is_observation_path_safe(goal,points,sweep):
            self.last_failure = self.executor.last_observation_path_rejection
            self.trace({"phase":"place_path_rejected","reason":self.last_failure,
                        "ik":getattr(self.executor,"last_plan_diagnostic",{}),"goal":goal.tolist()})
            return False
        return True

    def opening_safe(self, held, evidence):
        pose = self.robot_state().T_base_ee
        sweep = HeldSweep(None,self.gripper.state().width_m,opening=True)
        if sweep.collision_count(self._points(evidence,held),pose):
            self.last_failure = "opening_sweep_collision"
            return False
        retreat = pose.copy()
        retreat[2,3] += .045
        # Reserve a retreat corridor before release. Revalidate it after opening.
        return self._safe(retreat,None,evidence,opening=True)

    def move_checked(self, goal, held, evidence, *, opening=False):
        self.executor.clear_grasp_plan()
        if not self._safe(goal,held,evidence,opening):
            return False
        # A slow IK check cannot authorize motion against an old observation.
        if self.refresh_destination is None:
            self.last_failure = "missing_fresh_observation_provider"
            return False
        fresh = self.refresh_destination()
        if np.linalg.norm(fresh.center_base_m-evidence.center_base_m)>.008:
            self.last_failure = "destination_changed_during_plan"
            return False
        if not self._safe(goal,held,fresh,opening):
            return False
        if time.monotonic()-fresh.observation.timestamp_s > .35:
            self.last_failure = "scene_stale_after_plan"
            return False
        # A bounded step only. Scene/world/self collision checks remain active.
        ok=self.executor.move_to(goal,speed_scale=.20)
        if not ok:
            self.last_failure="checked_motion_endpoint_not_reached"
        return ok
