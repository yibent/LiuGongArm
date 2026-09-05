"""Isaac placement adapter: actual joint path + payload/jaw swept volume.

V1 is restricted to compact payloads in the calibrated fingertip corridor.
Long tools/full attached-body self-collision are deliberately not advertised.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from mr_liu.grasp.contact import FingerGeometry
from mr_liu.grasp.transforms import invert_transform, transform_points
from .geometry import box_corners, scene_cloud


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
        if held is None:
            return points
        pose = self.robot_state().T_base_ee @ held.T_ee_object
        reference = transform_points(pose,held.reference_points_object)
        distance, _ = cKDTree(reference).query(points)
        colors = evidence.observation.rgb[rows,cols].astype(float)
        colors /= np.maximum(colors.sum(axis=1,keepdims=True),12)
        own = (distance < .008) & (np.linalg.norm(colors-held.chromaticity,axis=1)<.13)
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
        return self.executor.move_to(goal,speed_scale=.20)
