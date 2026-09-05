"""Bounded non-fast-loop placement. No automatic release on failure."""
from __future__ import annotations
import time
import numpy as np
from mr_liu.grasp.transforms import invert_transform, transform_points
from .contracts import PlaceError, PlaceResult, PlaceSettings
from .geometry import box_corners, select_site, footprint_supported


class GeneralPlaceNode:
    name = "general_place"

    def __init__(self, *, perception, motion, gripper, settings=None,
                 clock=time.monotonic, trace=lambda e: None, backend=None):
        self.perception, self.motion, self.gripper = perception,motion,gripper
        self.settings = settings or PlaceSettings()
        self.clock, self.trace = clock,trace
        self.cancelled = False
        self.backend = backend

    def cancel(self):
        self.cancelled = True
        self.motion.stop()

    def execute(self, request, held):
        cfg = self.settings
        started = self.clock()
        released, phase, metrics = False,"validate",{}
        sequences = {}

        def fresh(obs):
            if self.cancelled:
                raise PlaceError("cancelled")
            key = obs.camera_frame
            if obs.sequence <= sequences.get(key,-1) or not -.02 <= self.clock()-obs.timestamp_s <= cfg.max_age_s:
                raise PlaceError("stale_observation")
            sequences[key] = obs.sequence

        def event(name, **extra):
            nonlocal phase
            phase = name
            self.trace({"phase":name, **extra})

        try:
            if self.cancelled:
                raise PlaceError("cancelled")
            self.motion.stop()
            if not held.stable_base_verified:
                raise PlaceError("stable_base_unverified")
            if not -.02 <= started-held.verified_at_s <= 3.:
                raise PlaceError("held_state_stale")
            grip = self.gripper.state()
            if (grip.width_m is None or not np.isfinite(grip.width_m)
                    or not .002 < grip.width_m < cfg.open_width_m-.008
                    or not -.02 <= self.clock()-grip.timestamp_s <= cfg.max_age_s):
                raise PlaceError("held_state_unverified")
            initial_ee = self.motion.robot_state().T_base_ee.copy()
            held_pose = initial_ee @ held.T_ee_object
            if np.degrees(np.arccos(np.clip(held_pose[2,2],-1,1))) > cfg.max_object_tilt_deg:
                raise PlaceError("object_orientation_infeasible")
            event("place_find")
            destination = self.perception.destination(request)
            fresh(destination.observation)
            initial_corners=transform_points(held_pose,box_corners(held.half_extents_m))
            initial_radius=np.max(np.abs(initial_corners[:,:2]-held_pose[:2,3]),axis=0)
            if self.backend is None:
                # Five-axis approach-axis IK may change yaw during translation.
                # Reserve the entire upright yaw envelope, including allowed
                # tilt, instead of selecting a barely-fitting initial footprint.
                reserved_radius = (np.linalg.norm(held.half_extents_m[:2])
                    + held.half_extents_m[2]*np.sin(np.deg2rad(cfg.max_object_tilt_deg)))
                xy = select_site(destination, np.array([reserved_radius,reserved_radius,held.half_extents_m[2]]),
                                 cfg.boundary_margin_m+held.uncertainty_m)
                metrics['reserved_footprint_radius_m']=float(reserved_radius)
                metrics['place_backend']='geometric'
                if request.relation == "relative":
                    xy=destination.center_base_m[:2].copy()
                    if not footprint_supported(destination,xy,initial_radius,cfg.boundary_margin_m+held.uncertainty_m):
                        raise PlaceError("requested_relative_position_infeasible")
            else:
                try:
                    xy=self.backend.select(request,held,destination,initial_ee,
                                           cfg.boundary_margin_m+held.uncertainty_m)
                finally:
                    metrics.update(getattr(self.backend,'last_metrics',{}))
            local_xy = xy - destination.center_base_m[:2]
            anchor = destination.center_base_m.copy()
            goal_object = held_pose.copy()
            goal_object[:3,3] = [*xy,destination.support_z_m+held.half_extents_m[2]+cfg.release_gap_m]
            goal_ee = goal_object @ invert_transform(held.T_ee_object)
            if np.linalg.norm(goal_ee[:3,3]-initial_ee[:3,3]) > cfg.max_fine_travel_m:
                raise PlaceError("requires_fast_loop_handoff")
            metrics["target_object_position_m"] = goal_object[:3,3].tolist()
            if request.dry_run:
                return PlaceResult(False,"planned",request.request_id,message="Perception-only plan; path/release not executed",
                                   metrics=metrics)
            # Align above the surface, then descend. Never descend first while
            # the held object is still laterally outside the selected region.
            stage = "align"
            final_payload = None
            for iteration in range(cfg.max_iterations):
                if self.clock()-started > cfg.timeout_s:
                    raise PlaceError("place_timeout")
                event("place_"+stage, iteration=iteration)
                destination = self.perception.destination(request)
                fresh(destination.observation)
                if np.linalg.norm(destination.center_base_m-anchor) > cfg.max_destination_shift_m:
                    raise PlaceError("destination_moved")
                xy = destination.center_base_m[:2]+local_xy
                state = self.motion.robot_state()
                expected = state.T_base_ee @ held.T_ee_object
                payload = self.perception.payload(held,expected)
                fresh(payload.observation)
                drift = np.linalg.norm(payload.T_base_object[:3,3]-expected[:3,3])
                if drift > cfg.max_slip_m:
                    raise PlaceError("payload_slipped")
                corners = transform_points(payload.T_base_object,box_corners(held.half_extents_m))
                radius = np.max(np.abs(corners[:,:2]-payload.T_base_object[:2,3]),axis=0)
                if not footprint_supported(destination,xy,radius,cfg.boundary_margin_m+held.uncertainty_m):
                    raise PlaceError("support_no_longer_valid")
                tilt = np.degrees(np.arccos(np.clip(payload.T_base_object[2,2],-1,1)))
                if tilt > cfg.max_object_tilt_deg:
                    raise PlaceError("payload_tilted")
                bottom = float(corners[:,2].min())
                gap = bottom-destination.support_z_m
                error_xy = xy-payload.T_base_object[:2,3]
                metrics.update(iterations=iteration+1, xy_error_m=float(np.linalg.norm(error_xy)),
                               release_gap_m=gap, payload_residual_m=payload.residual_m)
                if stage == "align" and np.linalg.norm(error_xy) <= cfg.xy_tolerance_m and gap >= cfg.preplace_height_m-.004:
                    stage = "descend"
                if (stage == "descend" and np.linalg.norm(error_xy) <= cfg.xy_tolerance_m
                        and abs(gap-cfg.release_gap_m) <= cfg.release_gap_tolerance_m):
                    final_payload = payload
                    break
                if gap < -held.uncertainty_m:
                    raise PlaceError("support_penetration")
                desired_gap = cfg.preplace_height_m if stage == "align" else cfg.release_gap_m
                delta = np.r_[error_xy, desired_gap-gap]
                # If below clearance, lift before any lateral displacement.
                if stage == "align" and gap < cfg.preplace_height_m-.004:
                    delta[:2] = 0
                norm = np.linalg.norm(delta)
                delta *= min(1.,cfg.max_step_m/max(norm,1e-12))
                goal = state.T_base_ee.copy()
                goal[:3,3] += delta
                if np.linalg.norm(goal[:3,3]-initial_ee[:3,3]) > cfg.max_fine_travel_m:
                    raise PlaceError("fine_workspace_exceeded")
                if not self.motion.move_checked(goal,held,destination):
                    raise PlaceError("held_path_infeasible",getattr(self.motion,"last_failure","") or "")
            if final_payload is None:
                raise PlaceError("place_servo_timeout")
            self.motion.stop()
            event("place_release_check")
            if not self.motion.opening_safe(held,destination):
                raise PlaceError("opening_or_retreat_infeasible",getattr(self.motion,"last_failure","") or "")
            # Planning can be slow: acquire both destination and payload again.
            latest = self.perception.destination(request)
            fresh(latest.observation)
            if np.linalg.norm(latest.center_base_m-anchor) > cfg.max_destination_shift_m:
                raise PlaceError("destination_moved_before_release")
            expected = self.motion.robot_state().T_base_ee @ held.T_ee_object
            payload = self.perception.payload(held,expected)
            fresh(payload.observation)
            corners = transform_points(payload.T_base_object,box_corners(held.half_extents_m))
            gap = corners[:,2].min()-latest.support_z_m
            if (np.linalg.norm(payload.T_base_object[:2,3]-xy)>cfg.xy_tolerance_m
                    or not cfg.release_gap_m-cfg.release_gap_tolerance_m <= gap <= cfg.release_gap_m+cfg.release_gap_tolerance_m
                    or np.linalg.norm(payload.T_base_object[:3,3]-expected[:3,3])>cfg.max_slip_m):
                raise PlaceError("release_pose_changed")
            event("place_open")
            if self.cancelled:
                raise PlaceError("cancelled")
            if not self.motion.opening_safe(held,latest):
                raise PlaceError("release_space_changed")
            if max(self.clock()-payload.observation.timestamp_s,
                   self.clock()-latest.observation.timestamp_s) > cfg.max_age_s:
                raise PlaceError("release_observation_stale_after_check")
            # released means opening may have begun, even if actuator fails.
            released = True
            if not self.gripper.open(cfg.open_width_m,speed_mps=cfg.open_speed_mps):
                raise PlaceError("release_actuator_failed")
            state = self.gripper.state()
            if (state.width_m is None or not np.isfinite(state.width_m) or state.width_m < cfg.open_width_m-.003
                    or not -.02 <= self.clock()-state.timestamp_s <= cfg.max_age_s):
                raise PlaceError("gripper_not_open")
            event("place_retreat")
            retreat = self.motion.retreat_target()
            # Closed-loop retreat: payload is now a world obstacle; no attachment.
            for _ in range(12):
                state = self.motion.robot_state()
                delta=retreat[:3,3]-state.T_base_ee[:3,3]
                distance=np.linalg.norm(delta)
                if distance < .003:
                    break
                step = state.T_base_ee.copy()
                step[:3,3] += delta*min(1.,cfg.max_step_m/max(distance,1e-12))
                if not self.motion.move_checked(step,None,latest,opening=True):
                    raise PlaceError("released_retreat_infeasible")
            else:
                raise PlaceError("released_retreat_timeout")
            event("place_verify")
            positions, times = [],[]
            expected = payload.T_base_object.copy()
            expected[2,3] -= gap
            for _ in range(max(cfg.settle_frames*4,20)):
                evidence = self.perception.payload(held,expected,released=True)
                fresh(evidence.observation)
                p = evidence.T_base_object[:3,3]
                actual_bottom = transform_points(evidence.T_base_object,box_corners(held.half_extents_m))[:,2].min()
                if np.linalg.norm(p[:2]-xy)>.010 or abs(actual_bottom-latest.support_z_m)>.006:
                    raise PlaceError("placement_outside_tolerance")
                positions.append(p.copy()); times.append(evidence.observation.timestamp_s)
                if len(positions)>=cfg.settle_frames and times[-1]-times[0]>=cfg.settle_min_s:
                    break
            if len(positions)<cfg.settle_frames or times[-1]-times[0]<cfg.settle_min_s:
                raise PlaceError("placement_observation_window_short")
            spread = float(np.max(np.linalg.norm(np.asarray(positions)-positions[-1],axis=1)))
            if spread > .004:
                raise PlaceError("placement_unstable")
            metrics.update(stability_spread_m=spread,verification_frames=len(positions),
                           final_object_position_m=positions[-1].tolist(),elapsed_s=self.clock()-started)
            event("place_succeeded",**metrics)
            return PlaceResult(True,phase,request.request_id,released=True,verified=True,metrics=metrics)
        except PlaceError as exc:
            metrics["elapsed_s"] = self.clock()-started
            self.trace({"phase":"place_failed","failed_phase":phase,"failure":exc.code,"released":released})
            return PlaceResult(False,phase,request.request_id,exc.code,str(exc),released,False,metrics)
        except Exception as exc:
            return PlaceResult(False,phase,request.request_id,"internal_error",repr(exc),released,False,metrics)
        finally:
            self.motion.stop()
