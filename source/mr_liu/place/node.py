"""Bounded non-fast-loop placement. No automatic release on failure."""
from __future__ import annotations
import time
import numpy as np
from mr_liu.grasp.transforms import invert_transform, transform_points, interpolate_pose_step, rotation_angle_rad
from .contracts import PlaceError, PlaceResult, PlaceSettings
from .geometry import box_corners, select_site, footprint_supported
from .release import stationary_release_state


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
            if request.relation in {"insert", "hang"}:
                metrics.update(requested_relation=request.relation,
                    missing_capabilities=["6d_payload_tracking", "contact_motion_control", "task_release_verification"])
                raise PlaceError("placement_relation_not_executable", request.relation)
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
            held_object_rotation = held_pose[:3, :3].copy()
            full_pose = bool(getattr(self.backend, "supports_full_pose", False)
                             and getattr(self.motion, "tracks_orientation", False))
            if (not full_pose and
                    np.degrees(np.arccos(np.clip(held_pose[2,2],-1,1))) > cfg.max_object_tilt_deg):
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
                try:
                    xy = select_site(destination, np.array([reserved_radius,reserved_radius,held.half_extents_m[2]]),
                                     cfg.boundary_margin_m+held.uncertainty_m)
                    metrics['footprint_policy']='all_yaw_envelope'
                except PlaceError as exc:
                    if exc.code!='no_supported_placement_region':raise
                    # A destination may fit the actual upright orientation but
                    # not every possible yaw. Use fresh-footprint feedback in
                    # that case, without weakening the support/boundary checks.
                    xy=select_site(destination,np.r_[initial_radius+cfg.xy_tolerance_m,held.half_extents_m[2]],
                                   cfg.boundary_margin_m+held.uncertainty_m)
                    metrics['footprint_policy']='observed_orientation_reselect'
                metrics['preferred_all_yaw_radius_m']=float(reserved_radius)
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
            # A full-pose backend owns the object orientation.  Keep the
            # selected SE(3) transform intact; only its vertical translation
            # is recomputed from the observed support plane and the requested
            # release gap.  Translation-only controllers must explicitly
            # reject this path rather than silently discarding rotation.
            target_object_rotation = held_object_rotation.copy()
            selected_pose = getattr(self.backend, "selected_object_pose", None) if self.backend is not None else None
            if selected_pose is not None:
                selected_pose = np.asarray(selected_pose, dtype=np.float64)
                if selected_pose.shape != (4, 4) or not np.all(np.isfinite(selected_pose)):
                    raise PlaceError("invalid_selected_object_pose")
                target_object_rotation = selected_pose[:3, :3].copy()
                if not full_pose and rotation_angle_rad(held_object_rotation, target_object_rotation) > np.deg2rad(.5):
                    raise PlaceError("rotation_requires_pose_controller")
            local_xy = xy - destination.center_base_m[:2]
            anchor = destination.center_base_m.copy()
            goal_object = np.eye(4, dtype=np.float64)
            goal_object[:3, :3] = target_object_rotation
            goal_object[:2,3] = xy
            object_bottom_offset = float(transform_points(goal_object, box_corners(held.half_extents_m))[:,2].min()
                                         - goal_object[2,3])
            goal_object[2,3] = destination.support_z_m + cfg.release_gap_m - object_bottom_offset
            goal_ee = goal_object @ invert_transform(held.T_ee_object)
            if np.linalg.norm(goal_ee[:3,3]-initial_ee[:3,3]) > cfg.max_fine_travel_m:
                raise PlaceError("requires_fast_loop_handoff")
            metrics["target_object_position_m"] = goal_object[:3,3].tolist()
            metrics["target_object_pose"] = goal_object.tolist()
            metrics["target_object_rotation_change_deg"] = float(np.degrees(
                rotation_angle_rad(held_object_rotation, target_object_rotation)))
            metrics["full_pose_controller"] = full_pose
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
                shared_payload=getattr(self.perception,'payload_at_destination',None)
                payload=(shared_payload(held,expected,destination)
                         if stage=='descend' and shared_payload is not None
                         else self.perception.payload(held,expected))
                if payload.observation is not destination.observation:
                    fresh(payload.observation)
                drift = np.linalg.norm(payload.T_base_object[:3,3]-expected[:3,3])
                if drift > cfg.max_slip_m:
                    raise PlaceError("payload_slipped")
                payload_rotation_slip_deg = float(np.degrees(payload.orientation_residual_rad))
                if payload_rotation_slip_deg > cfg.max_payload_rotation_slip_deg:
                    raise PlaceError("payload_rotated_in_gripper")
                corners = transform_points(payload.T_base_object,box_corners(held.half_extents_m))
                radius = np.max(np.abs(corners[:,:2]-payload.T_base_object[:2,3]),axis=0)
                if not footprint_supported(destination,xy,radius,cfg.boundary_margin_m+held.uncertainty_m):
                    if self.backend is not None or request.relation=='relative':
                        raise PlaceError("support_no_longer_valid")
                    xy=select_site(destination,np.r_[radius+cfg.xy_tolerance_m,held.half_extents_m[2]],
                                   cfg.boundary_margin_m+held.uncertainty_m)
                    local_xy=xy-destination.center_base_m[:2]
                    # If a new site is needed after descending, return to the
                    # clearance stage (vertical lift first) before translating.
                    stage='align'
                    metrics['footprint_policy']='observed_orientation_reselect'
                    metrics['site_reselections']=metrics.get('site_reselections',0)+1
                    event('place_reselect_supported_site',xy_base_m=xy.tolist())
                    metrics['target_object_position_m'][:2]=xy.tolist()
                tilt = np.degrees(np.arccos(np.clip(payload.T_base_object[2,2],-1,1)))
                if not full_pose and tilt > cfg.max_object_tilt_deg:
                    raise PlaceError("payload_tilted")
                bottom = float(corners[:,2].min())
                gap = bottom-destination.support_z_m
                error_xy = xy-payload.T_base_object[:2,3]
                orientation_error_deg = float(np.degrees(rotation_angle_rad(
                    payload.T_base_object[:3,:3], target_object_rotation)))
                metrics.update(iterations=iteration+1, xy_error_m=float(np.linalg.norm(error_xy)),
                               release_gap_m=gap, payload_residual_m=payload.residual_m,
                               payload_rotation_slip_deg=payload_rotation_slip_deg,
                               object_orientation_error_deg=orientation_error_deg,
                               object_tilt_deg=float(tilt))
                if (stage == "align" and np.linalg.norm(error_xy) <= cfg.xy_tolerance_m
                        and abs(gap-cfg.preplace_height_m) <= .004):
                    stage = ("rotate" if orientation_error_deg > cfg.orientation_tolerance_deg
                             else "descend")
                if (stage == "rotate" and np.linalg.norm(error_xy) <= cfg.xy_tolerance_m
                        and abs(gap-cfg.preplace_height_m) <= .004
                        and orientation_error_deg <= cfg.orientation_tolerance_deg):
                    stage = "descend"
                if (stage == "descend" and np.linalg.norm(error_xy) <= cfg.xy_tolerance_m
                        and abs(gap-cfg.release_gap_m) <= cfg.release_gap_tolerance_m
                        and orientation_error_deg <= cfg.orientation_tolerance_deg):
                    final_payload = payload
                    break
                if gap < -held.uncertainty_m:
                    raise PlaceError("support_penetration")
                desired_gap = cfg.preplace_height_m if stage in {"align", "rotate"} else cfg.release_gap_m
                # Establish clearance before rotating or translating laterally.
                # Once above the preplace plane, move toward the complete
                # AnyPlace object orientation in bounded 4-degree tokens.
                below_clearance = stage == "align" and gap < cfg.preplace_height_m-.004
                desired_rotation = (target_object_rotation if stage in {"rotate", "descend"}
                                    else held_object_rotation)
                desired_center = (payload.T_base_object[:2,3].copy() if below_clearance
                                  else xy.copy())
                desired_object = np.eye(4, dtype=np.float64)
                desired_object[:3,:3] = desired_rotation
                desired_object[:2,3] = desired_center
                desired_object[2,3] = destination.support_z_m + desired_gap - float(
                    transform_points(desired_object, box_corners(held.half_extents_m))[:,2].min()
                    - desired_object[2,3])
                desired_ee = desired_object @ invert_transform(held.T_ee_object)
                goal = interpolate_pose_step(
                    state.T_base_ee, desired_ee,
                    max_translation_m=cfg.max_step_m,
                    max_rotation_rad=np.deg2rad(cfg.max_rotation_step_deg),
                )
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
            # A slow check can be retried only by acquiring new evidence, never
            # by refreshing its timestamp or skipping geometry/held-state checks.
            for release_attempt in range(3):
                release_state = self.motion.robot_state()
                release_grip = self.gripper.state()
                latest = self.perception.destination(request)
                fresh(latest.observation)
                if np.linalg.norm(latest.center_base_m-anchor) > cfg.max_destination_shift_m:
                    raise PlaceError("destination_moved_before_release")
                expected = self.motion.robot_state().T_base_ee @ held.T_ee_object
                shared_payload=getattr(self.perception,'payload_at_destination',None)
                payload=(shared_payload(held,expected,latest) if shared_payload is not None
                         else self.perception.payload(held,expected))
                if payload.observation is not latest.observation:
                    fresh(payload.observation)
                corners = transform_points(payload.T_base_object,box_corners(held.half_extents_m))
                gap = corners[:,2].min()-latest.support_z_m
                release_xy_error = float(np.linalg.norm(payload.T_base_object[:2,3]-xy))
                release_gap_error = float(gap-cfg.release_gap_m)
                release_slip = float(np.linalg.norm(payload.T_base_object[:3,3]-expected[:3,3]))
                metrics.update(release_xy_error_m=release_xy_error,
                               release_gap_error_m=release_gap_error,
                               release_slip_m=release_slip,
                               release_orientation_error_deg=float(np.degrees(
                                   rotation_angle_rad(payload.T_base_object[:3,:3], target_object_rotation))),
                               release_payload_orientation_residual_deg=float(np.degrees(
                                   payload.orientation_residual_rad)),
                               release_xy_ok=bool(release_xy_error <= cfg.xy_tolerance_m),
                               release_gap_ok=bool(cfg.release_gap_m-cfg.release_gap_tolerance_m <= gap <= cfg.release_gap_m+cfg.release_gap_tolerance_m),
                               release_slip_ok=bool(release_slip <= cfg.max_slip_m))
                pose_outside_tolerance = (
                    release_xy_error > cfg.xy_tolerance_m
                    or not cfg.release_gap_m-cfg.release_gap_tolerance_m <= gap <= cfg.release_gap_m+cfg.release_gap_tolerance_m
                    or release_slip > cfg.max_slip_m
                    or np.degrees(rotation_angle_rad(payload.T_base_object[:3,:3], target_object_rotation))
                       > cfg.orientation_tolerance_deg
                )
                if pose_outside_tolerance:
                    # A Panda can stop a few millimetres short after the last
                    # Cartesian step.  When the payload is still demonstrably
                    # attached and the measured error fits one bounded step,
                    # close the loop once before declaring the release pose
                    # invalid.  This is a new, collision-checked motion token;
                    # it never widens the final acceptance tolerances.
                    correction = np.r_[xy-payload.T_base_object[:2,3],
                                       cfg.release_gap_m-gap]
                    orientation_error = rotation_angle_rad(
                        payload.T_base_object[:3,:3], target_object_rotation)
                    correction_norm = float(np.linalg.norm(correction))
                    can_correct = (
                        release_slip <= cfg.max_slip_m
                        and release_xy_error <= cfg.xy_tolerance_m + cfg.max_step_m
                        and abs(gap-cfg.release_gap_m) <= cfg.release_gap_tolerance_m + cfg.max_step_m
                        and correction_norm <= cfg.max_step_m
                        and orientation_error <= np.deg2rad(cfg.max_rotation_step_deg)
                        and release_attempt < 2
                    )
                    metrics.update(release_correction_m=correction_norm,
                                   release_correction_attempted=bool(can_correct))
                    if can_correct:
                        def correction_pose(state, delta):
                            pose = state.T_base_ee.copy()
                            pose[:3,:3] = target_object_rotation @ held.T_ee_object[:3,:3].T
                            pose[:3,3] += delta
                            return interpolate_pose_step(
                                state.T_base_ee, pose,
                                max_translation_m=max(float(np.linalg.norm(delta)), 1e-6),
                                max_rotation_rad=np.deg2rad(cfg.max_rotation_step_deg),
                            )

                        event("place_release_correction", attempt=release_attempt+1,
                              correction_m=correction.tolist(),
                              measured_error_m=release_xy_error)
                        # Lateral motion at the release height can put the
                        # finger sweep into the support surface.  Use three
                        # bounded Cartesian tokens: a short lift, the lateral
                        # correction while clear, and a final vertical return.
                        # Each token is independently revalidated by
                        # ``move_checked``; the final acceptance test below is
                        # unchanged.
                        state = release_state
                        if release_xy_error > cfg.xy_tolerance_m:
                            lift_m = min(.004, cfg.max_step_m)
                            lift = np.array([0., 0., lift_m])
                            if not self.motion.move_checked(
                                correction_pose(state, lift), held, latest
                            ):
                                raise PlaceError("release_correction_infeasible",
                                                 getattr(self.motion, "last_failure", "") or "")
                            state = self.motion.robot_state()
                            lateral = np.array([correction[0], correction[1], 0.])
                            if np.linalg.norm(lateral) > 1e-9 and not self.motion.move_checked(
                                correction_pose(state, lateral), held, latest
                            ):
                                raise PlaceError("release_correction_infeasible",
                                                 getattr(self.motion, "last_failure", "") or "")
                            state = self.motion.robot_state()
                            down = np.array([0., 0., correction[2] - lift_m])
                            if np.linalg.norm(down) > 1e-9 and not self.motion.move_checked(
                                correction_pose(state, down), held, latest
                            ):
                                raise PlaceError("release_correction_infeasible",
                                                 getattr(self.motion, "last_failure", "") or "")
                        else:
                            goal = correction_pose(state, correction)
                            if not self.motion.move_checked(goal, held, latest):
                                raise PlaceError("release_correction_infeasible",
                                                 getattr(self.motion, "last_failure", "") or "")
                        continue
                    raise PlaceError("release_pose_changed")
                if self.cancelled:
                    raise PlaceError("cancelled")
                if not self.motion.opening_safe(held,latest):
                    raise PlaceError("release_space_changed",getattr(self.motion,"last_failure","") or "")
                age = max(self.clock()-payload.observation.timestamp_s,
                          self.clock()-latest.observation.timestamp_s)
                if age <= cfg.max_age_s:
                    metrics['release_observation_age_s'] = age
                    metrics['release_age_policy'] = 'fresh'
                    break
                if age <= cfg.stationary_release_max_age_s:
                    if not stationary_release_state(release_state,self.motion.robot_state(),
                            release_grip,self.gripper.state(),now=self.clock(),max_age_s=cfg.max_age_s):
                        raise PlaceError('release_not_stationary')
                    metrics['release_observation_age_s'] = age
                    metrics['release_age_policy'] = 'stationary_bounded'
                    break
                event("place_release_reobserve",attempt=release_attempt+1)
            else:
                raise PlaceError("release_observation_stale_after_check")
            event("place_open")
            if self.cancelled:
                raise PlaceError("cancelled")
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
            # A checked endpoint can advance less than the nominal 6 mm step
            # when cuMotion settles a Panda joint trajectory. Keep the same
            # bounded Cartesian increment but allow enough iterations for the
            # full 45 mm withdrawal to converge.
            for retreat_iteration in range(24):
                state = self.motion.robot_state()
                delta=retreat[:3,3]-state.T_base_ee[:3,3]
                distance=np.linalg.norm(delta)
                metrics['retreat_iterations'] = retreat_iteration + 1
                metrics['retreat_distance_m'] = float(distance)
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
