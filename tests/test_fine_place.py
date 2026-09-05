from __future__ import annotations
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from mr_liu.place.contracts import *
from mr_liu.place.geometry import footprint_supported, select_site, support_evidence, polygon_mask
from mr_liu.place.node import GeneralPlaceNode
from mr_liu.place.isaac_motion import HeldSweep
from mr_liu.grasp.contracts import CameraIntrinsics, RGBDObservation, GripperState


class Rig:
    def __init__(self):
        self.t=10.; self.seq=0; self.moves=0; self.opened=False
        self.pose=np.eye(4); self.pose[:3,3]=[0,0,1.07]
        self.block=False; self.slip=0.; self.shift=0.; self.stale=False; self.open_safe=True
        self.after_release_shift=0.; self.stop_count=0
        self.held=HeldObject("test",np.eye(4),np.array([.018]*3),
            np.random.default_rng(1).uniform(-.018,.018,(64,3)),np.array([.7,.15,.15]),10.,True)
        self.surface=np.array([[x,y,1.] for x,y in product(np.arange(-.10,.101,.003),repeat=2)])
    def obs(self,frame="scene"):
        self.seq+=1; self.t+=.1
        pose=np.diag([1.,-1.,-1.,1.]); pose[2,3]=2
        return RGBDObservation(self.seq,self.t-1 if self.stale else self.t,
            np.zeros((80,100,3),np.uint8),np.ones((80,100)),CameraIntrinsics(100,80,100,100,50,40),
            pose,camera_frame=frame)
    def destination(self,request):
        obs=self.obs()
        return DestinationEvidence(obs,self.surface,self.surface,np.array([.03+self.shift,0,1.]),1.,np.ones((80,100),bool))
    def payload(self,held,expected,*,released=False):
        obs=self.obs("scene" if released else "wrist")
        pose=expected.copy()
        pose[0,3]+=self.after_release_shift if released else self.slip
        return PayloadEvidence(obs,pose,held.reference_points_object,.001)
    def robot_state(self): return SimpleNamespace(T_base_ee=self.pose.copy())
    def move_checked(self,goal,held,evidence,**kwargs):
        if self.block:return False
        self.pose=goal.copy(); self.moves+=1
        return True
    def opening_safe(self,*a):return self.open_safe
    def stop(self):self.stop_count+=1
    def state(self):return GripperState(self.t,.075 if self.opened else .036)
    def open(self,*a,**kw):self.opened=True;return True
    def node(self):return GeneralPlaceNode(perception=self,motion=self,gripper=self,clock=lambda:self.t)


class FinePlaceTests(unittest.TestCase):
    def test_closed_loop_release_retreat_and_multiframe_verification(self):
        rig=Rig(); result=rig.node().execute(PlaceRequest("tray"),rig.held)
        self.assertTrue(result.success,result)
        self.assertTrue(result.verified); self.assertTrue(result.released)
        self.assertGreater(rig.moves,10); self.assertGreaterEqual(result.metrics["verification_frames"],5)
    def test_stale_held_state_never_opens(self):
        r=Rig(); result=r.node().execute(PlaceRequest("tray"),replace(r.held,verified_at_s=1.))
        self.assertEqual(result.failure,"held_state_stale"); self.assertFalse(r.opened)
    def test_unknown_stability_is_not_guessed_from_label(self):
        r=Rig(); result=r.node().execute(PlaceRequest("tray"),replace(r.held,stable_base_verified=False))
        self.assertEqual(result.failure,"stable_base_unverified"); self.assertEqual(r.moves,0)
    def test_stale_rgbd_stops_before_move(self):
        r=Rig(); r.stale=True; result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"stale_observation"); self.assertEqual(r.moves,0)
    def test_slip_never_releases(self):
        r=Rig();r.slip=.02;result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"payload_slipped");self.assertFalse(r.opened)
    def test_collision_never_releases(self):
        r=Rig();r.block=True;result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"held_path_infeasible");self.assertFalse(r.opened)
    def test_opening_collision_preserves_payload(self):
        r=Rig();r.open_safe=False;result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"opening_or_retreat_infeasible");self.assertFalse(r.opened)
    def test_open_does_not_imply_success(self):
        r=Rig();r.after_release_shift=.025;result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertFalse(result.success);self.assertTrue(result.released);self.assertFalse(result.verified)
    def test_dry_run_does_not_grant_execution_or_success(self):
        r=Rig();result=r.node().execute(PlaceRequest("tray",dry_run=True),r.held)
        self.assertEqual(result.phase,"planned");self.assertFalse(result.success);self.assertEqual(r.moves,0)
    def test_cancel_is_latched(self):
        r=Rig();node=r.node();node.cancel();result=node.execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"cancelled");self.assertEqual(r.moves,0)
    def test_no_fast_loop_transit(self):
        r=Rig();r.shift=.4;result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertFalse(result.success);self.assertEqual(r.moves,0)
    def test_relative_frame_is_explicit(self):
        with self.assertRaises(ValueError):PlaceRequest("cup",offset_base_m=(.1,0))
        self.assertEqual(PlaceRequest("cup",relation="relative",offset_base_m=(.1,0)).relation,"relative")
    def test_missing_depth_cannot_be_free_support(self):
        r=Rig();e=r.destination(None)
        self.assertTrue(footprint_supported(e,[0,0],[.018,.018]))
        e.surface_points=e.surface_points[np.linalg.norm(e.surface_points[:,:2],axis=1)>.02]
        self.assertFalse(footprint_supported(e,[0,0],[.018,.018]))
    def test_surface_estimate_in_calibrated_base_frame(self):
        r=Rig();obs=r.obs();e=support_evidence(obs,np.ones(obs.depth_m.shape,bool))
        self.assertAlmostEqual(e.support_z_m,1.)
        moved=obs.T_base_camera.copy();moved[2,3]+=.25
        self.assertAlmostEqual(support_evidence(replace(obs,T_base_camera=moved),e.mask).support_z_m,1.25)
    def test_flat_region_is_not_a_container(self):
        r=Rig();obs=r.obs()
        with self.assertRaises(PlaceError):support_evidence(obs,np.ones(obs.depth_m.shape,bool),relation="inside")
    def test_bbox_alone_is_not_free_space(self):
        with self.assertRaises(PlaceError):polygon_mask({"xyxy":[1,1,50,50]},(80,100))
    def test_payload_volume_catches_obstacle_clear_of_fingers(self):
        r=Rig();sweep=HeldSweep(r.held,.036)
        self.assertGreater(sweep.collision_count(np.array([[.012,.012,0.]]),np.eye(4)),0)
    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):PlaceSettings(max_step_m=.5)
        with self.assertRaises(ValueError):PlaceSettings(timeout_s=float('nan'))
    def test_destination_motion_during_loop_does_not_open(self):
        r=Rig();original=r.destination;count=0
        def moving(request):
            nonlocal count
            count+=1
            if count>2:r.shift=.025
            return original(request)
        r.destination=moving
        result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"destination_moved");self.assertFalse(r.opened)
    def test_repeated_observation_does_not_open(self):
        r=Rig();original=r.destination
        def repeated(request):
            e=original(request);e.observation=replace(e.observation,sequence=1);return e
        r.destination=repeated
        result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"stale_observation");self.assertFalse(r.opened)
    def test_partial_open_failure_still_reports_release_attempt(self):
        r=Rig()
        def failed_open(*a,**kw):r.opened=True;return False
        r.open=failed_open
        result=r.node().execute(PlaceRequest("tray"),r.held)
        self.assertEqual(result.failure,"release_actuator_failed");self.assertTrue(result.released)
    def test_unequal_polygon_lengths_are_valid(self):
        item={"xyxy":[0,0,80,80],"segmentation":{"<REGION_TO_SEGMENTATION>":{
            "polygons":[[[5,5,20,5,20,20,5,20],[40,40,60,40,50,60]]]}}}
        self.assertGreater(polygon_mask(item,(80,100)).sum(),100)
    def test_florence_description_is_real_task_not_a_detection_alias(self):
        from find_and_track.florence_finder import FlorenceFinder
        f=FlorenceFinder(device="cpu");f.model=object();f.processor=object();f._run_task=Mock(return_value={})
        f.describe(np.zeros((20,30,3),np.uint8),detail="more")
        self.assertEqual(f._run_task.call_args.args[1],"<MORE_DETAILED_CAPTION>")
        f.segment_box(np.zeros((20,30,3),np.uint8),[0,0,30,20])
        self.assertEqual(f._run_task.call_args.args[1],"<REGION_TO_SEGMENTATION>")
        self.assertEqual(f._run_task.call_args.args[2],"<loc_0><loc_0><loc_999><loc_999>")


if __name__ == "__main__": unittest.main()
