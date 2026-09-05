from __future__ import annotations
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock,patch
import ast
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from mr_liu.place.contracts import *
from mr_liu.place.geometry import footprint_supported, select_site, support_evidence, polygon_mask
from mr_liu.place.node import GeneralPlaceNode
from mr_liu.place.isaac_motion import HeldSweep,IsaacPlaceMotion
from mr_liu.place.perception import RGBDPlacePerception
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
    def opening_safe(self,*a):
        self.retreat=self.pose.copy();self.retreat[2,3]+=.045
        return self.open_safe
    def retreat_target(self):return self.retreat.copy()
    def stop(self):self.stop_count+=1
    def state(self):return GripperState(self.t,.075 if self.opened else .036)
    def open(self,*a,**kw):self.opened=True;return True
    def node(self):return GeneralPlaceNode(perception=self,motion=self,gripper=self,clock=lambda:self.t)


class FinePlaceTests(unittest.TestCase):
    def test_place_step_requests_progress_on_exact_checked_path(self):
        source=Path(__file__).resolve().parents[1]/'source/mr_liu/grasp/adapters/isaac_motion.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IsaacCumotionExecutor')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='move_place_step')
        class PathPlan:
            waypoints=[np.zeros(1),np.array([.01875])]
        env={'np':np,'_JointWaypointPath':PathPlan}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
        executor=SimpleNamespace(arm=SimpleNamespace(T_base_ee=lambda:np.eye(4)),
            controller=SimpleNamespace(set_track_orientation=Mock()),plan_orientation=True,
            _plan=Mock(return_value=PathPlan()),_drive_joint_waypoint=Mock(return_value=True),
            _plan_cache={1:2},_target_reached=Mock(return_value=True),max_move_steps=120)
        target=np.eye(4);target[0,3]=.004
        self.assertTrue(env['move_place_step'](executor,target,speed_scale=.2))
        self.assertEqual(executor._drive_joint_waypoint.call_args.kwargs['minimum_cartesian_progress_m'],.001)
        np.testing.assert_array_equal(executor._drive_joint_waypoint.call_args.args[0],PathPlan.waypoints[-1])
        target[0,3]=.02
        self.assertFalse(env['move_place_step'](executor,target,speed_scale=.2))
        self.assertEqual(executor._plan.call_count,1)
        target[0,3]=.004;executor._plan.return_value=None
        self.assertFalse(env['move_place_step'](executor,target,speed_scale=.2))

    def test_small_place_step_requires_physical_progress_despite_joint_tolerance(self):
        source=Path(__file__).resolve().parents[1]/'source/mr_liu/grasp/adapters/isaac_motion.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IsaacCumotionExecutor')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_drive_joint_waypoint')
        env={'np':np,'robot_config':lambda:{'tool_frame':'ee'}}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
        q=np.zeros(1)
        drive=SimpleNamespace(_stopped=False,joint_tolerance_rad=.020,waypoint_settle_steps=0,
            arm_dof_indices=[0],_arm_values=lambda getter:getter(),advance_frame=lambda:None,
            controller=SimpleNamespace(cumotion_robot=SimpleNamespace(kinematics=SimpleNamespace(
                position=lambda joints,frame:np.array([joints[0]*.25,0,0])))))
        def command(positions,**kwargs):q[:]=positions
        drive.articulation=SimpleNamespace(get_dof_positions=lambda:q.copy(),
            get_dof_velocities=lambda:np.zeros(1),get_dof_efforts=lambda:np.zeros(1),
            set_dof_position_targets=Mock(side_effect=command))
        method=env['_drive_joint_waypoint'];goal=np.array([.01875])
        self.assertTrue(method(drive,goal,speed_scale=.2,max_steps=10))
        drive.articulation.set_dof_position_targets.assert_not_called()
        self.assertTrue(method(drive,goal,speed_scale=.2,max_steps=10,minimum_cartesian_progress_m=.001))
        self.assertGreaterEqual(q[0]*.25,.001)
        self.assertGreater(drive.articulation.set_dof_position_targets.call_count,0)

    def test_sim_shutdown_watchdog_precedes_native_teardown(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_fine_grasp_demo.py'
        tree=ast.parse(script.read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        watchdog=next(n for n in calls if n.func.attr=='dump_traceback_later')
        close=next(n for n in calls if n.func.attr=='close' and isinstance(n.func.value,ast.Name)
                   and n.func.value.id=='simulation_app')
        self.assertLess(watchdog.lineno,close.lineno)
        self.assertEqual(ast.literal_eval(watchdog.args[0]),45.)
        self.assertTrue(ast.literal_eval(next(k.value for k in watchdog.keywords if k.arg=='exit')))

    def test_stationary_release_budget_allows_slow_check_only_when_still(self):
        r=Rig();opening=r.opening_safe
        r.robot_state=lambda:SimpleNamespace(T_base_ee=r.pose.copy(),timestamp_s=r.t,
                                             joint_positions={'joint':0.})
        def slow(*args):
            ok=opening(*args);r.t+=.5;return ok
        r.opening_safe=slow
        node=r.node();node.settings=PlaceSettings(stationary_release_max_age_s=1.)
        result=node.execute(PlaceRequest('region'),r.held)
        self.assertTrue(result.success,result)
        self.assertEqual(result.metrics['release_age_policy'],'stationary_bounded')
        self.assertGreater(result.metrics['release_observation_age_s'],.35)

    def test_extended_release_rejects_motion_and_missing_proprioception(self):
        from mr_liu.place.release import stationary_release_state
        before=SimpleNamespace(T_base_ee=np.eye(4),timestamp_s=9.5,joint_positions={'j':0.})
        after=SimpleNamespace(T_base_ee=np.eye(4),timestamp_s=10.,joint_positions={'j':0.})
        grip=GripperState(10.,.036)
        check=lambda:stationary_release_state(before,after,grip,grip,now=10.,max_age_s=.35)
        self.assertTrue(check())
        after.T_base_ee[0,3]=.001;self.assertFalse(check())
        after.T_base_ee=np.eye(4);after.joint_positions={'j':.01};self.assertFalse(check())
        after.joint_positions={};self.assertFalse(check())
        after.joint_positions={'j':0.};after.timestamp_s=9.;self.assertFalse(check())

    def test_release_budget_does_not_relax_acquisition_or_allow_unbounded_age(self):
        r=Rig();r.stale=True
        node=r.node();node.settings=PlaceSettings(stationary_release_max_age_s=1.)
        self.assertEqual(node.execute(PlaceRequest('region'),r.held).failure,'stale_observation')
        self.assertFalse(r.opened)
        with self.assertRaises(ValueError):PlaceSettings(stationary_release_max_age_s=2.)
        with self.assertRaises(ValueError):PlaceSettings(stationary_release_max_age_s=.1)

    def test_tracked_contour_collision_requires_new_model_and_full_preflight(self):
        obs=SimpleNamespace(timestamp_s=10.)
        e=SimpleNamespace(observation=obs,center_base_m=np.zeros(3))
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,move_to=Mock(return_value=True))
        refiner=SimpleNamespace(request_model_refresh=Mock(return_value=True))
        refresh=Mock(return_value=e)
        motion=IsaacPlaceMotion(executor,None,refresh_destination=refresh,mask_refiner=refiner)
        def rejected(*a,**k):motion.last_failure='observed_surface_finger_collision';return False
        motion._safe=Mock(side_effect=rejected)
        with patch('mr_liu.place.isaac_motion.time.monotonic',return_value=10.):
            self.assertFalse(motion.move_checked(np.eye(4),Rig().held,e))
        refiner.request_model_refresh.assert_called_once();executor.move_to.assert_not_called()
        self.assertEqual(motion._safe.call_count,2)

    def test_payload_flow_is_current_frame_bounded_and_identity_checked(self):
        from mr_liu.place.payload_mask import Sam2PayloadRefiner
        r=Rig();seed=np.zeros((80,100),bool);seed[20:40,20:40]=True
        transport=Mock();transport._request.side_effect=lambda request:{'sequence':request['sequence'],'mask':seed,'segmentation_score':.95}
        refiner=Sam2PayloadRefiner(transport,track_between_inference=True)
        refiner.flow=SimpleNamespace(initialize=Mock(),update=Mock(return_value=seed),diagnostic={'flow_reason':'lk_forward_backward'})
        refiner(r.obs(),seed)
        for _ in range(3):np.testing.assert_array_equal(refiner(r.obs(),seed),seed)
        self.assertEqual(transport._request.call_count,1)
        refiner(r.obs(),seed);self.assertEqual(transport._request.call_count,2)
        refiner.flow.update.return_value=np.ones(seed.shape,bool)
        refiner(r.obs(),seed);self.assertEqual(transport._request.call_count,3)
        refiner.flow.update.return_value=seed;r.t+=3
        refiner(r.obs(),seed);self.assertEqual(transport._request.call_count,4)

    def test_retreat_proposal_preserves_exact_pose_but_rechecks_safety(self):
        from scipy.spatial.transform import Rotation
        pose=np.diag([1.,-1.,-1.,1.])
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose.copy()))
        motion=IsaacPlaceMotion(executor,SimpleNamespace(state=lambda:SimpleNamespace(width_m=.03)))
        motion._points=Mock(return_value=np.empty((0,3)));motion._safe=Mock(return_value=True)
        self.assertTrue(motion.opening_safe(Rig().held,None));target=motion.retreat_target()
        pose[:3,:3]=Rotation.from_euler('z',.02,degrees=True).as_matrix()@pose[:3,:3]
        pose[0,3]+=.00002
        self.assertTrue(motion.opening_safe(Rig().held,None))
        np.testing.assert_array_equal(motion._safe.call_args.args[0],target)
        self.assertEqual(motion._safe.call_count,2)

    def test_release_can_share_destination_frame_without_duplicate_rejection(self):
        r=Rig();calls=[]
        def shared(held,expected,destination):
            calls.append(destination.observation.sequence)
            return PayloadEvidence(destination.observation,expected,held.reference_points_object,.001)
        r.payload_at_destination=shared
        result=r.node().execute(PlaceRequest('region'),r.held)
        self.assertTrue(result.success,result);self.assertGreater(len(calls),0)

    def test_reselect_after_descent_lifts_before_lateral_motion(self):
        r=Rig();changed=[False];first=[];move=r.move_checked
        def supported(*a,**k):
            if not changed[0] and r.pose[2,3]<1.055:
                changed[0]=True;return False
            return True
        def moving(goal,*a,**k):
            if changed[0] and not first:first.append(goal[:3,3]-r.pose[:3,3])
            return move(goal,*a,**k)
        r.move_checked=moving
        with patch('mr_liu.place.node.footprint_supported',side_effect=supported), \
             patch('mr_liu.place.node.select_site',side_effect=[np.array([.03,0]),np.array([.05,0])]):
            result=r.node().execute(PlaceRequest('region'),r.held)
        self.assertTrue(result.success,result)
        np.testing.assert_allclose(first[0][:2],[0,0]);self.assertGreater(first[0][2],0)
        self.assertEqual(result.metrics['site_reselections'],1)

    def test_partial_region_can_use_observed_orientation_with_full_margin(self):
        r=Rig()
        from mr_liu.place.geometry import select_site as actual
        calls=[0]
        def choose(e,half,margin):
            calls[0]+=1
            if calls[0]==1:raise PlaceError('no_supported_placement_region')
            self.assertAlmostEqual(margin,.013)
            return actual(e,half,margin)
        with patch('mr_liu.place.node.select_site',side_effect=choose):
            result=r.node().execute(PlaceRequest('region'),r.held)
        self.assertTrue(result.success,result)
        self.assertEqual(result.metrics['footprint_policy'],'observed_orientation_reselect')

    def test_release_reobserves_after_slow_check_but_never_opens_stale(self):
        r=Rig();original=r.opening_safe;calls=[0]
        def delayed(*a):
            calls[0]+=1;ok=original(*a)
            if calls[0]==2:r.t+=.4
            return ok
        r.opening_safe=delayed
        result=r.node().execute(PlaceRequest('region'),r.held)
        self.assertTrue(result.success,result);self.assertEqual(calls[0],3)
        r=Rig();original=r.opening_safe
        def always_slow(*a):
            ok=original(*a);r.t+=.4;return ok
        r.opening_safe=always_slow
        result=r.node().execute(PlaceRequest('region'),r.held)
        self.assertFalse(r.opened)
        self.assertEqual(result.failure,'release_observation_stale_after_check')

    def test_slow_preflight_reobserves_without_moving_on_stale_depth(self):
        now=[1.]
        def evidence():return SimpleNamespace(observation=SimpleNamespace(timestamp_s=now[0]),center_base_m=np.zeros(3))
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,move_to=Mock(return_value=True))
        refresh=Mock(side_effect=evidence)
        motion=IsaacPlaceMotion(executor,None,refresh_destination=refresh)
        checks=[0]
        def safe(*a,**k):
            checks[0]+=1
            now[0]+=.4 if checks[0]==2 else .01
            return True
        motion._safe=Mock(side_effect=safe)
        with patch('mr_liu.place.isaac_motion.time.monotonic',side_effect=lambda:now[0]):
            self.assertTrue(motion.move_checked(np.eye(4),Rig().held,evidence()))
        self.assertEqual(refresh.call_count,2);executor.move_to.assert_called_once()
        executor.move_to.reset_mock();refresh.reset_mock()
        def always_slow(*a,**k):now[0]+=.4;return True
        motion._safe.side_effect=always_slow
        with patch('mr_liu.place.isaac_motion.time.monotonic',side_effect=lambda:now[0]):
            self.assertFalse(motion.move_checked(np.eye(4),Rig().held,evidence()))
        self.assertEqual(refresh.call_count,3);executor.move_to.assert_not_called()
        self.assertEqual(motion.last_failure,'scene_stale_after_plan')

    def test_collision_point_cache_requires_same_snapshot_and_robot_pose(self):
        pose=np.eye(4);pose[2,3]=1.1
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose.copy()))
        motion=IsaacPlaceMotion(executor,None)
        obs=SimpleNamespace(metadata={})
        evidence=SimpleNamespace(observation=obs,support_z_m=1.)
        points=np.array([[0.,0,1.1]])
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,np.array([0]),np.array([0]))) as cloud:
            motion._points(evidence,None);motion._points(evidence,None)
            self.assertEqual(cloud.call_count,1)
            pose[0,3]=.001;motion._points(evidence,None)
            self.assertEqual(cloud.call_count,2)
            evidence.observation=SimpleNamespace(metadata={});motion._points(evidence,None)
            self.assertEqual(cloud.call_count,3)

    def test_robot_guard_does_not_destroy_contact_edge_identity(self):
        rgb=np.zeros((18,18,3),np.uint8);rgb[:]=[20,40,200]
        rgb[3:16,3:8]=[180,80,60]
        robot=np.zeros((18,18),bool);robot[5:14,6]=True
        rows,cols=np.nonzero(~robot)
        points=np.c_[(cols-8)*.001,(rows-9)*.001,np.full(len(rows),1.)]
        seed=(rows>=3)&(rows<16)&(cols>=3)&(cols<8)
        points[seed,2]=1.1
        fringe=(rows==9)&(cols==8);points[fringe,2]=1.1
        pose=np.eye(4);pose[2,3]=1.1
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        obs=SimpleNamespace(rgb=rgb,depth_m=np.ones((18,18)),metadata={'robot_self_mask':robot})
        held=SimpleNamespace(T_ee_object=np.eye(4),half_extents_m=np.array([.02]*3),chromaticity=np.array([180,80,60])/320)
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            remaining=IsaacPlaceMotion(executor,None)._points(SimpleNamespace(observation=obs,support_z_m=1.),held)
        self.assertFalse(np.any(np.all(remaining==points[fringe][0],axis=1)))
        self.assertTrue(np.all(remaining[:,2]==1.))

    def test_declared_contact_uncertainty_allows_escape_not_pressing(self):
        points=np.array([[-.0034,0,-.015]]) # 1.1 mm inside approximate jaw
        sweep=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.01,
                        contact_mask=np.array([True]),contact_uncertainty_m=.003)
        self.assertEqual(sweep.collision_count(points,np.eye(4)),0)
        escape=np.eye(4);escape[0,3]=.003
        self.assertEqual(sweep.collision_count(points,escape),0)
        press=np.eye(4);press[0,3]=-.001
        self.assertGreater(sweep.collision_count(points,press),0)
        self.assertGreater(sweep.collision_count(np.array([[0.,0,-.015]]),np.eye(4)),0)
        outside=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.01,
                          contact_mask=np.array([False]),contact_uncertainty_m=.003)
        self.assertGreater(outside.collision_count(points,np.eye(4)),0)
        with self.assertRaises(ValueError):HeldSweep(None,.075,contact_uncertainty_m=.004)

    def test_rgb_mask_background_depth_is_kept_as_obstacle(self):
        rgb=np.zeros((16,16,3),np.uint8);rgb[:]=[20,40,200];rgb[5:10,5:10]=[180,80,60]
        rows,cols=np.indices((16,16));rows=rows.ravel();cols=cols.ravel()
        points=np.c_[(cols-8)*.001,(rows-8)*.001,np.full(256,1.)]
        own=(rows>=5)&(rows<10)&(cols>=5)&(cols<10);points[own,2]=1.10
        pose=np.eye(4);pose[2,3]=1.1
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        held=SimpleNamespace(T_ee_object=np.eye(4),half_extents_m=np.array([.02]*3),chromaticity=np.array([180,80,60])/320)
        refined=np.zeros((16,16),bool);refined[4:11,4:11]=True
        obs=SimpleNamespace(rgb=rgb,depth_m=np.ones((16,16)),sequence=1)
        motion=IsaacPlaceMotion(executor,None,mask_refiner=lambda *a:refined)
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            remaining=motion._points(SimpleNamespace(observation=obs,support_z_m=1.),held)
        self.assertEqual(len(remaining),231)
        self.assertTrue(np.all(remaining[:,2]==1.))
        # Even a refiner missing part of the validated appearance component
        # cannot turn that component into an attached-body collision.
        refined[5:7,5:10]=False
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            remaining=motion._points(SimpleNamespace(observation=obs,support_z_m=1.),held)
        self.assertEqual(len(remaining),231)

    def test_site_reserves_all_yaws_before_moving(self):
        r=Rig()
        from mr_liu.place.geometry import select_site as actual
        with patch('mr_liu.place.node.select_site',wraps=actual) as selector:
            result=r.node().execute(PlaceRequest('region',dry_run=True),r.held)
        expected=np.linalg.norm(r.held.half_extents_m[:2])+r.held.half_extents_m[2]*np.sin(np.deg2rad(5))
        np.testing.assert_allclose(selector.call_args.args[1][:2],[expected,expected])
        self.assertEqual(result.phase,'planned')

    def test_retreat_alternatives_must_pass_unchanged_checks(self):
        pose=np.diag([1.,-1.,-1.,1.])
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        motion=IsaacPlaceMotion(executor,SimpleNamespace(state=lambda:SimpleNamespace(width_m=.03)))
        motion._points=Mock(return_value=np.empty((0,3)))
        motion._safe=Mock(side_effect=[False,False,True])
        self.assertTrue(motion.opening_safe(Rig().held,None))
        self.assertEqual(motion._safe.call_count,3)
        self.assertAlmostEqual(motion.retreat_target()[2,3],.045)
        self.assertLessEqual(np.linalg.norm(motion.retreat_target()[:2,3]),.02)

    def test_florence_service_error_is_actionable_and_does_not_return_geometry(self):
        from mr_liu.place.perception import FlorencePlaceLocator
        locator=FlorencePlaceLocator()
        response=Mock(ok=False,reason='Internal Server Error')
        response.json.return_value={'message':'model load: insufficient memory'}
        locator.session=SimpleNamespace(post=Mock(return_value=response))
        with self.assertRaisesRegex(PlaceError,'destination_model_unavailable: model load: insufficient memory'):
            locator.locate(Rig().obs(),'blue region')

    def test_retreat_uses_preflighted_tool_axis_not_world_vertical(self):
        from scipy.spatial.transform import Rotation
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[178,0,30],degrees=True).as_matrix()
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        gripper=SimpleNamespace(state=lambda:SimpleNamespace(width_m=.03))
        motion=IsaacPlaceMotion(executor,gripper)
        motion._points=Mock(return_value=np.empty((0,3)));motion._safe=Mock(return_value=True)
        with self.assertRaisesRegex(PlaceError,'retreat_not_preflighted'):motion.retreat_target()
        self.assertTrue(motion.opening_safe(Rig().held,None))
        expected=pose.copy();expected[:3,3]-=pose[:3,2]*.045
        np.testing.assert_allclose(motion.retreat_target(),expected)
        np.testing.assert_allclose(motion._safe.call_args.args[0],expected)
        snapshot=motion.retreat_target();snapshot[0,3]=10
        np.testing.assert_allclose(motion.retreat_target(),expected)
        pose[:3,:3]=np.eye(3)
        self.assertFalse(motion.opening_safe(Rig().held,None))
        self.assertEqual(motion.last_failure,'retreat_axis_not_upward')

    def test_path_fk_cache_reuses_exact_samples_but_rechecks_world_and_base(self):
        import time
        path=Path(__file__).resolve().parents[1]/"source/mr_liu/grasp/adapters/isaac_motion.py"
        tree=ast.parse(path.read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="IsaacCumotionExecutor")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="is_observation_path_safe")
        namespace={"np":np,"time":time,"_JointStepPlan":type("JointStep",(),{})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),"exec"),namespace)
        plan=namespace["_JointStepPlan"]();plan.q_target=np.array([.01,0.])
        base=[np.zeros(3),np.array([1.,0.,0.,0.])]
        inspector=SimpleNamespace(in_self_collision=Mock(return_value=False),in_collision_with_obstacle=Mock(return_value=False))
        executor=SimpleNamespace(_plan=lambda target:plan,_arm_values=lambda getter:np.zeros(2),
            articulation=SimpleNamespace(get_dof_positions=lambda:np.zeros(2)),_collision_inspector=inspector,
            controller=SimpleNamespace(synchronize_world=Mock(),world_binding=SimpleNamespace(get_world_interface=lambda:
                SimpleNamespace(get_world_to_robot_base_transform=lambda:base))),
            _world_ee_pose_from_joints=Mock(return_value=np.eye(4)))
        finger=SimpleNamespace(path_collision_count=Mock(return_value=0))
        check=lambda:namespace["is_observation_path_safe"](executor,np.eye(4),np.empty((0,3)),finger)
        self.assertTrue(check());calls=executor._world_ee_pose_from_joints.call_count
        self.assertGreater(calls,1)
        self.assertTrue(check());self.assertEqual(executor._world_ee_pose_from_joints.call_count,calls)
        self.assertEqual(inspector.in_self_collision.call_count,6)
        self.assertEqual(inspector.in_collision_with_obstacle.call_count,6)
        self.assertEqual(finger.path_collision_count.call_count,6)
        self.assertEqual(executor.controller.synchronize_world.call_count,2)
        base[0][0]=.001
        self.assertTrue(check());self.assertEqual(executor._world_ee_pose_from_joints.call_count,2*calls)

    def test_release_contact_only_allows_nonincreasing_initial_contact(self):
        point=np.array([[-.00448,0,-.015]])
        sweep=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.010,contact_mask=np.array([True]))
        self.assertEqual(sweep.collision_count(point,np.eye(4)),0)
        away=np.eye(4);away[0,3]=.002
        self.assertEqual(sweep.collision_count(point,away),0)
        pressing=np.eye(4);pressing[0,3]=-.002
        self.assertGreater(sweep.collision_count(point,pressing),0)
        unrelated=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.010,contact_mask=np.array([False]))
        self.assertGreater(unrelated.collision_count(point,np.eye(4)),0)

    def test_release_contact_cannot_exempt_deep_penetration(self):
        point=np.array([[0.,0,-.015]])
        sweep=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.010,contact_mask=np.array([True]))
        self.assertGreater(sweep.collision_count(point,np.eye(4)),0)

    def test_swept_volume_broad_phase_matches_exhaustive_collision_test(self):
        from scipy.spatial.transform import Rotation
        from mr_liu.grasp.contact import FingerGeometry
        from mr_liu.grasp.transforms import transform_points,invert_transform
        r=Rig();random=np.random.default_rng(42)
        for opening in (False,True):
            pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[12,4,75],degrees=True).as_matrix()
            points=random.uniform(-.15,.15,(3000,3));widths=np.linspace(.03,.075,12) if opening else [.03]
            expected=max(FingerGeometry(open_width_m=w).collision_count(points,pose) for w in widths)
            local=transform_points(invert_transform(pose@r.held.T_ee_object),points)
            expected+=np.all(np.abs(local)<r.held.half_extents_m+.001,axis=1).sum()
            sweep=HeldSweep(r.held,.03,opening=opening)
            self.assertEqual(sweep.collision_count(points,pose),expected)
            self.assertEqual(sweep.collision_count(points,pose),expected)

    def test_background_recording_drains_and_uses_no_pickled_optional_pose(self):
        from tempfile import TemporaryDirectory
        from mr_liu.place.recording import AsyncRGBDRecorder
        r=Rig()
        with TemporaryDirectory() as directory:
            writer=AsyncRGBDRecorder(directory,max_pending=2)
            for _ in range(4):writer(r.obs())
            self.assertEqual(writer.close(),[])
            paths=list(Path(directory).glob('*.npz'));self.assertEqual(len(paths),4)
            for path in paths:
                with np.load(path) as data:
                    for key in data.files:np.asarray(data[key])

    def test_near_endpoint_does_not_allow_a_large_ik_detour(self):
        sweep=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.010)
        pose=np.eye(4);pose[0,3]=.011
        with self.assertRaisesRegex(PlaceError,'nonlocal_ik_path'):
            sweep.collision_count(np.empty((0,3)),pose)

    def test_near_endpoint_does_not_allow_a_large_rotation_detour(self):
        from scipy.spatial.transform import Rotation
        sweep=HeldSweep(None,.075,reference_pose=np.eye(4),max_distance_m=.010)
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()
        with self.assertRaisesRegex(PlaceError,'nonlocal_ik_rotation'):
            sweep.collision_count(np.empty((0,3)),pose)

    def test_warmup_mask_is_discarded_and_cannot_authorize_live_use(self):
        from mr_liu.place.payload_mask import Sam2PayloadRefiner
        r=Rig();obs=r.obs();seed=np.ones(obs.depth_m.shape,bool)
        transport=Mock();transport._request.return_value={'sequence':obs.sequence,'mask':None,'segmentation_score':.3}
        refiner=Sam2PayloadRefiner(transport);refiner.warm(obs,seed)
        self.assertIsNone(refiner.mask);self.assertIsNone(refiner.key)
        with self.assertRaisesRegex(PlaceError,'held_mask_model_uncertain'):refiner(obs,seed)
        self.assertEqual(transport._request.call_count,2)

    def test_payload_model_checks_frame_identity_and_caches_same_observation(self):
        from mr_liu.place.payload_mask import Sam2PayloadRefiner
        r=Rig();obs=r.obs();seed=np.zeros(obs.depth_m.shape,bool);seed[20:40,20:40]=True
        transport=Mock();transport._request.return_value={'sequence':obs.sequence,'mask':seed,'segmentation_score':.95}
        refiner=Sam2PayloadRefiner(transport)
        np.testing.assert_array_equal(refiner(obs,seed),seed)
        refiner(obs,seed);self.assertEqual(transport._request.call_count,1)
        transport._request.return_value['sequence']=-1
        with self.assertRaisesRegex(PlaceError,'held_mask_frame_mismatch'):refiner(r.obs(),seed)

    def test_payload_model_cannot_mask_the_whole_table(self):
        from mr_liu.place.payload_mask import Sam2PayloadRefiner
        r=Rig();obs=r.obs();seed=np.zeros(obs.depth_m.shape,bool);seed[20:40,20:40]=True
        transport=Mock();transport._request.return_value={'sequence':obs.sequence,
            'mask':np.ones(seed.shape,bool),'segmentation_score':.99}
        with self.assertRaisesRegex(PlaceError,'held_mask_identity_mismatch'):
            Sam2PayloadRefiner(transport)(obs,seed)

    def test_payload_model_low_score_is_not_a_motion_permit(self):
        from mr_liu.place.payload_mask import Sam2PayloadRefiner
        r=Rig();obs=r.obs();seed=np.ones(obs.depth_m.shape,bool)
        transport=Mock();transport._request.return_value={'sequence':obs.sequence,'mask':seed,'segmentation_score':.3}
        with self.assertRaisesRegex(PlaceError,'held_mask_model_uncertain'):
            Sam2PayloadRefiner(transport)(obs,seed)

    def test_isolated_depth_ghost_is_missing_not_replaced_by_background(self):
        from mr_liu.place.geometry import depth_consistency_mask
        depth=np.ones((20,20));depth[10,10]=.85
        valid=depth_consistency_mask(depth)
        self.assertFalse(valid[10,10]);self.assertTrue(valid[10,11])
        self.assertEqual(depth[10,10],.85)
        depth[8:12,8:12]=.85
        self.assertTrue(depth_consistency_mask(depth)[8,8])

    def test_payload_occlusion_does_not_translate_destination_center(self):
        r=Rig();obs=r.obs();mask=np.ones(obs.depth_m.shape,bool)
        original=support_evidence(obs,mask)
        depth=obs.depth_m.copy();depth[:,60:]=.95
        occluded=support_evidence(replace(obs,depth_m=depth),mask)
        np.testing.assert_allclose(original.center_base_m,occluded.center_base_m,atol=1e-6)
        self.assertLess(len(occluded.surface_points),len(original.surface_points))

    def test_robot_boundary_guard_is_one_pixel_and_not_a_large_roi(self):
        rows,cols=np.indices((8,8));rows=rows.ravel();cols=cols.ravel()
        points=np.c_[(cols-4)*.001,(rows-4)*.001,np.full(64,1.1)]
        robot=np.zeros((8,8),bool);robot[4,4]=True
        pose=np.eye(4);pose[2,3]=1.1
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        evidence=SimpleNamespace(observation=SimpleNamespace(metadata={'robot_self_mask':robot}),support_z_m=1.)
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            result=IsaacPlaceMotion(executor,None)._points(evidence,None)
        self.assertEqual(len(result),55)
        self.assertTrue(np.any(np.all(result==points[(rows==4)&(cols==6)][0],axis=1)))

    def test_payload_mask_keeps_support_and_depth_separated_obstacle(self):
        rgb=np.zeros((16,16,3),np.uint8);rgb[:]=[20,40,200]
        rgb[5:10,5:10]=[180,80,60]
        rows,cols=np.indices((16,16));rows=rows.ravel();cols=cols.ravel()
        points=np.c_[(cols-8)*.001,(rows-8)*.001,np.ones(256)]
        own=(rows>=5)&(rows<10)&(cols>=5)&(cols<10)
        points[own,2]=1.10
        edge=(rows==8)&(cols==10);points[edge,2]=1.10
        obstacle=(rows==8)&(cols==11);points[obstacle,2]=1.108
        pose=np.eye(4);pose[2,3]=1.10
        held=SimpleNamespace(T_ee_object=np.eye(4),half_extents_m=np.array([.02]*3),
                             chromaticity=np.array([180,80,60])/320)
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        motion=IsaacPlaceMotion(executor,None)
        evidence=SimpleNamespace(observation=SimpleNamespace(rgb=rgb,depth_m=np.ones((16,16))),support_z_m=1.)
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            result=motion._points(evidence,held)
        self.assertFalse(np.any(np.all(result==points[edge][0],axis=1)))
        self.assertTrue(np.any(np.all(result==points[obstacle][0],axis=1)))
        self.assertEqual((result[:,2]==1.).sum(),229)

    def test_two_held_appearance_instances_are_not_excluded_as_one(self):
        rgb=np.zeros((16,16,3),np.uint8);rgb[:]=[20,40,200]
        rgb[2:6,2:6]=[180,80,60];rgb[10:14,10:14]=[180,80,60]
        rows,cols=np.indices((16,16));rows=rows.ravel();cols=cols.ravel()
        points=np.c_[(cols-8)*.001,(rows-8)*.001,np.full(256,1.1)]
        pose=np.eye(4);pose[2,3]=1.1
        executor=SimpleNamespace(clear_grasp_plan=lambda:None,robot_state=lambda:SimpleNamespace(T_base_ee=pose))
        held=SimpleNamespace(T_ee_object=np.eye(4),half_extents_m=np.array([.02]*3),chromaticity=np.array([180,80,60])/320)
        evidence=SimpleNamespace(observation=SimpleNamespace(rgb=rgb,depth_m=np.ones((16,16))),support_z_m=1.)
        with patch('mr_liu.place.isaac_motion.scene_cloud',return_value=(points,rows,cols)):
            with self.assertRaisesRegex(PlaceError,'held_collision_mask_uncertain'):
                IsaacPlaceMotion(executor,None)._points(evidence,held)

    def test_appearance_handles_shading_without_accepting_other_hues(self):
        from mr_liu.place.appearance import appearance_matches
        rgb=np.array([[180,80,60],[90,24,10],[220,190,20],[20,50,220],[5,2,1],[240,240,240]])
        self.assertEqual(appearance_matches(rgb,np.array([.55,.25,.20])).tolist(),
                         [True,True,False,False,False,False])

    def test_neutral_appearance_does_not_use_undefined_hue(self):
        from mr_liu.place.appearance import appearance_matches
        self.assertEqual(appearance_matches(np.array([[80,80,80],[200,30,10]]),
                         np.array([1/3]*3)).tolist(),[True,False])

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
    def test_slow_florence_result_requires_new_frame(self):
        r=Rig();texture=np.random.default_rng(3).integers(0,255,(80,100,3),dtype=np.uint8)
        camera=Mock()
        camera.capture.side_effect=lambda t:replace(r.obs(),rgb=texture)
        locator=Mock()
        locator.locate.return_value={"boxes":[{"xyxy":[10,10,90,70],"segmentation":{
            "<REGION_TO_SEGMENTATION>":{"polygons":[[[10,10,90,10,90,70,10,70]]]}}}]}
        def slow(*a):r.t+=12;return locator.locate.return_value
        locator.locate.side_effect=slow
        p=RGBDPlacePerception(camera,None,locator,clock=lambda:r.t)
        e=p.destination(PlaceRequest("region"))
        self.assertEqual(camera.capture.call_count,2)
        self.assertEqual(e.observation.sequence,2)
        self.assertAlmostEqual(e.observation.timestamp_s,r.t)
    def test_ambiguous_destinations_stop_before_tracking(self):
        r=Rig();camera=Mock();camera.capture.side_effect=lambda t:r.obs()
        locator=Mock();locator.locate.return_value={"boxes":[{},{}]}
        p=RGBDPlacePerception(camera,None,locator,clock=lambda:r.t)
        with self.assertRaisesRegex(PlaceError,"ambiguous_destination"):
            p.destination(PlaceRequest("tray"))
    def test_negative_bbox_does_not_slice_to_end_of_image(self):
        item={"xyxy":[-100,-100,-1,-1],"segmentation":{"<REGION_TO_SEGMENTATION>":{
            "polygons":[[[0,0,90,0,90,70,0,70]]]}}}
        with self.assertRaisesRegex(PlaceError,"invalid_destination_box"):
            polygon_mask(item,(80,100))
    def test_arm_stop_preserves_force_limited_gripper_target(self):
        # Exercise the actual adapter method without importing Isaac/Kit in a
        # CPU unit test. C++ SDK construction is covered by the physical demo.
        path=Path(__file__).resolve().parents[1]/"source/mr_liu/grasp/adapters/isaac_motion.py"
        tree=ast.parse(path.read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="IsaacCumotionExecutor")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="stop")
        module=ast.Module(body=[method],type_ignores=[])
        namespace={};exec(compile(ast.fix_missing_locations(module),str(path),"exec"),namespace)
        adapter=Mock();adapter.arm_dof_indices=[0,1,2,3,4]
        adapter._arm_values.return_value=np.array([.1,.2,.3,.4,.5])
        namespace["stop"](adapter)
        call=adapter.articulation.set_dof_position_targets.call_args
        self.assertEqual(call.kwargs["dof_indices"],[0,1,2,3,4])
        self.assertEqual(len(call.args[0]),5)


if __name__ == "__main__": unittest.main()
