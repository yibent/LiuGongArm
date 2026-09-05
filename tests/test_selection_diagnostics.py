"""Candidate diagnostics must preserve screening decisions and per-pose evidence."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.contracts import GraspCandidate
from mr_liu.grasp.selection import SelectionConfig, select_grasp
from mr_liu.grasp.transforms import axis_alignment_error
from mr_liu.motion.kinematics import ArmKinematics


class SelectionDiagnosticsTests(unittest.TestCase):
    def run_selection(self, answers, count=1, feasible_limit=0):
        calls = []
        motion = SimpleNamespace(last_plan_diagnostic={})
        def reachable(pose):
            passed, reason = answers[len(calls)]
            calls.append(pose.copy())
            # Mutating the same object tests that logs take independent snapshots.
            motion.last_plan_diagnostic.clear()
            motion.last_plan_diagnostic.update(reason=reason, attempts=[{"seed_index": len(calls)}])
            return passed
        motion.is_reachable = reachable
        motion.is_collision_free = lambda *args, **kwargs: True
        pose = np.eye(4)
        pose[2, 3] = 0.2
        candidates = [GraspCandidate(pose, .04, .9, 1, "test") for _ in range(count)]
        result = select_grasp(
            candidates, SimpleNamespace(sequence=1, T_base_camera=np.eye(4)),
            SimpleNamespace(points_base=np.zeros((4, 3)), T_base_object=np.eye(4),
                            extents_m=np.ones(3)),
            motion, SimpleNamespace(required_table_clearance_m=.01),
            SelectionConfig(.1, .01, .08, .06, .01, 0,
                            max_feasible_candidates=feasible_limit))
        json.dumps(result.diagnostics, allow_nan=False)
        return result, calls

    def test_pregrasp_failure_short_circuits_and_records_reason(self):
        result, calls = self.run_selection([(False, "ik_tolerance")])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.rejected, {"ik_unreachable": 1})
        self.assertEqual(result.diagnostics[0]["failed_stage"], "pregrasp_reachability")
        self.assertEqual(result.diagnostics[0]["checks"][0]["planning"]["reason"], "ik_tolerance")

    def test_grasp_path_collision_not_confused_with_pregrasp(self):
        result, calls = self.run_selection([(True, "reachable"), (False, "obstacle_collision")])
        self.assertEqual(len(calls), 2)
        entry = result.diagnostics[0]
        self.assertEqual(entry["failed_stage"], "grasp_reachability")
        self.assertEqual(entry["checks"][0]["planning"]["reason"], "reachable")
        self.assertEqual(entry["checks"][1]["planning"]["reason"], "obstacle_collision")
        self.assertNotEqual(entry["checks"][0]["target_world_matrix"],
                            entry["checks"][1]["target_world_matrix"])

    def test_all_128_candidates_logged_without_extra_planning(self):
        result, calls = self.run_selection([(False, "ik_tolerance")] * 128, 128)
        self.assertEqual(len(result.diagnostics), 128)
        self.assertEqual(len(calls), 128)
        self.assertEqual(result.diagnostics[-1]["rank"], 127)

    def test_successful_candidate_selection_is_unchanged(self):
        result, calls = self.run_selection([(True, "reachable"), (True, "reachable")])
        self.assertIsNotNone(result.grasp)
        self.assertEqual(result.rejected, {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result.diagnostics[0]["checks"]), 4)

    def test_live_search_stops_only_after_a_fully_validated_candidate(self):
        result, calls = self.run_selection([(True, "reachable"), (True, "reachable")],
                                          count=3, feasible_limit=1)
        self.assertEqual(result.considered, 1)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(result.grasp)
        self.assertEqual(len(result.diagnostics[0]["checks"]), 4)


def actual_method(name):
    # Importing Isaac SDK requires Kit; compile the real method with explicit fakes.
    source = ROOT / "source/mr_liu/grasp/adapters/isaac_motion.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IsaacCumotionExecutor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "time": time, "deepcopy": deepcopy,
                 "least_squares": least_squares, "axis_alignment_error": axis_alignment_error}
    deferred = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[deferred, method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


class PlanDiagnosticCacheTests(unittest.TestCase):
    def test_coarse_and_grasp_solver_modes_do_not_share_cached_plans(self):
        instance = SimpleNamespace(_fine_grasp_constraints=False,
            _plan_cache={"coarse": object()}, _transition_cache={"coarse": object()})
        configure = MethodType(actual_method("set_grasp_constraints"), instance)
        configure(True)
        self.assertTrue(instance._fine_grasp_constraints)
        self.assertEqual(instance._plan_cache, {})
        instance._plan_cache["fine"] = object()
        configure(True)
        self.assertIn("fine", instance._plan_cache)
        configure(False)
        self.assertEqual(instance._plan_cache, {})

    def test_new_plan_captures_diagnostics_and_clears_expired_entries(self):
        pose = np.eye(4)
        instance = SimpleNamespace(
            _pose_key=lambda p: tuple(p.reshape(-1)), _plan_cache={},
            _plan_diagnostic_cache={"old": {"reason": "old"}},
            controller=SimpleNamespace(synchronize_world=lambda: None,
                cumotion_robot=SimpleNamespace(controlled_joint_names=["j1"])),
            articulation=SimpleNamespace(get_dof_positions=lambda: np.zeros(1)),
            arm_dof_indices=[0], arm=SimpleNamespace(T_base_ee=lambda: pose),
            _active_grasp_q=None, orientation_mode="approach_axis")
        def planner(q, target, key):
            instance.last_plan_diagnostic = {"reason": "ik_tolerance", "attempts": []}
            return None
        instance._plan_axis_constrained = planner
        self.assertIsNone(MethodType(actual_method("_plan"), instance)(pose))
        self.assertNotIn("old", instance._plan_diagnostic_cache)
        self.assertFalse(instance.last_plan_diagnostic["cache_hit"])
        self.assertFalse(instance.last_plan_diagnostic["reachable"])
        self.assertEqual(instance.last_plan_diagnostic["joint_names"], ["j1"])
        json.dumps(instance.last_plan_diagnostic, allow_nan=False)

    def test_cached_failure_restores_own_reason_without_replanning(self):
        pose = np.eye(4)
        key = tuple(pose.reshape(-1))
        instance = SimpleNamespace(
            _pose_key=lambda p: tuple(p.reshape(-1)), _plan_cache={key: None},
            _plan_diagnostic_cache={key: {"reason": "ik_tolerance", "attempts": [{"position_error_mm": 5.0}]}},
            last_plan_diagnostic={"reason": "obstacle_collision"})
        self.assertIsNone(MethodType(actual_method("_plan"), instance)(pose))
        self.assertEqual(instance.last_plan_diagnostic["reason"], "ik_tolerance")
        self.assertTrue(instance.last_plan_diagnostic["cache_hit"])
        instance.last_plan_diagnostic["attempts"][0]["position_error_mm"] = 99
        self.assertEqual(instance._plan_diagnostic_cache[key]["attempts"][0]["position_error_mm"], 5)

    def test_collision_records_exact_path_sample(self):
        instance = SimpleNamespace(_collision_inspector=SimpleNamespace(
            in_self_collision=lambda q: q[0] >= .1,
            in_collision_with_obstacle=lambda q: False))
        result = MethodType(actual_method("_linear_cspace_path"), instance)(
            np.zeros(5), np.array([.2, 0, 0, 0, 0]), collision_step_rad=.05)
        self.assertIsNone(result)
        self.assertEqual(instance.last_collision_diagnostic["reason"], "self_collision")
        self.assertEqual(instance.last_collision_diagnostic["sample_index"], 2)
        self.assertEqual(instance.last_collision_diagnostic["path_fraction"], .5)


class PayloadLiftTests(unittest.TestCase):
    def motion(self, collision=False):
        robot = ArmKinematics(ROOT / "assets/robots/so101/robot.urdf")
        q = np.array([.7049, .75954, -.4863, .69894, -1.566])
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        base[:3, 3] = [.2, 0, 1.02]
        fk = lambda joints: robot.forward(dict(zip(robot.names, joints)), base)
        def limits(i):
            lo, hi = robot.limits[robot.names[i]]
            return SimpleNamespace(lower=lo, upper=hi)
        paths = []
        def path(start, end):
            paths.append((start.copy(), end.copy()))
            return None if collision else (start.copy(), end.copy())
        motion = SimpleNamespace(
            controller=SimpleNamespace(synchronize_world=lambda: None,
                cumotion_robot=SimpleNamespace(kinematics=SimpleNamespace(cspace_coord_limits=limits))),
            articulation=SimpleNamespace(get_dof_positions=lambda: q.copy()),
            _arm_values=lambda getter: getter(), _world_ee_pose_from_joints=fk,
            _linear_cspace_path=path, last_collision_reason="obstacle_collision")
        motion._plan_payload_lift = MethodType(actual_method("_plan_payload_lift"), motion)
        return motion, q, paths

    def test_lift_preserves_contact_orientation_along_entire_path(self):
        motion, q, _ = self.motion()
        start = motion._world_ee_pose_from_joints(q)
        target = start.copy()
        target[2, 3] += .08
        paths = motion._plan_payload_lift(target)
        self.assertIsNotNone(paths)
        self.assertGreaterEqual(len(paths), 16)
        previous_z = start[2, 3]
        for begin, end in paths:
            for alpha in np.linspace(0, 1, 5):
                pose = motion._world_ee_pose_from_joints(begin+alpha*(end-begin))
                self.assertLess(np.linalg.norm(pose[:2, 3]-start[:2, 3]), .003)
                self.assertLess(axis_alignment_error(pose[:3, 0], start[:3, 0]), np.deg2rad(3))
                self.assertGreaterEqual(pose[2, 3]+.0001, previous_z)
                previous_z = pose[2, 3]
            self.assertLess(abs(end[-1]-q[-1]), np.deg2rad(3))
        self.assertLess(np.linalg.norm(pose[:3, 3]-target[:3, 3]), .003)

    def test_blocked_lift_has_no_unchecked_retraction_fallback(self):
        motion, q, paths = self.motion(collision=True)
        target = motion._world_ee_pose_from_joints(q)
        target[2, 3] += .08
        self.assertIsNone(motion._plan_payload_lift(target))
        self.assertEqual(len(paths), 1)
        self.assertEqual(motion.last_lift_diagnostic["reason"], "obstacle_collision")


if __name__ == "__main__":
    unittest.main()
