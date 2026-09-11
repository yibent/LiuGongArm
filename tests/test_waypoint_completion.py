"""Exercise the actual executor loop with deterministic joints, without starting Kit.

Only the methods are compiled from the SDK-bound module; their implementation
is not copied into the test. Physics and kinematics are explicit test doubles.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Callable
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'source/mr_liu/grasp/adapters/isaac_motion.py'
tree = ast.parse(SOURCE.read_text())
executor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IsaacCumotionExecutor')
method = next(n for n in executor.body if isinstance(n, ast.FunctionDef) and n.name == '_drive_joint_waypoint')
namespace = {'np': np, 'json': json, 'Callable': Callable, 'robot_config': lambda: {'tool_frame': 'tool'}}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)


def rig():
    state = SimpleNamespace(q=np.zeros(5), commanded=np.zeros(5), frames=0, stop_after=None)
    def set_targets(q, **kwargs):
        state.commanded = np.array(q).copy()
    articulation = SimpleNamespace(
        get_dof_positions=lambda: state.q.copy(),
        get_dof_velocities=lambda: np.zeros(5),
        get_dof_efforts=lambda: np.zeros(5),
        set_dof_position_targets=set_targets)
    instance = SimpleNamespace(
        articulation=articulation, arm_dof_indices=list(range(5)), _stopped=False,
        joint_tolerance_rad=.02, waypoint_settle_steps=3,
        controller=SimpleNamespace(cumotion_robot=SimpleNamespace(
            controlled_joint_names=['base','shoulder','elbow','wrist','roll'],
            kinematics=SimpleNamespace(position=lambda q, tool: np.array([q[0]*.25, 0, 0])))),
        _arm_values=lambda getter: getter())
    def advance():
        state.frames += 1
        state.q = state.commanded.copy()
        if state.stop_after == state.frames: instance._stopped = True
    instance.advance_frame = advance
    instance.drive = MethodType(namespace['_drive_joint_waypoint'], instance)
    return instance, state


class WaypointCompletionTests(unittest.TestCase):
    def drive(self, instance, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return instance.drive(np.array([.017,0,0,0,0]), speed_scale=.65, max_steps=8, **kwargs)

    def test_joint_tolerance_cannot_skip_unsent_trajectory(self):
        instance, state = rig()
        # 0.017 rad is below the legacy joint threshold, but the 4.25 mm
        # position error is above both Cartesian thresholds. Old code exited
        # at frame zero without issuing any target; now it must send the end.
        self.assertTrue(self.drive(instance))
        self.assertGreater(state.frames, 0)
        self.assertAlmostEqual(state.commanded[0], .017)
        self.assertTrue(instance.last_waypoint_diagnostic['command_complete'])

    def test_waits_within_budget_for_original_pose_not_only_joint_goal(self):
        instance, state = rig()
        self.assertTrue(self.drive(instance, endpoint_check=lambda: state.frames >= 4))
        self.assertEqual(state.frames, 4)
        self.assertTrue(instance.last_waypoint_diagnostic['requested_endpoint_passed'])

    def test_unreachable_endpoint_still_fails_without_relaxing_threshold(self):
        instance, state = rig()
        self.assertFalse(self.drive(instance, endpoint_check=lambda: False))
        self.assertEqual(state.frames, 11)
        self.assertEqual(instance.last_waypoint_diagnostic['outcome'], 'budget_exhausted')

    def test_partial_servo_progress_cannot_override_original_pose_requirement(self):
        instance, state = rig()
        self.assertFalse(self.drive(instance, endpoint_check=lambda: False,
                                    accept_partial_cartesian_progress=True))

    def test_stop_still_interrupts_settling(self):
        instance, state = rig()
        state.stop_after = 1
        self.assertFalse(self.drive(instance, endpoint_check=lambda: False))
        self.assertEqual(state.frames, 1)
