"""Official Franka PickPlaceController on cube, cylinder, and sphere objects."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shape", choices=("cube", "cylinder", "sphere"), default="cube")
    p.add_argument("--start", type=float, nargs=3, default=(0.3, 0.25, 0.03))
    p.add_argument("--target", type=float, nargs=3, default=(-0.25, -0.25, 0.03))
    p.add_argument("--size", type=float, nargs=3, default=(0.06, 0.06, 0.06),
                   help="cube dimensions, or cylinder [radius, radius, height], or sphere [radius, radius, radius]")
    p.add_argument("--mass-kg", type=float, default=0.03)
    p.add_argument("--max-frames", type=int, default=5000)
    p.add_argument("--settle-frames", type=int, default=120)
    return p.parse_args()


A = args()
if A.output.exists() and any(A.output.iterdir()):
    raise SystemExit("Use a new output directory")
A.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": A.headless, "shutdown_watchdog_timeout": 45.0})
import isaacsim.core.experimental.utils.app as app_utils
if not app_utils.enable_extension("isaacsim.robot.manipulators.examples"):
    raise RuntimeError("official manipulator extension unavailable")
app.update()

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

started = time.perf_counter()
report = {
    "implementation": "isaacsim.robot.manipulators.examples.franka.PickPlaceController",
    "controller_modified": False,
    "shape": A.shape,
    "start": list(A.start),
    "target": list(A.target),
    "size": list(A.size),
    "mass_kg": A.mass_kg,
}
try:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    robot = world.scene.add(Franka(prim_path="/World/Franka", name="franka_shape_test"))
    start = np.asarray(A.start, dtype=float)
    target = np.asarray(A.target, dtype=float)
    size = np.asarray(A.size, dtype=float)
    if A.shape == "cube":
        obj = DynamicCuboid(prim_path="/World/Object", name="object", position=start,
                           scale=size, size=1.0, color=np.array([0.8, 0.1, 0.1]), mass=A.mass_kg)
    elif A.shape == "cylinder":
        obj = DynamicCylinder(prim_path="/World/Object", name="object", position=start,
                              radius=float(size[0]), height=float(size[2]),
                              color=np.array([0.1, 0.3, 0.9]), mass=A.mass_kg)
    else:
        obj = DynamicSphere(prim_path="/World/Object", name="object", position=start,
                            radius=float(size[0]), color=np.array([0.9, 0.6, 0.1]), mass=A.mass_kg)
    obj = world.scene.add(obj)
    world.reset()
    # Match the official PickPlace task's post_reset hook.  Creating a scene
    # without that task wrapper otherwise leaves the fingers at the USD state.
    robot.gripper.set_joint_positions(robot.gripper.joint_opened_positions)
    controller = PickPlaceController(name="official_shape_pick_place", gripper=robot.gripper,
                                     robot_articulation=robot)
    ac = robot.get_articulation_controller()
    for _ in range(10):
        world.step(render=not A.headless)
    initial, _ = obj.get_world_pose()
    max_z = float(initial[2])
    events = []
    previous = int(controller.get_current_event())
    frames = 0
    while not controller.is_done() and frames < A.max_frames:
        current_pos, _ = obj.get_world_pose()
        action = controller.forward(
            picking_position=np.asarray(current_pos),
            placing_position=target,
            current_joint_positions=robot.get_joint_positions(),
        )
        ac.apply_action(action)
        world.step(render=not A.headless)
        frames += 1
        current_pos, _ = obj.get_world_pose()
        max_z = max(max_z, float(current_pos[2]))
        event = int(controller.get_current_event())
        if event != previous:
            events.append({"completed_phase": previous, "frame": frames,
                           "object_position_m": np.asarray(current_pos).tolist()})
            previous = event
    sequence_end, sequence_orientation = obj.get_world_pose()
    for _ in range(A.settle_frames):
        world.step(render=not A.headless)
    final, final_orientation = obj.get_world_pose()
    initial = np.asarray(initial, dtype=float)
    final = np.asarray(final, dtype=float)
    target_error = float(np.linalg.norm(final - target))
    settle = float(np.linalg.norm(final - np.asarray(sequence_end, dtype=float)))
    lift = max_z - float(initial[2])
    report.update(
        sequence_completed=bool(controller.is_done()), frames=frames,
        phase=int(controller.get_current_event()), phase_transitions=events,
        initial_position_m=initial.tolist(), final_position_m=final.tolist(),
        final_orientation_wxyz=np.asarray(final_orientation).tolist(),
        target_error_m=target_error, observed_lift_m=lift, settle_displacement_m=settle,
        physical_pick_place_observed=bool(controller.is_done() and target_error <= .05 and lift >= .06 and settle <= .01),
        elapsed_s=time.perf_counter() - started,
    )
except Exception as exc:
    report.update(sequence_completed=False, error=repr(exc), elapsed_s=time.perf_counter() - started)
    raise
finally:
    (A.output / "official_franka_shape_pick_place_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    app.close()
