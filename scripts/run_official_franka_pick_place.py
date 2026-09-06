"""Run Isaac Sim's official experimental FrankaPickPlace example unchanged.

This entry point deliberately delegates robot motion, grasping, transport, and
release to NVIDIA's ``FrankaPickPlace`` state machine.  Project perception and
placement validators are not part of this execution path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-position", type=float, nargs=3, default=(0.5, 0.0, 0.0258))
    parser.add_argument("--target-position", type=float, nargs=3, default=(0.0, 0.5, 0.14))
    parser.add_argument("--cube-size", type=float, nargs=3, default=(0.0515, 0.0515, 0.0515))
    parser.add_argument("--ik-method", default="damped-least-squares",
                        choices=("damped-least-squares", "pseudoinverse", "transpose",
                                 "singular-value-decomposition"))
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--settle-frames", type=int, default=120)
    return parser.parse_args()


ARGS = parse_args()
if ARGS.output.exists() and any(ARGS.output.iterdir()):
    raise SystemExit("Use a new output directory; previous evidence is never overwritten")
ARGS.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "shutdown_watchdog_timeout": 45.0})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
if not app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples"):
    raise RuntimeError("Could not enable Isaac Sim's official manipulators examples extension")
simulation_app.update()
from isaacsim.robot.experimental.manipulators.examples.franka.pick_place import FrankaPickPlace


def pose(prim) -> tuple[list[float], list[float]]:
    positions, orientations = prim.get_world_poses()
    return positions.numpy()[0].tolist(), orientations.numpy()[0].tolist()


started = time.perf_counter()
report: dict[str, object] = {
    "implementation": "isaacsim.robot.experimental.manipulators.examples.franka.FrankaPickPlace",
    "controller_modified": False,
    "project_perception_or_release_gate_used": False,
    "arguments": {
        "cube_position": list(ARGS.cube_position),
        "target_position": list(ARGS.target_position),
        "cube_size": list(ARGS.cube_size),
        "ik_method": ARGS.ik_method,
        "max_frames": ARGS.max_frames,
        "settle_frames": ARGS.settle_frames,
    },
}

try:
    stage_utils.create_new_stage()
    controller = FrankaPickPlace()
    controller.setup_scene(
        cube_initial_position=np.asarray(ARGS.cube_position, dtype=float),
        target_position=np.asarray(ARGS.target_position, dtype=float),
        cube_size=np.asarray(ARGS.cube_size, dtype=float),
    )
    SimulationManager.set_physics_dt(1.0 / 60.0)
    app_utils.play()
    for _ in range(10):
        simulation_app.update()

    initial_cube_position, initial_cube_orientation = pose(controller.cube)
    transitions: list[dict[str, int]] = []
    previous_event = int(controller._event)
    frame_count = 0
    while not controller.is_done() and frame_count < ARGS.max_frames:
        controller.forward(ik_method=ARGS.ik_method)
        simulation_app.update()
        frame_count += 1
        current_event = int(controller._event)
        if current_event != previous_event:
            transitions.append({"completed_phase": previous_event, "frame": frame_count})
            previous_event = current_event

    sequence_completed = bool(controller.is_done())
    release_position, release_orientation = pose(controller.cube)
    for _ in range(ARGS.settle_frames):
        simulation_app.update()
    final_cube_position, final_cube_orientation = pose(controller.cube)
    final_ee_position, final_ee_orientation = pose(controller.robot.end_effector_link)
    target = np.asarray(ARGS.target_position, dtype=float)
    final = np.asarray(final_cube_position, dtype=float)

    report.update(
        sequence_completed=sequence_completed,
        frames=frame_count,
        official_phase=int(controller._event),
        phase_transitions=transitions,
        initial_cube_position_m=initial_cube_position,
        initial_cube_orientation_wxyz=initial_cube_orientation,
        cube_position_at_sequence_end_m=release_position,
        cube_orientation_at_sequence_end_wxyz=release_orientation,
        final_cube_position_m=final_cube_position,
        final_cube_orientation_wxyz=final_cube_orientation,
        final_end_effector_position_m=final_ee_position,
        final_end_effector_orientation_wxyz=final_ee_orientation,
        final_cube_to_requested_target_m=float(np.linalg.norm(final - target)),
        settle_displacement_m=float(np.linalg.norm(final - np.asarray(release_position))),
        elapsed_s=time.perf_counter() - started,
    )
except Exception as exc:
    report.update(
        sequence_completed=False,
        error=repr(exc),
        elapsed_s=time.perf_counter() - started,
    )
    raise
finally:
    (ARGS.output / "official_franka_pick_place_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    simulation_app.close()

print(json.dumps(report, indent=2))
