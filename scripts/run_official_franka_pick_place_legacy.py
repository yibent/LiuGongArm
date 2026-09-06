"""Run NVIDIA's official Franka task and PickPlaceController stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-position", type=float, nargs=3, default=(0.3, 0.3, 0.02575))
    parser.add_argument("--cube-orientation", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--target-position", type=float, nargs=3, default=(-0.3, -0.3, 0.02575))
    parser.add_argument("--cube-size", type=float, nargs=3, default=(0.0515, 0.0515, 0.0515))
    parser.add_argument("--mass-kg", type=float, default=0.02)
    parser.add_argument("--target-platform-height", type=float, default=0.0)
    parser.add_argument("--target-platform-size", type=float, nargs=2, default=(0.18, 0.18))
    parser.add_argument("--gripper-yaw-deg", type=float, default=0.0)
    parser.add_argument("--success-tolerance-m", type=float, default=0.01)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--industrial-scene", action="store_true")
    parser.add_argument("--move-target-phase", type=int, default=-1,
                        help="Inject a target move in official phase 0-9; -1 disables it")
    parser.add_argument("--move-target-step", type=int, default=30)
    parser.add_argument("--move-target-delta", type=float, nargs=3, default=(0.08, 0.0, 0.0))
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--settle-frames", type=int, default=120)
    return parser.parse_args()


ARGS = parse_args()
if ARGS.output.exists() and any(ARGS.output.iterdir()):
    raise SystemExit("Use a new output directory; previous evidence is never overwritten")
ARGS.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "shutdown_watchdog_timeout": 45.0})

import isaacsim.core.experimental.utils.app as app_utils

if not app_utils.enable_extension("isaacsim.robot.manipulators.examples"):
    raise RuntimeError("Could not enable Isaac Sim's official manipulators examples extension")
simulation_app.update()

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.grasp.transforms import look_at_opencv, matrix_to_quaternion_wxyz


class VideoWriter:
    def __init__(self, path: Path, fps: int = 20):
        import imageio_ffmpeg
        self.path = path
        self.frames = 0
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def add(self, rgba) -> None:
        if rgba is None:
            return
        frame = np.asarray(rgba)
        if frame.ndim < 3 or frame.shape[-1] < 3:
            return
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError(f"video encoder exited {code}")


started = time.perf_counter()
report: dict[str, object] = {
    "implementation": "isaacsim.robot.manipulators.examples.franka.PickPlaceController",
    "motion_controller": "official RMPFlowController",
    "controller_modified": False,
    "project_perception_or_release_gate_used": False,
    "arguments": {
        "cube_position": list(ARGS.cube_position),
        "cube_orientation": list(ARGS.cube_orientation),
        "target_position": list(ARGS.target_position),
        "cube_size": list(ARGS.cube_size),
        "mass_kg": ARGS.mass_kg,
        "target_platform_height": ARGS.target_platform_height,
        "gripper_yaw_deg": ARGS.gripper_yaw_deg,
        "record_video": ARGS.record_video,
        "industrial_scene": ARGS.industrial_scene,
        "move_target_phase": ARGS.move_target_phase,
        "move_target_step": ARGS.move_target_step,
        "move_target_delta": list(ARGS.move_target_delta),
        "max_frames": ARGS.max_frames,
        "settle_frames": ARGS.settle_frames,
    },
}
video = None

try:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    task = PickPlace(
        name="official_franka_pick_place",
        cube_initial_position=np.asarray(ARGS.cube_position, dtype=float),
        cube_initial_orientation=np.asarray(ARGS.cube_orientation, dtype=float),
        target_position=np.asarray(ARGS.target_position, dtype=float),
        cube_size=np.asarray(ARGS.cube_size, dtype=float),
    )
    world.add_task(task)
    industrial_objects = []
    if ARGS.industrial_scene:
        industrial_specs = [
            ("block_a", [0.48, -0.30, 0.0225], [0.045, 0.045, 0.045], [0.8, 0.2, 0.1]),
            ("block_b", [0.05, 0.42, 0.04], [0.04, 0.04, 0.08], [0.2, 0.7, 0.2]),
            ("block_c", [-0.38, 0.25, 0.015], [0.07, 0.04, 0.03], [0.8, 0.7, 0.1]),
            ("block_d", [-0.47, -0.02, 0.0275], [0.055, 0.055, 0.055], [0.5, 0.2, 0.8]),
            ("block_e", [0.42, 0.02, 0.025], [0.08, 0.03, 0.05], [0.1, 0.7, 0.8]),
        ]
        for name, position, scale, color in industrial_specs:
            industrial_objects.append(world.scene.add(DynamicCuboid(
                prim_path=f"/World/Industrial/{name}", name=name,
                position=np.asarray(position), scale=np.asarray(scale), size=1.0,
                color=np.asarray(color), mass=0.03,
            )))
        industrial_objects.append(world.scene.add(DynamicCylinder(
            prim_path="/World/Industrial/cylinder", name="industrial_cylinder",
            position=np.array([0.10, -0.42, 0.045]), radius=0.03, height=0.09,
            color=np.array([0.3, 0.4, 0.9]), mass=0.04,
        )))
    if ARGS.target_platform_height > 0.0:
        target = np.asarray(ARGS.target_position, dtype=float)
        world.scene.add(FixedCuboid(
            prim_path="/World/TargetPlatform",
            name="target_platform",
            position=np.array([target[0], target[1], ARGS.target_platform_height / 2.0]),
            scale=np.array([
                ARGS.target_platform_size[0],
                ARGS.target_platform_size[1],
                ARGS.target_platform_height,
            ]),
            size=1.0,
            color=np.array([0.2, 0.7, 0.2]),
        ))
    world.reset()

    camera = None
    if ARGS.record_video:
        camera = Camera(prim_path="/World/OfficialDemoCamera", frequency=60, resolution=(960, 540))
        camera.initialize()
        camera.set_focal_length(1.8)
        camera.set_clipping_range(0.01, 10.0)
        camera_pose = look_at_opencv([1.10, -1.10, 0.95], [0.0, 0.0, 0.12])
        camera.set_world_pose(position=camera_pose[:3, 3],
                              orientation=matrix_to_quaternion_wxyz(camera_pose[:3, :3]),
                              camera_axes="ros")
        video = VideoWriter(ARGS.output / "official_pick_place.mp4")

    params = task.get_params()
    cube_name = params["cube_name"]["value"]
    robot_name = params["robot_name"]["value"]
    robot = world.scene.get_object(robot_name)
    cube = world.scene.get_object(cube_name)
    cube.set_mass(ARGS.mass_kg)
    controller = PickPlaceController(
        name="official_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
    )
    articulation_controller = robot.get_articulation_controller()

    for warmup in range(30):
        world.step(render=ARGS.record_video or not ARGS.headless)
        if video is not None and warmup % 3 == 0:
            video.add(camera.get_rgba())
    initial_position, initial_orientation = cube.get_world_pose()
    transitions: list[dict[str, int]] = []
    previous_event = int(controller.get_current_event())
    frame_count = 0
    max_cube_z = float(initial_position[2])
    yaw = np.radians(ARGS.gripper_yaw_deg)
    end_effector_orientation = np.array([0.0, -np.sin(yaw / 2.0), np.cos(yaw / 2.0), 0.0])
    event_step = 0
    target_move = None

    while not controller.is_done() and frame_count < ARGS.max_frames:
        current_event_before = int(controller.get_current_event())
        if (target_move is None and current_event_before == ARGS.move_target_phase
                and event_step >= ARGS.move_target_step):
            moved_position, moved_orientation = cube.get_world_pose()
            moved_position = np.asarray(moved_position) + np.asarray(ARGS.move_target_delta)
            cube.set_world_pose(position=moved_position, orientation=moved_orientation)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            target_move = {"phase": current_event_before, "event_step": event_step,
                           "new_position_m": moved_position.tolist()}
        observations = world.get_observations()
        action = controller.forward(
            picking_position=observations[cube_name]["position"],
            placing_position=observations[cube_name]["target_position"],
            current_joint_positions=observations[robot_name]["joint_positions"],
            end_effector_orientation=end_effector_orientation,
        )
        articulation_controller.apply_action(action)
        world.step(render=ARGS.record_video or not ARGS.headless)
        frame_count += 1
        if video is not None and frame_count % 3 == 0:
            video.add(camera.get_rgba())
        observed_cube_position, _ = cube.get_world_pose()
        max_cube_z = max(max_cube_z, float(observed_cube_position[2]))
        current_event = int(controller.get_current_event())
        if current_event != previous_event:
            transitions.append({
                "completed_phase": previous_event,
                "frame": frame_count,
                "cube_position_m": np.asarray(observed_cube_position).tolist(),
            })
            previous_event = current_event
            event_step = 0
        else:
            event_step += 1

    sequence_completed = bool(controller.is_done())
    sequence_end_position, sequence_end_orientation = cube.get_world_pose()
    for settle_frame in range(ARGS.settle_frames):
        world.step(render=ARGS.record_video or not ARGS.headless)
        if video is not None and settle_frame % 3 == 0:
            video.add(camera.get_rgba())
    final_position, final_orientation = cube.get_world_pose()
    ee_position, ee_orientation = robot.end_effector.get_world_pose()
    initial = np.asarray(initial_position, dtype=float)
    final = np.asarray(final_position, dtype=float)
    target = np.asarray(ARGS.target_position, dtype=float)
    target_error = float(np.linalg.norm(final - target))
    displacement = float(np.linalg.norm(final - initial))
    settle_displacement = float(np.linalg.norm(final - np.asarray(sequence_end_position, dtype=float)))
    lift_height = max_cube_z - float(initial[2])
    required_displacement = min(0.1, float(np.linalg.norm(target - initial)) * 0.5)

    report.update(
        sequence_completed=sequence_completed,
        physical_pick_place_observed=bool(
            sequence_completed
            and target_error <= ARGS.success_tolerance_m
            and displacement >= required_displacement
            and lift_height >= 0.08
            and settle_displacement <= 0.01
        ),
        frames=frame_count,
        official_phase=int(controller.get_current_event()),
        phase_transitions=transitions,
        initial_cube_position_m=np.asarray(initial_position).tolist(),
        initial_cube_orientation_wxyz=np.asarray(initial_orientation).tolist(),
        cube_position_at_sequence_end_m=np.asarray(sequence_end_position).tolist(),
        cube_orientation_at_sequence_end_wxyz=np.asarray(sequence_end_orientation).tolist(),
        final_cube_position_m=np.asarray(final_position).tolist(),
        final_cube_orientation_wxyz=np.asarray(final_orientation).tolist(),
        final_end_effector_position_m=np.asarray(ee_position).tolist(),
        final_end_effector_orientation_wxyz=np.asarray(ee_orientation).tolist(),
        final_cube_to_requested_target_m=target_error,
        cube_displacement_m=displacement,
        max_cube_z_m=max_cube_z,
        observed_lift_m=lift_height,
        settle_displacement_m=settle_displacement,
        target_move=target_move,
        video="official_pick_place.mp4" if video is not None else None,
        video_frames=video.frames if video is not None else 0,
        elapsed_s=time.perf_counter() - started,
    )
except Exception as exc:
    report.update(sequence_completed=False, error=repr(exc), elapsed_s=time.perf_counter() - started)
    raise
finally:
    if video is not None:
        video.close()
    (ARGS.output / "official_franka_pick_place_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    simulation_app.close()
