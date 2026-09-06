"""Isaac asset, FK and moving RGB-D calibration check; no grasp commands."""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--robot', choices=('so101', 'franka'), default='franka')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ['MR_LIU_ROBOT'] = args.robot
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'shutdown_watchdog_timeout': 30.})
import cv2
import numpy as np
from isaacsim.core.experimental.utils import stage as stage_utils, app as app_utils
from isaacsim.core.experimental.objects import GroundPlane, DistantLight
from isaacsim.core.simulation_manager import SimulationManager
from mr_liu.config import robot_config
from mr_liu.robot.so101 import RobotArm
from mr_liu.sim.spawn import spawn_table_and_robot
from mr_liu.perception.camera import spawn_configured_cameras
from mr_liu.grasp.adapters.isaac_motion import IsaacCumotionExecutor
from mr_liu.grasp.adapters.isaac_camera import IsaacWristCamera
from mr_liu.grasp.adapters.isaac_gripper import create_isaac_gripper
from mr_liu.grasp.contracts import TargetSpec

try:
    stage_utils.create_new_stage()
    GroundPlane('/World/GroundPlane', positions=[0, 0, 0])
    DistantLight('/World/Light').set_intensities(1200.)
    spawn_table_and_robot(include_tabletop_props=False)
    app.update()
    cameras = spawn_configured_cameras()
    SimulationManager.setup_simulation(dt=1/120, device='cpu')
    arm = RobotArm()
    app.update()
    app_utils.play()
    for _ in range(30):
        app.update()
    arm.validate_joint_mapping()
    arm.apply_configured_pose()
    for _ in range(45):
        app.update()
    executor = IsaacCumotionExecutor(arm, advance_frame=lambda: SimulationManager.step(steps=1),
                                    physics_dt_s=1/120, track_orientation=True)
    gripper = create_isaac_gripper(arm.articulation, advance_frame=lambda: SimulationManager.step(steps=1),
                                   physics_dt_s=1/120)
    camera = IsaacWristCamera(cameras['wrist'], ee_pose_provider=arm.T_base_ee, advance_frame=app.update)
    report = {'robot': robot_config(), 'dof_names': arm.dof_names, 'samples': []}
    for index, width in enumerate((.08, .04, .08)):
        moved = gripper.open(width, speed_mps=.04)
        obs = camera.capture(TargetSpec('calibration'))
        if obs is None:
            raise RuntimeError('No wrist RGB-D frame')
        fk = executor._world_ee_pose_from_joints(executor._arm_values(arm.articulation.get_dof_positions))
        actual = arm.T_base_ee()
        error = float(np.linalg.norm(fk[:3,3]-actual[:3,3]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(fk[:3,:3]@actual[:3,:3].T)-1)/2,-1,1))))
        sample = {'open_success': moved, 'width_m': gripper.state().width_m,
                  'fk_translation_error_m': error, 'fk_rotation_error_deg': angle,
                  'T_base_ee': actual.tolist(), 'T_ee_camera': obs.T_ee_camera.tolist(),
                  'sequence': obs.sequence, 'depth_valid_pixels': int(np.isfinite(obs.depth_m).sum())}
        report['samples'].append(sample)
        for name, cam in cameras.items():
            rgba = cam.rgb_rgba()
            if rgba is not None:
                cv2.imwrite(str(args.output/f'{name}_{index}.png'), rgba[:,:,:3][:,:,::-1])
        np.savez_compressed(args.output/f'rgbd_{index}.npz', rgb=obs.rgb, depth_m=obs.depth_m,
                            T_base_camera=obs.T_base_camera, T_ee_camera=obs.T_ee_camera,
                            T_base_ee=obs.T_base_ee, K=obs.intrinsics.matrix)
        print('PROFILE_SAMPLE '+json.dumps(sample), flush=True)
    report['passed'] = all(s['open_success'] and s['fk_translation_error_m']<.001
                           and s['fk_rotation_error_deg']<.5 for s in report['samples'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PROFILE_PASSED', report['passed'], flush=True)
finally:
    app.close()
