"""Composable Arena scene with the official Panda embodiment and IK action.

Import only after AppLauncher. No project IK solver, joint servo or attachment
constraint is used. Primitive assets make the first regression self-contained.
Isaac Lab 3 uses XYZW quaternions throughout this module.
"""
from argparse import Namespace

import numpy as np
from mr_liu.arena.arrays import numpy_data
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.no_task import NoTask


class ConfiguredAsset(Asset):
    def __init__(self, name, cfg):
        super().__init__(name)
        self.cfg = cfg

    def get_object_cfg(self):
        return self.name, self.cfg

    def get_event_cfg(self):
        return self.name, None


class PandaTask(NoTask):
    """Physical evaluation belongs to the Arena task, never to model scores."""
    name = "liugong_pick_place"

    def __init__(self):
        super().__init__()
        self.episode_length_s = 600
        self.task_description = "Pick the selected industrial part and place it on the selected support."

    def get_viewer_cfg(self):
        return ViewerCfg(eye=(1.3, 1.2, 1.1), lookat=(.4, 0., .05))

    def support_contact(self, env, name, destination):
        sensor = env.scene[f"{name}_{destination['name']}_contact"]
        forces = numpy_data(sensor.data.force_matrix_w)
        return bool(np.any(np.linalg.norm(forces, axis=-1) > 0.))

    def evaluate(self, env, name, initial_z, destination, *, released, max_lift, stability):
        body = env.scene[name]
        position = numpy_data(body.data.root_pos_w)[0]
        velocity = numpy_data(body.data.root_lin_vel_w)[0]
        xy_error = None
        supported = False
        support_gap = None
        if destination is not None:
            xy_error = float(np.linalg.norm(position[:2] - np.asarray(destination["position"])[:2]))
            half = np.asarray(destination["size"])[:2] / 2
            size = np.asarray(self.object_sizes[name])
            rotation = Rotation.from_quat(numpy_data(body.data.root_quat_w)[0]).as_matrix()
            world_half = np.abs(rotation) @ (size / 2)
            support_gap = float(position[2] - world_half[2] - destination["position"][2] - destination["size"][2] / 2)
            supported = bool(np.all(np.abs(position[:2] - np.asarray(destination["position"])[:2]) + world_half[:2] < half)
                             and abs(support_gap) < .006)
        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints("panda_finger_joint.*")
        opening = float(numpy_data(robot.data.joint_pos)[0, finger_ids].sum())
        tcp = numpy_data(env.scene["ee_frame"].data.target_pos_w)[0, 0]
        released = bool(released and opening > .065 and np.linalg.norm(tcp - position) > .08)
        lifted = max_lift >= .04
        success = bool(lifted and (released and supported and stability < .003 and np.linalg.norm(velocity) < .03
                                   if destination is not None else position[2] - initial_z > .04))
        return {"physical_success": success, "lifted": bool(lifted), "released": bool(released),
                "max_lift_m": float(max_lift), "final_position_world_m": position.tolist(),
                "destination_xy_error_m": xy_error, "supported_region": supported,
                "support_gap_m": support_gap, "gripper_opening_m": opening,
                "settle_displacement_m": float(stability), "linear_speed_mps": float(np.linalg.norm(velocity)),
                "evaluation_source": "arena_task_simulation_ground_truth"}


def _camera(path, width, height, position, lookat):
    forward = np.asarray(lookat) - np.asarray(position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    quat = Rotation.from_matrix(np.stack([right, down, forward], axis=1)).as_quat()
    return CameraCfg(
        prim_path=path, update_period=0., width=width, height=height,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        spawn=sim.PinholeCameraCfg(focal_length=18., horizontal_aperture=20.955, clipping_range=(.02, 5.)),
        offset=CameraCfg.OffsetCfg(pos=position, rot=tuple(quat), convention="ros"),
    )


def build_environment(config, *, device="cuda:0"):
    embodiment = FrankaIKEmbodiment(enable_cameras=False)
    # Reuse the branch's official Franka Panda, without Arena's optional stand.
    embodiment.scene_config.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # The Arena/Lab release points at an unavailable staging Panda asset. Use
    # the Sim 6 production asset verified by the existing Franka branch.
    embodiment.scene_config.robot.spawn.usd_path = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
        "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")
    embodiment.scene_config.robot.spawn.variants = {"Gripper": "Default", "Mesh": "Performance"}
    embodiment.scene_config.robot.spawn.activate_contact_sensors = True
    # Match the Panda branch's 20 N per-finger grasp. The training asset's
    # 200 N default excites tilted contacts on these light parts in Sim 6.
    embodiment.scene_config.robot.actuators["panda_hand"].effort_limit_sim = 20.
    embodiment.scene_config.robot.spawn.articulation_props.solver_position_iteration_count = 32
    embodiment.scene_config.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    embodiment.set_initial_joint_pose([0., -.785398, 0., -2.356194, 0., 1.570796, .785398, .04, .04])
    embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.
    # Configure the existing Isaac Lab IK action for absolute full poses.
    embodiment.action_config.arm_action.controller.use_relative_mode = False
    embodiment.action_config.arm_action.scale = 1.
    embodiment.action_config.arm_action.body_offset.pos = [0., 0., .1034]
    assets = []
    for row in config["objects"] + config["destinations"]:
        dynamic = row in config["objects"]
        shape = dict(
            activate_contact_sensors=True,
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=not dynamic,
                                                  disable_gravity=not dynamic,
                                                  solver_position_iteration_count=32,
                                                  solver_velocity_iteration_count=4),
            mass_props=sim.MassPropertiesCfg(mass=row.get("mass", .1)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(row["color"])),
            semantic_tags=[("class", row["name"])],
        )
        spawner = (sim.CylinderCfg(radius=row["size"][0] / 2, height=row["size"][2], **shape)
                   if row.get("shape") == "cylinder" else sim.CuboidCfg(size=tuple(row["size"]), **shape))
        cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + row["name"], spawn=spawner,
                             init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(row["position"])))
        assets.append(ConfiguredAsset(row["name"], cfg))
    assets += [
        ConfiguredAsset("floor", AssetBaseCfg(prim_path="/World/Ground", spawn=sim.CuboidCfg(
            size=(2., 2., .04), collision_props=sim.CollisionPropertiesCfg(),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.22, .25, .24))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(.3, 0., -.02)))),
        ConfiguredAsset("light", AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=2500.))),
    ]
    task = PandaTask()
    task.object_sizes = {row["name"]: row["size"] for row in config["objects"]}

    def configure(cfg):
        for obj in config["objects"]:
            for destination in config["destinations"]:
                setattr(cfg.scene, f"{obj['name']}_{destination['name']}_contact", ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/" + obj["name"],
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/" + destination["name"]], update_period=0.))
        width, height = config["camera"]["width"], config["camera"]["height"]
        cfg.scene.scene_camera = _camera("{ENV_REGEX_NS}/SceneCamera", width, height,
                                         (.52, -.02, .78), (.46, 0., 0.))
        cfg.scene.side_camera = _camera("{ENV_REGEX_NS}/SideCamera", width, height,
                                        (.95, .35, .55), (.45, 0., .03))
        cfg.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristRGBD", update_period=0.,
            update_latest_camera_pose=True,
            width=width, height=height, data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            colorize_semantic_segmentation=True,
            spawn=sim.PinholeCameraCfg(focal_length=12., horizontal_aperture=20.955, clipping_range=(.02, 4.)),
            offset=CameraCfg.OffsetCfg(pos=(.05, 0., .02), rot=(0., 0., 0., 1.), convention="ros"))
        cfg.sim.dt = 1 / 120
        cfg.seed = 0
        cfg.decimation = 2
        cfg.sim.render_interval = 2
        cfg.sim.device = device
        cfg.wait_for_textures = False
        return cfg

    description = IsaacLabArenaEnvironment(
        name="LiuGong-Arena-Panda-v0", embodiment=embodiment, scene=Scene(assets=assets),
        task=task, env_cfg_callback=configure)
    options = Namespace(num_envs=1, env_spacing=2., solve_relations=False, mimic=False,
                        device=device, disable_fabric=False, presets=None)
    env = ArenaEnvBuilder(description, options).make_registered()
    env.reset()
    # RTX annotators need several render frames after attachment before depth
    # buffers exist. Prime them through Lab's renderer before reading cameras.
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    for _ in range(20):
        env.unwrapped.sim.step()
        ensure_isaac_rtx_render_update()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
    return env, task
