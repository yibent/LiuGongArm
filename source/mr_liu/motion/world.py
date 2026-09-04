from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation
from isaacsim.robot_motion.cumotion import CumotionWorldInterface

from mr_liu.config import motion_config, robot_config


def bind_world(articulation: Articulation, root_poses=None) -> mg.WorldBinding:
    motion = motion_config()
    robot_prim = robot_config()["usd_prim_path"]
    exclude = list(motion.get("exclude_from_obstacles") or [])
    if robot_prim not in exclude:
        exclude.append(robot_prim)

    poses = root_poses if root_poses is not None else articulation.get_world_poses()
    robot_pos, _robot_ori = poses
    objects = mg.SceneQuery().get_prims_in_aabb(
        search_box_origin=robot_pos.numpy()[0],
        search_box_minimum=[-10.0, -10.0, -10.0],
        search_box_maximum=[10.0, 10.0, 10.0],
        tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
        exclude_prim_paths=exclude,
    )

    margin = float(motion.get("obstacle_obb_margin", 0.02))
    obstacle_strategy = mg.ObstacleStrategy()
    for prim_type in (Mesh, Cone, Cylinder, Cube):
        obstacle_strategy.set_default_configuration(prim_type, mg.ObstacleConfiguration("obb", margin))

    world_binding = mg.WorldBinding(
        world_interface=CumotionWorldInterface(),
        obstacle_strategy=obstacle_strategy,
        tracked_prims=objects,
        tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
    )
    world_binding.initialize()
    world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=poses)
    world_binding.synchronize_transforms()
    return world_binding
