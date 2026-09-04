# SO-101 cuMotion description

- `so101_new_calib.urdf` — upstream TheRobotStudio URDF (mesh references, not used by the planner).
- `robot.urdf` — kinematics-only URDF for cuMotion (`load_cumotion_robot`).
- `robot.xrdf` — cspace, tool frame `gripper_frame_link`, collision spheres.
- `rmp_flow.yaml` — RMPflow gains for the 5 arm joints.

Joint names match the Isaac Sim 6.0 USD asset `so101_new_calib`.
