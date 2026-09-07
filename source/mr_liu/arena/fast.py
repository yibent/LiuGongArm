"""NVIDIA's existing PickPlaceController phase machine, driven through Arena IK.

Only the Cartesian/gripper adapters live here. The SDK supplies the phase
interpolation; this module neither builds a second robot nor solves joint IK.
"""
import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.controllers.pick_place_controller import PickPlaceController

from mr_liu.arena.cascade import FastPathFailure
from mr_liu.grasp.transforms import invert_transform, transform_points


class ArenaCartesian:
    def __init__(self, runtime):
        self.runtime = runtime

    def forward(self, target_end_effector_position, target_end_effector_orientation):
        pose = np.eye(4)
        pose[:3, 3] = target_end_effector_position
        # The legacy controller API uses WXYZ; the Arena action uses XYZW.
        pose[:3, :3] = Rotation.from_quat(np.roll(target_end_effector_orientation, -1)).as_matrix()
        self.runtime.goal = pose
        return ArticulationAction()

    def reset(self):
        pass


class ArenaGripper:
    def __init__(self, runtime):
        self.runtime = runtime

    def forward(self, action):
        self.runtime.gripper = -1. if action == "close" else 1.
        return ArticulationAction()


PHASES = ["pregrasp", "approach", "settle_grasp", "close_gripper", "lift",
          "transport", "place_approach", "release", "retreat", "finish"]


def verify_release_pose(runtime, position, orientation_wxyz):
    """The SDK advances on time; release must also wait for actual Arena IK arrival."""
    if not runtime.holding_status()['verified']:
        raise FastPathFailure('夹持状态已改变，释放前未确认持物。')
    pose=np.eye(4);pose[:3,3]=position
    pose[:3,:3]=Rotation.from_quat(np.roll(orientation_wxyz,-1)).as_matrix()
    runtime.move(pose,label='place_approach')


def fast_pick_place(runtime, request):
    row, destination = runtime.prepare_task(request)
    points = runtime.cloud(row["name"])
    observation_ref = runtime.visual_result['request_id']
    geometry = None
    low, high = np.quantile(points, [.02, .98], axis=0)
    pick = (low + high) / 2
    if request.cell_ref and high[2]-low[2] > max(high[:2]-low[:2]):
        # Pinch above short dividers instead of putting the fingers between them.
        pick[2] = low[2]+.8*(high[2]-low[2])
    place = pick.copy()
    if destination:
        support = runtime.placement_support(request, destination, points, runtime.tcp_pose()[:3, 3])
        place[:2] = support[:2]
        place[2] = support[2] + pick[2]-low[2] + .003
    runtime.event("planning", route=request.route(), object_points=len(points),
                  algorithm="isaacsim PickPlaceController / Arena IK")
    steps = runtime.config["fast"]["phase_steps"]
    controller = PickPlaceController("arena_fast_pick_place", ArenaCartesian(runtime), ArenaGripper(runtime),
        end_effector_initial_height=max(pick[2], place[2]) + .14,
        events_dt=[1. / count for count in steps])
    place_orientation = np.array([0., 1., 0., 0.])
    if destination and destination.get('yaw_delta_rad'):
        rotation = Rotation.from_euler('z', destination['yaw_delta_rad']) * Rotation.from_quat([1., 0., 0., 0.])
        place_orientation = np.roll(rotation.as_quat(), 1)
    previous = -1
    while not controller.is_done():
        phase = controller.get_current_event()
        if phase != previous:
            if phase == 4:
                # Preserve measured pregrasp geometry at closure, before lift
                # and jaw occlusion make another segmentation unreliable.
                geometry = {'points_tcp': transform_points(invert_transform(runtime.tcp_pose()), points),
                            'observation_ref': observation_ref}
                runtime.remember_hold(row, geometry)
            if phase == 5:
                if runtime.max_lift < .04:
                    raise FastPathFailure("Fast grasp did not lift the target")
                if not runtime.holding_status()['verified']:
                    raise FastPathFailure('夹持状态已改变，抬升后物体未跟随夹爪。')
                runtime.held_context['max_lift_m'] = runtime.max_lift
                runtime.event("lift_verified", lift_m=runtime.max_lift)
                if destination is None:
                    return runtime.task.evaluate(runtime.env, row["name"], runtime.initial_z, None,
                        released=False, max_lift=runtime.max_lift, stability=0.)
            if phase == 7: verify_release_pose(runtime,place,place_orientation)
            runtime.event(PHASES[phase], backend="official_pick_place")
            if phase == 3: runtime.held = row["name"]
            if phase == 8: runtime.clear_hold()
            previous = phase
        controller.forward(pick, place, np.zeros(9), end_effector_orientation=place_orientation if phase >= 5 else np.array([0., 1., 0., 0.]))
        runtime.tick()
    before = runtime.object_pose(row["name"])[:3, 3]
    for _ in range(45): runtime.tick()
    stability = np.linalg.norm(runtime.object_pose(row["name"])[:3, 3] - before)
    return runtime.task.evaluate(runtime.env, row["name"], runtime.initial_z, destination,
        released=True, max_lift=runtime.max_lift, stability=stability)


def fast_place_held(runtime, request, row, destination):
    """Resume the official controller at transport, preserving the real grasp.

Warm the controller's first phases with detached adapters, without stepping
the robot. Then use its original transport/release/retreat interpolation.
"""
    child = runtime.held_cloud()
    tcp = runtime.tcp_pose()
    support = runtime.placement_support(request, destination, child, tcp[:3, 3])
    low, high = np.quantile(child, [.02, .98], axis=0)
    centre = (low + high) / 2
    place = tcp[:3, 3].copy()
    delta = Rotation.from_euler('z', destination.get('yaw_delta_rad', 0.)).apply(tcp[:3, 3]-centre)
    place[:2] = support[:2]+delta[:2]
    place[2] = support[2] + tcp[2, 3] - low[2] + .003
    pick = tcp[:3, 3].copy()
    rotation = Rotation.from_euler('z', destination.get('yaw_delta_rad', 0.)) * Rotation.from_matrix(tcp[:3, :3])
    orientation = np.roll(rotation.as_quat(), 1)
    detached = SimpleNamespace()
    arm, gripper = ArenaCartesian(detached), ArenaGripper(detached)
    steps = runtime.config['fast']['phase_steps']
    controller = PickPlaceController('arena_fast_place_held', arm, gripper,
        end_effector_initial_height=max(pick[2], place[2] + .13),
        events_dt=[1.] * 5 + [1. / count for count in steps[5:]])
    while controller.get_current_event() < 5:
        controller.forward(pick, place, np.zeros(9), end_effector_orientation=orientation)
    arm.runtime = gripper.runtime = runtime
    runtime.event('planning', route={**request.route(), 'grasp': 'retained_from_previous_command'},
                  algorithm='isaacsim PickPlaceController transport/release/retreat', object_points=len(child))
    previous = -1
    while not controller.is_done():
        phase = controller.get_current_event()
        if phase != previous:
            if phase == 7: verify_release_pose(runtime,place,orientation)
            runtime.event(PHASES[phase], backend='official_pick_place')
            if phase == 8: runtime.clear_hold()
            previous = phase
        controller.forward(pick, place, np.zeros(9), end_effector_orientation=orientation)
        runtime.tick()
    before = runtime.object_pose(row['name'])[:3, 3]
    for _ in range(45): runtime.tick()
    stability = np.linalg.norm(runtime.object_pose(row['name'])[:3, 3] - before)
    return runtime.task.evaluate(runtime.env, row['name'], runtime.initial_z, destination,
        released=True, max_lift=runtime.max_lift, stability=stability)
