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
from mr_liu.arena.orientation import endpoint_vector
from mr_liu.arena.grasp_geometry import top_grasp_orientation


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


def verify_release_pose(runtime, position, orientation_wxyz, row=None, destination=None):
    """The SDK advances on time; release must also wait for actual Arena IK arrival."""
    if not runtime.holding_status()['verified']:
        raise FastPathFailure('夹持状态已改变，释放前未确认持物。')
    pose=np.eye(4);pose[:3,3]=position
    pose[:3,:3]=Rotation.from_quat(np.roll(orientation_wxyz,-1)).as_matrix()
    contact = (lambda: runtime.task.support_contact(
        runtime.env, row['name'], destination)) if row is not None and destination is not None else None
    runtime.move(pose,label='place_approach', until_contact=contact)


def fast_pick_place(runtime, request):
    row, destination = runtime.prepare_task(request)
    points = runtime.cloud(row["name"])
    observation_ref = runtime.visual_result['request_id']
    scene = runtime.perception.scene_cloud(runtime.visual_result)
    binding = runtime.bind_orientation(request, row, points)
    if binding and endpoint_vector(binding['axis_world'],binding['endpoint'],binding['direction'])[2] < -.7:
        raise FastPathFailure('端点翻转需要侧向抓法，使用 GraspGenX 选择可翻转的抓取姿态。')
    geometry = None
    low, high = np.quantile(points, [.02, .98], axis=0)
    pick = (low + high) / 2
    grasp_orientation = np.array([0., 1., 0., 0.])
    if binding and abs(binding['axis_world'][2]) < .5:
        # Pinch the upper half of a horizontal part; keep the closing
        # direction perpendicular to its observed long axis.
        axis = np.asarray(binding['axis_world'])
        # Pinch toward the end that will be uppermost. After standing the
        # part up, the fingers and wrist remain above the container walls.
        upper = binding['endpoint'] if binding['direction']=='up' else 1-binding['endpoint']
        pick += axis*(np.quantile(points@axis,.78 if upper else .22)-pick@axis)
        pick[2] = high[2]-.35*(high[2]-low[2])
        yaw = np.arctan2(binding['axis_world'][1],binding['axis_world'][0])
        # An oblique pinch leaves the palm above the rim after the part stands
        # up. Jaws still close across the observed shaft, not along its length.
        tilt = np.deg2rad(30 if upper == 0 else -30)
        grasp_orientation = np.roll((Rotation.from_euler('z',yaw)*Rotation.from_euler('x',np.pi)
                                     *Rotation.from_euler('y',tilt)).as_quat(),1)
    if request.cell_ref and high[2]-low[2] > max(high[:2]-low[:2]):
        # Pinch above short dividers instead of putting the fingers between them.
        pick[2] = low[2]+.8*(high[2]-low[2])
    if not binding or abs(binding['axis_world'][2]) >= .5:
        grasp_orientation, clearance = top_grasp_orientation(pick,scene,runtime.tcp_pose()[:3,:3])
        runtime.event('grasp_clearance_selected', **clearance)
    place = pick.copy()
    if destination and not request.orientation and request.relation in {'on', 'inside'}:
        support = runtime.placement_support(request, destination, points, runtime.tcp_pose()[:3, 3])
        place[:2] = support[:2]
        place[2] = support[2] + pick[2]-low[2] + .003
    runtime.event("planning", route=request.route(), object_points=len(points),
                  algorithm="isaacsim PickPlaceController / Arena IK",
                  grasp_position_world_m=pick.tolist(), grasp_orientation_wxyz=grasp_orientation.tolist())
    steps = runtime.config["fast"]["phase_steps"]
    controller = PickPlaceController("arena_fast_pick_place", ArenaCartesian(runtime), ArenaGripper(runtime),
        end_effector_initial_height=max(pick[2], place[2]) + .14,
        events_dt=[1. / count for count in steps])
    place_orientation = grasp_orientation.copy()
    if destination and destination.get('yaw_delta_rad'):
        rotation = Rotation.from_euler('z', destination['yaw_delta_rad']) * Rotation.from_quat(np.roll(grasp_orientation,-1))
        place_orientation = np.roll(rotation.as_quat(), 1)
    previous = -1
    while not controller.is_done():
        phase = controller.get_current_event()
        if phase != previous:
            if phase == 3:
                # The SDK advances by elapsed steps even if IK is lagging.
                # Finish the observed approach before closing the fingers.
                pose=np.eye(4);pose[:3,3]=pick
                pose[:3,:3]=Rotation.from_quat(np.roll(grasp_orientation,-1)).as_matrix()
                runtime.move(pose,label='approach')
            if phase == 4:
                # Preserve measured pregrasp geometry at closure, before lift
                # and jaw occlusion make another segmentation unreliable.
                geometry = {'points_tcp': transform_points(invert_transform(runtime.tcp_pose()), points),
                            'observation_ref': observation_ref}
                runtime.remember_hold(row, geometry)
                runtime.event('grasp_closure_measured', holding_measurement=runtime.holding_status())
            if phase == 5:
                if runtime.max_lift < .04:
                    raise FastPathFailure("Fast grasp did not lift the target")
                if not runtime.holding_status()['verified']:
                    raise FastPathFailure('夹持状态已改变，抬升后物体未跟随夹爪。')
                runtime.held_context['max_lift_m'] = runtime.max_lift
                runtime.event("lift_verified", lift_m=runtime.max_lift, holding_measurement=runtime.holding_status())
                if destination is None:
                    return runtime.task.evaluate(runtime.env, row["name"], runtime.initial_z, None,
                        released=False, max_lift=runtime.max_lift, stability=0.)
                if request.relation in {'insert', 'sleeve_on_peg', 'hang'}:
                    return runtime.contact_place_held(request, row, destination)
                if request.orientation:
                    return runtime.oriented_place_held(request, row, destination)
            if phase == 7: verify_release_pose(runtime,place,place_orientation,row,destination)
            runtime.event(PHASES[phase], backend="official_pick_place")
            if phase == 3: runtime.held = row["name"]
            if phase == 8: runtime.clear_hold()
            previous = phase
        controller.forward(pick, place, np.zeros(9), end_effector_orientation=place_orientation if phase >= 5 else grasp_orientation)
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
    if request.relation in {'insert', 'sleeve_on_peg', 'hang'}:
        return runtime.contact_place_held(request, row, destination)
    if request.orientation:
        return runtime.oriented_place_held(request, row, destination)
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
            if phase == 7: verify_release_pose(runtime,place,orientation,row,destination)
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
