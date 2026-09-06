"""NVIDIA's existing PickPlaceController phase machine, driven through Arena IK.

Only the Cartesian/gripper adapters live here. The SDK supplies the phase
interpolation; this module neither builds a second robot nor solves joint IK.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.controllers.pick_place_controller import PickPlaceController

from mr_liu.arena.cascade import FastPathFailure


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


def fast_pick_place(runtime, request):
    row, destination = runtime.prepare_task(request)
    points = runtime.cloud(row["name"])
    low, high = np.quantile(points, [.02, .98], axis=0)
    pick = (low + high) / 2
    place = pick.copy()
    if destination:
        parent = runtime.cloud(destination["name"])
        place[:2] = np.asarray(destination["position"])[:2]
        place[2] = np.quantile(parent[:, 2], .95) + (high[2] - low[2]) / 2 + .003
    runtime.event("planning", route=request.route(), object_points=len(points),
                  algorithm="isaacsim PickPlaceController / Arena IK")
    steps = runtime.config["fast"]["phase_steps"]
    controller = PickPlaceController("arena_fast_pick_place", ArenaCartesian(runtime), ArenaGripper(runtime),
        end_effector_initial_height=max(pick[2], place[2]) + .14,
        events_dt=[1. / count for count in steps])
    previous = -1
    while not controller.is_done():
        phase = controller.get_current_event()
        if phase != previous:
            if phase == 5:
                if runtime.max_lift < .04:
                    raise FastPathFailure("Fast grasp did not lift the target")
                runtime.event("lift_verified", lift_m=runtime.max_lift)
                if destination is None:
                    return runtime.task.evaluate(runtime.env, row["name"], runtime.initial_z, None,
                        released=False, max_lift=runtime.max_lift, stability=0.)
            runtime.event(PHASES[phase], backend="official_pick_place")
            if phase == 3: runtime.held = row["name"]
            if phase == 7: runtime.held = None
            previous = phase
        controller.forward(pick, place, np.zeros(9), end_effector_orientation=np.array([0., 1., 0., 0.]))
        runtime.tick()
    before = runtime.object_pose(row["name"])[:3, 3]
    for _ in range(45): runtime.tick()
    stability = np.linalg.norm(runtime.object_pose(row["name"])[:3, 3] - before)
    return runtime.task.evaluate(runtime.env, row["name"], runtime.initial_z, destination,
        released=True, max_lift=runtime.max_lift, stability=stability)
