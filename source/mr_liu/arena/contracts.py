"""Semantic commands and explicit SE(3) conventions, independent of Isaac/CUDA."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mr_liu.grasp.transforms import assert_transform, invert_transform


@dataclass(frozen=True)
class ManipulationRequest:
    target: str
    destination: str | None = None
    mode: Literal["auto", "basic", "enhanced"] = "auto"
    unfamiliar: bool = False
    cluttered: bool = False
    precise: bool = False
    relation: str = "on"
    placement_selection: Literal['center', 'free_space'] = 'center'

    def __post_init__(self):
        if not self.target.strip() or len(self.target) > 256:
            raise ValueError("A target label is required")
        if self.destination is not None and (not self.destination.strip() or len(self.destination) > 256):
            raise ValueError("A destination label is required for placement")
        if self.mode not in {"auto", "basic", "enhanced"}:
            raise ValueError("mode must be auto, basic or enhanced")
        if self.placement_selection not in {'center', 'free_space'}:
            raise ValueError('placement_selection must be center or free_space')
        if self.relation not in {"on", "inside"}:
            raise ValueError("Insertion and hanging require contact skills that are not yet available")

    @property
    def enhanced(self):
        return self.mode == "enhanced" or (
            self.mode == "auto" and (self.unfamiliar or self.cluttered or self.precise)
        )

    def route(self):
        return {
            "decision": "busagent",
            "grasp": "graspgenx" if self.enhanced else "official_pick_place",
            "placement": ("anyplace" if self.enhanced else "official_pick_place") if self.destination else None,
            "execution": "isaaclab_arena.franka_ik",
            "robot": "franka_panda",
        }


def placement_to_tcp(relative, object_input, tcp_to_object):
    """AnyPlace transforms the input child cloud; it does not return an EE pose."""
    for value in (relative, object_input, tcp_to_object):
        assert_transform(np.asarray(value))
    return np.asarray(relative) @ object_input @ invert_transform(tcp_to_object)


def model_grasp_to_tcp(model_pose, fingertip_depth=0.1034):
    """GraspGenX +X closing/+Z approach -> Panda hand +Y closing/+Z approach.

    Model origin is the gripper base; Arena's controlled TCP is at the pinch
    centre 103.4 mm along panda_hand Z. All input/output transforms are world SE(3).
    """
    assert_transform(np.asarray(model_pose))
    offset = np.eye(4)
    offset[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    offset[2, 3] = fingertip_depth
    return np.asarray(model_pose) @ offset
