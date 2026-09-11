"""Injectable Isaac Sim world-truth coordinates for deterministic tests.

The provider deliberately depends only on a small body lookup callable, so tests
can use a fake body and production code can pass ``env.scene[name]`` without
coupling the perception stack to Isaac imports.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from mr_liu.arena.arrays import numpy_data


@dataclass(frozen=True)
class TruthTarget:
    entity_name: str
    position_world_m: tuple[float, float, float]
    quaternion_world_xyzw: tuple[float, float, float, float]
    frame: str = "world"
    source: str = "isaac_world_truth"
    test_only: bool = True


class IsaacWorldTruthProvider:
    """Read the current rigid-body root pose on the simulation thread."""

    def __init__(self, body_lookup: Callable[[str], object]):
        self._body_lookup = body_lookup

    def locate(self, entity_name: str) -> TruthTarget:
        if not entity_name or not isinstance(entity_name, str):
            raise ValueError("entity_name must be a non-empty string")
        body = self._body_lookup(entity_name)
        data = getattr(body, "data", body)
        try:
            position = np.asarray(numpy_data(data.root_pos_w))[0].reshape(-1)[:3]
            quaternion = np.asarray(numpy_data(data.root_quat_w))[0].reshape(-1)[:4]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"body {entity_name!r} has no valid root pose") from exc
        if position.size != 3 or quaternion.size != 4:
            raise ValueError(f"body {entity_name!r} has invalid root pose dimensions")
        return TruthTarget(entity_name, tuple(float(x) for x in position),
                           tuple(float(x) for x in quaternion))
