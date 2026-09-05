"""BusAgent-facing FineGrasp skill facade.

BusAgent should depend on this module and the contracts, not on a particular
neural model, simulator or robot driver.
"""

from __future__ import annotations

from mr_liu.grasp.contracts import FineGraspRequest, FineGraspResult
from mr_liu.grasp.node import GeneralGraspNode


class FineGraspSkill:
    name = "fine_grasp"
    aliases = ("general_grasp", "grasp")

    def __init__(self, node: GeneralGraspNode) -> None:
        self._node = node

    def execute(self, request: FineGraspRequest) -> FineGraspResult:
        return self._node.execute(request)

    def cancel(self) -> None:
        self._node.cancel()
