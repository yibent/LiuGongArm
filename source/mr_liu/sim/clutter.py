"""Deterministic clutter and collision fixtures for Isaac Sim benchmarks.

The fixture deliberately keeps clutter outside ``/World/TabletopProps``.  The
normal motion configuration treats that path as candidate targets and excludes
it from the cuMotion world.  Clutter is therefore authored below dedicated
roots so every fixed body becomes a real planning obstacle while the selected
FineGrasp target remains separately excluded by the executor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ClutterBody:
    """One visible, collision-bearing box or cylinder in world coordinates."""

    name: str
    shape: str
    position_m: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float] = (0.35, 0.38, 0.42)
    semantic_label: str = "obstacle"
    yaw_deg: float = 0.0
    dynamic: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("clutter body name is required")
        if self.shape not in {"box", "cylinder"}:
            raise ValueError(f"unsupported clutter shape: {self.shape}")
        if len(self.position_m) != 3 or len(self.dimensions_m) != 3:
            raise ValueError("clutter position and dimensions must have three values")
        if any(float(value) <= 0.0 for value in self.dimensions_m):
            raise ValueError("clutter dimensions must be positive")
        if len(self.color_rgb) != 3 or any(float(value) < 0.0 or float(value) > 1.0 for value in self.color_rgb):
            raise ValueError("clutter color must contain RGB values in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClutterBody":
        data = dict(value)
        for key in ("position_m", "dimensions_m", "color_rgb"):
            if key in data:
                data[key] = tuple(float(item) for item in data[key])
        data["yaw_deg"] = float(data.get("yaw_deg", 0.0))
        data["dynamic"] = bool(data.get("dynamic", False))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClutterScene:
    """A named table layout and the expected safety outcome for a trial."""

    name: str
    description: str
    bodies: tuple[ClutterBody, ...] = field(default_factory=tuple)
    expected_outcome: str = "grasp_or_replan"
    expected_failure: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("clutter scene name is required")
        if self.expected_outcome not in {"grasp_or_replan", "safe_reject", "recover_or_reject"}:
            raise ValueError(f"unsupported expected clutter outcome: {self.expected_outcome}")
        names = [body.name for body in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("clutter body names must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClutterScene":
        data = dict(value)
        data["bodies"] = tuple(ClutterBody.from_mapping(item) for item in data.get("bodies", []))
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "bodies": [body.to_dict() for body in self.bodies],
            "expected_outcome": self.expected_outcome,
            "expected_failure": self.expected_failure,
            "metadata": dict(self.metadata),
        }


def load_scene(path: str | Path) -> ClutterScene:
    return ClutterScene.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def write_scene(path: str | Path, scene: ClutterScene) -> None:
    Path(path).write_text(json.dumps(scene.to_dict(), indent=2), encoding="utf-8")


def default_clutter_scenes() -> list[ClutterScene]:
    """Return deterministic layouts covering clutter, occlusion and blocked paths.

    Coordinates are in the same world frame as ``configs/scene.yaml`` and the
    target benchmark cases.  Bodies are static by default so tests do not rely
    on a settling race; ``dynamic`` is reserved for explicit moving-obstacle
    fault experiments.
    """

    return [
        ClutterScene(
            name="mixed_tools",
            description="Six compact industrial distractors around the target.",
            bodies=(
                ClutterBody("red_block", "box", (0.42, -0.22, 1.075), (0.055, 0.055, 0.050), (0.78, 0.08, 0.05), "block"),
                ClutterBody("wrench", "box", (0.49, -0.08, 1.060), (0.120, 0.018, 0.018), (0.42, 0.46, 0.50), "wrench", yaw_deg=18.0),
                ClutterBody("drill", "box", (0.55, 0.18, 1.085), (0.070, 0.045, 0.070), (0.08, 0.20, 0.55), "power_drill", yaw_deg=-12.0),
                ClutterBody("bolt", "cylinder", (0.66, -0.18, 1.065), (0.022, 0.022, 0.030), (0.55, 0.56, 0.58), "bolt"),
                ClutterBody("nut", "cylinder", (0.67, -0.05, 1.062), (0.028, 0.028, 0.018), (0.65, 0.67, 0.70), "nut"),
                ClutterBody("gear", "cylinder", (0.60, 0.10, 1.068), (0.050, 0.050, 0.022), (0.20, 0.22, 0.25), "gear"),
            ),
            expected_outcome="recover_or_reject",
            expected_failure="target_not_visible",
            metadata={"category": "clutter", "same_label_instances": 0},
        ),
        ClutterScene(
            name="occluded_target",
            description="A low blocker partially hides the target approach sector.",
            bodies=(
                ClutterBody("front_blocker", "box", (0.24, -0.20, 1.095), (0.090, 0.070, 0.090), (0.12, 0.35, 0.70), "blocker"),
                ClutterBody("side_blocker", "box", (0.40, -0.08, 1.075), (0.045, 0.110, 0.050), (0.70, 0.25, 0.06), "blocker", yaw_deg=22.0),
                ClutterBody("distractor_cube", "box", (0.52, -0.26, 1.075), (0.050, 0.050, 0.050), (0.80, 0.12, 0.06), "red cube"),
                ClutterBody("distractor_cylinder", "cylinder", (0.59, 0.24, 1.075), (0.045, 0.045, 0.050), (0.08, 0.55, 0.28), "cylinder"),
            ),
            expected_outcome="recover_or_reject",
            expected_failure="target_not_visible",
            metadata={"category": "occlusion", "requires_reobserve": True},
        ),
        ClutterScene(
            name="narrow_corridor",
            description="Two tall rails leave a narrow, collision-sensitive corridor.",
            bodies=(
                ClutterBody("left_rail", "box", (0.34, -0.12, 1.18), (0.035, 0.34, 0.26), (0.25, 0.28, 0.32), "rail"),
                ClutterBody("right_rail", "box", (0.34, 0.22, 1.18), (0.035, 0.34, 0.26), (0.25, 0.28, 0.32), "rail"),
                ClutterBody("rear_stop", "box", (0.52, 0.05, 1.14), (0.035, 0.40, 0.18), (0.30, 0.33, 0.38), "rear_stop"),
                ClutterBody("corridor_target_marker", "box", (0.34, 0.05, 1.070), (0.038, 0.038, 0.038), (0.82, 0.08, 0.05), "target_marker"),
            ),
            expected_outcome="safe_reject",
            expected_failure="gripper_open_blocked",
            metadata={"category": "narrow_corridor", "self_collision_risk": True},
        ),
        ClutterScene(
            name="duplicate_label",
            description="Two visually similar red cubes force semantic disambiguation.",
            bodies=(
                ClutterBody("red_cube_a", "box", (0.43, -0.17, 1.075), (0.050, 0.050, 0.050), (0.80, 0.08, 0.05), "red cube"),
                ClutterBody("red_cube_b", "box", (0.43, 0.02, 1.075), (0.050, 0.050, 0.050), (0.80, 0.08, 0.05), "red cube"),
                ClutterBody("rear_tool", "box", (0.57, 0.18, 1.075), (0.100, 0.025, 0.025), (0.45, 0.47, 0.50), "tool", yaw_deg=-20.0),
            ),
            expected_outcome="safe_reject",
            expected_failure="ambiguous_label_instances",
            metadata={"category": "semantic_ambiguity", "same_label_instances": 2},
        ),
        ClutterScene(
            name="moving_blocker",
            description="A dynamic blocker is used to exercise motion-time guard stops.",
            bodies=(
                ClutterBody("moving_blocker", "box", (0.36, -0.04, 1.10), (0.070, 0.070, 0.100), (0.65, 0.18, 0.08), "moving_blocker", dynamic=True),
                ClutterBody("rear_blocker", "box", (0.56, -0.04, 1.085), (0.080, 0.080, 0.070), (0.10, 0.30, 0.65), "blocker"),
                ClutterBody("side_tool", "box", (0.58, 0.24, 1.075), (0.120, 0.022, 0.022), (0.43, 0.45, 0.48), "tool", yaw_deg=12.0),
            ),
            expected_outcome="recover_or_reject",
            expected_failure="target_moved_during_transit",
            metadata={"category": "dynamic_obstacle", "inject_motion_fault": True},
        ),
    ]


def _set_color(geom, color: tuple[float, float, float]) -> None:
    from pxr import Gf

    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def spawn_clutter_scene(scene: ClutterScene) -> list[str]:
    """Author all fixture bodies and return their prim paths.

    The function is intentionally Isaac-only at call time.  It uses static
    PhysX colliders for repeatability and applies rigid-body APIs only to
    explicitly dynamic fault bodies.
    """

    import omni.usd
    from isaacsim.core.experimental.prims import RigidPrim
    from pxr import Gf, Semantics, UsdGeom, UsdPhysics, Vt

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/ClutterProps")
    UsdGeom.Xform.Define(stage, "/World/ClutterObstacles")
    paths: list[str] = []

    def box_mesh(path: str, dimensions: tuple[float, float, float], color: tuple[float, float, float]):
        """Create an exact-size box with identity USD scale.

        cuMotion 1.3 rejects non-uniform xform scales for OBB obstacles.  The
        dimensions therefore live in mesh points, leaving the xform scale
        explicitly uniform at one.
        """
        hx, hy, hz = (float(value) * 0.5 for value in dimensions)
        points = Vt.Vec3fArray([
            Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, -hy, -hz),
            Gf.Vec3f(hx, hy, -hz), Gf.Vec3f(-hx, hy, -hz),
            Gf.Vec3f(-hx, -hy, hz), Gf.Vec3f(hx, -hy, hz),
            Gf.Vec3f(hx, hy, hz), Gf.Vec3f(-hx, hy, hz),
        ])
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4, 4, 4, 4, 4, 4]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([
            0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
            1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7,
        ]))
        mesh.CreateSubdivisionSchemeAttr("none")
        _set_color(mesh, color)
        UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        return mesh

    for body in scene.bodies:
        root = "/World/ClutterProps" if body.semantic_label not in {"blocker", "rail", "rear_stop", "moving_blocker"} else "/World/ClutterObstacles"
        path = f"{root}/{body.name}"
        if body.shape == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr("Z")
            geom.CreateRadiusAttr(float(body.dimensions_m[0]) * 0.5)
            geom.CreateHeightAttr(float(body.dimensions_m[2]))
            # WorldBinding's OBB adapter reads a local scale on every tracked
            # prim, including cylinders whose radius/height are authored as
            # native attributes.
            UsdGeom.Xformable(geom).AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        else:
            geom = box_mesh(path, body.dimensions_m, body.color_rgb)
        _set_color(geom, body.color_rgb)
        xform = UsdGeom.Xformable(geom)
        xform.AddTranslateOp().Set(Gf.Vec3d(*body.position_m))
        if body.yaw_deg:
            xform.AddRotateZOp().Set(float(body.yaw_deg))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        if body.dynamic:
            RigidPrim(path, masses=[0.10], reset_xform_op_properties=False)
        semantic = Semantics.SemanticsAPI.Apply(geom.GetPrim(), "Semantics")
        semantic.CreateSemanticTypeAttr().Set("class")
        semantic.CreateSemanticDataAttr().Set(body.semantic_label)
        paths.append(path)
    return paths
