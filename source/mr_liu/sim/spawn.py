"""USD scene authoring: table, SO-101, loose tabletop props, and HDRI."""

from __future__ import annotations

from mr_liu.config import robot_config, scene_config
from mr_liu.sim.assets import resolve_isaac_asset

MOUNT_JOINT_PATH = "/World/SO101Mount"
TABLETOP_PROPS_ROOT = "/World/TabletopProps"


def apply_environment_map() -> None:
    from isaacsim.core.experimental.objects import DomeLight
    from pxr import Sdf

    cfg = scene_config()
    hdri = resolve_isaac_asset(cfg["hdri_rel"])
    dome = DomeLight(
        "/World/DomeLight",
        texture_files=hdri,
        texture_formats="latlong",
    )
    dome.set_intensities(float(cfg["dome_intensity"]))
    dome.prims[0].CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
    print(f"[mr_liu] Environment HDRI: {hdri}")


def _find_so101_base(stage):
    from pxr import UsdPhysics

    robot_cfg = robot_config()
    prim_root = robot_cfg["usd_prim_path"]
    for path in (f"{prim_root}/base", f"{prim_root}/base_link"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prim_root + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetName() in ("base", "base_link"):
                return prim
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prim_root + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return None


def mount_so101_to_table() -> None:
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    base = _find_so101_base(stage)
    if base is None:
        raise RuntimeError("SO-101 base rigid body not found. Load the robot first.")

    cfg = robot_config()
    root = stage.GetPrimAtPath(cfg["articulation_prim_path"])
    # The asset already has a world-fixed articulation root. Reuse it instead
    # of constraining the base a second time at a different world frame.
    if root.IsValid() and root.IsA(UsdPhysics.FixedJoint):
        joint = UsdPhysics.FixedJoint(root)
        duplicate = stage.GetPrimAtPath(MOUNT_JOINT_PATH)
        if duplicate.IsValid():
            duplicate.SetActive(False)
    else:
        joint = UsdPhysics.FixedJoint.Define(stage, MOUNT_JOINT_PATH)
        root = joint.GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
    joint.CreateBody0Rel().ClearTargets(True)
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    world = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    quat = world.ExtractRotationQuat()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary())))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    joint.CreateBreakForceAttr().Set(float("inf"))
    joint.CreateBreakTorqueAttr().Set(float("inf"))
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
    articulation.CreateSolverPositionIterationCountAttr().Set(int(cfg.get("solver_position_iterations", 32)))
    articulation.CreateSolverVelocityIterationCountAttr().Set(int(cfg.get("solver_velocity_iterations", 1)))
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(cfg["usd_prim_path"]+"/") and prim.IsA(UsdPhysics.RevoluteJoint):
            cap = cfg.get("gripper_max_velocity_deg_s", 120) if prim.GetName() == cfg["gripper_joint"] else cfg.get("drive_max_velocity_deg_s", 60)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(float(cap))
    print(f"[mr_liu] Mounted {base.GetPath()} using single fixed joint {joint.GetPath()}")


def _subtree_prims(root):
    """Yield a root and its composed descendants, including instance proxies."""
    from pxr import Usd

    return Usd.PrimRange(root, Usd.TraverseInstanceProxies())


def _ensure_dynamic_rigid_body(stage, prim_path: str, mass: float) -> tuple[int, int]:
    """Ensure a referenced prop has a rigid body, colliders, and explicit mass."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"Tabletop prop did not compose: {prim_path}")

    prims = list(_subtree_prims(root))
    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]

    if not rigid_bodies:
        UsdPhysics.RigidBodyAPI.Apply(root)
        rigid_bodies = [root]

    if not colliders:
        for prim in prims:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            colliders.append(prim)

    # Factory task assets may be authored as one-body articulations.  A loose
    # tabletop prop should instead participate as a regular gravity-driven body.
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for prim in rigid_bodies:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False).Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False).Set(False)

    mass_api = UsdPhysics.MassAPI.Apply(rigid_bodies[0])
    mass_api.CreateMassAttr(float(mass))
    return len(rigid_bodies), len(colliders)


def _set_display_color(gprim, color=(0.32, 0.34, 0.38)) -> None:
    from pxr import Gf

    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _spawn_procedural_wrench(stage, prop: dict) -> None:
    """Author a compact compound rigid-body wrench when no library asset exists."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim_path = str(prop["prim_path"])
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()

    def cube(name: str, translation, size, angle_deg: float = 0.0) -> None:
        geom = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        geom.CreateSizeAttr(1.0)
        _set_display_color(geom)
        xform = UsdGeom.Xformable(geom)
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        if angle_deg:
            xform.AddRotateZOp().Set(float(angle_deg))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    # 130 mm long, low-profile wrench sized for the SO-101 gripper.
    cube("Handle", (-0.005, 0.0, 0.0), (0.090, 0.014, 0.008))
    cube("JawUpper", (0.047, 0.014, 0.0), (0.040, 0.012, 0.008), 28.0)
    cube("JawLower", (0.047, -0.014, 0.0), (0.040, 0.012, 0.008), -28.0)
    cylinder = UsdGeom.Cylinder.Define(stage, f"{prim_path}/RingEnd")
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(0.018)
    cylinder.CreateHeightAttr(0.008)
    _set_display_color(cylinder)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(-0.058, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(prop["mass"]))


def spawn_tabletop_props() -> list[str]:
    """Spawn configured loose objects as dynamic, labelled tabletop props."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.usd
    from isaacsim.core.experimental.prims import XformPrim
    from pxr import Semantics, UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, TABLETOP_PROPS_ROOT)
    spawned: list[str] = []

    for prop in scene_config().get("tabletop_props", []):
        name = str(prop["name"])
        kind = str(prop.get("kind", "asset"))
        prim_path = str(prop["prim_path"])
        print(f"[mr_liu] Loading prop {name} ({kind}) at {prim_path}")
        if kind == "asset":
            asset_path = resolve_isaac_asset(str(prop["asset_rel"]))
            stage_utils.add_reference_to_stage(usd_path=asset_path, path=prim_path)
        elif kind == "procedural_wrench":
            asset_path = "procedural compound rigid body"
            _spawn_procedural_wrench(stage, prop)
        else:
            raise ValueError(f"Unsupported tabletop prop kind {kind!r} for {name!r}")

        xform_kwargs = {
            "positions": [prop["position"]],
            "orientations": [prop.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])],
            "reset_xform_op_properties": True,
        }
        if "scale" in prop:
            xform_kwargs["scales"] = [prop["scale"]]
        XformPrim(prim_path, **xform_kwargs)

        rigid_count, collider_count = _ensure_dynamic_rigid_body(
            stage, prim_path, float(prop["mass"])
        )
        root = stage.GetPrimAtPath(prim_path)
        semantics = Semantics.SemanticsAPI.Apply(root, "Semantics")
        semantics.CreateSemanticTypeAttr().Set("class")
        semantics.CreateSemanticDataAttr().Set(str(prop.get("semantic_label", name)))
        spawned.append(prim_path)
        print(
            f"[mr_liu] Prop {name}: {asset_path} at {prim_path} "
            f"rigid_bodies={rigid_count} colliders={collider_count}"
        )
    return spawned


def spawn_table_and_so101() -> None:
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import XformPrim

    scene = scene_config()
    robot = robot_config()
    apply_environment_map()

    table_usd = resolve_isaac_asset(scene["table_usd_rel"])
    so101_usd = resolve_isaac_asset(robot["usd_asset_rel"])
    stage_utils.add_reference_to_stage(usd_path=table_usd, path="/World/Table")
    stage_utils.add_reference_to_stage(usd_path=so101_usd, path=robot["usd_prim_path"])

    XformPrim(
        "/World/Table",
        positions=[scene["table_pos"]],
        orientations=[scene["table_orient_wxyz"]],
        reset_xform_op_properties=True,
    )
    XformPrim(
        robot["usd_prim_path"],
        positions=[scene["robot_pos"]],
        orientations=[scene.get("robot_orient_wxyz", [1.0, 0.0, 0.0, 0.0])],
        reset_xform_op_properties=True,
    )
    print(f"[mr_liu] Table: {table_usd}")
    print(f"[mr_liu] SO-101: {so101_usd}")
    mount_so101_to_table()
    spawn_tabletop_props()
