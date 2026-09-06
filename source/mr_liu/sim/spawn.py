"""USD scene authoring: table, SO-101, loose tabletop props, and HDRI."""

from __future__ import annotations

from mr_liu.config import robot_config, scene_config
from mr_liu.sim.assets import resolve_isaac_asset

MOUNT_JOINT_PATH = "/World/SO101Mount"
TABLETOP_PROPS_ROOT = "/World/TabletopProps"


def apply_grasp_contact_materials(*, target_path: str | None = None) -> dict[str, object]:
    """Bind the explicit Panda contact material used by grasp experiments."""
    import omni.usd
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema, Tf

    from mr_liu.config import robot_config
    cfg = robot_config()
    if cfg.get("name") != "franka":
        return {"enabled": False, "reason": "non_franka_profile"}
    stage = omni.usd.get_context().get_stage()
    robot_root = stage.GetPrimAtPath(cfg.get("usd_prim_path", "/World/Franka"))
    # Official Panda is shipped as an instanceable reference. De-instance the
    # single benchmark copy before authoring per-finger physics bindings.
    if robot_root.IsValid() and robot_root.IsInstance():
        robot_root.SetInstanceable(False)
    settings = cfg.get("contact_material", {})
    material_path = "/World/PhysicsMaterials/PandaGraspContact"
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    physics = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics.CreateStaticFrictionAttr().Set(float(settings.get("static_friction", 1.2)))
    physics.CreateDynamicFrictionAttr().Set(float(settings.get("dynamic_friction", 1.0)))
    physics.CreateRestitutionAttr().Set(float(settings.get("restitution", 0.0)))
    PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    roots = ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]
    if target_path:
        roots.append(str(target_path))
    bound = []
    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            try:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics")
                bound.append(str(prim.GetPath()))
            except Tf.ErrorException:
                # Referenced Panda geometry is commonly an instance proxy and
                # cannot be edited in place. Bind the nearest editable link/root
                # so the material still propagates to its collision descendants.
                editable = root
                while editable.IsInstanceProxy() and editable.GetParent().IsValid():
                    editable = editable.GetParent()
                UsdShade.MaterialBindingAPI.Apply(editable).Bind(
                    material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics")
                bound.append(str(editable.GetPath()))
    return {
        "enabled": True,
        "material_path": material_path,
        "static_friction": float(physics.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(physics.GetDynamicFrictionAttr().Get()),
        "restitution": float(physics.GetRestitutionAttr().Get()),
        "bound_collision_prims": bound,
    }


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
    base_name = robot_cfg.get("base_link_name", "base")
    for path in (f"{prim_root}/{base_name}", f"{prim_root}/base_link"):
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
    mount_path = "/World/FrankaMount" if cfg.get("name") == "franka" else MOUNT_JOINT_PATH
    root = stage.GetPrimAtPath(cfg["articulation_prim_path"])
    if cfg.get("name") == "franka":
        for prim in stage.Traverse():
            if (str(prim.GetPath()).startswith(cfg["usd_prim_path"] + "/")
                    and prim.IsA(UsdPhysics.FixedJoint)
                    and not UsdPhysics.FixedJoint(prim).GetBody0Rel().GetTargets()):
                root = prim
                break
    # The asset already has a world-fixed articulation root. Reuse it instead
    # of constraining the base a second time at a different world frame.
    if root.IsValid() and root.IsA(UsdPhysics.FixedJoint):
        joint = UsdPhysics.FixedJoint(root)
        duplicate = stage.GetPrimAtPath(mount_path)
        if duplicate.IsValid():
            duplicate.SetActive(False)
    else:
        joint = UsdPhysics.FixedJoint.Define(stage, mount_path)
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
        elif str(prim.GetPath()).startswith(cfg["usd_prim_path"]+"/") and prim.IsA(UsdPhysics.PrismaticJoint):
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(
                float(cfg.get("gripper_max_velocity_m_s", .04)))
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


def spawn_table_and_so101(*, include_tabletop_props: bool = True) -> None:
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import XformPrim

    scene = scene_config()
    robot = robot_config()
    apply_environment_map()

    table_usd = resolve_isaac_asset(scene["table_usd_rel"])
    so101_usd = resolve_isaac_asset(robot["usd_asset_rel"])
    stage_utils.add_reference_to_stage(usd_path=table_usd, path="/World/Table")
    kwargs = {"usd_path": so101_usd, "path": robot["usd_prim_path"]}
    if robot.get("usd_variants"):
        kwargs["variants"] = [tuple(pair) for pair in robot["usd_variants"]]
    stage_utils.add_reference_to_stage(**kwargs)

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
    print(f"[mr_liu] {robot['name']}: {so101_usd}")
    mount_so101_to_table()
    if include_tabletop_props:
        spawn_tabletop_props()


spawn_table_and_robot = spawn_table_and_so101
