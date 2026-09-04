"""USD scene authoring: table, SO-101, mount joint, HDRI."""

from __future__ import annotations

from mr_liu.config import robot_config, scene_config
from mr_liu.sim.assets import resolve_isaac_asset

MOUNT_JOINT_PATH = "/World/SO101Mount"


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
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    base = _find_so101_base(stage)
    if base is None:
        raise RuntimeError("SO-101 base rigid body not found. Load the robot first.")

    existing = stage.GetPrimAtPath(MOUNT_JOINT_PATH)
    if existing.IsValid() and existing.IsA(UsdPhysics.FixedJoint):
        joint = UsdPhysics.FixedJoint(existing)
    else:
        joint = UsdPhysics.FixedJoint.Define(stage, MOUNT_JOINT_PATH)

    table = stage.GetPrimAtPath("/World/Table")
    if table.IsValid() and table.HasAPI(UsdPhysics.RigidBodyAPI):
        joint.CreateBody0Rel().SetTargets([table.GetPath()])
    else:
        joint.CreateBody0Rel().ClearTargets(True)
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateBreakForceAttr().Set(float("inf"))
    joint.CreateBreakTorqueAttr().Set(float("inf"))
    print(f"[mr_liu] Mounted {base.GetPath()} with {MOUNT_JOINT_PATH}")


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
        reset_xform_op_properties=True,
    )
    print(f"[mr_liu] Table: {table_usd}")
    print(f"[mr_liu] SO-101: {so101_usd}")
    mount_so101_to_table()
