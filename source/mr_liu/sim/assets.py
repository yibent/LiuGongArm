from mr_liu.config import scene_config


def resolve_isaac_asset(rel_path: str) -> str:
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim asset root is not available. Check network or ISAACSIM_ASSET_ROOT.")
    return root.rstrip("/") + rel_path


def scene_asset(key: str) -> str:
    return resolve_isaac_asset(scene_config()[key])
