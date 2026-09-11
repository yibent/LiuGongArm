from pathlib import Path


def test_local_arena_scene_disables_optional_registry_before_franka_import():
    source = (Path(__file__).resolve().parents[1]/'source/mr_liu/arena/environment.py').read_text()
    guard = source.index('arena_asset_registry._assets_registered = True')
    franka = source.index('from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment')
    assert guard < franka
