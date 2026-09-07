import json
import pytest
from mr_liu.arena.perception import PerceptionBridge


def test_stable_cell_uses_latest_observed_frame_and_rejects_missing_identity(tmp_path):
    bridge = PerceptionBridge(tmp_path, 'unused')
    stable = 'grid:' + 'c' * 32 + ':2:3'
    for frame in ['a' * 32, 'b' * 32]:
        path = tmp_path / frame
        path.mkdir()
        ref = f'obs:{frame}:side_camera:6'
        row = {'ref': ref, 'cell_id': stable, 'kind': 'cell', 'row': 2, 'column': 3}
        (path / 'request.json').write_text(json.dumps({'scene_id': bridge.scene_id}))
        (path / 'result.json').write_text(json.dumps({'references': [row]}))
        bridge.world.geometry_memory['observed-container'] = {'geometry': {'cells': [row]}}
        assert bridge.resolve_reference(stable) == row
    bridge.world.geometry_memory.clear()
    with pytest.raises(ValueError, match='重新观察'):
        bridge.resolve_reference(stable)
