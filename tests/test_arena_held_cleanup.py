from pathlib import Path


def test_stale_held_identity_is_cleared_from_every_command_boundary():
    source = (Path(__file__).resolve().parents[1]/'source/mr_liu/arena/runtime.py').read_text()
    assert source.count("if self.held is not None and not self.holding_status()['verified']:") >= 2
    assert "if self.held_context and not self.holding_status()['verified']:" not in source
