from pathlib import Path


def test_contact_execution_accepts_seating_before_exact_tcp_goal():
    source = (Path(__file__).resolve().parents[1]/'source/mr_liu/arena/runtime.py').read_text()
    assert "witness['satisfied'] and self.holding_status()['verified']" in source
    assert "reason='relation_seated_before_tcp_goal'" in source
    assert "retry_scope='full_contact_path'" in source
