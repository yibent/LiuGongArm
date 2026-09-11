from pathlib import Path


def test_fast_release_uses_measured_destination_contact_before_exact_tcp_goal():
    source = (Path(__file__).resolve().parents[1]/'source/mr_liu/arena/fast.py').read_text()
    assert "runtime.task.support_contact" in source
    assert "until_contact=contact" in source
    assert "verify_release_pose(runtime,place,place_orientation,row,destination)" in source
    assert "verify_release_pose(runtime,place,orientation,row,destination)" in source
