import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from mr_liu.arena.cascade import run_cascade, FastPathFailure
from mr_liu.arena.contracts import ManipulationRequest


def test_fast_success_never_calls_models():
    enhanced = Mock()
    result, route, attempts = run_cascade(ManipulationRequest('part', 'pad'),
        lambda _: {'physical_success': True}, enhanced, Mock(), Mock())
    assert result['physical_success'] and route['grasp'] == 'official_pick_place'
    enhanced.assert_not_called()
    assert len(attempts) == 1


def test_physical_failure_escalates_once_after_recovery():
    order = []
    def recover(_): order.append('recover')
    def enhanced(request):
        assert request.enhanced
        order.append('model')
        return {'physical_success': True}
    result, route, attempts = run_cascade(ManipulationRequest('part', 'pad'),
        lambda _: {'physical_success': False}, enhanced, recover, Mock())
    assert order == ['recover', 'model']
    assert route['placement'] == 'anyplace' and len(attempts) == 2


def test_stop_does_not_start_recovery_or_models():
    recover, enhanced = Mock(), Mock()
    with pytest.raises(InterruptedError):
        run_cascade(ManipulationRequest('part'), Mock(side_effect=InterruptedError()), enhanced, recover, Mock())
    recover.assert_not_called(); enhanced.assert_not_called()


def test_held_object_upgrades_only_placement():
    full_model, placement = Mock(), Mock(return_value={'physical_success': True})
    _, route, _ = run_cascade(ManipulationRequest('part', 'pad'),
        Mock(side_effect=FastPathFailure('placement')), full_model, lambda _: placement, Mock())
    full_model.assert_not_called(); placement.assert_called_once()
    assert route['grasp'] == 'official_pick_place'


def test_complex_task_skips_fast_and_model_failure_does_not_loop():
    fast, recover = Mock(), Mock()
    enhanced = Mock(side_effect=RuntimeError('model failure'))
    with pytest.raises(RuntimeError, match='model failure'):
        run_cascade(ManipulationRequest('part', 'pad', precise=True), fast, enhanced, recover, Mock())
    fast.assert_not_called(); recover.assert_not_called(); enhanced.assert_called_once()


def test_failed_escalation_preserves_attempt_history():
    attempts = []
    with pytest.raises(RuntimeError, match='model unavailable'):
        run_cascade(ManipulationRequest('part', 'pad'),
            Mock(side_effect=FastPathFailure('missed grasp')),
            Mock(side_effect=RuntimeError('model unavailable')), lambda _: None, Mock(), attempts=attempts)
    assert [row['backend'] for row in attempts] == ['official_pick_place', 'models']
    assert all(not row['ok'] and row['elapsed_s'] >= 0 for row in attempts)
    assert attempts[1]['error'] == 'model unavailable'
