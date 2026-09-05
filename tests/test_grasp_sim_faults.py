import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from mr_liu.sim.grasp_faults import OneShotPreCloseShift


class SimGraspFaultTests(unittest.TestCase):
    def test_explicit_fault_only_shifts_once_at_first_close(self):
        shift = Mock()
        fault = OneShotPreCloseShift(.04, shift)
        self.assertIsNone(fault.on_event({"phase": "servo"}))
        result = fault.on_event({"phase": "close"})
        self.assertTrue(result['simulation_only'])
        self.assertIsNone(fault.on_event({"phase": "close"}))
        shift.assert_called_once_with((.04, 0., 0.))

    def test_default_zero_is_inert_and_displacements_are_bounded(self):
        shift = Mock()
        self.assertIsNone(OneShotPreCloseShift(0., shift).on_event({"phase": "close"}))
        shift.assert_not_called()
        for distance in (float('nan'), float('inf'), .051, -.051):
            with self.assertRaises(ValueError):
                OneShotPreCloseShift(distance, shift)


if __name__ == '__main__':
    unittest.main()
