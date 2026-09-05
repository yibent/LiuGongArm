import unittest
from mr_liu.sim.timing import render_interval


class SimTimingTests(unittest.TestCase):
    def test_four_physics_substeps_per_render(self):
        self.assertAlmostEqual(render_interval(1/120, 30)/(1/120), 4)

    def test_invalid_render_schedule_is_rejected(self):
        for hz in (0, -1, 240, 31, float('nan')):
            with self.assertRaises(ValueError):
                render_interval(1/120, hz)
