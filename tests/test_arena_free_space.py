import unittest
import numpy as np
from mr_liu.arena.free_space import choose_free_support


def plane(xmin=-.3, xmax=.3, ymin=-.3, ymax=.3, z=0.):
    x, y = np.meshgrid(np.arange(xmin, xmax, .003), np.arange(ymin, ymax, .003))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


class FreeSupportTests(unittest.TestCase):
    child = np.array([[-.02, -.02, .10], [.02, .02, .14], [-.02, .02, .13], [.02, -.02, .10]])

    def test_occupied_centre_is_not_a_destination(self):
        table = plane()
        obstacle = plane(-.06, .06, -.06, .06, .04)
        target, details = choose_free_support(table, np.r_[table, obstacle], self.child, [0, 0, .2])
        self.assertGreater(np.linalg.norm(target[:2]), .10)
        self.assertGreaterEqual(details['clearance_m'], details['footprint_radius_m'])
        self.assertAlmostEqual(target[2], 0.)

    def test_edges_and_unseen_gaps_do_not_count_as_free(self):
        table = np.r_[plane(-.30, -.10), plane(.10, .30)]
        target, _ = choose_free_support(table, table, self.child, [0, 0, .2])
        self.assertGreater(abs(target[0]), .15)
        self.assertLess(abs(target[0]), .25)

    def test_surface_too_small_does_not_release(self):
        table = plane(-.03, .03, -.03, .03)
        with self.assertRaisesRegex(RuntimeError, '没有.*空位'):
            choose_free_support(table, table, self.child, [0, 0, .2])

    def test_raised_rim_is_not_table_height(self):
        table = plane(z=.4)
        rim = plane(-.3, -.28, -.3, .3, z=.45)
        target, _ = choose_free_support(np.r_[table, rim], np.r_[table, rim], self.child, [0, 0, .7])
        self.assertAlmostEqual(target[2], .4)

    def test_large_payload_needs_a_larger_patch(self):
        table = plane()
        _, small = choose_free_support(table, table, self.child, [.29, 0, .2])
        child = self.child.copy(); child[:, :2] *= 5
        target, large = choose_free_support(table, table, child, [.29, 0, .2])
        self.assertGreater(large['footprint_radius_m'], small['footprint_radius_m'])
        self.assertLess(target[0], .15)


if __name__ == '__main__': unittest.main()
