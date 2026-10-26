import unittest
import numpy as np
from ss_precursor.circular_stacks import cap_memberships, cap_weights


class CircularStackValidation(unittest.TestCase):
    def test_surface_caps_cross_dateline_and_pole_and_are_nested(self):
        centers = [[0, 179], [89, 0]]
        bounce = [[0, -179], [0, 179], [89, 180], [0, 0]]
        small = cap_memberships(centers, bounce, 100)
        large = cap_memberships(centers, bounce, 300)
        np.testing.assert_array_equal(small[0], [1])
        np.testing.assert_array_equal(large[0], [0, 1])
        np.testing.assert_array_equal(large[1], [2])
        for a, b in zip(small, large):
            self.assertTrue(set(a).issubset(set(b)))

    def test_membership_is_phase_specific_and_event_weights_use_selected_cap(self):
        center = [[0, 0]]
        a = cap_memberships(center, [[0, 0], [0, 20]], 900)[0]
        b = cap_memberships(center, [[0, 20], [0, 0]], 900)[0]
        np.testing.assert_array_equal(a, [0]); np.testing.assert_array_equal(b, [1])
        rows = [{'event_id':e} for e in ['a', 'a', 'b', 'b', 'b']]
        np.testing.assert_allclose(cap_weights(rows, [0, 1, 2], 'event_equal'), [.5, .5, 1])


if __name__ == '__main__':
    unittest.main()
