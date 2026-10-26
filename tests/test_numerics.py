import unittest
import numpy as np
from ss_precursor.stacking import slant_stack, sample_trace, weighted_samples
from ss_precursor.depth_recovery import depth_stack
from ss_precursor.geometry import geographic_bin, point_on_ray


def pulse(time, center, width=3):
    x = (time-center)/width
    return (1-x*x)*np.exp(-x*x/2)


class NumericalValidation(unittest.TestCase):
    def test_moveout_and_intercept(self):
        t = np.arange(-230, -70, 0.5)
        distances = np.linspace(95, 155, 25)
        true_p, intercept = -0.45, -150.25
        traces = [pulse(t, intercept+true_p*(d-125)) for d in distances]
        tau = np.arange(-180, -120, 0.25)
        p = np.arange(-0.8, -0.1, 0.025)
        result = slant_stack([t]*len(traces), traces, distances, tau, p, 125)
        i, j = np.unravel_index(np.nanargmax(result['stack']), result['stack'].shape)
        self.assertAlmostEqual(p[i], true_p, delta=0.025)
        self.assertAlmostEqual(tau[j], intercept, delta=0.25)
        self.assertEqual(result['count'][i, j], len(traces))
        self.assertGreater(result['semblance'][i, j], 0.99)

    def test_valid_support_and_fractional_interpolation(self):
        np.testing.assert_allclose(sample_trace([0, 1, 2], [0, 2, 4], [0.25, 1.5]), [0.5, 3])
        self.assertTrue(np.isnan(sample_trace([0, 1], [1, 1], [-1, 2])).all())
        out = weighted_samples([[2, np.nan], [4, 8]], [1, 3])
        np.testing.assert_allclose(out['stack'], [3.5, 8])
        np.testing.assert_array_equal(out['count'], [2, 1])
        self.assertTrue(np.isnan(weighted_samples([[np.nan]])['stack'][0]))
        self.assertTrue(np.isnan(sample_trace([0, 1, 2], [1, np.nan, 3], [0.5, 1.5])).all())

    def test_differential_correction_sign_and_depth_recovery(self):
        t = np.arange(-260, -80, 0.25)
        depths = np.arange(350, 451, 0.5)
        distances = np.linspace(90, 150, 12)
        reference = -150 - 0.3*(depths-400)
        predictions = reference[None, :] + 0.1*(distances[:, None]-120)
        imposed = 6.0
        traces = [pulse(t, -150+0.1*(d-120)+imposed) for d in distances]
        raw = depth_stack([t]*len(traces), traces, depths, predictions)
        corrected = depth_stack([t]*len(traces), traces, depths, predictions, np.full_like(predictions, imposed))
        self.assertAlmostEqual(depths[np.argmax(raw['stack'])], 380, delta=0.5)
        self.assertAlmostEqual(depths[np.argmax(corrected['stack'])], 400, delta=0.5)
        self.assertGreater(corrected['semblance'][np.argmax(corrected['stack'])], 0.99)

    def test_geographic_boundaries(self):
        self.assertEqual(geographic_bin(0, 180), geographic_bin(0, -180))
        self.assertEqual(geographic_bin(0, 540), geographic_bin(0, -180))
        self.assertEqual(geographic_bin(90, 0)[0], 35)
        self.assertEqual(geographic_bin(-90, 0)[0], 0)
        self.assertEqual(geographic_bin(0, 179.9)[1], 71)
        with self.assertRaises(ValueError):
            geographic_bin(91, 0)
        lat, lon = point_on_ray(0, 170, 90, 20)
        self.assertAlmostEqual(lat, 0)
        self.assertAlmostEqual(lon, -170)

    def test_invalid_weights(self):
        for weights in ([0], [-1], [np.nan]):
            with self.assertRaises(ValueError):
                weighted_samples([[1]], weights)


if __name__ == '__main__':
    unittest.main()
