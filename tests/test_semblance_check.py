import unittest
import numpy as np
from ss_precursor.semblance_check import sample_terms,window_semblance


class SemblanceValidation(unittest.TestCase):
    def test_sum_terms_before_division_not_mean_of_ratios(self):
        terms=sample_terms([[10,1,10],[-10,1,-10]],[1,1])
        point=window_semblance(**terms,length=1,minimum_support=2,energy_floor=0)
        window=window_semblance(**terms,length=3,minimum_support=2,energy_floor=0)
        self.assertAlmostEqual(window['semblance'][1],4/804)
        self.assertAlmostEqual(np.mean(point['semblance']),1/3)
        self.assertTrue(np.isnan(window['semblance'][[0,2]]).all())

    def test_identical_valid_samples_and_weights_and_full_window_support(self):
        terms=sample_terms([[1,np.nan,3],[2,4,np.nan]],[1,2])
        np.testing.assert_array_equal(terms['A'],[5,8,3])
        np.testing.assert_array_equal(terms['E'],[9,32,9])
        np.testing.assert_array_equal(terms['W'],[3,2,1])
        result=window_semblance(**terms,length=3,minimum_support=1,energy_floor=0)
        self.assertAlmostEqual(result['semblance'][1],98/100)
        strict=window_semblance(**terms,length=3,minimum_support=2,energy_floor=0)
        self.assertTrue(np.isnan(strict['semblance']).all())

    def test_low_energy_is_masked_without_stabilizing_constant(self):
        for amplitude,expected in [(0,False),(1e-12,False),(1e-8,True)]:
            terms=sample_terms(np.full((3,7),amplitude),[1,2,3])
            result=window_semblance(**terms,length=3,minimum_support=3,energy_floor=1e-20)
            if expected: np.testing.assert_allclose(result['semblance'][1:-1],1)
            else: self.assertTrue(np.isnan(result['semblance']).all())
