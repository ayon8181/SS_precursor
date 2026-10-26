import unittest
import numpy as np
from ss_precursor.vespagram_followups import half_width,coherence,phase_samples,timing_fit,conservative_timing_fit,event_excluded_template,variance_split,amplitude_shape_metrics

class FollowupTests(unittest.TestCase):
    def test_fwhm_crossings(self):
        x=np.arange(-10.,10.01,.01);y=np.exp(-x*x/2)
        width,clipped=half_width(x,y,np.argmax(y))
        self.assertAlmostEqual(width,2*np.sqrt(2*np.log(2)),places=4)
        self.assertFalse(clipped)

    def test_signed_coherence_retains_cancellation(self):
        t=np.arange(-100.,101.);u=np.exp(-(t/8)**2)
        self.assertAlmostEqual(coherence(np.array([u,u]),t),1.)
        self.assertAlmostEqual(coherence(np.array([u,-u]),t),0.)

    def test_phase_sampling_aligns_individual_predictions(self):
        t=np.arange(-700.,301.);off=np.arange(-60.,61.);arrivals=np.array([-150.,-170.])
        data=np.array([np.exp(-((t-ti)/8)**2) for ti in arrivals])
        values=phase_samples(t,data,arrivals,off)
        np.testing.assert_allclose(values[0],values[1],atol=1e-14)
        self.assertAlmostEqual(coherence(values,off),1.)

    def test_fit_recovers_timing_but_does_not_flip_negative_waveform(self):
        t=np.arange(-100.,101.);tt=np.arange(-25.,26.)
        pulse=lambda x:(1-2*(x/9)**2)*np.exp(-(x/9)**2)
        u=-.03*pulse(t-7.2)
        result=timing_fit(t,u,tt,pulse(tt))
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['lag_s'],7.2,delta=.06)
        self.assertLess(result['signed_correlation_after'],0.)
        self.assertEqual(result['additional_polarity_multiplier'],1.)
        np.testing.assert_array_equal(u,-.03*pulse(t-7.2))

    def test_conservative_fit_recovers_small_delay_without_polarity_changes(self):
        t=np.arange(-100.,101.);tt=np.arange(-25.,26.)
        pulse=lambda x:(1-2*(x/9)**2)*np.exp(-(x/9)**2)
        result=conservative_timing_fit(t,.03*pulse(t-4.2),tt,pulse(tt))
        self.assertTrue(result['accepted']);self.assertAlmostEqual(result['lag_s'],4.2,delta=.06)
        reversed_fit=conservative_timing_fit(t,-.03*pulse(t-4.2),tt,pulse(tt))
        self.assertFalse(reversed_fit['accepted'])

    def test_reference_excludes_the_whole_target_event(self):
        t=np.arange(-100.,101.);values=np.array([np.ones(len(t))*v for v in [100,200,2,4]])
        template=event_excluded_template(values,np.array(['a','a','b','c']),'a',t,np.arange(-25.,26.))
        np.testing.assert_array_equal(template,np.full(51,3.))

    def test_energy_decomposition_distinguishes_between_and_within_groups(self):
        t=np.arange(-100.,101.)
        x=np.array([np.full(len(t),v) for v in [1,3,-1,-3]])
        split=variance_split(x,['a','a','b','b'],t)
        self.assertAlmostEqual(split['global_coherent_fraction'],0.)
        self.assertAlmostEqual(split['between_group_fraction'],.8)
        self.assertAlmostEqual(split['within_group_fraction'],.2)
        self.assertAlmostEqual(sum(split.values()),1.)

    def test_amplitude_ceiling_does_not_confuse_aligned_shapes_with_incoherence(self):
        t=np.arange(-100.,101.);u=np.exp(-(t/9)**2)
        m=amplitude_shape_metrics(np.array([u,2*u]),t)
        self.assertAlmostEqual(m['raw_coherence'],.9)
        self.assertAlmostEqual(m['amplitude_only_coherence_ceiling'],.9)
        self.assertAlmostEqual(m['coherence_fraction_of_amplitude_ceiling'],1.)
        self.assertAlmostEqual(m['unit_norm_shape_coherence'],1.)

    def test_negligible_energy_is_not_given_a_timing_shift(self):
        t=np.arange(-100.,101.);tt=np.arange(-25.,26.)
        result=timing_fit(t,np.zeros(len(t)),tt,np.exp(-(tt/8)**2))
        self.assertFalse(result['accepted']);self.assertIsNone(result['lag_s'])

if __name__=='__main__':unittest.main()
