import unittest
import numpy as np
from ss_precursor.paired_prem_alignment import absolute_template_lag
from ss_precursor.stacking import sample_trace

class PairedPREMAlignmentTests(unittest.TestCase):
    def pulse(self,t):
        return (1-2*(t/9)**2)*np.exp(-(t/9)**2)+.12*np.exp(-((t+180)/7)**2)

    def test_fractional_timing_keeps_both_polarities_and_precursor_separation(self):
        t=np.arange(-500.,301.); tt=np.arange(-40.,41.)
        for sign in [-1.,1.]:
            for delay in [-8.3,7.2]:
                target=sign*self.pulse(t-delay)
                fit=absolute_template_lag(t,target,tt,self.pulse(tt))
                self.assertAlmostEqual(fit['lag_s'],delay,delta=.06)
                self.assertEqual(np.sign(fit['signed_correlation_after']),sign)
                self.assertEqual(fit['additional_polarity_multiplier'],1.)
                shifted=sample_trace(t,target,t+fit['lag_s'])
                self.assertEqual(np.sign(shifted[np.argmin(abs(t))]),sign)
                near=(t>=-190)&(t<=-170)
                self.assertAlmostEqual(t[near][np.argmax(abs(shifted[near]))],-180,delta=.6)
                self.assertAlmostEqual(shifted[t==-180][0],sign*.12,delta=.004)
                self.assertTrue(np.isnan(shifted[0] if delay<0 else shifted[-1]))

    def test_ambiguous_periodic_pulse_has_no_clear_peak_margin(self):
        t=np.arange(-200.,201.); tt=np.arange(-40.,41.)
        fit=absolute_template_lag(t,np.cos(t*np.pi/10),tt,np.cos(tt*np.pi/10))
        self.assertIsNotNone(fit['peak_margin'])
        self.assertLess(fit['peak_margin'],1e-10)

    def test_constant_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            absolute_template_lag(np.arange(-200.,201.),np.ones(401),np.arange(-40.,41.),np.ones(81))

if __name__=='__main__':unittest.main()
