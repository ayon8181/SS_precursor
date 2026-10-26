from types import SimpleNamespace
import unittest
import numpy as np
from obspy import Trace,UTCDateTime
from ss_precursor.cap_filter_experiment import fixed_filter

class FrozenFilterTests(unittest.TestCase):
    def setUp(self):
        self.origin=SimpleNamespace(time=UTCDateTime(0))
        self.dt=.5;self.time=5+np.arange(5000)*self.dt
        self.trace=Trace(np.sin(2*np.pi*(self.time-1000)/15),header={'starttime':UTCDateTime(5),'delta':self.dt})
        self.cfg=dict(minimum_synthetic_period_s=10,detrend='linear',taper_fraction_per_end=.05,
                      filter_order=2,filter_padding_s=300,edge_guard_s=150)
        self.tau=np.arange(-150.,151.)

    def test_frozen_scale_and_sign_no_peak_renormalization(self):
        a,start=fixed_filter(self.trace,self.origin,self.cfg,(10,100),1000,2,1,self.tau)
        b,_=fixed_filter(self.trace,self.origin,self.cfg,(10,100),1000,4,-1,self.tau)
        np.testing.assert_allclose(b,-a/2,rtol=1e-13,atol=1e-14)
        self.assertEqual(start,5.)

    def test_band_changes_spectrum_and_does_not_restore_amplitude(self):
        short,_=fixed_filter(self.trace,self.origin,self.cfg,(10,50),1000,1,1,self.tau)
        long,_=fixed_filter(self.trace,self.origin,self.cfg,(30,100),1000,1,1,self.tau)
        self.assertLess(np.linalg.norm(long)/np.linalg.norm(short),.2)

    def test_frozen_clock_shift_is_not_reestimated(self):
        base,_=fixed_filter(self.trace,self.origin,self.cfg,(10,100),1000,1,1,self.tau)
        shifted,_=fixed_filter(self.trace,self.origin,self.cfg,(10,100),1005,1,1,self.tau)
        np.testing.assert_allclose(shifted[:-5],base[5:],rtol=1e-12,atol=1e-13)

    def test_missing_support_does_not_change_cohort_silently(self):
        with self.assertRaises(ValueError):
            fixed_filter(self.trace,self.origin,self.cfg,(10,100),20,1,1,self.tau)

if __name__=='__main__':unittest.main()
