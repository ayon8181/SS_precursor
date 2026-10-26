import unittest
import numpy as np
from ss_precursor.broad_vespagrams import broad_beam


def pulse(t):return (1-2*(t/8)**2)*np.exp(-(t/8)**2)

class BroadBeamValidation(unittest.TestCase):
    def test_known_positive_and_negative_moveout(self):
        d=np.linspace(100,160,25);ref=d.mean();time=np.arange(-1000.,801.)
        display=np.arange(-350.,151.);p=np.arange(-2.,2.01,.1)
        for intercept,slope in [(-220.,-1.2),(30.,1.3)]:
            data=np.array([pulse(time-intercept-slope*(di-ref)) for di in d])
            r=broad_beam(time,data,d,display,p,ref,(10,100),padding=200,analytic_guard=100)
            i,j=np.unravel_index(np.nanargmax(r['envelope']),r['envelope'].shape)
            self.assertAlmostEqual(p[i],slope,places=6)
            self.assertAlmostEqual(display[j],intercept,delta=1.)
            self.assertGreater(r['signed_stack'][i,j],.98)
            self.assertEqual(r['trace_support'][i,j],len(d))

    def test_reference_distance_changes_intercept_with_correct_sign(self):
        d=np.linspace(100,160,25);time=np.arange(-900.,801.);ref=d.mean();slope=-1.
        data=np.array([pulse(time+200-slope*(di-ref)) for di in d])
        display=np.arange(-240.,-159.)
        r=broad_beam(time,data,d,display,[slope],ref+7,(10,100),padding=200,analytic_guard=100)
        self.assertEqual(display[np.nanargmax(r['envelope'][0])],-207.)

    def test_zero_amplitude_is_valid_and_distinct_from_missing_support(self):
        t=np.arange(-600.,601.);display=np.arange(-100.,101.)
        r=broad_beam(t,np.zeros((2,len(t))),[110,150],display,[0.],130,(10,100),padding=200,analytic_guard=100)
        self.assertTrue(r['envelope_valid'].all());np.testing.assert_array_equal(r['envelope'],0.)
        self.assertTrue(np.isnan(r['windowed_semblance']).all())
        missing=broad_beam(t,np.full((2,len(t)),np.nan),[110,150],display,[0.],130,(10,100),padding=200,analytic_guard=100)
        self.assertFalse(missing['envelope_valid'].any());self.assertTrue(np.isnan(missing['envelope']).all())
        self.assertTrue((missing['trace_support']==0).all())

    def test_windowed_semblance_uses_sums_before_division(self):
        t=np.arange(-600.,601.);display=np.arange(-100.,101.)
        data=np.array([np.ones(len(t)),np.full(len(t),2.)])
        r=broad_beam(t,data,[110,150],display,[0.],130,(10,100),padding=200,analytic_guard=100)
        np.testing.assert_allclose(r['windowed_semblance'],.9,atol=1e-14)
        # Local opposite signs make instantaneous semblance zero; window totals remain finite.
        data=np.array([np.ones(len(t)),np.where(t==0,-1.,1.)])
        r=broad_beam(t,data,[110,150],display,[0.],130,(10,100),padding=200,analytic_guard=100)
        width=int(r['semblance_window_samples'])
        self.assertAlmostEqual(r['windowed_semblance'][0,display==0][0],(width-1)/width,places=12)

    def test_hilbert_padding_converges_for_pulses_near_display_edges(self):
        t=np.arange(-1600.,1201.);d=np.linspace(100,160,17);display=np.arange(-350.,151.)
        data=np.array([pulse(t+340-.7*(di-130))+.3*pulse(t-140+.5*(di-130)) for di in d])
        a=broad_beam(t,data,d,display,[-.5,.7],130,(10,100),padding=400,analytic_guard=300)
        b=broad_beam(t,data,d,display,[-.5,.7],130,(10,100),padding=700,analytic_guard=300)
        self.assertTrue(a['envelope_valid'].all() and b['envelope_valid'].all())
        self.assertLess(np.max(abs(a['envelope']-b['envelope'])),.002)

if __name__=='__main__':unittest.main()
