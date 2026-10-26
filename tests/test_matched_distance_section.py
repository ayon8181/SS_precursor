import unittest
import numpy as np
from ss_precursor.distance_section import match_global

class MatchedPolarityTests(unittest.TestCase):
    def dataset(self, tag, signs, names, weighting='trace'):
        cfg=dict(waveform_tag=tag,period_band_s=[10,100],filter_order=2,detrend='linear',
                 taper_fraction_per_end=.05,filter_padding_s=300,relative_time_s=[-1,1,1],
                 horizontal_convention='RT',input_units='displacement_mm',stack_weighting=weighting)
        rows=[dict(event_id='event',trace_id=name,station_coordinates={'latitude':0,'longitude':0},
                   distance_degree=120.,polarity_multiplier=sign) for name,sign in zip(names,signs)]
        return ({'configuration':cfg},None,[{'trace_id':r['trace_id']} for r in rows],
                np.arange(-1,2),[np.array([1.,2.,3.])*sign for sign in signs],rows,
                np.ones(len(rows))*10,np.ones(len(rows)))

    def test_replace_signs_and_intersect(self):
        glad=self.dataset('proc_glad',[1,-1,1],['a','b','extra'])
        prem=self.dataset('proc_prem',[-1,-1],['a','b'])
        result=match_global(glad,prem,True)
        self.assertEqual([r['trace_id'] for r in result[5]],['a','b'])
        np.testing.assert_array_equal(result[4],[[-1,-2,-3],[-1,-2,-3]])
        self.assertEqual([r['cached_waveform_multiplier'] for r in result[5]],[-1,1])
        self.assertEqual([r['polarity_multiplier'] for r in result[5]],[-1,-1])
        np.testing.assert_array_equal(result[3],glad[3])

    def test_recompute_event_weights_after_intersection(self):
        main=self.dataset('proc_prem',[1,1,1],['a','b','extra'],'event_equal')
        reference=self.dataset('proc_glad',[1,1],['a','b'])
        np.testing.assert_array_equal(match_global(main,reference)[7],[.5,.5])

    def test_require_prem_reference_and_equal_filters(self):
        glad=self.dataset('proc_glad',[1],['a'])
        with self.assertRaises(ValueError):match_global(glad,glad,True)
        prem=self.dataset('proc_prem',[1],['a'])
        prem[0]['configuration']['filter_order']=4
        with self.assertRaises(ValueError):match_global(glad,prem,True)

if __name__=='__main__':unittest.main()
