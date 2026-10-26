import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from ss_precursor.event_database import template_lag,align_event,prepare_event_database,index_database
from ss_precursor.batch import save_plan,event_cache
from ss_precursor.pipeline import grid


def pulse(t):
    return (1-2*(t/9)**2)*np.exp(-(t/9)**2)+.1*np.exp(-((t+180)/6)**2)


class EventReferenceValidation(unittest.TestCase):
    def test_published_event_database_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'input'; source.write_bytes(b'input')
            stat=source.stat(); cfg={'relative_time_s':[-500,300,1],'event_reference_alignment':{}}
            plan=root/'events.json'
            save_plan(plan,dict(kind='events-v1',network='SY',configuration=cfg,tasks=[dict(index=0,
                event_id='event',source=str(source),source_size_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns)]))
            def fake_prepare(source,configuration,output,**kwargs):
                t=grid(configuration['relative_time_s'])
                rows=[dict(event_id='event',station=f'SY.S{i}',trace_id=f'SY.S{i}.T',distance_degree=120,
                           predicted_ss_s=1000.,measured_ss_s=1000.,alignment_residual_s=0.) for i in range(2)]
                np.save(output/'tau.npy',t); np.save(output/'waveforms.npy',np.array([pulse(t),pulse(t-3.2)]))
                (output/'traces.json').write_text(json.dumps({'accepted':rows,'rejected':[]}))
                (output/'provenance.json').write_text(json.dumps({'configuration':configuration,'run_selection':{'network':'SY'}}))
            with patch('ss_precursor.event_database.prepare',side_effect=fake_prepare) as mocked:
                result=prepare_event_database(plan,root/'database',0)
                self.assertEqual(result['status'],'complete')
                cached=prepare_event_database(plan,root/'database',0)
                self.assertEqual(cached['status'],'cached'); self.assertEqual(mocked.call_count,1)
            cache=event_cache(root/'database',0)
            self.assertEqual(np.load(cache/'waveforms.npy').shape,(2,801))
            reference=json.loads((cache/'reference.json').read_text())
            np.testing.assert_array_equal(np.load(cache/'reference.npy'),np.load(cache/'waveforms.npy')[reference['accepted_row_index']])
            prov=json.loads((cache/'provenance.json').read_text())
            expected=dict(cfg,event_id='event')
            self.assertEqual(prov['configuration'],expected)
            self.assertEqual(prov['configuration_sha256'],hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest())
            self.assertEqual(index_database(plan,root/'database')['completed'],1)

    def test_fractional_lag_sign_and_whole_trace_precursor_separation(self):
        tau=np.arange(-532.,333.)
        template_time=np.arange(-40.,41.)
        for shift in [8.3,-7.3]:
            fit=template_lag(tau,pulse(tau-shift),template_time,pulse(template_time),30,10)
            self.assertAlmostEqual(fit['lag_s'],shift,delta=.05)
            self.assertGreater(fit['correlation'],.99)
            self.assertFalse(fit['lag_at_boundary'])

    def test_event_database_keeps_individual_traces_and_rejects_bad_match(self):
        tau=np.arange(-532.,333.); rng=np.random.RandomState(12)
        data=np.array([pulse(tau),pulse(tau),pulse(tau-8.3),rng.normal(0,.2,len(tau))])
        rows=[dict(event_id='one_event',trace_id=f'SY.S{i}.T',station=f'SY.S{i}',distance_degree=120,
                   measured_ss_s=1000.,predicted_ss_s=1000.,alignment_residual_s=0,polarity_multiplier=1.,ss_amplitude_mm=2.)
              for i in range(4)]
        cfg={'relative_time_s':[-500,300,1]}
        t,aligned,accepted,rejected,reference,qc=align_event(tau,data,rows,cfg)
        self.assertEqual(aligned.shape,(3,801)); self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]['station'],'SY.S3')
        self.assertFalse(rejected[0]['alignment_applied'])
        self.assertEqual(rejected[0]['measured_ss_s'],1000.)
        row=next(r for r in accepted if r['station']=='SY.S2')
        self.assertAlmostEqual(row['measured_ss_s'],1008.3,delta=.05)
        index=accepted.index(row); near=(t>=-190)&(t<=-170)
        precursor_time=t[near][np.argmax(aligned[index,near])]
        self.assertAlmostEqual(precursor_time,-180,delta=.6)
        self.assertAlmostEqual(aligned[index,np.argmin(abs(t+180))],.1,delta=.003)
        self.assertEqual(row['ss_amplitude_mm'],2.)
        self.assertEqual(reference['event_id'],'one_event')
        self.assertEqual(accepted[reference['accepted_row_index']]['trace_id'],reference['trace_id'])

    def test_separate_events_cannot_share_a_reference(self):
        tau=np.arange(-532.,333.); data=np.array([pulse(tau),pulse(tau)])
        with self.assertRaisesRegex(ValueError,'multiple events'):
            align_event(tau,data,[{'event_id':'a'},{'event_id':'b'}],{'relative_time_s':[-500,300,1]})

    def test_periodic_secondary_peaks_are_identified(self):
        tau=np.arange(-200.,201.); tt=np.arange(-40.,41.)
        fit=template_lag(tau,np.cos(2*np.pi*tau/20),tt,np.cos(2*np.pi*tt/20),30,10)
        self.assertIsNotNone(fit['secondary_correlation'])
        self.assertLess(fit['peak_margin'],1e-10)
