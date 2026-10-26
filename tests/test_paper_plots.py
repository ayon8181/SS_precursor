"""Absolute slowness and source-depth correction are tested independently of plotting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ss_precursor.batch import save_plan, read_plan
from ss_precursor.paper_plots import (absolute_slant_stack, envelope_display, paper_plots,
                                      settings_with_defaults, curve_slant_stack)


class PaperPlotValidation(unittest.TestCase):
    def test_curved_moveout_recovers_known_pulse_and_residual_sign(self):
        t = np.arange(-400, 101, 0.5)
        d = np.linspace(85, 155, 29)
        predicted = -150-0.012*(d-120)**2
        traces = [np.exp(-((t-(p+5+0.3*(di-120)))/3)**2)
                  for p, di in zip(predicted, d)]
        q = np.arange(11, 15.01, 0.1)
        offsets = np.arange(-40, 40.01, 0.5)
        result = curve_slant_stack([t]*len(d), traces, d, predicted, offsets, q, 120, 12)
        i, j = np.unravel_index(np.argmax(result['stack']), result['stack'].shape)
        self.assertAlmostEqual(q[i], 12.3)
        self.assertAlmostEqual(offsets[j], 5)
        self.assertGreater(result['stack'][i, j], 0.99)
        self.assertAlmostEqual(result['residual_slownesses'][i], 0.3)

    def synthetic(self, n=25):
        tau = np.arange(-400, 100.1, 0.5)
        distances = np.linspace(90, 150, n)
        ss_ref = 2000+np.linspace(0, 30, n)
        ss_actual = ss_ref+12*(distances-120)
        phase_ref = ss_ref-150-np.linspace(0, 6, n)
        phase_centers = phase_ref-ss_actual+11.5*(distances-120)+7.5
        traces = [np.exp(-((tau-center)/3)**2) for center in phase_centers]
        return tau, distances, ss_ref, ss_actual, phase_ref, traces

    def test_absolute_slowness_and_event_specific_reference_times(self):
        tau, distances, ss_ref, ss_actual, phase_ref, traces = self.synthetic()
        q = np.arange(0, 20.01, 0.1); offsets = np.arange(-40, 40.1, 0.5)
        result = absolute_slant_stack([tau]*len(traces), traces, distances, phase_ref-ss_actual,
                                      offsets, q, 120)
        full_support = np.where(result['count'] == len(traces), result['stack'], np.nan)
        i, j = np.unravel_index(np.nanargmax(full_support), full_support.shape)
        self.assertAlmostEqual(q[i], 11.5, delta=0.1)
        self.assertAlmostEqual(offsets[j], 7.5, delta=0.5)
        self.assertGreater(q[i], 10)  # Not the SS-relative value of -0.5 s/degree.
        self.assertAlmostEqual(full_support[i, j], 1, places=6)

    def test_overview_restores_ss_moveout(self):
        tau, distances, ss_ref, ss_actual, phase_ref, traces = self.synthetic()
        traces = [np.exp(-(tau/3)**2) for _ in distances]
        q = np.arange(10, 14.01, 0.1); time = np.arange(-20, 20.1, 0.5)
        result = absolute_slant_stack([tau]*len(traces), traces, distances, ss_ref-ss_actual,
                                      time, q, 120)
        i, j = np.unravel_index(np.nanargmax(result['stack']), result['stack'].shape)
        self.assertAlmostEqual(q[i], 12, delta=0.1)
        self.assertAlmostEqual(time[j], 0, delta=0.5)

    def test_envelope_uses_one_panel_scale_and_masks_missing_support(self):
        t = np.linspace(-30, 30, 121)
        waveform = np.exp(-(t/5)**2)*np.cos(t)
        amplitude = np.array([waveform, waveform*0.2, waveform])
        count = np.full_like(amplitude, 10, dtype=int); count[2, :] = 2
        amplitude[:, 55:65] = np.nan
        result = {'stack': amplitude, 'count': count, 'time_s': t}
        display, meta = envelope_display(result, 10, settings_with_defaults())
        self.assertAlmostEqual(np.nanmax(display[1])/np.nanmax(display[0]), 0.2, places=6)
        self.assertTrue(np.isnan(display[:, 55:65]).all())
        self.assertTrue(np.isnan(display[2]).all())
        self.assertEqual(meta['minimum_contributing_traces'], 8)

    def test_phase_window_excludes_strong_ss_instead_of_zero_padding(self):
        tau = np.arange(-300, 101, 1)
        u = np.exp(-(tau/3)**2)+0.01*np.exp(-((tau+150)/3)**2)
        # The q=15 query at D-D0=10 would sample SS despite being a precursor scan.
        result = absolute_slant_stack([tau], [u], [130], [-150], [0], [0, 15], 120,
                                      support_centers=[-150], support_half_window=60)
        self.assertAlmostEqual(result['stack'][0, 0], 0.01)
        self.assertTrue(np.isnan(result['stack'][1, 0]))
        self.assertEqual(result['count'][1, 0], 0)

    def test_three_phase_export_and_resume_without_asdf(self):
        tau, distances, ss_ref, ss_actual, phase_ref, precursor = self.synthetic(n=7)
        rows = []; refs = {phase: [] for phase in ['SS', 'S^400S', 'S^670S']}
        traces = []
        for i, d in enumerate(distances):
            arrivals = {}
            u = np.exp(-(tau/3)**2)
            for phase, q, dt in [('SS', 12, 0), ('S^400S', 11.5, -150), ('S^670S', 11, -230)]:
                refs[phase].append({'time_s': float(ss_ref[i]+dt), 'ray_parameter_s_degree': q})
                actual = ss_ref[i]+dt+q*(d-120)
                arrivals[phase] = {'time_s': float(actual)}
                if phase != 'SS': u += 0.1*np.exp(-((tau-(actual-ss_actual[i]))/3)**2)
            rows.append({'event_id': f'event{i}', 'trace_id': f'SY.S{i}.S3.BXT', 'station': f'SY.S{i}',
                         'distance_degree': float(d), 'predicted_ss_s': float(ss_actual[i]), 'arrivals': arrivals})
            traces.append(u)
        task = {'index': 0, 'phase': 'S^400S', 'latitude_index': 18, 'longitude_index': 36,
                'latitude_center': 2.5, 'longitude_center': 2.5, 'trace_count': 7, 'event_count': 7}
        cfg = {'reference_model': 'prem', 'stack_weighting': 'trace', 'period_band_s': [25, 100]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan_path = root/'plan.json'
            save_plan(plan_path, {'kind': 'bins-v1', 'configuration': cfg, 'tasks': [task],
                                  'missing_event_tasks': [], 'cache_root': str(root/'cache')})
            plan = read_plan(plan_path, 'bins-v1')
            payload = (plan, task, [], tau, traces, rows, np.full(7, 20), np.ones(7))
            settings = {'formats': ['png'], 'dpi': 60, 'absolute_slowness_s_degree': [8, 14, 0.5],
                        'focus_time_s': [-40, 40, 1], 'record_max_traces': 5}
            with patch('ss_precursor.paper_plots.load_members', return_value=payload), \
                    patch('ss_precursor.paper_plots.reference_predictions', return_value=refs):
                result = paper_plots(plan_path, root/'out', 0, settings)
                directory = root/'out'/result['bin']
                for phase in ['SS', 'S400S', 'S670S']:
                    self.assertTrue((directory/f'vespagram_{phase}.png').exists())
                    self.assertTrue((directory/f'{phase}_absolute.npz').exists())
                self.assertTrue((directory/'vespagram_overview.png').exists())
                self.assertTrue((directory/'vespagram_phases.png').exists())
                self.assertTrue((directory/'record_section.png').exists())
                prov = json.loads((directory/'plot_provenance.json').read_text())
                self.assertEqual(len(prov['record_section_trace_ids']), 5)
                self.assertEqual(paper_plots(plan_path, root/'out', 0, settings)['status'], 'cached')


if __name__ == '__main__':
    unittest.main()
