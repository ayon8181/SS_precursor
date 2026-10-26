"""Controlled job-array integration tests; never read the real ASDF collection."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from obspy import UTCDateTime
from obspy.core.event import Event, Origin
from ss_precursor.batch import (event_cache, fingerprint, make_bin_plan, make_event_plan,
                                prepare_event, read_plan, save_plan, stack_bin, task_destination)
from ss_precursor.pipeline import dump_json, grid, prepare


class BatchValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = yaml.safe_load(Path('example_config.yaml').read_text())
        self.cfg.update(minimum_bin_traces=3, minimum_bin_events=3,
                        slowness_s_degree=[-1, 1, 0.5], precursor_stack_offsets_s=[-10, 10, 1])

    def caches(self, name='run', weighting='trace', count=3, duplicate=False):
        root = self.root/name; root.mkdir()
        cfg = dict(self.cfg, stack_weighting=weighting)
        tasks = []
        for i in range(3):
            source = root/f'input{i}.h5'; source.write_text('controlled source identity')
            stat = source.stat()
            tasks.append({'index': i, 'source': str(source), 'event_id': f'smi:local/event{i}',
                          'source_size_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns})
        plan_path = root/'events.json'
        payload = {'kind': 'events-v1', 'configuration': cfg, 'network': 'SY', 'tasks': tasks}
        save_plan(plan_path, payload)
        for i, task in enumerate(tasks[:count]):
            cache = event_cache(root, i); cache.mkdir(parents=True)
            tau = grid(cfg['relative_time_s']); data, rows = [], []
            for j in range(i+1):
                dt400, dt670 = -150+2*i+j, -230+i+j
                waveform = (i+1)*np.exp(-((tau-dt400)/3)**2)+(i+1)*0.5*np.exp(-((tau-dt670)/3)**2)
                data.append(waveform)
                rows.append({'event_id': task['event_id'], 'station': f'SY.S{j}', 'trace_id': f'SY.S{j}.S3.BXT',
                    'distance_degree': 100+10*i+2*j, 'azimuth_degree': 30*i,
                    'predicted_ss_s': 2000, 'predicted_interference_flags': [],
                    'station_coordinates': {'latitude': 80, 'longitude': 50},
                    'arrivals': {'S^400S': {'time_s': 2000+dt400, 'reflection_latitude': 0.5,
                                          'reflection_longitude': [-180, 180, 540][i]},
                                 'S^670S': {'time_s': 2000+dt670, 'reflection_latitude': 6,
                                          'reflection_longitude': -170}}})
            if duplicate and i == 0:
                rows.append(dict(rows[0])); data.append(data[0].copy())
            np.save(cache/'tau.npy', tau); np.save(cache/'waveforms.npy', np.asarray(data))
            dump_json(cache/'traces.json', {'accepted': rows, 'rejected': []})
            dump_json(cache/'provenance.json', {'configuration': dict(cfg, event_id=task['event_id']),
                      'run_selection': {'network': 'SY'}, 'selected_station_count': len(rows)})
            dump_json(cache/'complete.json', {'plan': fingerprint(payload), 'task_index': i})
        return root, plan_path

    def test_pool_events_at_phase_reflections_and_stack_known_arrivals(self):
        for weighting, expected in [('trace', 14/6), ('event_equal', 2)]:
            root, events = self.caches(name=weighting, weighting=weighting)
            result = make_bin_plan(events, root, root/'bins')
            self.assertEqual(result['task_count'], 2)
            plan = read_plan(root/'bins/plan.json', 'bins-v1')
            first = plan['tasks'][0]
            self.assertEqual(first['phase'], 'S^400S')
            self.assertEqual((first['latitude_index'], first['longitude_index']), (18, 0))
            self.assertEqual(first['trace_count'], 6)
            self.assertEqual(first['event_count'], 3)
            stacked = stack_bin(root/'bins/plan.json', root/'stacks', 0)
            directory = root/'stacks'/stacked['bin']
            with np.load(directory/'predicted_stack.npz') as a:
                k = np.argmax(a['stack'])
                self.assertEqual(a['offsets'][k], 0)
                self.assertAlmostEqual(a['stack'][k], expected, places=6)
                self.assertEqual(a['count'][k], 6)
                self.assertAlmostEqual(a['weight_sum'][k], 6 if weighting == 'trace' else 3)
            self.assertEqual(stack_bin(root/'bins/plan.json', root/'stacks', 0)['status'], 'cached')
            members = json.loads((directory/'members.json').read_text())
            self.assertEqual({m['event_id'] for m in members}, {f'smi:local/event{i}' for i in range(3)})

    def test_missing_event_tasks_fail_by_default_and_partial_plan_is_explicit(self):
        root, events = self.caches(count=2)
        with self.assertRaisesRegex(ValueError, 'not completed'):
            make_bin_plan(events, root, root/'bins')
        self.assertFalse((root/'bins').exists())
        result = make_bin_plan(events, root, root/'bins', allow_incomplete=True)
        self.assertEqual(result['missing_event_tasks'], [2])
        self.assertEqual(result['task_count'], 0)  # Event-count threshold is still enforced.
        coverage = json.loads((root/'bins/coverage.json').read_text())
        self.assertTrue(all('event_count' in b['ineligible_reasons'] for b in coverage))
        relaxed = make_bin_plan(events, root, root/'bins_two_events', allow_incomplete=True,
                                stack_settings={'minimum_bin_events': 2})
        self.assertEqual(relaxed['task_count'], 2)
        with self.assertRaisesRegex(ValueError, 'override only'):
            make_bin_plan(events, root, root/'invalid', allow_incomplete=True,
                          stack_settings={'period_band_s': [10, 50]})

    def test_duplicates_cannot_inflate_bin_counts(self):
        root, events = self.caches(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate event/trace'):
            make_bin_plan(events, root, root/'bins')
        self.assertFalse((root/'bins').exists())

    def test_failed_writer_never_publishes_partial_cache_and_unlocks(self):
        with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
            with task_destination(self.root, 'task', {'id': 1}) as work:
                (work/'partial').write_text('incomplete')
                raise RuntimeError('controlled failure')
        self.assertFalse((self.root/'task').exists())
        self.assertFalse((self.root/'task.lock').exists())
        with task_destination(self.root, 'task', {'id': 1}) as work:
            (work/'data').write_text('complete')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            with task_destination(self.root, 'task', {'id': 2}):
                pass

    def test_plan_enumerates_multiple_events_and_prepare_passes_network(self):
        source = self.root/'multi.h5'; source.write_text('controlled')
        class Dataset:
            events = [Event(resource_id=f'smi:local/e{i}') for i in range(2)]
        @contextmanager
        def mocked(path): yield Dataset()
        with patch('ss_precursor.batch.open_asdf', mocked):
            result = make_event_plan([str(source)], self.cfg, self.root/'events.json')
        self.assertEqual(result['task_count'], 2)
        def mocked_prepare(path, cfg, output, **kwargs):
            self.assertEqual(cfg['event_id'], 'smi:local/e1')
            self.assertEqual(kwargs, {'network': 'SY', 'cache_format': 'npy'})
            return {'accepted': 0, 'rejected': 0}
        with patch('ss_precursor.batch.prepare', side_effect=mocked_prepare) as mock:
            self.assertEqual(prepare_event(self.root/'events.json', self.root/'run', 1)['status'], 'complete')
            self.assertEqual(prepare_event(self.root/'events.json', self.root/'run', 1)['status'], 'cached')
            self.assertEqual(mock.call_count, 1)
        plan = json.loads((self.root/'events.json').read_text()); plan['network'] = 'IU'
        (self.root/'events.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'modified'):
            read_plan(self.root/'events.json', 'events-v1')

    def test_exact_network_selection_precedes_station_access(self):
        class Group:
            coordinates = {'latitude': 0, 'longitude': 0, 'elevation_in_m': 0}
        class Stations:
            def list(self): return ['IU.REAL', 'SY.ONE', 'SY.BAD', 'SY1.OTHER']
            def __getitem__(self, name):
                self_outer.assertIn(name, ['SY.ONE', 'SY.BAD'])
                if name == 'SY.BAD':
                    class BadGroup:
                        coordinates = {'latitude': float('nan'), 'longitude': 0}
                    return BadGroup()
                return Group()
        origin = Origin(time=UTCDateTime(0), latitude=0, longitude=0, depth=10000)
        event = Event(origins=[origin], preferred_origin_id=origin.resource_id)
        class Dataset:
            events = [event]
            waveforms = Stations()
        @contextmanager
        def mocked(path): yield Dataset()
        self_outer = self
        source = self.root/'input'; source.write_text('controlled')
        with patch('ss_precursor.pipeline.open_asdf', mocked):
            result = prepare(source, self.cfg, self.root/'out', network='SY', cache_format='npy')
        self.assertEqual(result['rejected'], 2)  # Bad metadata and out-of-range station both logged.
        prov = json.loads((self.root/'out/provenance.json').read_text())
        self.assertEqual(prov['selected_station_count'], 2)
        self.assertEqual(np.load(self.root/'out/waveforms.npy').shape, (0, 501))

    def test_parallel_cli_and_array_task_retry(self):
        root, events = self.caches()
        make_bin_plan(events, root, root/'bins')
        command = [sys.executable, '-m', 'ss_precursor', 'stack-bins',
                   '--plan', str(root/'bins/plan.json'), '--output', str(root/'stacks')]
        result = subprocess.run(command+['--workers', '2'], check=True, capture_output=True, text=True)
        outputs = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual({r['task_index'] for r in outputs}, {0, 1})
        self.assertTrue(all(r['status'] == 'complete' for r in outputs))
        result = subprocess.run(command+['--task-index', '0'], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'cached')


if __name__ == '__main__':
    unittest.main()
