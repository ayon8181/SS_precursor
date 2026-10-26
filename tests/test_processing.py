import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import numpy as np
import yaml
from obspy import Trace, Stream, UTCDateTime
from obspy.core.event import Origin, Event, ResourceIdentifier
from ss_precursor.asdf_io import associated_stream, select_event
from ss_precursor.preprocessing import align, transverse
from ss_precursor.arrivals import unique_reflection
from ss_precursor.pipeline import prepare
from obspy.taup import TauPyModel


class ProcessingValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(Path('example_config.yaml').read_text())

    def test_start_offset_and_whole_trace_polarity(self):
        origin = Origin(time=UTCDateTime(0), latitude=0, longitude=0, depth=10000)
        time = 5+np.arange(4000)*0.5
        u = -np.exp(-((time-1207)/12)**2)
        tr = Trace(u, header={'starttime': origin.time+5, 'delta': 0.5})
        t, data, meta = align(tr, origin, 1200, self.cfg)
        self.assertAlmostEqual(meta['measured_ss_s'], 1207, delta=1)
        self.assertAlmostEqual(meta['trace_start_from_origin_s'], 5)
        self.assertEqual(meta['polarity_multiplier'], -1)
        self.assertAlmostEqual(t[np.argmax(data)], 0, delta=0.5)
        with self.assertRaises(ValueError):
            align(Trace(np.zeros_like(u), header=tr.stats), origin, 1200, self.cfg)

    def test_component_timing(self):
        r = Trace(np.zeros(20), header={'channel': 'BXR', 'delta': 1})
        t = Trace(np.ones(20), header={'channel': 'BXT', 'delta': 1})
        np.testing.assert_array_equal(transverse(Stream([r, t]), 0, 'RT').data, t.data)
        t.stats.starttime += 1
        with self.assertRaises(ValueError):
            transverse(Stream([r, t]), 0, 'RT')

    def test_missing_multi_event_association(self):
        tr = Trace(np.ones(20))
        with self.assertRaises(ValueError):
            associated_stream({'tag': Stream([tr])}, 'tag', 'event', 2)

    def test_ne_rotation_convention(self):
        n = Trace(np.ones(20), header={'channel': 'BXN'})
        e = Trace(np.zeros(20), header={'channel': 'BXE'})
        np.testing.assert_allclose(transverse(Stream([n, e]), 90, 'NE').data, 1)
        self.assertEqual(n.stats.channel, 'BXN')  # Original stream is preserved.

    def test_ambiguous_catalog(self):
        class Dataset:
            events = [Event(resource_id=ResourceIdentifier('smi:local/a')), Event(resource_id=ResourceIdentifier('smi:local/b'))]
        with self.assertRaises(ValueError):
            select_event(Dataset())

    def test_real_prem_reflection_branches(self):
        model = TauPyModel('prem')
        ss = unique_reflection(model, 20, 120, 'SS', 0, 0, 0, 90)
        for h in (400, 670):
            a = unique_reflection(model, 20, 120, f'S^{h}S', h, 0, 0, 90)
            self.assertLess(a['time_s'], ss['time_s'])
            self.assertAlmostEqual(a['reflection_longitude'], 60, delta=1)

    def test_failed_station_does_not_terminate_run(self):
        from contextlib import contextmanager
        class BadStations:
            def list(self): return ['BAD.ONE', 'BAD.TWO']
            def __getitem__(self, name): raise ValueError('deliberate bad station')
        origin = Origin(time=UTCDateTime(0), latitude=0, longitude=0, depth=10000)
        event = Event(origins=[origin], preferred_origin_id=origin.resource_id)
        class Dataset:
            waveforms = BadStations()
            events = [event]
        @contextmanager
        def mocked(path): yield Dataset()
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)/'input'; source.write_text('test')
            with patch('ss_precursor.pipeline.open_asdf', mocked):
                result = prepare(source, self.cfg, Path(root)/'out')
            self.assertEqual(result['rejected'], 2)
            self.assertEqual(result['accepted'], 0)
            rows = json.loads((Path(root)/'out/traces.json').read_text())
            self.assertEqual(len(rows['rejected']), 2)


if __name__ == '__main__':
    unittest.main()
