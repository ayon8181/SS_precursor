import unittest
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ss_precursor.distance_section import distance_stack, load_global, moving_distance_stack


class DistanceSectionValidation(unittest.TestCase):
    def test_moving_windows_overlap_and_normalize_by_their_own_support(self):
        # Boundary at 110 leaves the first half-open window and enters later
        # windows. The final range endpoint 112 remains included.
        data = [[1, .1], [1, .3], [1, -.5], [1, np.nan], [1, .9]]
        stack, counts, support, membership, bounds = moving_distance_stack(
            [0, 1], data, [100, 105, 110, 111, 112], np.ones(5), 100, 112, 10, 1)
        np.testing.assert_array_equal(bounds, [[100, 110], [101, 111], [102, 112]])
        np.testing.assert_array_equal(counts, [2, 2, 4])
        np.testing.assert_allclose(stack[0], 1)
        np.testing.assert_allclose(stack[1], [.2, -.1, (.3-.5+.9)/3])
        np.testing.assert_array_equal(support[1], [2, 2, 3])
        self.assertTrue(membership[:, 1].all())  # Shared, rather than partitioned.

    def test_global_selects_all_cells_once_and_keeps_range_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {'configuration': {'interface_depths_km': [400, 670]}, 'tasks': [],
                    'cache_root': tmp, 'network': 'SY', 'event_plan_fingerprint': 'original'}
            rows = [dict(event_id='e', trace_id='t'+str(i), station='SY.A', distance_degree=d)
                    for i, d in enumerate([99, 100, 110, 160, 161])]
            (root/'complete.json').write_text(json.dumps({'plan': 'original', 'task_index': 0}))
            (root/'traces.json').write_text(json.dumps({'accepted': rows}))
            (root/'provenance.json').write_text(json.dumps({'origin_depth_km': 20}))
            con = sqlite3.connect(str(root/'index.sqlite'))
            con.execute('CREATE TABLE members(phase TEXT, task INTEGER, row_index INTEGER, '
                        'event_id TEXT, trace_id TEXT, distance REAL)')
            for phase in ['S^400S', 'S^670S']:
                for i, row in enumerate(rows):
                    con.execute('INSERT INTO members VALUES (?,?,?,?,?,?)',
                                (phase, 0, i, 'e', row['trace_id'], row['distance_degree']))
            con.commit(); con.close()
            with patch('ss_precursor.distance_section.read_plan', return_value=plan), \
                    patch('ss_precursor.distance_section.event_cache', return_value=root), \
                    patch('ss_precursor.distance_section.load_preprocessed',
                          return_value=(np.arange(3), np.ones((5, 3)))):
                payload = load_global(root/'plan.json', [100, 160])
            selected = payload[5]
            self.assertEqual([r['distance_degree'] for r in selected], [100, 110, 160])
            self.assertEqual(len(payload[4]), 3)  # Not doubled by phase membership.
            self.assertEqual(len(payload[2]), 3)  # Includes cells with no eligible tasks.

    def test_signed_weighted_means_edges_and_empty_intervals(self):
        data = [[1, .2, np.nan], [1, -.2, .4], [1, -.1, .2]]
        stack, counts, support, columns = distance_stack(
            [0, 1, 2], data, [100, 101, 130], [1, 3, 1], [100, 110, 120, 130])
        np.testing.assert_array_equal(counts, [2, 0, 1])
        np.testing.assert_array_equal(columns, [0, 0, 2])
        np.testing.assert_allclose(stack[:, 0], [1, -.1, .4])
        self.assertTrue(np.isnan(stack[:, 1]).all())
        np.testing.assert_array_equal(support[:, 0], [2, 2, 1])
        self.assertAlmostEqual(stack[1, 2], -.1)
