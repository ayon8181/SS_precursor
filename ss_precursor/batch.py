"""Independent event and geographic-bin tasks for job arrays or local processes.

Plans freeze configuration and input identities. Each task writes its own cache
transactionally; bins read selected waveform rows through memory maps.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import glob
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
from .asdf_io import open_asdf
from .geometry import geographic_bin
from .pipeline import dump_json, grid, load_preprocessed, prepare
from .stacking import slant_stack, predicted_stack


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_plan(path, payload):
    dump_json(path, dict(payload, fingerprint=fingerprint(payload)))


def read_plan(path, kind):
    payload = json.loads(Path(path).read_text())
    digest = payload.pop('fingerprint')
    if digest != fingerprint(payload) or payload.get('kind') != kind:
        raise ValueError('Plan is modified or has the wrong kind; create a new plan')
    return dict(payload, fingerprint=digest)


def make_event_plan(patterns, cfg, output, network='SY'):
    """Read event catalogs only; expand every event explicitly, even in multi-event files."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new event-plan filename to keep job-array indices stable')
    if not network or '.' in network:
        raise ValueError('Network must be an exact network code, e.g. SY')
    paths = sorted({Path(p).resolve() for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise ValueError('No input ASDF files matched')
    tasks = []
    for path in paths:
        stat = path.stat()
        with open_asdf(path) as ds:
            ids = [str(e.resource_id) for e in ds.events]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'{path}: missing events or duplicate event resource IDs')
        if cfg.get('event_id') is not None:
            ids = [eid for eid in ids if eid == cfg['event_id']]
            if not ids:
                raise ValueError(f'{path}: configured event ID not found')
        for event_id in ids:
            tasks.append({'index': len(tasks), 'source': str(path), 'event_id': event_id,
                          'source_size_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns})
        print(json.dumps({'catalog': str(path), 'event_tasks': len(tasks)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_plan(output, {'kind': 'events-v1', 'configuration': cfg, 'network': network, 'tasks': tasks})
    return {'plan': str(output), 'task_count': len(tasks)}


@contextmanager
def task_destination(root, name, identity):
    """Yield None for completed tasks; prevent concurrent writers and partial caches."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    destination = root/name
    if destination.exists():
        marker = destination/'complete.json'
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError(f'Cache identity mismatch or incomplete cache: {destination}; use a new output root')
        yield None
        return
    lock = root/(name+'.lock')
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Task already locked: {lock}; remove only after confirming no job is active')
    try:
        with tempfile.TemporaryDirectory(prefix=name+'.tmp-', dir=str(root)) as staging:
            work = Path(staging)/'result'; work.mkdir()
            yield work
            dump_json(work/'complete.json', identity)
            work.rename(destination)
    finally:
        lock.rmdir()


def event_cache(root, index):
    return Path(root)/'events'/f'event_{index:06d}'


def prepare_event(plan_path, root, index):
    plan = read_plan(plan_path, 'events-v1')
    if not 0 <= index < len(plan['tasks']):
        raise ValueError('Event task index out of range')
    task = plan['tasks'][index]
    identity = {'plan': plan['fingerprint'], 'task_index': index}
    destination = event_cache(root, index)
    with task_destination(destination.parent, destination.name, identity) as work:
        if work is None:
            return {'task_index': index, 'status': 'cached', 'output': str(destination)}
        source = Path(task['source']); stat = source.stat()
        if stat.st_size != task['source_size_bytes'] or stat.st_mtime_ns != task['source_mtime_ns']:
            raise ValueError(f'Input changed since planning: {source}')
        cfg = dict(plan['configuration'], event_id=task['event_id'])
        result = prepare(source, cfg, work, network=plan['network'], cache_format='npy')
    return dict(result, task_index=index, status='complete', output=str(destination))


def make_bin_plan(event_plan, root, output, allow_incomplete=False, stack_settings=None):
    """Index metadata in SQLite without loading all events' waveforms or metadata at once."""
    plan = read_plan(event_plan, 'events-v1')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new bin-plan directory; existing array indices must remain stable')
    completed, missing = [], []
    for task in plan['tasks']:
        cache = event_cache(root, task['index'])
        marker = cache/'complete.json'
        if not marker.exists():
            missing.append(task['index'])
            continue
        if json.loads(marker.read_text()) != {'plan': plan['fingerprint'], 'task_index': task['index']}:
            raise ValueError(f'Cache belongs to a different plan: {cache}')
        completed.append(task)
    if missing and not allow_incomplete:
        raise ValueError(f'Event tasks not completed: {missing}. Finish them before planning bins, or explicitly use --allow-incomplete')
    if not completed:
        raise ValueError('No completed event caches')
    allowed = {'bin_size_degree', 'minimum_bin_traces', 'minimum_bin_events',
               'minimum_distance_aperture_degree', 'stack_weighting', 'slowness_s_degree',
               'reference_distance_degree', 'precursor_stack_offsets_s'}
    settings = stack_settings or {}
    if not isinstance(settings, dict) or set(settings)-allowed:
        raise ValueError('Stack settings may override only binning, coverage, weighting, and stack grids')
    cfg = dict(plan['configuration'], **settings); size = cfg['bin_size_degree']
    geographic_bin(0, 0, size)
    if (cfg['minimum_bin_traces'] < 1 or cfg['minimum_bin_events'] < 1
            or not np.isfinite(cfg['minimum_distance_aperture_degree'])
            or cfg['minimum_distance_aperture_degree'] < 0):
        raise ValueError('Coverage thresholds must be positive counts and a nonnegative aperture')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bin-plan-', dir=str(output.parent)) as staging:
        work = Path(staging)/'result'; work.mkdir()
        con = sqlite3.connect(str(work/'index.sqlite'))
        summary = {'selected_stations': 0, 'accepted_traces': 0, 'rejected_traces': 0, 'rejection_reasons': Counter(),
                   'events_without_selected_stations': [], 'events_with_zero_accepted': []}
        try:
            con.executescript('''
                CREATE TABLE trace_keys(event_id TEXT, trace_id TEXT, PRIMARY KEY(event_id, trace_id));
                CREATE TABLE members(phase TEXT, lat INTEGER, lon INTEGER, task INTEGER, row_index INTEGER,
                    event_id TEXT, trace_id TEXT, station TEXT, distance REAL, azimuth REAL,
                    reflection_lat REAL, reflection_lon REAL, relative_s REAL, interference_flags INTEGER);
            ''')
            for task in completed:
                cache = event_cache(root, task['index'])
                prov = json.loads((cache/'provenance.json').read_text())
                expected_cfg = dict(plan['configuration'], event_id=task['event_id'])
                if prov['configuration'] != expected_cfg or prov['run_selection'].get('network') != plan['network']:
                    raise ValueError(f'Incompatible preprocessing configuration/network: {cache}')
                rows = json.loads((cache/'traces.json').read_text())
                tau, data = load_preprocessed(cache)
                if not np.array_equal(tau, grid(cfg['relative_time_s'])) or data.shape != (len(rows['accepted']), len(tau)):
                    raise ValueError(f'Cache time grid or row count mismatch: {cache}')
                del data
                summary['selected_stations'] += prov.get('selected_station_count', len(rows['accepted'])+len(rows['rejected']))
                if prov.get('selected_station_count', len(rows['accepted'])+len(rows['rejected'])) == 0:
                    summary['events_without_selected_stations'].append(task['index'])
                if not rows['accepted']:
                    summary['events_with_zero_accepted'].append(task['index'])
                summary['accepted_traces'] += len(rows['accepted'])
                summary['rejected_traces'] += len(rows['rejected'])
                summary['rejection_reasons'].update(r['reason'] for r in rows['rejected'])
                for i, row in enumerate(rows['accepted']):
                    if row['event_id'] != task['event_id'] or row['station'].split('.', 1)[0] != plan['network']:
                        raise ValueError(f'Wrong event or network in {cache}')
                    try:
                        con.execute('INSERT INTO trace_keys VALUES (?,?)', (row['event_id'], row['trace_id']))
                    except sqlite3.IntegrityError:
                        raise ValueError(f'Duplicate event/trace would be counted twice: {row["event_id"]}, {row["trace_id"]}')
                    for h in cfg['interface_depths_km']:
                        phase = f'S^{h:g}S'; arrival = row['arrivals'][phase]
                        lat, lon = geographic_bin(arrival['reflection_latitude'], arrival['reflection_longitude'], size)
                        con.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (phase, lat, lon, task['index'], i, row['event_id'], row['trace_id'], row['station'],
                             row['distance_degree'], row['azimuth_degree'], arrival['reflection_latitude'],
                             arrival['reflection_longitude'], arrival['time_s']-row['predicted_ss_s'],
                             len(row['predicted_interference_flags'])))
                con.commit()
            con.execute('CREATE INDEX by_bin ON members(phase,lat,lon,task,row_index)')
            bins = []
            for phase, lat, lon, n, events, low, high, mean, square in con.execute('''
                SELECT phase,lat,lon,COUNT(*),COUNT(DISTINCT event_id),MIN(distance),MAX(distance),
                       AVG(distance),AVG(distance*distance) FROM members GROUP BY phase,lat,lon ORDER BY phase,lat,lon
            '''):
                reasons = []
                if n < cfg['minimum_bin_traces']: reasons.append('trace_count')
                if events < cfg['minimum_bin_events']: reasons.append('event_count')
                if high-low < cfg['minimum_distance_aperture_degree']: reasons.append('distance_aperture')
                bins.append({'phase': phase, 'latitude_index': lat, 'longitude_index': lon,
                    'latitude_center': -90+(lat+0.5)*size, 'longitude_center': -180+(lon+0.5)*size,
                    'trace_count': n, 'event_count': events, 'distance_min_degree': low,
                    'distance_max_degree': high, 'distance_aperture_degree': high-low,
                    'distance_std_degree': math.sqrt(max(0, square-mean*mean)),
                    'eligible': not reasons, 'ineligible_reasons': reasons})
            con.commit()
        finally:
            con.close()
        summary['rejection_reasons'] = dict(summary['rejection_reasons'])
        dump_json(work/'coverage.json', bins)
        dump_json(work/'selection_summary.json', summary)
        tasks = [dict(b, index=i) for i, b in enumerate(b for b in bins if b['eligible'])]
        save_plan(work/'plan.json', {'kind': 'bins-v1', 'event_plan_fingerprint': plan['fingerprint'],
            'configuration': cfg, 'network': plan['network'], 'cache_root': str(root),
            'completed_event_tasks': [t['index'] for t in completed], 'missing_event_tasks': missing,
            'tasks': tasks})
        work.rename(output)
    return {'plan': str(output/'plan.json'), 'task_count': len(tasks), 'occupied_bins': len(bins),
            'missing_event_tasks': missing, **summary}


def stack_bin(plan_path, output, index, plots=False):
    plan_path = Path(plan_path).resolve(); plan = read_plan(plan_path, 'bins-v1')
    if not 0 <= index < len(plan['tasks']):
        raise ValueError('Bin task index out of range')
    task = plan['tasks'][index]; cfg = plan['configuration']
    name = f'{task["phase"].replace("^", "")}_lat{task["latitude_index"]:02d}_lon{task["longitude_index"]:02d}'
    identity = {'plan': plan['fingerprint'], 'task_index': index, 'plots': plots}
    with task_destination(output, name, identity) as work:
        if work is None:
            return {'task_index': index, 'status': 'cached', 'bin': name}
        con = sqlite3.connect((plan_path.parent/'index.sqlite').as_uri()+'?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            members = [dict(r) for r in con.execute('SELECT * FROM members WHERE phase=? AND lat=? AND lon=? ORDER BY task,row_index',
                (task['phase'], task['latitude_index'], task['longitude_index']))]
        finally:
            con.close()
        if len(members) != task['trace_count']:
            raise ValueError('Bin index is inconsistent with plan')
        traces, tau, loaded_task, data = [], None, None, None
        for member in members:
            if member['task'] != loaded_task:
                cache = event_cache(plan['cache_root'], member['task'])
                marker = json.loads((cache/'complete.json').read_text())
                if marker != {'plan': plan['event_plan_fingerprint'], 'task_index': member['task']}:
                    raise ValueError('Event cache identity changed after bin planning')
                current_tau, data = load_preprocessed(cache)
                if tau is not None and not np.array_equal(tau, current_tau):
                    raise ValueError('Mixed time grids in bin')
                tau, loaded_task = current_tau, member['task']
            traces.append(np.array(data[member['row_index']], copy=True))
        event_counts = Counter(m['event_id'] for m in members)
        weighting = cfg.get('stack_weighting', 'trace')
        if weighting not in ('trace', 'event_equal'):
            raise ValueError('stack_weighting must be trace or event_equal')
        weights = np.array([1 if weighting == 'trace' else 1/event_counts[m['event_id']] for m in members])
        times = [tau]*len(traces)
        result = slant_stack(times, traces, [m['distance'] for m in members], tau,
                             grid(cfg['slowness_s_degree']), cfg['reference_distance_degree'], weights)
        offsets = grid(cfg.get('precursor_stack_offsets_s', [-60, 60, cfg['relative_time_s'][2]]))
        predicted = predicted_stack(times, traces, [m['relative_s'] for m in members], offsets, weights)
        np.savez_compressed(work/'vespagram.npz', **result)
        np.savez_compressed(work/'predicted_stack.npz', offsets=offsets, **predicted)
        dump_json(work/'members.json', members)
        dump_json(work/'provenance.json', {'bin': task, 'configuration': cfg, 'network': plan['network'],
            'bin_plan': str(plan_path), 'bin_plan_fingerprint': plan['fingerprint'],
            'event_plan_fingerprint': plan['event_plan_fingerprint'], 'cache_root': plan['cache_root'],
            'weighting': weighting, 'weights': weights.tolist(), 'missing_event_tasks': plan['missing_event_tasks']})
        if plots:
            from .diagnostics import plot_stacks
            plot_stacks(result, {task['phase']: predicted}, offsets, work/'stacks.png')
    return {'task_index': index, 'status': 'complete', 'bin': name, 'trace_count': len(traces)}


def run_tasks(function, arguments, workers=1):
    """Local process parallelism; scheduler arrays call the same single-task functions."""
    if workers < 1:
        raise ValueError('workers must be positive')
    failures = []
    if workers == 1:
        for args in arguments:
            try:
                print(json.dumps(function(*args)), flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(json.dumps({'error': str(exc), 'task_index': args[2]}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(function, *args): args[2] for args in arguments}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    failures.append(str(exc))
                    print(json.dumps({'error': str(exc), 'task_index': futures[future]}), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; successful caches are reusable; rerun to retry failures')
