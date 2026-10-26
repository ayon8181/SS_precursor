"""Incremental station processing with restartable, provenance-bearing outputs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from .asdf_io import open_asdf, select_event, associated_stream
from .arrivals import predictions
from .geometry import geographic_bin, spherical_geometry
from .preprocessing import transverse, align
from .stacking import sample_trace, slant_stack, predicted_stack


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def grid(spec):
    lo, hi, step = spec
    if step <= 0 or hi <= lo:
        raise ValueError('Invalid grid')
    return np.arange(lo, hi+step*0.5, step)


def provenance(cfg, source):
    stat = source.stat()
    return {'configuration': cfg, 'configuration_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            'source': str(source.resolve()), 'source_size_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'python': platform.python_version(), 'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'obspy', 'pyasdf', 'matplotlib']}}


def load_preprocessed(directory):
    """Memory-map batch caches; retain compatibility with original NPZ caches."""
    directory = Path(directory)
    if (directory/'waveforms.npy').exists():
        return (np.load(directory/'tau.npy', allow_pickle=False),
                np.load(directory/'waveforms.npy', mmap_mode='r', allow_pickle=False))
    with np.load(directory/'preprocessed.npz', allow_pickle=False) as cached:
        return cached['tau'], cached['data']


def prepare(path, cfg, output, limit=None, station_ids=None, network=None, cache_format='npz'):
    from obspy.taup import TauPyModel
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if cfg['input_units'] != 'displacement_mm':
        raise ValueError('This baseline expects displacement_mm; unit conversion is not implicit')
    if 2*cfg['relative_time_s'][2] >= cfg['period_band_s'][0]:
        raise ValueError('Output time grid does not resolve the filter passband')
    if not 0 <= cfg['taper_fraction_per_end'] <= 0.5:
        raise ValueError('Taper fraction per end must lie in [0, 0.5]')
    if cache_format not in ('npz', 'npy'):
        raise ValueError('Unknown preprocessing cache format')
    tau = grid(cfg['relative_time_s']); data = []; accepted = []; rejected = []
    model = TauPyModel(cfg['reference_model'])
    prov = provenance(cfg, path)
    prov['run_selection'] = {'station_limit': limit, 'station_ids': station_ids, 'network': network}
    with open_asdf(path) as ds:
        event, origin = select_event(ds, cfg.get('event_id'), cfg['origin'])
        prov.update(event_id=str(event.resource_id), origin_id=str(origin.resource_id),
            origin_time=str(origin.time), origin_latitude=origin.latitude, origin_longitude=origin.longitude,
            origin_depth_km=origin.depth/1000,
            reference_discontinuities_km=model.model.s_mod.v_mod.get_discontinuity_depths().tolist())
        names = ds.waveforms.list()
        if station_ids is not None and set(station_ids)-set(names):
            raise ValueError('Requested station IDs absent from ASDF: '+str(sorted(set(station_ids)-set(names))))
        stations = [name for name in names if (station_ids is None or name in station_ids)
                    and (network is None or name.split('.', 1)[0] == network)][:limit]
        prov.update(available_station_count=len(names), selected_station_count=len(stations))
        for station_index, station in enumerate(stations):
            row = {'station': station, 'event_id': str(event.resource_id), 'source': str(path.resolve())}
            try:
                group = ds.waveforms[station]; coords = group.coordinates
                if (not np.all(np.isfinite([coords['latitude'], coords['longitude']]))
                        or not -90 <= coords['latitude'] <= 90):
                    raise ValueError('Invalid station coordinates')
                distance, az, baz = spherical_geometry(origin.latitude, origin.longitude, coords['latitude'], coords['longitude'])
                row.update(station_coordinates=coords, distance_degree=distance, azimuth_degree=az, backazimuth_degree=baz)
                if not cfg['distance_range_degree'][0] <= distance <= cfg['distance_range_degree'][1]:
                    raise ValueError('Outside configured exploratory distance range')
                if not cfg['source_depth_range_km'][0] <= origin.depth/1000 <= cfg['source_depth_range_km'][1]:
                    raise ValueError('Outside configured source-depth range')
                stream = associated_stream(group, cfg['waveform_tag'], str(event.resource_id), len(ds.events))
                tr = transverse(stream, baz, cfg['horizontal_convention'])
                arrivals, interference = predictions(model, origin, distance, az, cfg['interface_depths_km'], cfg['interfering_phases'])
                row.update(trace_id=tr.id, arrivals=arrivals, interference=interference)
                collisions = []
                for phase, arrival in arrivals.items():
                    if phase == 'SS':
                        continue
                    for a in interference:
                        if abs(a['time_s']-arrival['time_s']) < cfg['interference_separation_s']:
                            collisions.append({'precursor': phase, 'interference': a['phase'], 'separation_s': a['time_s']-arrival['time_s']})
                row['predicted_interference_flags'] = collisions
                if collisions and cfg['reject_predicted_interference']:
                    raise ValueError('Predicted precursor interference')
                time, waveform, measured = align(tr, origin, arrivals['SS']['time_s'], cfg)
                row.update(measured)
                aligned = sample_trace(time, waveform, tau)
                if not np.all(np.isfinite(aligned)):
                    raise ValueError('Configured output window exceeds reliable trace support')
                row['bins'] = {phase: list(geographic_bin(a['reflection_latitude'], a['reflection_longitude'], cfg['bin_size_degree']))
                               for phase, a in arrivals.items() if phase != 'SS'}
                row['status'] = 'accepted'
                accepted.append(row); data.append(aligned)
            except Exception as exc:
                row.update(status='rejected', reason=f'{type(exc).__name__}: {exc}')
                rejected.append(row)
            if (station_index+1) % 50 == 0:
                print(json.dumps({'processed': station_index+1, 'total': len(stations),
                                  'accepted': len(accepted), 'rejected': len(rejected)}), flush=True)
    dump_json(output/'provenance.json', prov)
    dump_json(output/'traces.json', {'accepted': accepted, 'rejected': rejected})
    matrix = np.asarray(data).reshape(len(data), len(tau))
    if cache_format == 'npy':
        np.save(output/'tau.npy', tau, allow_pickle=False)
        np.save(output/'waveforms.npy', matrix, allow_pickle=False)
    else:
        np.savez_compressed(output/'preprocessed.npz', tau=tau, data=matrix)
    dump_json(output/'coverage.json', coverage(accepted, cfg))
    write_run_report(output, prov, accepted, rejected)
    return {'accepted': len(accepted), 'rejected': len(rejected), 'output': str(output)}


def write_run_report(output, prov, accepted, rejected):
    from collections import Counter
    cfg = prov['configuration']
    lines = ['# First-event processing report', '',
        f"Source: `{prov['source']}`", '',
        f"Event: `{prov['event_id']}`; origin: `{prov['origin_id']}`.",
        f"Origin depth: {prov['origin_depth_km']:.3f} km. Units: displacement mm (user confirmed).",
        f"Accepted {len(accepted)} stations; rejected {len(rejected)} stations.", '',
        f"Reference model: {cfg['reference_model']}; target interfaces: {cfg['interface_depths_km']} km.",
        f"Processing band: {cfg['period_band_s']} s; minimum reliable synthetic period: {cfg['minimum_synthetic_period_s']} s.", '']
    if accepted:
        residuals = np.array([r['alignment_residual_s'] for r in accepted])
        distances = np.array([r['distance_degree'] for r in accepted])
        lines += [f"SS alignment residual (measured minus predicted): median {np.median(residuals):.3f} s; "
                  f"5th–95th percentiles {np.percentile(residuals, 5):.3f}–{np.percentile(residuals, 95):.3f} s.",
                  f"Distance range: {distances.min():.2f}–{distances.max():.2f} degrees.",
                  f"Stations with predicted precursor interference flags: {sum(bool(r['predicted_interference_flags']) for r in accepted)}.", '']
    lines += ['Rejection reasons:', '']
    lines += [f'- {reason}: {n}' for reason, n in Counter(r['reason'] for r in rejected).most_common()]
    bins = coverage(accepted, cfg)
    lines += ['', f"Eligible geographic bins: {sum(r['eligible'] for r in bins)} of {len(bins)} occupied phase-specific bins.",
        '', 'The first-event stacks are processing diagnostics. No real PREM interface depth recovery is claimed. '
        'Review the record section for alignment and interference before selecting multi-event bins. '
        'All source ASDF files were opened read-only.', '']
    (Path(output)/'report.md').write_text('\n'.join(lines))


def coverage(rows, cfg):
    groups = {}
    for r in rows:
        for phase, cell in r['bins'].items():
            groups.setdefault((phase, *cell), []).append(r)
    result = []
    for (phase, lat, lon), members in sorted(groups.items()):
        distances = np.array([r['distance_degree'] for r in members]); events = {r['event_id'] for r in members}
        result.append({'phase': phase, 'latitude_index': lat, 'longitude_index': lon,
            'trace_count': len(members), 'event_count': len(events),
            'distance_min_degree': float(distances.min()), 'distance_max_degree': float(distances.max()),
            'distance_std_degree': float(distances.std()), 'distance_aperture_degree': float(np.ptp(distances)),
            'azimuths_degree': [r['azimuth_degree'] for r in members],
            'reflection_coordinates': [[r['arrivals'][phase]['reflection_latitude'], r['arrivals'][phase]['reflection_longitude']] for r in members],
            'eligible': len(members) >= cfg['minimum_bin_traces'] and len(events) >= cfg['minimum_bin_events'] and np.ptp(distances) >= cfg['minimum_distance_aperture_degree']})
    return result


def stack_prepared(directory):
    from .diagnostics import plot_record_section, plot_stacks
    directory = Path(directory)
    rows = json.loads((directory/'traces.json').read_text())['accepted']
    cfg = json.loads((directory/'provenance.json').read_text())['configuration']
    if not rows:
        raise ValueError('No accepted traces; inspect traces.json rejection reasons')
    tau, data = load_preprocessed(directory)
    times = [tau]*len(data)
    result = slant_stack(times, data, [r['distance_degree'] for r in rows], tau, grid(cfg['slowness_s_degree']), cfg['reference_distance_degree'])
    np.savez_compressed(directory/'vespagram.npz', **result)
    offsets = grid([-60, 60, cfg['relative_time_s'][2]])
    predicted = {}
    for phase in rows[0]['arrivals']:
        if phase == 'SS':
            continue
        # Differential predictions remain referenced to theoretical SS. Measured
        # SS defines each trace's time origin; its residual is not added again.
        relative = [r['arrivals'][phase]['time_s']-r['predicted_ss_s'] for r in rows]
        predicted[phase] = predicted_stack(times, data, relative, offsets)
        np.savez_compressed(directory/f'{phase.replace("^", "")}_predicted_stack.npz', offsets=offsets, **predicted[phase])
    plot_record_section(tau, data, rows, directory/'record_section.png')
    plot_stacks(result, predicted, offsets, directory/'stacks.png')
    return {'trace_count': len(rows), 'reference_distance_degree': result['reference_distance'], 'output': str(directory)}
