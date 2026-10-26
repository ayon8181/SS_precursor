"""Signed SS-aligned stacks in fixed or moving epicentral-distance windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from .batch import read_plan, event_cache
from .paper_plots import load_members
from .pipeline import dump_json, load_preprocessed
from .stacking import weighted_samples


def load_global(plan_path, distance_range):
    """Select accepted traces once from all cells, including ineligible cells.

    The index contains one entry per reflection phase, so select exactly one
    phase to avoid counting the same waveform twice.
    """
    plan_path = Path(plan_path).resolve()
    plan = read_plan(plan_path, 'bins-v1')
    if plan.get('missing_event_tasks'):
        raise ValueError('Global stack requires a complete event-cache plan')
    phase = 'S^{:g}S'.format(plan['configuration']['interface_depths_km'][0])
    con = sqlite3.connect((plan_path.parent/'index.sqlite').as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        members = [dict(row) for row in con.execute(
            'SELECT * FROM members WHERE phase=? AND distance>=? AND distance<=? ORDER BY task,row_index',
            (phase, float(distance_range[0]), float(distance_range[1])))]
    finally:
        con.close()
    if not members:
        raise ValueError('No accepted waveforms in requested distance range')
    traces, rows, depths, seen = [], [], [], set()
    current, tau = None, None
    for member in members:
        if current != member['task']:
            cache = event_cache(plan['cache_root'], member['task'])
            marker = json.loads((cache/'complete.json').read_text())
            if marker != {'plan': plan['event_plan_fingerprint'], 'task_index': member['task']}:
                raise ValueError('Event cache identity changed')
            ctau, data = load_preprocessed(cache)
            metadata = json.loads((cache/'traces.json').read_text())['accepted']
            prov = json.loads((cache/'provenance.json').read_text())
            if tau is not None and not np.array_equal(ctau, tau):
                raise ValueError('Mixed cache time grids')
            if data.shape != (len(metadata), len(ctau)):
                raise ValueError('Cache shape/metadata mismatch')
            tau, current = ctau, member['task']
        row = metadata[member['row_index']]
        key = (row['event_id'], row['trace_id'])
        if key != (member['event_id'], member['trace_id']) or key in seen:
            raise ValueError('Duplicate or mismatched event/trace identity')
        if (row['station'].split('.', 1)[0] != plan['network']
                or not distance_range[0] <= row['distance_degree'] <= distance_range[1]):
            raise ValueError('Wrong network or distance selection')
        seen.add(key)
        traces.append(np.array(data[member['row_index']], copy=True))
        rows.append(row)
        depths.append(prov['origin_depth_km'])
    # Match existing full-selection event weighting if requested by the plan.
    weighting = plan['configuration'].get('stack_weighting', 'trace')
    if weighting not in ('trace', 'event_equal'):
        raise ValueError('Unknown stack weighting')
    counts = Counter(row['event_id'] for row in rows)
    weights = np.array([1 if weighting == 'trace' else 1/counts[row['event_id']] for row in rows])
    return plan, None, members, tau, traces, rows, np.asarray(depths), weights


def match_global(primary, reference, inherit_polarity=False):
    """Intersect accepted event/trace identities, optionally inherit PREM signs.

    Cached data already contain their own polarity multiplier. Multiplying by
    PREM_multiplier / original_multiplier replaces that choice exactly, while
    retaining the existing filtering, SS normalization and timing alignment.
    """
    plan, task, members, tau, traces, rows, depths, weights = primary
    other_plan, _, _, other_tau, _, other_rows, _, _ = reference
    if not np.array_equal(tau, other_tau):
        raise ValueError('Matched models must use identical time grids')
    if inherit_polarity and other_plan['configuration']['waveform_tag'] != 'proc_prem':
        raise ValueError('Inherited polarity must come from PREM')
    for key in ('period_band_s', 'filter_order', 'detrend', 'taper_fraction_per_end',
                'filter_padding_s', 'relative_time_s', 'horizontal_convention', 'input_units'):
        if plan['configuration'][key] != other_plan['configuration'][key]:
            raise ValueError('Processing differs between models: '+key)
    lookup = {(r['event_id'], r['trace_id']): r for r in other_rows}
    selected = [i for i, r in enumerate(rows) if (r['event_id'], r['trace_id']) in lookup]
    if not selected:
        raise ValueError('No common accepted event/station waveforms')
    kept_rows, kept_traces, kept_members = [], [], []
    for i in selected:
        row = dict(rows[i]); ref = lookup[row['event_id'], row['trace_id']]
        if (abs(row['distance_degree']-ref['distance_degree']) > 1e-6
                or row['station_coordinates'] != ref['station_coordinates']):
            raise ValueError('Matched trace geometry differs')
        own, inherited = row['polarity_multiplier'], ref['polarity_multiplier']
        if own not in (-1., 1.) or inherited not in (-1., 1.):
            raise ValueError('Invalid polarity multiplier')
        factor = inherited/own if inherit_polarity else 1.
        row.update(original_polarity_multiplier=own,
                   polarity_multiplier=inherited if inherit_polarity else own,
                   cached_waveform_multiplier=factor)
        member = dict(members[i], cached_waveform_multiplier=factor,
                      original_polarity_multiplier=own,
                      applied_polarity_multiplier=row['polarity_multiplier'])
        kept_rows.append(row); kept_traces.append(traces[i]*factor); kept_members.append(member)
    counts = Counter(r['event_id'] for r in kept_rows)
    weights = np.array([1 if plan['configuration'].get('stack_weighting', 'trace') == 'trace'
                        else 1/counts[r['event_id']] for r in kept_rows])
    return plan, task, kept_members, tau, kept_traces, kept_rows, depths[selected], weights


def distance_stack(tau, traces, distances, weights, edges):
    """Weighted means without column normalization; empty cells remain NaN."""
    data = np.asarray(traces, dtype=float)
    distances, weights, edges = map(np.asarray, (distances, weights, edges))
    if (data.shape != (len(distances), len(tau)) or weights.shape != distances.shape
            or np.any(~np.isfinite(distances)) or np.any(~np.isfinite(weights))
            or np.any(weights <= 0) or len(edges) < 2 or np.any(np.diff(edges) <= 0)):
        raise ValueError('Invalid distance-stack inputs')
    columns = np.searchsorted(edges, distances, side='right')-1
    columns[distances == edges[-1]] = len(edges)-2
    if np.any((columns < 0) | (columns >= len(edges)-1)):
        raise ValueError('Distance outside supplied edges')
    stack = np.full((len(tau), len(edges)-1), np.nan)
    support = np.zeros_like(stack, dtype=int)
    counts = np.bincount(columns, minlength=len(edges)-1)
    for col in range(len(edges)-1):
        keep = columns == col
        if not np.any(keep):
            continue
        values, w = data[keep], weights[keep, None]
        valid = np.isfinite(values)
        denom = np.sum(valid*w, axis=0)
        np.divide(np.sum(np.where(valid, values, 0)*w, axis=0), denom,
                  out=stack[:, col], where=denom > 0)
        support[:, col] = np.sum(valid, axis=0)
    return stack, counts, support, columns


def moving_distance_stack(tau, traces, distances, weights, low, high, width, step):
    """Full-width windows only; half-open bounds, with the final upper end included."""
    data = np.asarray(traces, dtype=float)
    distances, weights = np.asarray(distances), np.asarray(weights)
    if (not np.isfinite([low, high, width, step]).all() or not 0 < width <= high-low
            or step <= 0 or step > width
            or not np.isclose((high-low-width)/step, round((high-low-width)/step))):
        raise ValueError('Moving window must fit range and advance by a compatible positive step')
    if (data.shape != (len(distances), len(tau)) or not len(distances)
            or weights.shape != distances.shape or not np.isfinite(distances).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or np.any((distances < low) | (distances > high))):
        raise ValueError('Invalid moving-stack inputs')
    starts = low+np.arange(int(round((high-low-width)/step))+1)*step
    bounds = np.column_stack((starts, starts+width))
    membership = (distances[None, :] >= bounds[:, :1]) & (distances[None, :] < bounds[:, 1:])
    membership[-1] |= (distances == high)
    stack = np.full((len(tau), len(bounds)), np.nan)
    support = np.zeros_like(stack, dtype=int)
    counts = membership.sum(axis=1)
    for col, keep in enumerate(membership):
        if np.any(keep):
            result = weighted_samples(data[keep], weights[keep])
            stack[:, col], support[:, col] = result['stack'], result['count']
    return stack, counts, support, membership, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', help='Dataset name displayed in the figure title')
    parser.add_argument('--match-plan', type=Path, help='Use only accepted event/trace pairs common to both global plans')
    parser.add_argument('--prem-polarity', action='store_true', help='Replace cached polarity with the PREM multipliers from --match-plan')
    parser.add_argument('--task-index', type=int, help='Default: eligible bin with most traces')
    parser.add_argument('--global-stack', action='store_true', help='Pool accepted traces worldwide once, rather than one geographic bin')
    parser.add_argument('--distance-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Default for global stacks: 100 160 degrees')
    parser.add_argument('--distance-step', type=float, help='Default: global 10 degrees; geographic 5 degrees')
    parser.add_argument('--window-width', type=float, help='Enable moving windows of this width; use --distance-step for their advance')
    parser.add_argument('--amplitude-limit', type=float, default=0.3)
    parser.add_argument('--zoom-amplitude-limit', type=float, default=0.08)
    args = parser.parse_args()
    if args.distance_step is None:
        args.distance_step = 1 if args.window_width is not None else (10 if args.global_stack else 5)
    if args.window_width is not None and (not np.isfinite(args.window_width) or args.window_width <= 0):
        parser.error('Window width must be finite and positive')
    if not all(np.isfinite(x) and x > 0 for x in
               (args.distance_step, args.amplitude_limit, args.zoom_amplitude_limit)):
        parser.error('Distance step and amplitude limits must be finite and positive')
    if args.global_stack and args.task_index is not None:
        parser.error('--global-stack cannot be combined with --task-index')
    if args.global_stack and args.distance_range is None:
        args.distance_range = [100, 160]
    if args.distance_range is not None:
        lo, hi = args.distance_range
        if not (np.isfinite([lo, hi]).all() and 0 <= lo < hi <= 180
                and np.isclose((hi-lo)/args.distance_step, round((hi-lo)/args.distance_step))):
            parser.error('Distance range must be within 0..180 and divisible by distance step')
    if args.match_plan is not None and not args.global_stack:
        parser.error('--match-plan requires --global-stack')
    if args.prem_polarity and args.match_plan is None:
        parser.error('--prem-polarity requires --match-plan pointing to PREM')
    index = args.task_index
    if args.global_stack:
        plan, task, members, tau, traces, rows, depths, weights = load_global(args.plan, args.distance_range)
    else:
        plan = read_plan(args.plan, 'bins-v1')
        if not plan['tasks']:
            parser.error('No eligible bins')
        if index is None:
            index = max(range(len(plan['tasks'])), key=lambda i: plan['tasks'][i]['trace_count'])
        plan, task, members, tau, traces, rows, depths, weights = load_members(args.plan, index)
        if args.distance_range is not None:
            keep = [i for i, row in enumerate(rows)
                    if args.distance_range[0] <= row['distance_degree'] <= args.distance_range[1]]
            if not keep:
                parser.error('No selected traces in distance range')
            traces, rows, members = ([items[i] for i in keep] for items in (traces, rows, members))
            depths, weights = depths[keep], weights[keep]
    matched_summary = None
    if args.match_plan is not None:
        reference = load_global(args.match_plan, args.distance_range)
        original_count = len(rows)
        primary = (plan, task, members, tau, traces, rows, depths, weights)
        plan, task, members, tau, traces, rows, depths, weights = match_global(primary, reference, args.prem_polarity)
        matched_summary = dict(other_plan=str(args.match_plan.resolve()), other_plan_fingerprint=reference[0]['fingerprint'],
            original_trace_count=original_count, common_trace_count=len(rows), excluded_trace_count=original_count-len(rows),
            inherit_prem_polarity=args.prem_polarity, changed_cached_trace_signs=sum(r['cached_waveform_multiplier'] < 0 for r in rows),
            timing_note='Existing event-local SS alignment retained; no new timing shifts or amplitude normalization.',
            polarity_note='Applied sign equals the matching PREM preprocessing sign.' if args.prem_polarity else 'Original PREM signs retained.')
    d = np.array([r['distance_degree'] for r in rows])
    if args.distance_range is None:
        start = np.floor(d.min()/args.distance_step)*args.distance_step
        end = max(start+args.distance_step, np.ceil(d.max()/args.distance_step)*args.distance_step)
    else:
        start, end = args.distance_range
    if args.window_width is not None:
        stack, counts, support, membership, bounds = moving_distance_stack(
            tau, traces, d, weights, start, end, args.window_width, args.distance_step)
        centers = bounds.mean(axis=1)
        edges = np.r_[centers-args.distance_step/2, centers[-1]+args.distance_step/2]
        description = f'{args.window_width:g}° moving windows, {args.distance_step:g}° step'
    else:
        edges = start+np.arange(int(round((end-start)/args.distance_step))+1)*args.distance_step
        stack, counts, support, columns = distance_stack(tau, traces, d, weights, edges)
        centers = (edges[:-1]+edges[1:])/2
        bounds = np.column_stack((edges[:-1], edges[1:]))
        membership = np.arange(len(centers))[:, None] == columns[None, :]
        description = f'{args.distance_step:g}° distance intervals'
    predictions = {}
    for phase in ('S^400S', 'S^670S'):
        curves = np.full((3, len(centers)), np.nan)
        for col in range(len(centers)):
            values = [r['arrivals'][phase]['time_s']-r['predicted_ss_s']
                      for i, r in enumerate(rows) if membership[col, i] and phase in r['arrivals']]
            if values:
                curves[:, col] = np.percentile(values, [0, 50, 100])
        predictions[phase] = curves
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 7])
    time_edges = np.r_[tau-(tau[1]-tau[0])/2, tau[-1]+(tau[1]-tau[0])/2]
    for col, (window, limit, title) in enumerate([
            ((tau[0], tau[-1]), args.amplitude_limit, 'Full cached time window'),
            ((max(tau[0], -300), min(tau[-1], -70)), args.zoom_amplitude_limit, 'Precursor zoom')]):
        ax = fig.add_subplot(gs[1, col])
        top = fig.add_subplot(gs[0, col], sharex=ax)
        if args.window_width is not None:
            top.plot(centers, counts, color='black', linewidth=1)
        else:
            top.stairs(counts, edges, color='black', linewidth=1)
        top.set(ylabel='Waveforms', ylim=(0, max(counts)*1.15))
        top.tick_params(labelbottom=False)
        mesh = ax.pcolormesh(edges, time_edges, stack, shading='flat', cmap='RdBu',
                             vmin=-limit, vmax=limit, rasterized=True)
        ax.set_facecolor('0.88')
        ax.axhline(0, color='black', lw=0.8, label='Aligned SS')
        for phase, curves in predictions.items():
            ax.plot(centers, curves[1], '--', lw=1, label=phase.replace('^', '')+' prediction')
            ax.fill_between(centers, curves[0], curves[2], color='black', alpha=0.07)
        ax.set(xlim=(edges[0], edges[-1]), ylim=window, xlabel='Epicentral distance (degrees)',
               ylabel='Time relative to aligned SS (s)', title=title)
        if args.window_width is not None:
            ax.set_xlabel('Window centre: epicentral distance (degrees)')
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc='lower left', fontsize=8)
        fig.colorbar(mesh, ax=ax, label='Signed amplitude / individual SS peak', extend='both', shrink=0.7)
    heading = ('Global accepted '+plan['network']+' waveforms' if args.global_stack else
               f'{task["phase"].replace("^", "")} reflection bin '
               f'({task["latitude_center"]:g}°, {task["longitude_center"]:g}°)')
    if args.label:
        heading = args.label+' · '+heading
    fig.suptitle(heading+' · '
                 f'N={len(rows)}, {len(set(r["event_id"] for r in rows))} events\n'+
                 description+' · linear means · '
                 'no precursor moveout correction or column normalization', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/('distance_section.'+suffix), dpi=180)
    plt.close(fig)
    np.savez_compressed(output/'distance_stack.npz', tau=tau, distance_edges=edges,
                        distance_centers=centers, distance_window_bounds=bounds,
                        stack=stack, trace_counts=counts, support=support,
                        **{p.replace('^', '')+'_prediction_min_median_max_s': v for p, v in predictions.items()})
    dump_json(output/'members.json', members)
    dump_json(output/'provenance.json', dict(bin_plan=str(args.plan.resolve()),
        bin_plan_fingerprint=plan['fingerprint'], task_index=index, selection_bin=task, matched_selection=matched_summary,
        distance_edges_degree=edges.tolist(), trace_counts=counts.tolist(),
        distance_centers_degree=centers.tolist(), distance_window_bounds_degree=bounds.tolist(),
        source_depths_km=depths.tolist(), arguments=
        {'label': args.label, 'distance_step': args.distance_step, 'window_width': args.window_width, 'amplitude_limit': args.amplitude_limit,
         'zoom_amplitude_limit': args.zoom_amplitude_limit},
        global_stack=args.global_stack, distance_range_degree=args.distance_range,
        overlapping_windows=args.window_width is not None,
        independence_note='Overlapping windows share waveforms; adjacent columns are not independent.'
                          if args.window_width is not None else 'Distance intervals are disjoint.',
        weighting=plan['configuration'].get('stack_weighting', 'trace'),
        period_band_s=plan['configuration']['period_band_s'],
        note='Signed SS-normalized cached waveforms; no precursor shifts or column normalization. '
             'Colors saturate at the stated limits. Gray cells have no data. '
             'Dashed curves show median model predictions; shading spans contributing predictions.'))
    if matched_summary is not None:
        dump_json(output/'matched_selection.json', matched_summary)
    print(json.dumps({'output': str(output), 'task_index': index, 'trace_count': len(rows),
                      'distance_interval_counts': counts.tolist()}))


if __name__ == '__main__':
    main()
