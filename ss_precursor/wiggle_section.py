"""Matched PREM/GLAD wiggle sections rebinned into fixed distance averages."""
import argparse
import json
from pathlib import Path
import numpy as np


def load_section(directory, step):
    directory = Path(directory).resolve()
    prov = json.loads((directory/'provenance.json').read_text())
    members = json.loads((directory/'members.json').read_text())
    if prov['weighting'] != 'trace' or prov['overlapping_windows']:
        raise ValueError('Requires disjoint trace-weighted source sections')
    with np.load(directory/'distance_stack.npz', allow_pickle=False) as data:
        tau, old_edges = data['tau'].copy(), data['distance_edges'].copy()
        values, support = data['stack'], data['support']
        counts = data['trace_counts']
        width = np.diff(old_edges)
        ratio = step/width[0]
        if (not np.allclose(width, width[0]) or not np.isclose(ratio, round(ratio))
                or ratio < 1 or len(counts) % int(round(ratio))):
            raise ValueError('New distance intervals must contain whole source intervals')
        factor = int(round(ratio)); n = len(counts)//factor
        valid = support > 0
        if not np.isfinite(values[valid]).all():
            raise ValueError('Source has supported nonfinite samples')
        numerator = np.where(valid, values, 0)*support
        denominator = support.reshape(len(tau), n, factor).sum(axis=2)
        stack = np.divide(numerator.reshape(len(tau), n, factor).sum(axis=2), denominator,
                          out=np.full((len(tau), n), np.nan), where=denominator > 0)
        new_counts = counts.reshape(n, factor).sum(axis=1)
        edges = old_edges[::factor]
    # Use the actual contributing traces, rather than medians of 1-degree medians.
    plan = json.loads(Path(prov['bin_plan']).read_text())
    predictions = {p: [[] for _ in new_counts] for p in ('S^400S', 'S^670S')}
    cache_rows = {}
    for member in members:
        task = member['task']
        if task not in cache_rows:
            cache = Path(plan['cache_root'])/'events'/('event_%06d' % task)
            cache_rows[task] = json.loads((cache/'traces.json').read_text())['accepted']
        row = cache_rows[task][member['row_index']]
        if (row['event_id'], row['trace_id']) != (member['event_id'], member['trace_id']):
            raise ValueError('Source member identity changed')
        col = min(int(np.searchsorted(edges, member['distance'], side='right')-1), len(new_counts)-1)
        for phase in predictions:
            if phase in row['arrivals']:
                predictions[phase][col].append(row['predicted_ss_s']-row['arrivals'][phase]['time_s'])
    predictions = {p: np.array([np.median(v) if v else np.nan for v in cells])
                   for p, cells in predictions.items()}
    return dict(source=str(directory), provenance=prov, members=members, tau=tau,
                stack=stack, support=denominator, counts=new_counts, edges=edges,
                centers=(edges[:-1]+edges[1:])/2, predictions=predictions)


def draw(datasets, output, precursor_gain, ss_gain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    fig = plt.figure(figsize=(8*len(datasets), 9), constrained_layout=True)
    gs = fig.add_gridspec(2, len(datasets), height_ratios=[1.8, 6])
    for column, (name, section) in enumerate(datasets):
        upper = fig.add_subplot(gs[0, column])
        lower = fig.add_subplot(gs[1, column], sharex=upper)
        before = -section['tau']
        order = np.argsort(before)
        time = before[order]
        for ax, window, gain in [(upper, (-40, 40), ss_gain), (lower, (40, 300), precursor_gain)]:
            keep = (time >= window[0]) & (time <= window[1])
            for i, center in enumerate(section['centers']):
                amplitude = section['stack'][order, i][keep]
                y = time[keep]
                x = center+gain*amplitude
                ax.fill_betweenx(y, center, x, where=amplitude > 0, interpolate=True,
                                 color='#ef2b16', linewidth=0, zorder=2)
                ax.plot(x, y, color='black', linewidth=.9, zorder=3)
            ax.set_ylim(window[1], window[0])
            ax.set_xlim(section['edges'][0]-3, section['edges'][-1]+8)
            ax.set_facecolor('white')
            ax.tick_params(direction='out', right=False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        upper.set_title(name, loc='left', fontsize=18, fontweight='bold', pad=15)
        upper.xaxis.tick_top(); upper.xaxis.set_label_position('top')
        upper.set_xlabel('Epicentral distance (degrees)', fontsize=13, labelpad=9)
        upper.set_xticks(np.arange(100, 161, 10)); upper.set_yticks([-30, 0, 30])
        upper.axhline(0, color='.6', linewidth=.7, linestyle=':', zorder=1)
        upper.text(164, 0, 'SS', fontsize=12, va='center')
        upper.spines['bottom'].set_visible(False)
        upper.text(.02, .03, 'SS gain: %g° / unit amplitude' % ss_gain,
                   transform=upper.transAxes, fontsize=9, va='bottom',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        lower.spines['top'].set_visible(False); lower.spines['bottom'].set_visible(False)
        lower.tick_params(axis='x', bottom=False, labelbottom=False)
        lower.set_yticks([60, 120, 180, 240, 300])
        lower.set_ylabel('Time before aligned SS (s)', fontsize=14)
        for phase, times in section['predictions'].items():
            lower.plot(section['centers'], times, '--', color='.55', lw=.8, zorder=1)
            finite = np.flatnonzero(np.isfinite(times))
            if len(finite):
                lower.text(163, times[finite[-1]], phase.replace('^', ''), va='center', fontsize=11)
        lower.text(.02, .02, 'Precursor gain: %g° / unit amplitude (×%g SS)\n'
                   'Dashed: PREM predictions' % (precursor_gain, precursor_gain/ss_gain),
                   transform=lower.transAxes, fontsize=9, va='bottom',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        for ax, y in [(upper, 0), (lower, 1)]:
            # Short diagonal marks identify the change in amplitude gain at 40 s.
            ax.plot([-.009, .009], [y-.012, y+.012], transform=ax.transAxes,
                    color='black', clip_on=False, linewidth=1)
    first = datasets[0][1]
    events = len({m['event_id'] for m in first['members']})
    fig.suptitle('SS precursors · fixed 5° averages · %s waveforms · %s events\n'
                 '10–100 s, 2 corners · PREM polarity · individual SS normalization · red: positive'
                 % (len(first['members']), events), fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(str(output)+'.'+suffix, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prem-section', type=Path, required=True)
    parser.add_argument('--glad-section', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precursor-gain', type=float, default=35)
    parser.add_argument('--ss-gain', type=float, default=2.5)
    args = parser.parse_args()
    if not np.isfinite([args.precursor_gain, args.ss_gain]).all() or min(args.precursor_gain, args.ss_gain) <= 0:
        parser.error('Gains must be finite and positive')
    sections = [(name, load_section(path, 5)) for name, path in
                [('PREM', args.prem_section), ('GLAD-M35 · PREM polarity', args.glad_section)]]
    a, b = [s for _, s in sections]
    ma = {(m['event_id'], m['trace_id']): m for m in a['members']}
    mb = {(m['event_id'], m['trace_id']): m for m in b['members']}
    if ma.keys() != mb.keys() or any(ma[k]['applied_polarity_multiplier'] != mb[k]['applied_polarity_multiplier'] for k in ma):
        raise ValueError('Requires matching waveform identities and PREM signs')
    if not np.array_equal(a['counts'], b['counts']) or not np.array_equal(a['tau'], b['tau']):
        raise ValueError('Model selections or time grids differ')
    args.output.mkdir(parents=True, exist_ok=False)
    for (name, section), filename in zip(sections, ['PREM', 'GLAD_M35']):
        draw([(name, section)], args.output/filename, args.precursor_gain, args.ss_gain)
        np.savez_compressed(args.output/(filename+'_distance_stack.npz'),
                            tau=section['tau'], stack=section['stack'], support=section['support'],
                            trace_counts=section['counts'], distance_edges=section['edges'],
                            distance_centers=section['centers'],
                            **{p.replace('^', '')+'_predicted_time_before_SS_s':v for p,v in section['predictions'].items()})
    draw(sections, args.output/'PREM_GLAD_M35_wiggles', args.precursor_gain, args.ss_gain)
    provenance = dict(sources=[s['source'] for _,s in sections],
        source_bin_plan_fingerprints=[s['provenance']['bin_plan_fingerprint'] for _,s in sections],
        waveform_count=len(ma), event_count=len({k[0] for k in ma}), distance_step_degree=5,
        distance_edges_degree=a['edges'].tolist(), trace_counts=a['counts'].tolist(),
        rebinning='Sum(source means * per-sample support) / sum(support); equal trace weights.',
        period_band_s=[10,100], filter_order=2, polarity='Both use matching PREM preprocessing multipliers.',
        precursor_gain_degree_per_unit_amplitude=args.precursor_gain,
        ss_gain_degree_per_unit_amplitude=args.ss_gain,
        display_time_before_aligned_SS_s=[-40,300], gain_change_time_before_SS_s=40,
        prediction_note='Bin medians of contributing PREM theoretical precursor-minus-SS times, sign reversed for time-before-SS axis.',
        processing_note='Existing timing alignment and individual SS normalization retained; no additional waveform filtering or column normalization.')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))

if __name__ == '__main__':
    main()
