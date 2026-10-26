"""Paper-style displays of conventional linear stacks from existing event caches.

Model-curve moveout is removed before scanning residual slowness by default.
No peak sharpening, phase-weighted filtering, or waveform reprocessing is used.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from scipy.signal import hilbert
from .arrivals import unique_reflection
from .batch import read_plan, event_cache, task_destination
from .pipeline import dump_json, grid, load_preprocessed
from .stacking import sample_trace, weighted_samples


DEFAULT_SETTINGS = {
    'moveout_mode': 'predicted_curve',
    'phases': ['SS', 'S^400S', 'S^670S'],
    'absolute_slowness_s_degree': [0, 20, 0.1],
    'focus_time_s': [-40, 40, 0.5],
    'phase_input_half_window_s': 60,
    'precursor_zoom_s': [-300, -70],
    'reference_distance_degree': None,
    'minimum_support_fraction': 0.8,
    'minimum_support_traces': 3,
    'contour_floor': 0.2,
    'record_max_traces': 140,
    'record_distance_range_degree': None,
    'record_gain_degree': 0.7,
    'record_zoom_gain_degree': 4.0,
    'formats': ['png', 'pdf'],
    'dpi': 180,
}


def settings_with_defaults(settings=None):
    settings = settings or {}
    if set(settings)-set(DEFAULT_SETTINGS):
        raise ValueError('Unknown paper-plot settings: '+str(sorted(set(settings)-set(DEFAULT_SETTINGS))))
    out = dict(DEFAULT_SETTINGS, **settings)
    if out['moveout_mode'] not in ('predicted_curve', 'linear_absolute'):
        raise ValueError('moveout_mode must be predicted_curve or linear_absolute')
    if not 0 < out['minimum_support_fraction'] <= 1 or out['minimum_support_traces'] < 1:
        raise ValueError('Invalid support threshold')
    if not 0 <= out['contour_floor'] < 1:
        raise ValueError('contour_floor must be in [0, 1)')
    if (out['record_max_traces'] < 1 or out['record_gain_degree'] <= 0
            or out['record_zoom_gain_degree'] <= 0 or out['phase_input_half_window_s'] <= 0):
        raise ValueError('Invalid record-section settings')
    if not out['phases'] or len(set(out['phases'])) != len(out['phases']):
        raise ValueError('Supply distinct phase names')
    if not out['formats'] or set(out['formats'])-{'png', 'pdf', 'svg'}:
        raise ValueError('Formats must be png, pdf, or svg')
    grid(out['absolute_slowness_s_degree']); grid(out['focus_time_s'])
    return out


def absolute_slant_stack(times, traces, distances, query_origins, time_s, slownesses,
                         reference_distance, weights=None, support_centers=None, support_half_window=None):
    """Sample u_i[a_i + t + q*(Delta_i-Delta_0)], with absolute q in s/degree.

    For overview a_i=T_SS(Delta_0,h_i)-T_SS(Delta_i,h_i).
    For phase focus a_i=T_phase(Delta_0,h_i)-T_SS(Delta_i,h_i).
    Inputs u_i are already aligned on measured SS. This preserves that alignment
    while restoring model-predicted SS distance moveout for the absolute scan.
    """
    distances, origins = np.asarray(distances), np.asarray(query_origins)
    if not len(traces) or not (len(times) == len(traces) == len(distances) == len(origins)):
        raise ValueError('Mismatched or empty absolute-stack inputs')
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(origins)):
        raise ValueError('Nonfinite distance or travel-time origin')
    if support_centers is not None:
        if (len(support_centers) != len(traces) or not np.all(np.isfinite(support_centers))
                or support_half_window is None or support_half_window <= 0):
            raise ValueError('Invalid phase-input support windows')
    rows = []
    for q in slownesses:
        samples = []
        for i, (t, u, a, d) in enumerate(zip(times, traces, origins, distances)):
            query = a+np.asarray(time_s)+q*(d-reference_distance)
            sampled = sample_trace(t, u, query)
            if support_centers is not None:
                sampled[np.abs(query-support_centers[i]) > support_half_window] = np.nan
            samples.append(sampled)
        rows.append(weighted_samples(samples, weights))
    if not rows:
        raise ValueError('Empty slowness scan')
    return {**{key: np.asarray([r[key] for r in rows]) for key in rows[0]},
            'time_s': np.asarray(time_s), 'slownesses': np.asarray(slownesses),
            'reference_distance': float(reference_distance)}


def curve_slant_stack(times, traces, distances, predicted_centers, time_s, slownesses,
                      reference_distance, reference_slowness, weights=None,
                      support_centers=None, support_half_window=None):
    """Remove each model arrival's curvature, then scan residual moveout.

    The displayed coordinate is model-reference slowness plus residual slowness;
    it is not an independent measurement of absolute ray parameter.
    """
    result = absolute_slant_stack(times, traces, distances, predicted_centers, time_s,
                                  np.asarray(slownesses)-reference_slowness, reference_distance,
                                  weights, support_centers, support_half_window)
    result['residual_slownesses'] = result['slownesses'].copy()
    result['slownesses'] = np.asarray(slownesses)
    result['reference_slowness'] = float(reference_slowness)
    result['moveout_mode'] = 'predicted_curve'
    return result


def envelope_display(result, trace_count, settings, window=None):
    """Envelope of the signed linear stack, globally normalized within this panel.

    Missing time segments are never bridged in the Hilbert transform. Masking is
    applied after computing each supported segment, with no smoothing in q.
    """
    amplitude = np.asarray(result['stack'])
    envelope = np.full_like(amplitude, np.nan)
    for i, row in enumerate(amplitude):
        valid = np.isfinite(row)
        starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
        ends = np.flatnonzero(valid & ~np.r_[valid[1:], False])+1
        for start, end in zip(starts, ends):
            if end-start < 3:
                continue
            segment = row[start:end]
            pad = min(40, len(segment)-1)
            envelope[i, start:end] = np.abs(hilbert(np.pad(segment, pad, mode='reflect')))[pad:-pad]
    minimum = max(settings['minimum_support_traces'],
                  int(np.ceil(trace_count*settings['minimum_support_fraction'])))
    envelope[result['count'] < minimum] = np.nan
    selected = np.ones(len(result['time_s']), dtype=bool)
    if window is not None:
        selected = (result['time_s'] >= window[0]) & (result['time_s'] <= window[1])
    finite = envelope[:, selected]
    scale = float(np.nanmax(finite)) if np.any(np.isfinite(finite)) else 0.0
    normalized = envelope/scale if scale > 0 else np.full_like(envelope, np.nan)
    return normalized, {'normalization_amplitude': scale, 'minimum_contributing_traces': minimum,
                        'quantity': 'envelope_of_signed_linear_stack', 'normalization': 'one_scale_per_panel'}


def load_members(plan_path, index):
    """Load one bin's selected waveform rows and metadata, not the full dataset."""
    plan_path = Path(plan_path).resolve(); plan = read_plan(plan_path, 'bins-v1')
    if not 0 <= index < len(plan['tasks']):
        raise ValueError('Bin task index out of range')
    task = plan['tasks'][index]
    con = sqlite3.connect((plan_path.parent/'index.sqlite').as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        members = [dict(r) for r in con.execute(
            'SELECT * FROM members WHERE phase=? AND lat=? AND lon=? ORDER BY task,row_index',
            (task['phase'], task['latitude_index'], task['longitude_index']))]
    finally:
        con.close()
    if len(members) != task['trace_count']:
        raise ValueError('Bin plan/member count mismatch')
    traces, rows, depths, tau, current, data, metadata, prov = [], [], [], None, None, None, None, None
    for member in members:
        if member['task'] != current:
            cache = event_cache(plan['cache_root'], member['task'])
            marker = json.loads((cache/'complete.json').read_text())
            if marker != {'plan': plan['event_plan_fingerprint'], 'task_index': member['task']}:
                raise ValueError('Event cache identity changed')
            ctau, data = load_preprocessed(cache)
            if tau is not None and not np.array_equal(ctau, tau):
                raise ValueError('Mixed cache time grids')
            tau, current = ctau, member['task']
            metadata = json.loads((cache/'traces.json').read_text())['accepted']
            prov = json.loads((cache/'provenance.json').read_text())
        row = metadata[member['row_index']]
        if row['event_id'] != member['event_id'] or row['trace_id'] != member['trace_id']:
            raise ValueError('Waveform/metadata identity mismatch')
        traces.append(np.array(data[member['row_index']], copy=True)); rows.append(row)
        depths.append(prov['origin_depth_km'])
    counts = Counter(row['event_id'] for row in rows)
    weighting = plan['configuration'].get('stack_weighting', 'trace')
    if weighting not in ('trace', 'event_equal'):
        raise ValueError('Unknown weighting')
    weights = np.array([1 if weighting == 'trace' else 1/counts[row['event_id']] for row in rows])
    return plan, task, members, tau, traces, rows, np.asarray(depths), weights


def reference_predictions(model, depths, distance, phases):
    """Compute reference arrivals at each event depth; reject ambiguous branches."""
    cache, predictions = {}, {phase: [] for phase in phases}
    for depth in depths:
        for phase in phases:
            key = (float(depth), phase)
            if key not in cache:
                if phase == 'SS': boundary = 0.0
                elif phase.startswith('S^') and phase.endswith('S'):
                    boundary = float(phase[2:-1])
                    if not np.any(np.isclose(model.model.s_mod.v_mod.get_discontinuity_depths(), boundary)):
                        raise ValueError(f'{phase}: interface absent from reference model')
                else:
                    raise ValueError('Phase must be SS or an underside reflection such as S^400S')
                cache[key] = unique_reflection(model, float(depth), distance, phase, boundary, 0, 0, 0)
            predictions[phase].append(cache[key])
    return predictions


def phase_label(phase):
    return phase.replace('^', '')


def save_figure(fig, stem, settings):
    import matplotlib.pyplot as plt
    for suffix in settings['formats']:
        fig.savefig(str(stem)+'.'+suffix, dpi=settings['dpi'], bbox_inches='tight')
    plt.close(fig)


def draw_contours(ax, result, display, settings, window, xlabel, title, markers):
    import matplotlib.colors as colors
    from matplotlib.ticker import MaxNLocator
    floor = settings['contour_floor']
    levels = np.linspace(floor, 1, 5)
    ax.set_facecolor('white')
    time = result['time_s']
    keep = (time >= window[0]) & (time <= window[1])
    values = np.ma.masked_invalid(np.clip(display[:, keep], 0, 1))
    if np.any(~np.ma.getmaskarray(values)) and np.ma.max(values) > floor and keep.sum() >= 2:
        mesh = ax.contourf(time[keep], result['slownesses'], values, levels=levels, cmap='Greys',
                           norm=colors.Normalize(floor, 1), antialiased=False)
    else:
        mesh = None
        ax.text(0.5, 0.5, 'No supported amplitude above display threshold', ha='center', transform=ax.transAxes, fontsize=8)
    for x, p, label in markers:
        if window[0] <= x <= window[1] and result['slownesses'][0] <= p <= result['slownesses'][-1]:
            ax.plot(x, p, marker='*', ms=10, mfc='white', mec='black', mew=0.9)
            if label:
                ax.annotate(label, (x, p), xytext=(6, 9), textcoords='offset points', fontsize=9)
    ax.set(xlim=window, ylim=(result['slownesses'][0], result['slownesses'][-1]),
           xlabel=xlabel, ylabel=('Model reference + residual slowness (s/degree)'
                                  if settings['moveout_mode'] == 'predicted_curve'
                                  else 'Absolute slowness (s/degree)'), title=title)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.minorticks_on()
    return mesh


def plot_record_sections(tau, traces, rows, phases, task, output, settings):
    import matplotlib.pyplot as plt
    distances = np.array([r['distance_degree'] for r in rows])
    selection = np.arange(len(rows))
    if settings['record_distance_range_degree'] is not None:
        lo, hi = settings['record_distance_range_degree']
        selection = selection[(distances >= lo) & (distances <= hi)]
    if not len(selection):
        raise ValueError('No traces within requested record-section distance range')
    selection = selection[np.argsort(distances[selection], kind='stable')]
    if len(selection) > settings['record_max_traces']:
        selection = selection[np.linspace(0, len(selection)-1, settings['record_max_traces'], dtype=int)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True, sharey=True)
    zoom = settings['precursor_zoom_s']
    gain = settings['record_gain_degree']
    zoom_gain = settings['record_zoom_gain_degree']
    for i in selection:
        for ax, panel_gain in zip(axes, [gain, zoom_gain]):
            ax.plot(tau, distances[i]+panel_gain*traces[i], color='black', lw=0.45, rasterized=True)
    colors = ['#35628d', '#b04c2e', '#3b7656']
    for k, phase in enumerate(phases):
        if not all(phase in rows[i]['arrivals'] for i in selection):
            continue
        x = np.array([rows[i]['arrivals'][phase]['time_s']-rows[i]['predicted_ss_s'] for i in selection])
        for ax in axes:
            ax.scatter(x, distances[selection], marker='|', color=colors[k % len(colors)], s=14, label=phase_label(phase))
    axes[0].set(xlim=(tau[0], tau[-1]), title='SS-aligned waveforms', ylabel='Epicentral distance (degrees)')
    axes[1].set(xlim=(max(tau[0], zoom[0]), min(tau[-1], zoom[1])), title=f'Precursor window — gain {zoom_gain:g}° per SS amplitude')
    axes[0].set_ylim(distances[selection].max()+2, distances[selection].min()-2)
    for ax in axes:
        ax.set_xlabel('Time relative to measured SS (s)')
        ax.tick_params(which='both', direction='in', top=True, right=True)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'{task["phase"].replace("^", "")} reflection bin ({task["latitude_center"]:g}°, '
                 f'{task["longitude_center"]:g}°); {len(selection)} of {len(rows)} traces shown; '
                 f'gain {gain:g}° per SS amplitude', fontsize=10)
    save_figure(fig, output/'record_section', settings)
    return [rows[i]['trace_id'] for i in selection]


def paper_plots(plan_path, output, index, supplied_settings=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from obspy.taup import TauPyModel
    settings = settings_with_defaults(supplied_settings)
    plan = read_plan(plan_path, 'bins-v1')
    if not 0 <= index < len(plan['tasks']):
        raise ValueError('Bin task index out of range')
    task = plan['tasks'][index]
    name = f'{phase_label(task["phase"])}_lat{task["latitude_index"]:02d}_lon{task["longitude_index"]:02d}'
    identity = {'plan': plan['fingerprint'], 'task_index': index, 'paper_plot_version': 2, 'settings': settings}
    with task_destination(output, name, identity) as work:
        if work is None:
            return {'task_index': index, 'status': 'cached', 'bin': name}
        plan, task, members, tau, traces, rows, depths, weights = load_members(plan_path, index)
        phases = settings['phases']
        distances = np.array([r['distance_degree'] for r in rows])
        ref = settings['reference_distance_degree']
        if ref is None: ref = plan['configuration'].get('reference_distance_degree')
        if ref is None: ref = float(np.mean(distances))
        if not distances.min() <= ref <= distances.max():
            raise ValueError('Reference distance must be inside contributing distance aperture')
        reference = reference_predictions(TauPyModel(plan['configuration']['reference_model']), depths, ref,
                                          list(dict.fromkeys(['SS']+phases)))
        ss_actual = np.array([r['predicted_ss_s'] for r in rows])
        ss_ref = np.array([a['time_s'] for a in reference['SS']])
        q = grid(settings['absolute_slowness_s_degree']); offsets = grid(settings['focus_time_s'])
        times = [tau]*len(traces)
        curved = settings['moveout_mode'] == 'predicted_curve'
        ss_q0 = float(np.median([a['ray_parameter_s_degree'] for a in reference['SS']]))
        if curved:
            overview = curve_slant_stack(times, traces, distances, np.zeros(len(rows)), tau, q,
                                         ref, ss_q0, weights)
        else:
            overview = absolute_slant_stack(times, traces, distances, ss_ref-ss_actual, tau, q, ref, weights)
        np.savez_compressed(work/'overview_absolute.npz', **overview)
        markers = [(float(np.median(np.array([a['time_s'] for a in reference[phase]])-ss_ref)),
                    float(np.median([a['ray_parameter_s_degree'] for a in reference[phase]])), phase_label(phase)) for phase in phases]
        full_display, full_meta = envelope_display(overview, len(rows), settings)
        zoom = settings['precursor_zoom_s']
        zoom = (max(float(tau[0]), zoom[0]), min(float(tau[-1]), zoom[1]))
        if zoom[0] >= zoom[1]: raise ValueError('Precursor window is outside cached time support')
        zoom_display, zoom_meta = envelope_display(overview, len(rows), settings, zoom)
        caption = (f'{plan["configuration"]["reference_model"].upper()} · {phase_label(task["phase"])} reflection bin '
                   f'({task["latitude_center"]:g}°, {task["longitude_center"]:g}°) · '
                   f'N={len(rows)}, events={len(set(r["event_id"] for r in rows))}, aperture={np.ptp(distances):.1f}°')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, display, window, title in [(axes[0], full_display, (tau[0], tau[-1]), 'SS and both precursors'),
                                           (axes[1], zoom_display, zoom, 'Precursor window (rescaled)')]:
            mesh = draw_contours(ax, overview, display, settings, window,
                                 ('Time relative to aligned SS (s)' if curved else
                                  'Time relative to predicted SS at reference distance (s)'), title, markers)
            if mesh is not None: fig.colorbar(mesh, ax=ax, label='Normalized linear-stack envelope', shrink=0.8)
        fig.suptitle(caption+'\nStars: model predictions; each panel normalized separately', fontsize=10)
        save_figure(fig, work/'vespagram_overview', settings)
        fig, axes = plt.subplots(1, len(phases), figsize=(4.7*len(phases), 4.7), constrained_layout=True, squeeze=False)
        panel_metadata = {}
        for ax, phase in zip(axes[0], phases):
            phase_ref = np.array([a['time_s'] for a in reference[phase]])
            if all(phase in row['arrivals'] for row in rows):
                phase_actual = np.array([row['arrivals'][phase]['time_s'] for row in rows])
            else:
                # Extra diagnostic phases can be predicted from the same rays
                # without rereading or changing the cached waveforms.
                model = TauPyModel(plan['configuration']['reference_model'])
                phase_actual = np.array([reference_predictions(model, [depth], float(distance), [phase])[phase][0]['time_s']
                                         for depth, distance in zip(depths, distances)])
            q0 = float(np.median([a['ray_parameter_s_degree'] for a in reference[phase]]))
            if curved:
                result = curve_slant_stack(times, traces, distances, phase_actual-ss_actual,
                                          offsets, q, ref, q0, weights,
                                          phase_actual-ss_actual, settings['phase_input_half_window_s'])
            else:
                result = absolute_slant_stack(times, traces, distances, phase_ref-ss_actual, offsets, q, ref, weights,
                                              phase_actual-ss_actual, settings['phase_input_half_window_s'])
            np.savez_compressed(work/(phase_label(phase)+'_absolute.npz'), **result)
            display, meta = envelope_display(result, len(rows), settings)
            marker = [(0, float(np.median([a['ray_parameter_s_degree'] for a in reference[phase]])), '')]
            xlabel = f'Time relative to predicted {phase_label(phase)} (s)'
            mesh = draw_contours(ax, result, display, settings, (offsets[0], offsets[-1]), xlabel, phase_label(phase), marker)
            if mesh is not None: fig.colorbar(mesh, ax=ax, label='Normalized amplitude', shrink=0.8)
            single, single_ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
            single_mesh = draw_contours(single_ax, result, display, settings, (offsets[0], offsets[-1]), xlabel,
                                        phase_label(phase)+' · '+plan['configuration']['reference_model'].upper(), marker)
            if single_mesh is not None: single.colorbar(single_mesh, ax=single_ax, label='Normalized linear-stack envelope', shrink=0.8)
            single.suptitle(caption+'\nStar: model prediction; source-depth-specific reference times', fontsize=8)
            save_figure(single, work/('vespagram_'+phase_label(phase)), settings)
            panel_metadata[phase] = dict(meta, predicted_absolute_slownesses_s_degree=[a['ray_parameter_s_degree'] for a in reference[phase]],
                                        reference_times_s=phase_ref.tolist(),
                                        input_window_centers_relative_to_measured_ss_s=(phase_actual-ss_actual).tolist(),
                                        input_half_window_s=settings['phase_input_half_window_s'])
        fig.suptitle(caption+'\nSame traces in every panel; each phase normalized separately', fontsize=10)
        save_figure(fig, work/'vespagram_phases', settings)
        shown = plot_record_sections(tau, traces, rows, phases, task, work, settings)
        dump_json(work/'members.json', members)
        dump_json(work/'plot_provenance.json', dict(identity, selection_bin=task, reference_distance_degree=ref,
            bin_plan_path=str(Path(plan_path).resolve()), cache_root=plan['cache_root'],
            source_depths_km=depths.tolist(), cache_time_support_s=[float(tau[0]), float(tau[-1])],
            cached_period_band_s=plan['configuration']['period_band_s'], weighting=plan['configuration'].get('stack_weighting', 'trace'),
            overview_display=full_meta, precursor_zoom_display=zoom_meta, phase_displays=panel_metadata,
            record_section_trace_ids=shown, missing_event_tasks=plan.get('missing_event_tasks', []),
            amplitude_note='Signed linear stacks are saved; display is their envelope. No smoothing across slowness. '
                           'Normalization differs by panel and does not preserve inter-phase amplitude ratios.',
            selection_note='All panels use the same traces selected by this bin phase. Other phases may reflect outside this cell.',
            moveout_note=('Model curve removed per trace; plotted slowness is model reference plus scanned residual. '
                          'Focused panels use each phase curve; overview uses the SS curve. '
                          'Measured SS alignment shifts remain removed.' if curved else
                          'Absolute linear scan restores predicted SS distance moveout; broad apertures can smear arrivals.')))
    return {'task_index': index, 'status': 'complete', 'bin': name, 'trace_count': len(rows)}
