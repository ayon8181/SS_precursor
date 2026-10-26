"""Vespagrams from saved circular-cap memberships, with exact PREM moveout."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .circular_stacks import cap_weights
from .distance_section import load_global
from .paper_plots import curve_slant_stack, envelope_display, draw_contours, settings_with_defaults
from .pipeline import dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', type=Path, required=True, help='Completed circular radius comparison directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--centers', nargs='+', help='Default: three example centres from cap provenance')
    parser.add_argument('--radii-km', type=float, nargs='+', help='Default: all saved radii')
    parser.add_argument('--reference-distance', type=float, default=130)
    args = parser.parse_args()
    capdir = args.caps.resolve()
    provenance = json.loads((capdir/'provenance.json').read_text())
    centers = {c['id']:c for c in json.loads((capdir/'centers.json').read_text())}
    chosen = args.centers or [c['id'] for c in provenance['examples']]
    radii = args.radii_km or provenance['radii_km']
    if (len(set(chosen)) != len(chosen) or any(c not in centers for c in chosen)
            or len(set(radii)) != len(radii) or any(r not in provenance['radii_km'] for r in radii)):
        parser.error('Choose distinct centres and radii present in the saved cap comparison')
    lo, hi = provenance['distance_range_degree']
    if not np.isfinite(args.reference_distance) or not lo <= args.reference_distance <= hi:
        parser.error('Reference distance must be inside the selected epicentral range')
    plan, _, members, tau, traces, rows, depths, _ = load_global(
        provenance['bin_plan'], provenance['distance_range_degree'])
    if plan['fingerprint'] != provenance['bin_plan_fingerprint']:
        raise ValueError('Cap plan identity changed')
    if members != json.loads((capdir/'waveform_members.json').read_text()):
        raise ValueError('Waveform membership/order changed since cap comparison')
    memberships = json.loads((capdir/'cap_members.json').read_text())
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings = settings_with_defaults()
    offsets = np.arange(-40., 40.01, .5)
    residual = np.arange(-2., 2.0001, .025)
    panels, records = {}, []
    for center in chosen:
        fig, axes = plt.subplots(2, len(radii), figsize=(4.5*len(radii), 8.5), constrained_layout=True, squeeze=False)
        semfig, semaxes = plt.subplots(2, len(radii), figsize=(4.5*len(radii), 8.5), constrained_layout=True, squeeze=False)
        for pi, phase in enumerate(provenance['phases']):
            for ri, radius in enumerate(radii):
                indices = np.array(memberships[phase][str(float(radius))][center], dtype=int)
                if not len(indices) or len(indices) != len(set(indices.tolist())):
                    raise ValueError('Empty or duplicate cap membership')
                d = np.array([rows[i]['distance_degree'] for i in indices])
                origins = np.array([rows[i]['arrivals'][phase]['time_s']-rows[i]['predicted_ss_s'] for i in indices])
                result = curve_slant_stack([tau]*len(indices), [traces[i] for i in indices], d,
                    origins, offsets, residual, args.reference_distance, 0.,
                    cap_weights(rows, indices, provenance['weighting']), origins, 60.)
                display, meta = envelope_display(result, len(indices), settings)
                result['moveout_mode'] = 'phase_curve_plus_residual_slowness'
                result['quantity_note'] = 'slownesses and residual_slownesses are residual relative to each trace model phase curve'
                name = f'{center}_{phase.replace("^", "")}_{radius:g}km'
                np.savez_compressed(output/(name+'.npz'), **result)
                n_events = len({rows[i]['event_id'] for i in indices})
                title = f'{phase.replace("^", "")} · {radius:g} km\nN={len(indices)}, events={n_events}'
                ax = axes[pi,ri]
                mesh = draw_contours(ax, result, display, settings, (-40,40),
                    'Time relative to predicted phase (s)', title, [(0,0,'')])
                ax.set_ylabel('Residual slowness relative to PREM (s/degree)')
                if mesh is not None:
                    fig.colorbar(mesh, ax=ax, shrink=.7, label='Normalized stack envelope')
                sem = np.ma.masked_where(result['count'] < meta['minimum_contributing_traces'], result['semblance'])
                sax = semaxes[pi,ri]
                smesh = sax.pcolormesh(offsets, residual, sem, shading='auto', cmap='viridis', vmin=0, vmax=1, rasterized=True)
                sax.plot(0,0,'*',mfc='white',mec='black',ms=9)
                sax.set(xlabel='Time relative to predicted phase (s)', ylabel='Residual slowness relative to PREM (s/degree)',
                        title=title, xlim=(-40,40), ylim=(-2,2))
                semfig.colorbar(smesh, ax=sax, shrink=.7, label='Semblance')
                records.append(dict(center_id=center, phase=phase, radius_km=float(radius), waveform_count=len(indices),
                    event_count=n_events, distance_aperture_degree=float(np.ptp(d)),
                    membership_indices=indices.tolist(), result_file=name+'.npz', **meta))
                panels[center, phase, float(radius)] = (result, display, title)
        point = centers[center]
        heading = (f'{center} ({point["latitude"]:.3f}°, {point["longitude"]:.3f}°) · '
                   f'model curvature removed · reference distance {args.reference_distance:g}°\n')
        fig.suptitle(heading+'Stars: model predictions · each panel normalized separately · at least 80% trace support', fontsize=11)
        semfig.suptitle(heading+'Semblance on a common 0–1 scale; unsupported samples masked', fontsize=11)
        for ext in ('png','pdf'):
            fig.savefig(output/(center+'_vespagrams.'+ext), dpi=180)
            semfig.savefig(output/(center+'_semblance.'+ext), dpi=180)
        plt.close(fig); plt.close(semfig)
    # A compact 900 km comparison across the selected centres.
    radius = float(min(radii, key=lambda r:abs(r-900)))
    fig, axes = plt.subplots(2,len(chosen),figsize=(4.8*len(chosen),8),constrained_layout=True,squeeze=False)
    for pi, phase in enumerate(provenance['phases']):
        for ci, center in enumerate(chosen):
            result, display, title = panels[center,phase,radius]
            ax=axes[pi,ci]
            mesh=draw_contours(ax,result,display,settings,(-40,40),'Time relative to predicted phase (s)',center+' · '+title,[(0,0,'')])
            ax.set_ylabel('Residual slowness relative to PREM (s/degree)')
            if mesh is not None: fig.colorbar(mesh,ax=ax,shrink=.7,label='Normalized stack envelope')
    fig.suptitle(f'{radius:g} km circular caps · phase-specific bounce points · model curvature removed\n'
                 'Each panel normalized separately; stars are model predictions',fontsize=11)
    for ext in ('png','pdf'): fig.savefig(output/('reference_radius_comparison.'+ext),dpi=180)
    plt.close(fig)
    dump_json(output/'provenance.json',dict(source_caps=str(capdir),
        source_cap_provenance_sha256=hashlib.sha256((capdir/'provenance.json').read_bytes()).hexdigest(),
        bin_plan=provenance['bin_plan'], bin_plan_fingerprint=plan['fingerprint'],
        centers=chosen,radii_km=list(map(float,radii)), phases=provenance['phases'],
        reference_distance_degree=args.reference_distance, residual_slowness_s_degree=[-2,2,.025],
        time_offsets_s=[-40,40,.5], input_phase_half_window_s=60, display_settings=settings, panels=records,
        note='Query = predicted phase-minus-SS + offset + residual_slowness*(distance-reference_distance). '
             'Residual zero follows each trace exact PREM curve. Axis is residual, not absolute slowness. '
             'Signed SS-normalized stacks retained. Envelope normalized once per panel, not per slowness row. '
             'At least 80% trace support, no slowness smoothing, no new polarity flips.'))
    print(json.dumps({'output':str(output),'centers':chosen,'radii_km':radii,'panels':len(records)}))


if __name__ == '__main__':
    main()
