"""Compare phase-specific circular bounce-point stacks about user station points."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .distance_section import load_global, distance_stack
from .pipeline import dump_json
from .stacking import sample_trace, weighted_samples


EARTH_RADIUS_KM = 6371.0
PHASES = ['S^400S', 'S^670S']


def read_centers(script_path, station_path):
    """Read the existing points and verify the script's Fibonacci spacing.

    Parsing constants avoids importing/running a script that writes a station file.
    Use the actual station-file coordinates, rounded as in that script.
    """
    constants = {}
    for node in ast.parse(Path(script_path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('target_km', 'EARTH_RADIUS_KM'):
                    constants[target.id] = ast.literal_eval(node.value)
    spacing = float(constants['target_km'])
    radius = float(constants['EARTH_RADIUS_KM'])
    if radius != EARTH_RADIUS_KM or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Unexpected station generator radius or spacing')
    n = max(2, int(round(4*math.pi*radius**2 / (math.sqrt(3)/2*spacing**2))))
    points = []
    for line in Path(station_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.replace('\\t', '\t').split()
        points.append({'id': fields[0], 'latitude': float(fields[2]), 'longitude': float(fields[3])})
    if len(points) != n or len({p['id'] for p in points}) != n:
        raise ValueError('Station file does not match generated point count/IDs')
    for i, p in enumerate(points):
        lat = math.degrees(math.asin(1-2*(i+.5)/n))
        lon = math.degrees((i*math.pi*(3-math.sqrt(5))) % (2*math.pi))-180
        if abs(p['latitude']-lat) > .00051 or abs((p['longitude']-lon+180) % 360-180) > .00051:
            raise ValueError('Station file differs from the Fibonacci generator')
    return points, spacing


def xyz(latlon):
    a = np.asarray(latlon, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or np.any(abs(a[:, 0]) > 90):
        raise ValueError('Expected finite latitude/longitude pairs')
    lat, lon = np.deg2rad(a).T
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def cap_memberships(centers, bounce_points, radius_km):
    """Surface great-circle caps; chord searches handle the dateline and poles."""
    if not np.isfinite(radius_km) or not 0 < radius_km <= math.pi*EARTH_RADIUS_KM:
        raise ValueError('Invalid surface radius')
    chord = 2*np.sin(radius_km/(2*EARTH_RADIUS_KM))
    matches = cKDTree(xyz(bounce_points)).query_ball_point(xyz(centers), chord+1e-14)
    return [np.asarray(sorted(indices), dtype=int) for indices in matches]


def cap_weights(rows, indices, weighting):
    if weighting == 'trace':
        return np.ones(len(indices))
    if weighting != 'event_equal':
        raise ValueError('Unsupported stack weighting')
    from collections import Counter
    counts = Counter(rows[i]['event_id'] for i in indices)
    return np.array([1/counts[rows[i]['event_id']] for i in indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--centers-script', type=Path, required=True)
    parser.add_argument('--centers-file', type=Path, help='Default: stations_fibonacci_100km.txt next to script')
    parser.add_argument('--radii-km', type=float, nargs='+', default=[600, 900, 1200, 1500])
    parser.add_argument('--distance-range', type=float, nargs=2, default=[100, 160])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    radii = np.asarray(sorted(set(args.radii_km)))
    if (not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0)
            or np.any(radii > math.pi*EARTH_RADIUS_KM)):
        parser.error('Supply finite, positive surface radii')
    if not (np.isfinite(args.distance_range).all() and 0 <= args.distance_range[0] < args.distance_range[1] <= 180):
        parser.error('Distance range must be inside 0..180')
    station_file = args.centers_file or args.centers_script.parent/'stations_fibonacci_100km.txt'
    centers, spacing = read_centers(args.centers_script, station_file)
    center_xy = np.array([[c['latitude'], c['longitude']] for c in centers])
    plan, _, members, tau, traces, rows, depths, _ = load_global(args.plan, args.distance_range)
    weighting = plan['configuration'].get('stack_weighting', 'trace')
    offsets = np.arange(-60., 61.)
    shape = (len(PHASES), len(radii), len(centers), len(offsets))
    stacks = np.full(shape, np.nan); semblance = np.full(shape, np.nan)
    support = np.zeros(shape, dtype=np.int32); weight_sums = np.zeros(shape)
    counts = np.zeros(shape[:-1], dtype=int); events = np.zeros_like(counts)
    groups, coverage = {}, []
    for pi, phase in enumerate(PHASES):
        bounce = [[r['arrivals'][phase]['reflection_latitude'], r['arrivals'][phase]['reflection_longitude']] for r in rows]
        predicted = np.array([r['arrivals'][phase]['time_s']-r['predicted_ss_s'] for r in rows])
        samples = np.array([sample_trace(tau, u, t+offsets) for u, t in zip(traces, predicted)])
        for ri, radius in enumerate(radii):
            groups[pi, ri] = cap_memberships(center_xy, bounce, radius)
            for ci, indices in enumerate(groups[pi, ri]):
                n = len(indices); counts[pi, ri, ci] = n
                events[pi, ri, ci] = len({rows[i]['event_id'] for i in indices})
                if n:
                    result = weighted_samples(samples[indices], cap_weights(rows, indices, weighting))
                    stacks[pi, ri, ci] = result['stack']; semblance[pi, ri, ci] = result['semblance']
                    support[pi, ri, ci] = result['count']; weight_sums[pi, ri, ci] = result['weight_sum']
                coverage.append(dict(center_id=centers[ci]['id'], latitude=center_xy[ci, 0], longitude=center_xy[ci, 1],
                    phase=phase, radius_km=float(radius), trace_count=n, event_count=int(events[pi, ri, ci]),
                    eligible=n >= 10 and events[pi, ri, ci] >= 3))
    # Choose a fixed centre using coverage, not the amplitude of a precursor.
    reference_ri = int(np.argmin(abs(radii-900)))
    scores = np.min(counts[:, reference_ri], axis=0)
    examples = np.argsort(-scores, kind='stable')[:3]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'circular_stacks.npz', phases=np.array(PHASES), radii_km=radii,
        center_ids=np.array([c['id'] for c in centers]), center_latlon=center_xy, offsets_s=offsets,
        stack=stacks, semblance=semblance, support=support, weight_sum=weight_sums,
        trace_counts=counts, event_counts=events)
    with (output/'coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0])); writer.writeheader(); writer.writerows(coverage)
    dump_json(output/'centers.json', centers)
    # Global row identities plus member indices permit exact reconstruction of any cap.
    dump_json(output/'waveform_members.json', members)
    dump_json(output/'cap_members.json', {PHASES[pi]: {str(float(radius)): {
        c['id']: groups[pi, ri][ci].tolist() for ci, c in enumerate(centers)}
        for ri, radius in enumerate(radii)} for pi in range(len(PHASES))})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(.1, .9, len(radii)))
    fig, axes = plt.subplots(2, len(examples), figsize=(15, 8), constrained_layout=True, squeeze=False)
    for pi, phase in enumerate(PHASES):
        limit = max(.01, float(np.nanmax(abs(stacks[pi, :, examples, :])))*1.1)
        for col, ci in enumerate(examples):
            ax = axes[pi, col]
            for ri, radius in enumerate(radii):
                ax.plot(offsets, stacks[pi, ri, ci], color=colors[ri], lw=1.2,
                    label=f'{radius:g} km: N={counts[pi, ri, ci]}, E={events[pi, ri, ci]}')
            ax.axvline(0, color='0.4', ls='--', lw=.8); ax.axhline(0, color='0.6', lw=.6)
            ax.set(xlim=(-60, 60), ylim=(-limit, limit), xlabel='Time relative to predicted phase (s)',
                   ylabel='Signed amplitude / individual SS peak',
                   title=f'{phase.replace("^", "")} · {centers[ci]["id"]} ({center_xy[ci, 0]:.1f}°, {center_xy[ci, 1]:.1f}°)')
            ax.legend(fontsize=8)
    fig.suptitle('Fixed centres across all radii · exact phase-specific PREM differential moveout\n'
                 'Top three centres by minimum S400S/S670S trace count at the radius nearest 900 km', fontsize=11)
    for ext in ('png', 'pdf'): fig.savefig(output/('radius_comparison.'+ext), dpi=180)
    plt.close(fig)
    # Native SS-aligned distance sections at the same best centre, with no precursor shifts.
    best = int(examples[0]); d = np.array([r['distance_degree'] for r in rows])
    edges = np.linspace(args.distance_range[0], args.distance_range[1], 13)
    fig, axes = plt.subplots(2, len(radii), figsize=(4*len(radii), 9), constrained_layout=True, squeeze=False)
    te = np.r_[tau-.5*(tau[1]-tau[0]), tau[-1]+.5*(tau[1]-tau[0])]
    section_arrays = {}; dc = (edges[:-1]+edges[1:])/2
    for pi, phase in enumerate(PHASES):
        for ri, radius in enumerate(radii):
            idx = groups[pi, ri][best]; ax = axes[pi, ri]
            if len(idx):
                section, ncol, scount, colidx = distance_stack(tau, [traces[i] for i in idx], d[idx],
                    cap_weights(rows, idx, weighting), edges)
                mesh = ax.pcolormesh(edges, te, section, shading='flat', cmap='RdBu', vmin=-.08, vmax=.08, rasterized=True)
                prediction = np.array([rows[i]['arrivals'][phase]['time_s']-rows[i]['predicted_ss_s'] for i in idx])
                median = [np.median(prediction[colidx == j]) if np.any(colidx == j) else np.nan for j in range(12)]
                ax.plot(dc, median, '--', color='black', lw=.8)
                key = f'{phase.replace("^", "")}_{radius:g}km'
                section_arrays[key+'_stack'] = section; section_arrays[key+'_trace_counts'] = ncol
                section_arrays[key+'_support'] = scount
            ax.set(ylim=(-300, -70), xlim=args.distance_range, xlabel='Epicentral distance (degrees)',
                   ylabel='Time relative to aligned SS (s)', title=f'{phase.replace("^", "")} · {radius:g} km · N={len(idx)}')
            ax.set_facecolor('0.88')
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label='Signed amplitude / individual SS peak', shrink=.6, extend='both')
    fig.suptitle(f'{centers[best]["id"]}: ({center_xy[best,0]:.3f}°, {center_xy[best,1]:.3f}°) · '
                 'phase-specific cap membership · 12 disjoint distance intervals\n'
                 'Native SS alignment; dashed curve is the model prediction; common color limits ±0.08', fontsize=11)
    for ext in ('png', 'pdf'): fig.savefig(output/('best_center_distance_sections.'+ext), dpi=180)
    plt.close(fig)
    np.savez_compressed(output/'best_center_distance_sections.npz', tau=tau, distance_edges=edges, **section_arrays)
    # All-radius geographic coverage, plotted with one common count scale.
    import cartopy.crs as ccrs
    from cartopy.io import shapereader
    coast = shapereader.natural_earth(resolution='110m', category='physical', name='coastline')
    coastlines = list(shapereader.Reader(coast).geometries())
    fig, axes = plt.subplots(2, len(radii), figsize=(4*len(radii), 6), constrained_layout=True,
                             subplot_kw={'projection': ccrs.Robinson()}, squeeze=False)
    for pi, phase in enumerate(PHASES):
        for ri, radius in enumerate(radii):
            ax = axes[pi, ri]; ax.set_global()
            ax.add_geometries(coastlines, ccrs.PlateCarree(), facecolor='none', edgecolor='0.4', linewidth=.3)
            mesh = ax.scatter(center_xy[:,1], center_xy[:,0], c=counts[pi,ri], s=8, cmap='viridis',
                              vmin=0, vmax=counts.max(), transform=ccrs.PlateCarree(), rasterized=True)
            ax.plot(center_xy[best,1], center_xy[best,0], marker='*', color='red', ms=7, transform=ccrs.PlateCarree())
            ax.set_title(f'{phase.replace("^", "")} · {radius:g} km', fontsize=10)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label='Waveforms per circular cap', orientation='horizontal', shrink=.6, pad=.02)
    fig.suptitle(f'{len(centers)} user centres · surface great-circle radius · red star: {centers[best]["id"]}', fontsize=12)
    for ext in ('png', 'pdf'): fig.savefig(output/('coverage_maps.'+ext), dpi=180)
    plt.close(fig)
    examples_out = [dict(centers[int(ci)], trace_counts=counts[:,:,ci].tolist(), event_counts=events[:,:,ci].tolist()) for ci in examples]
    dump_json(output/'provenance.json', dict(bin_plan=str(args.plan.resolve()), bin_plan_fingerprint=plan['fingerprint'],
        centers_script=str(args.centers_script.resolve()), centers_file=str(station_file.resolve()),
        centers_script_sha256=hashlib.sha256(args.centers_script.read_bytes()).hexdigest(),
        centers_file_sha256=hashlib.sha256(station_file.read_bytes()).hexdigest(), center_spacing_km=spacing,
        earth_radius_km=EARTH_RADIUS_KM, radii_km=radii.tolist(), phases=PHASES, center_count=len(centers),
        waveform_count=len(rows), distance_range_degree=args.distance_range, weighting=weighting,
        period_band_s=plan['configuration']['period_band_s'], examples=examples_out,
        reference_radius_km=float(radii[reference_ri]),
        note='Each phase selects its own bounce points within the surface cap. Caps/radii overlap and are not independent. '
             'No cap-peak amplitude normalization or extra polarity flips. Exact model differential-time correction for '
             'offset stacks; native SS alignment for distance sections. All centres retained, including low coverage. '
             'Eligibility flag in coverage CSV means N>=10 and events>=3, not a detection test. '
             '10-degree square plan is used only to access all accepted cache members, including ineligible square cells.'))
    print(json.dumps({'output':str(output), 'centers':len(centers), 'waveforms':len(rows),
                      'best_center':examples_out[0], 'radii_km':radii.tolist()}))


if __name__ == '__main__':
    main()
