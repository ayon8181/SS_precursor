"""Map great-circle midpoints and PREM underside-reflection points from caches."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
from .batch import read_plan, event_cache
from .geometry import point_on_ray
from .pipeline import dump_json


def collect_points(plan_path, distance_range):
    plan_path = Path(plan_path).resolve()
    plan = read_plan(plan_path, 'bins-v1')
    if plan.get('missing_event_tasks'):
        raise ValueError('A complete event-cache plan is required')
    phase = 'S^{:g}S'.format(plan['configuration']['interface_depths_km'][0])
    con = sqlite3.connect((plan_path.parent/'index.sqlite').as_uri()+'?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        members = [dict(r) for r in con.execute(
            'SELECT task,row_index,event_id,trace_id FROM members '
            'WHERE phase=? AND distance>=? AND distance<=? ORDER BY task,row_index',
            (phase, float(distance_range[0]), float(distance_range[1])))]
    finally:
        con.close()
    result, seen, current = [], set(), None
    for member in members:
        if current != member['task']:
            cache = event_cache(plan['cache_root'], member['task'])
            if json.loads((cache/'complete.json').read_text()) != {
                    'plan': plan['event_plan_fingerprint'], 'task_index': member['task']}:
                raise ValueError('Event cache identity changed')
            metadata = json.loads((cache/'traces.json').read_text())['accepted']
            prov = json.loads((cache/'provenance.json').read_text())
            current = member['task']
        row = metadata[member['row_index']]
        key = (row['event_id'], row['trace_id'])
        if key != (member['event_id'], member['trace_id']) or key in seen:
            raise ValueError('Duplicate or mismatched event/trace')
        if (row['station'].split('.', 1)[0] != plan['network'] or
                not distance_range[0] <= row['distance_degree'] <= distance_range[1]):
            raise ValueError('Wrong network or distance')
        seen.add(key)
        midlat, midlon = point_on_ray(prov['origin_latitude'], prov['origin_longitude'],
                                     row['azimuth_degree'], row['distance_degree']/2)
        record = dict(event_id=key[0], trace_id=key[1], distance_degree=row['distance_degree'],
                      source_depth_km=prov['origin_depth_km'],
                      source_latitude=prov['origin_latitude'], source_longitude=prov['origin_longitude'],
                      station_latitude=row['station_coordinates']['latitude'],
                      station_longitude=row['station_coordinates']['longitude'],
                      midpoint_latitude=midlat, midpoint_longitude=midlon)
        for phase, label in [('SS', 'SS'), ('S^400S', 'S400S'), ('S^670S', 'S670S')]:
            record[label+'_latitude'] = row['arrivals'][phase]['reflection_latitude']
            record[label+'_longitude'] = row['arrivals'][phase]['reflection_longitude']
        result.append(record)
    if not result:
        raise ValueError('No waveforms selected')
    return plan, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance-range', nargs=2, type=float, default=[100, 160])
    args = parser.parse_args()
    low, high = args.distance_range
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 180:
        parser.error('Distance range must be inside 0..180 degrees')
    plan, points = collect_points(args.plan, args.distance_range)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.io import shapereader
    # Resolve cached coastlines before creating output; no waveforms are loaded.
    coast = shapereader.natural_earth(resolution='110m', category='physical', name='coastline')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    distance = np.array([r['distance_degree'] for r in points])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True,
                             subplot_kw={'projection': ccrs.Robinson()})
    for ax, (key, title) in zip(axes.flat, [
            ('midpoint', 'Geographic great-circle midpoints'), ('SS', 'SS surface bounce points'),
            ('S400S', 'S400S bounce points · 400 km'), ('S670S', 'S670S bounce points · 670 km')]):
        ax.set_global()
        ax.set_facecolor('#f6f8fa')
        ax.gridlines(linewidth=0.35, color='0.65', alpha=0.6, zorder=0)
        ax.add_geometries(shapereader.Reader(coast).geometries(), crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='0.25', linewidth=0.45, zorder=3)
        mesh = ax.scatter([r[key+'_longitude'] for r in points],
                          [r[key+'_latitude'] for r in points], c=distance, s=2.5,
                          vmin=low, vmax=high, cmap='viridis', linewidths=0,
                          transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
        ax.set_title(title, fontsize=12)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.045,
                 pad=0.025, shrink=0.6, label='Epicentral distance (degrees)')
    events = len(set(r['event_id'] for r in points))
    fig.suptitle(f'Global {plan["network"]} coverage · {len(points):,} event–station pairs · '
                 f'{events} events · {low:g}–{high:g}°\n'
                 'Same accepted waveform selection in every panel; bounce points from cached PREM rays', fontsize=13)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/('global_midpoints_bounce_points.'+suffix), dpi=200)
    plt.close(fig)
    with (output/'points.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    dump_json(output/'provenance.json', dict(bin_plan=str(args.plan.resolve()),
        bin_plan_fingerprint=plan['fingerprint'], distance_range_degree=args.distance_range,
        trace_count=len(points), event_count=events, network=plan['network'],
        reference_model=plan['configuration']['reference_model'], coastline_path=str(coast),
        midpoint_definition='Half the minor great-circle epicentral arc, independent of source depth.',
        bounce_point_definition='Cached phase-specific unique underside reflection from TauP ray paths; source depth included.',
        selection_note='One entry per accepted event/station, including geographically ineligible cells; no phase duplicates.'))
    print(json.dumps({'output': str(output), 'waveforms': len(points), 'events': events}))


if __name__ == '__main__':
    main()
