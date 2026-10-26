# Global midpoints and bounce points

Use the existing `geo` environment, whose Cartopy imports successfully:

```bash
cd ~/WORK/SS_precursor/SS_precursor
~/miniconda3/envs/geo/bin/python -m ss_precursor.point_map \
  --plan outputs/sy_bin_plan_10deg/plan.json \
  --distance-range 100 160 --output outputs/global_point_map_repeat
```

The output folder must be new. The four panels show the geographic great-circle
midpoint and the separate PREM SS, S400S, and S670S reflection points for each
accepted event–station pair in the specified distance range. This includes all
geographic cells, independently of their stack eligibility thresholds. Each
waveform appears once per panel. Points are colored by epicentral distance.

The geometric midpoint is halfway along the minor great-circle source–receiver
arc. The bounce points come directly from the cached phase-specific TauP ray
paths and account for event depth. They need not coincide with the midpoint.
No ray paths or waveforms are recalculated. Coastlines use the locally cached
Natural Earth 110m data. PNG, PDF, a CSV of all point coordinates and identifiers,
and provenance are saved. Cartopy is needed only for this mapping command; the
existing ASDF environment currently has a Cartopy/Shapely import conflict.
