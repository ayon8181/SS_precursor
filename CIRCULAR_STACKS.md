# Circular stacks about the SMKS station points

The centres are the 727 points defined by `~/WORK/SMKS/create_station.py` at
900 km target spacing. The existing `stations_fibonacci_100km.txt` is read and
checked against that generator, despite its historical filename. The generator
is not imported or run, so its output file is not rewritten.

Run from `~/WORK/SS_precursor/SS_precursor` using the `geo` environment (Cartopy
works there):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/miniconda3/envs/geo/bin/python -m ss_precursor.circular_stacks \
  --plan outputs/sy_bin_plan_10deg/plan.json \
  --centers-script ../../SMKS/create_station.py \
  --radii-km 600 900 1200 1500 \
  --distance-range 100 160 \
  --output outputs/circular_caps_repeat
```

The output directory must be new. The square bin plan provides access to the
completed caches only: all accepted traces in the distance range are considered,
including traces from square cells below the former coverage thresholds. There
is no square-cell geographic restriction. Defaults compare all four radii.

A cap includes a waveform when the surface great-circle distance from its own
phase-specific bounce point to the centre is at most the radius. Earth radius
is 6371 km; 900 km is a radius, not a diameter, and is measured on the surface,
not along the 400/670 km interface. S400S and S670S membership is calculated
separately. A waveform can contribute to multiple overlapping caps and radii,
but only once within any cap/phase/radius. Results across radii are correlated.

Each input trace retains its existing SS peak normalization and SS-positive
polarity correction. Signed linear means are normalized by their own valid
support (or weights). No precursor sign flips or cap-peak normalization are
applied. With `event_equal`, weights are recomputed within each selected cap.

Outputs:

- `radius_comparison.png/pdf`: signed phase stacks for the top three centres by
  minimum S400S/S670S coverage at the radius nearest 900 km. Every radius uses
  the same centre. Model differential times for each trace remove predicted
  distance/source-depth moveout; zero is the phase prediction. The examples
  are selected by counts, not by amplitude or apparent detection.
- `best_center_distance_sections.png/pdf`: native SS-aligned distance sections
  at the best-populated centre, with 12 disjoint distance intervals and common
  color limits +/-0.08. No precursor time correction is applied to these sections.
  Gray columns lack data; colors outside limits saturate. Dashed curves show
  each selected phase's median prediction per distance interval.
- `coverage_maps.png/pdf`: counts at every centre for each phase and radius,
  on a common scale. The red star marks the best-populated example centre.
- `circular_stacks.npz`: arrays ordered **phase, radius, centre, time offset**,
  with signed stacks, sample-wise semblance/support/weight sums, and total
  waveform/event counts. Axes and centre IDs/coordinates are included.
- `coverage.csv`: counts and an eligibility flag (at least 10 traces, 3 events).
  This flag assesses coverage, not the significance of a detection. Every
  centre, including those below these thresholds, remains in the saved arrays.
- `waveform_members.json` and `cap_members.json`: exact reproducible membership.
  Cap indices refer to the zero-based row in `waveform_members.json`, which
  supplies cache task and row identities. Membership keys are phase, radius
  string (e.g. `900.0`), and centre ID (e.g. `S511`).
- `centers.json`, `provenance.json`: source point coordinates, original file
  paths/hashes, processing settings and selected examples.

The initial comparison uses 14,667 accepted waveforms from 54 events. At 900 km,
all 727 centres have at least 10 traces and 3 events for both phases. The median
count is 70 traces; the best-populated example S511 (-23.854, 108.960 degrees)
has 171 traces at 900 km. Increasing radius raises counts while reducing spatial
localization. The plotted examples are stable around their predicted arrivals
across the tested radii, but this is not a global significance or resolution test.
