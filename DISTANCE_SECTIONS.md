# Signed distance–time sections

## Overlapping distance windows

```bash
python -m ss_precursor.distance_section \
  --plan outputs/sy_bin_plan_10deg/plan.json --task-index 187 \
  --distance-range 100 160 --window-width 10 --distance-step 1 \
  --output outputs/best_10deg_moving_distance_stack_100_160
```

This uses the same best-populated regional cell as the fixed-interval example.
There are 51 full-width windows: 100–110, 101–111, ..., 150–160 degrees, plotted
at centres 105–155 degrees. Each window is averaged by its own valid support.
Bounds are lower-inclusive and upper-exclusive, with the final 160-degree
endpoint included. Overlapping windows share traces, so adjacent columns are
correlated. The unique total N must not be obtained by adding window counts.
Color scales and original waveform normalization are unchanged. The plot has
no additional interpolation or image smoothing. NPZ and provenance record both
window centres and actual input bounds; `distance_edges` are display pixel edges.
Omit `--task-index` and add `--global-stack` to apply moving windows globally.
Without `--window-width`, the original disjoint-interval behavior is retained.

## Global stack, 100–160 degrees

```bash
python -m ss_precursor.distance_section \
  --plan outputs/sy_bin_plan_10deg/plan.json \
  --global-stack --distance-range 100 160 --distance-step 10 \
  --output outputs/global_distance_stack_100_160
```

This selects each accepted event–station trace once across all geographic cells,
including cells below the geographic coverage thresholds. It does not stack
already averaged geographic stacks, and the multiple phase entries in the index
do not duplicate waveforms. It requires a complete cache plan. Six nonoverlapping
distance intervals are used: [100,110), [110,120), ..., [150,160], with 160 included.
The count plot and NPZ contain the actual number of contributing waveforms in
each interval. No parallel job or waveform preprocessing is needed. All other
normalization, polarity and display conventions below are retained. With trace
weighting each interval is divided by its own sample support. With event weighting
the weights use each event's count in the full selected set, matching the existing
geographic stack convention; this does not force equal event weights per interval.

Run from `~/WORK/SS_precursor/SS_precursor` in your ASDF Python environment:

```bash
python -m ss_precursor.distance_section \
  --plan outputs/sy_bin_plan_10deg/plan.json \
  --output outputs/distance_section_example
```

By default this selects the eligible geographic reflection bin with the most
traces. Use `--task-index INDEX` to select another bin. The output directory must
be new. This reuses event caches and does not read ASDF files again.

The image has epicentral distance horizontally, SS-aligned time vertically,
signed amplitudes as colors, and contributing waveform counts above. Traces are
averaged in nonoverlapping 5-degree distance intervals. Set `--distance-step 2`
for finer intervals, with fewer contributing waveforms per interval. The 10-degree
geographic selection and the 5-degree epicentral-distance intervals are separate.

Individual cached traces retain their SS amplitude normalization and SS-positive
polarity correction. There is no extra column normalization, precursor shift,
or precursor polarity flip. Linear means use the bin plan's weighting, including
the full-bin event weights if `event_equal` was selected. Each valid sample uses
its own support; empty cells are gray. No interpolation fills empty intervals.

The full view clips colors at +/-0.3 of SS; the precursor zoom clips at +/-0.08.
Change these using `--amplitude-limit` and `--zoom-amplitude-limit`. Color clipping
does not change the signed stacks saved in `distance_stack.npz`. Blue is positive.
Dashed curves show the median per-trace model phase-minus-SS predictions in each
distance interval; faint shading spans their minimum and maximum. These are
predictions, not picked arrivals. All phases use the same selected trace subset,
whose other-phase reflection points can fall outside the selection cell.

Outputs include PNG/PDF, signed arrays, per-interval counts, sample support,
model predictions, selected members, and provenance. Available time support is
the existing cache window (-400 to +100 seconds).
