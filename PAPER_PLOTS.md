# Vespagrams and waveform sections

## Moveout correction (version 2)

The default `moveout_mode: predicted_curve` removes each trace's predicted
phase moveout before scanning residual slowness. This is necessary for broad
distance apertures: in the 30-trace preview spanning 83.8–154.8 degrees, a single
straight moveout line misses the PREM curve by up to 47 seconds.

For each focused panel the samples are

```
u_i[T_phase(D_i,h_i) - T_SS(D_i,h_i) + offset + (q-q0)*(D_i-D0)]
```

Here `q0` is the median model ray parameter at the reference distance. The axis
is **model-reference plus residual slowness**, not an independently measured
absolute ray parameter. At `q=q0` this is the exact model-predicted stack.
The overview uses the aligned SS curve and `q0_SS`; precursor differential
curvature remains in that overview. Each focused panel uses its own phase curve.
Measured SS alignment shifts remain removed in both modes.

Use a new output root, such as `outputs/paper_plots_corrected`, for version 2.
Signed stacks, support counts and semblance remain in the NPZ files. The legacy
`*_absolute.npz` filenames are retained for compatibility; corrected files also
contain `residual_slownesses`, `reference_slowness`, and `moveout_mode`.

Setting `moveout_mode: linear_absolute` reproduces the former straight-line
scan described below. Its broad-aperture smearing must be assessed before
interpreting weak or displaced peaks. The absolute-slowness descriptions below
apply only to that legacy mode; the default axes use the corrected definition above.

Run from `~/WORK/SS_precursor/SS_precursor` after activating your Python
environment. These commands reuse the completed event caches; they do not read
ASDF or repeat preprocessing, relax rejection criteria, or apply nonlinear filters.

Start with one geographic bin:

```bash
python -m ss_precursor paper-plots \
  --plan outputs/sy_bin_plan/plan.json \
  --output outputs/paper_plots \
  --settings paper_plot_settings.yaml --task-index 0
```

For all eligible bins using six local processes, omit `--task-index`:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m ss_precursor paper-plots \
  --plan outputs/sy_bin_plan/plan.json \
  --output outputs/paper_plots \
  --settings paper_plot_settings.yaml --workers 6
```

Outputs in each phase/bin directory are:

- `vespagram_overview.png/pdf`: SS, S400S, and S670S in one absolute-slowness
  scan, with a separately scaled precursor zoom alongside it.
- `vespagram_phases.png/pdf`: three focused panels, one per phase, on the
  same 0–20 s/degree absolute-slowness axis and ±40 s phase-relative time axis.
- `vespagram_SS`, `vespagram_S400S`, `vespagram_S670S`: separate focused figures.
- `record_section.png/pdf`: distance-sorted, SS-aligned wiggles and a precursor
  zoom with an explicitly labelled larger display gain; distance increases downward, and only a deterministic subset is plotted
  when the bin has more than `record_max_traces` contributors. The stacks still
  use every contributing trace.
- `overview_absolute.npz`, `SS_absolute.npz`, `S400S_absolute.npz`, and
  `S670S_absolute.npz`: signed linear stacks, sample-wise semblance, valid
  counts/weight sums, time axes, and absolute slowness axes.
- `plot_provenance.json` and `members.json`: bin selection, exact contributors,
  settings, model predictions, reference distance/depths, support thresholds,
  and amplitude scales.

All panels use the **same event/station subset**, selected by the named bin's
precursor reflection points. For example, an S400S-selected bin's S670S panel
uses those same rays, whose S670S reflection points may lie outside that cell.
Use S670S-selected bins for S670S-specific geographic interpretation. SS is
included as the surface-reference phase, not as a mantle discontinuity.

## What the axes and colours mean

The former plots scanned relative moveout in SS-aligned time. The new plots
compute an **absolute** slowness scan rather than adding a fixed value to the
old axis. Let `u_i(tau)` be a waveform aligned on its measured SS maximum,
`D0` the bin's reference distance, and `h_i` its event depth. The overview samples

```
u_i[T_SS(D0,h_i) - T_SS(D_i,h_i) + t + q*(D_i-D0)]
```

where `q` is absolute slowness in seconds/degree and `t` is time relative to
predicted SS at the reference distance. The phase-focused panel instead samples

```
u_i[T_phase(D0,h_i) - T_SS(D_i,h_i) + offset + q*(D_i-D0)]
```

This restores predicted SS distance moveout while preserving measured SS
alignment. Reference predictions account for each event's source depth.
**It is not a scan of raw origin-time traces:** any measured SS alignment shift
remains removed. Reference rays must be unambiguous; a failed reference prediction
is reported instead of selecting an arbitrary branch. Linear moveout is still an
approximation across a broad distance range; curvature can broaden the beams.

Focused panels use only the input samples within ±60 s (configurable) of each
trace's own predicted phase-relative arrival. This prevents high or low trial
slownesses from sweeping the strong SS pulse into a precursor panel. Samples
outside that phase window are excluded, not filled with valid zeros. The
contributing-count threshold therefore also constrains the usable slowness
range; white pixels outside that range are unsupported, not evidence of absent
signal. The full overview remains ungated.

White stars show **model predictions**, not detected peaks. Model ray parameters
vary with event depth; each star uses their median at the reference distance.

The greyscale is the envelope of the signed linear stack, normalized once per
panel. No normalization by slowness row or peak sharpening is performed. Values
below `contour_floor` are white; the default contour range is 0.2–1.0. A precursor
zoom and each focused phase panel have independent amplitude scales, recorded
in provenance, so darkness cannot be used to compare SS and precursor amplitudes.
Use the saved signed arrays for amplitude comparisons. Hilbert envelopes are
calculated separately on contiguous supported segments, with no interpolation
across missing time samples. Pixels below the configured contributing-count
threshold are masked. Semblance is retained as a separate diagnostic in NPZ.

## Settings and limits

Edit `paper_plot_settings.yaml` for slowness/time limits, contour threshold,
minimum contributing count/fraction, wiggle gain, displayed distance range,
trace density, and export formats. `record_distance_range_degree` selects the
wiggle display only; it does not change the stack contributors. Use a new output
root after changing settings. Completed plots with matching settings are skipped
on retry. `phases` can include another reference-model underside phase such as
`S^220S`; this adds a diagnostic for the existing selected rays, not a new
geographic bin assignment. No waveform processing or new discontinuity is invented.

Existing caches cover only **−400 to +100 s** relative to measured SS. The
screenshots' −700 to +300 s record sections cannot be reconstructed from those
saved windows. For that wider view, change preprocessing `relative_time_s` to
`[-700, 300, 1]`, create a new event plan/cache root and bin plan, and rerun
preprocessing. These plotting commands automatically use the available cache
window and never fill missing parts as valid waveforms.

These figures retain a conventional linear baseline. Similar-looking published
figures may use different filters, local apertures, or nonlinear coherence
estimators. A smooth or compact-looking contour is not evidence of good slowness
resolution: interpret it with period band, distance aperture, event count, and
the saved support/coherence diagnostics.
