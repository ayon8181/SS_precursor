# 10–100 second bandpass, two corners

New configuration: `sy_config_10_100s_2corners.yaml`.

- `period_band_s: [10, 100]` means frequencies 0.01–0.1 Hz.
- `filter_order: 2` sets the Butterworth prototype order (the usual two-corner
  parameter), retaining the existing forward/reverse zero-phase application.
- Detrending, taper, padding, SS alignment, SS-positive polarity, and individual
  SS amplitude normalization remain configured as before. Normalization and SS
  alignment are recalculated on the newly filtered waveform.

The frozen event plan is `outputs/sy_10_100s_2corners/events.json`. It uses the same
54 original ASDF event tasks as the 25–100 s run, with their unchanged input
identities verified. There are additional ASDF files in `../proc`; they are not
silently added to this comparison. The original 25–100 s caches/plots remain
available. The new plan changes only `period_band_s` and `filter_order`.

The 25–100 s caches cannot provide frequencies removed by their original filter.
Reprocessing reads the original ASDF files and creates separate event caches.
One previously accepted event/station waveform was processed successfully under
the new configuration; its diagnostic is in the new run's `validation_sample`.
The full 54-event reprocessing has not been launched by this setup step.

Run locally:

```bash
cd ~/WORK/SS_precursor/SS_precursor
SS_WORKERS=6 bash scripts/run_10_100s_2corners.sh
```

The runner uses your `asdf` Python for preprocessing/stacking and `geo` Python
for the circular-cap maps, where Cartopy works. Override `SS_PYTHON` or
`SS_MAP_PYTHON` if necessary. It limits numeric library threads to one per worker.

Outputs under `outputs/sy_10_100s_2corners` include:

- `cache`: new 10–100 s event waveforms, prepared with six local workers.
- `bin_plan`: an index of completed waveforms; square-cell eligibility does not
  restrict the subsequent global or circular selections.
- `global_1deg`: global 100–160 degree section with disjoint 1-degree intervals.
- `circular_caps`: all 727 SMKS centres and 600/900/1200/1500 km radii.
- `vespagrams`: S511/S545/S579 fixed for comparison with the earlier band.
- `semblance_check`: temporal-window comparisons using a newly estimated SS
  period in the new band. The spectrum display follows the configured band.

Event preparation resumes completed tasks after verifying identity. Complete
plot stages are skipped. A partial plot output without provenance is reported
by the plotting command; inspect it and choose a new output folder rather than
overwriting it. Do not edit the frozen plan or configuration to change a started
run. Filtering can change accepted traces and therefore cap membership/counts;
consult the new provenance when comparing bands.
