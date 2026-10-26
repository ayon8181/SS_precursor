# GLAD-M35 analysis preparation

The GLAD inputs are `../proc/*.glad_m35.T010-250s.h5`, with waveform tag
`proc_glad`. Inspection found 61 files/events and SY stations in each file.
They have existing R/T/Z channels. Units (displacement mm), R/T convention,
and minimum reliable period (10 s) were confirmed by the user.

The files already contain displacement synthetics, so instrument-response
removal, unit conversion, and a second R/T rotation are unnecessary. For the
same analysis as PREM, use the common 10–100 s two-corner zero-phase bandpass,
detrend/taper, measured SS alignment, SS-positive whole-trace polarity, and
normalization by each trace's own SS peak. Filtering alone does not perform
the alignment and amplitude standardization needed by these stacks.

`glad_m35_config.yaml` has `relative_time_s: [-500, 300, 1]`: each accepted
waveform cache contains 801 samples from 500 s before to 300 s after measured
SS. These are signed SS-normalized analysis windows, not a new raw ASDF copy.
Original ASDF files remain available unchanged. The valid time window must
fit inside reliable waveform support; missing samples are rejected rather
than filled with zeros. A real GLAD waveform passed this complete-window test.

Run locally from `~/WORK/SS_precursor/SS_precursor`:

```bash
SS_WORKERS=6 bash scripts/run_glad_m35.sh
```

The frozen plan at `outputs/glad_m35_10_100s_long/events.json` includes all 61
GLAD events. The runner processes event caches with six workers, indexes them,
then creates the global 1-degree distance section, circular cap/radius stacks,
S511/S545/S579 vespagrams, and temporal-semblance comparisons. The same 727 SMKS
centres and 600/900/1200/1500 km radii are used, with 100–160 degree selection
for plotted global/circular stacks. Preprocessing retains the PREM baseline's
80–160 degree acceptance range and other rejection settings.

The script uses `asdf` Python for analysis and `geo` Python for circular-map
plots. Override `SS_PYTHON`, `SS_MAP_PYTHON`, or `SS_WORKERS` as needed. It is a
local script, not a Slurm submission. It resumes completed event tasks after
checking their identities, and skips complete plotting stages. A partially
written plot folder without provenance requires inspection and a new output
folder; it is not silently overwritten. Current outputs are preserved.

All new outputs go to `outputs/glad_m35_10_100s_long`:

- `cache/events/event_*`: waveforms.npy, tau.npy and per-trace metadata.
- `bin_plan`: complete cache index. Its square-bin thresholds do not restrict
  the global or circular-cap selection, which includes all accepted traces.
- `global_1deg`: native SS-aligned distance section over the full saved window.
- `circular_caps`: phase-specific circular selections and radius comparisons.
- `vespagrams`: residual-slowness diagnostics about each model phase curve.
- `semblance_check`: instantaneous and finite-window coherence comparisons.
- `input_inventory.json`: file/tag coverage and common/extra event IDs.
- `validation_sample`: the tested one-waveform, 801-sample example.

PREM remains the **reference prediction model**, for shared travel-time curves
and nominal S400S/S670S bounce geometry. These are not 3-D ray paths through
GLAD-M35 or a verification of GLAD's exact interface depths. Reference-based
moveout retains GLAD travel-time anomalies; stars remain predictions, not picks.
The existing +/-30 s SS search is retained for consistency. After processing,
check rejection reasons and SS alignment residuals before changing that window.

There are 54 shared PREM/GLAD events and 7 GLAD-only events. Full GLAD stacks
include all 61. A controlled model comparison must use the common accepted
event/station pairs and identical cap membership/weights as appropriate; it
cannot assume the two full stacks have matched sampling. Existing PREM caches
span -400 to +100 s. Comparing the wider -500 to +300 s window for both models
would require re-preparing PREM with the same longer output window.

The full run has not been launched by this setup step. The config, frozen
61-event plan, input/tag inventory, and real-waveform validation are ready.
