# One SS reference per event, individual station waveforms retained

This workflow prepares the GLAD-M35 event database only. It does not average
an event into one trace, combine different events, or create cross-event stacks.
It uses the existing 61-event GLAD input set and opens ASDF files read-only.

Run from `~/WORK/SS_precursor/SS_precursor`:

```bash
SS_WORKERS=6 bash scripts/prepare_glad_event_database.sh
```

The new frozen plan is `outputs/glad_m35_event_reference/events.json`; its
configuration is `glad_m35_event_reference.yaml`. These are separate from the
earlier peak-aligned processing run. Use your `asdf` environment, or override
`SS_PYTHON`. Numeric library threads are limited to one per worker. The full
61-event job has not been launched by this setup step.

For each event:

1. Filter existing T displacement with the 10–100 s, two-corner zero-phase
   bandpass. Preserve SS amplitude normalization and whole-trace SS-positive
   polarity. Use PREM only to locate an initial +/-60 s SS search interval and
   retain the shared phase/bounce-point metadata. No PREM synthetic is used as
   a waveform reference.
2. Coarsely align each trace on its SS peak. Save temporary extra time support
   so the subsequent shift does not introduce extrapolated/padded samples.
3. Choose **one actual GLAD station trace** as the event's reference. Prefer
   candidates at 100–140 degrees, when available. Among up to 20 candidates
   with the strongest SS-window/coda power ratios, choose the one with the
   highest median waveform correlation to that event's station windows. It
   remains a single trace; no reference stack or reference from another event
   is substituted. Inspect `reference.json` for the choice and scores.
4. Cross-correlate each SS window (-40 to +40 s) with this reference. Search
   additional lags within +/-30 s, using demeaned, normalized correlation and
   a parabolic subsample refinement. Positive lag means sample target(t+lag),
   shifting a late target pulse earlier. Polarity is already fixed by coarse
   SS normalization; this step uses positive correlation and no extra flips.
5. Accept a correlation of at least 0.7 with an interior lag maximum. Flag
   secondary positive local peaks at least 10 s away when their correlation
   is within 0.05 of the best. The reference itself anchors zero lag. These
   initial QC settings are configurable in a **new plan before processing**.
6. Apply the accepted shift to the **whole waveform**, not to precursor windows
   independently. Crop once to -500..+300 s at 1 s sampling, and retain every
   accepted station as its own row. Correlation preserves precursor–SS
   separation within each trace; it does not force precursors to predictions.

Each completed event folder is
`outputs/glad_m35_event_reference/database/events/event_XXXXXX/`, containing:

- `waveforms.npy`: accepted aligned stations by 801 time samples, with signed
  amplitudes relative to each trace's original coarse SS peak.
- `tau.npy`: -500 to +300 s, relative to the chosen event reference SS pulse.
- `reference.npy`: the chosen reference trace on this same 801-sample grid.
- `reference.json`: reference event/station/trace ID, row index in `waveforms`,
  SS template samples/time grid, selection scores and alignment settings.
- `traces.json`: accepted row identities and original metadata, plus all
  rejection reasons. Accepted rows follow `waveforms.npy` exactly.
- `alignment_qc.json`: proposed lag, normalized correlation, secondary-peak
  margin, boundary flag and acceptance for every coarsely prepared trace.
- `unaligned_waveforms.npy`, `unaligned_tau.npy`, `unaligned_traces.json`:
  coarsely aligned waveforms that did not pass the reference fit, retained
  for later review. Their proposed correlation shifts were **not** applied.
  These have the extra coarse time support and are not stack-ready. Records
  rejected during initial preparation lack a reliable waveform window; their
  metadata remains in `traces.json`, and original ASDF files are preserved.
- `provenance.json`: original file, event, filter, geometry, time grids, shift
  convention and identities. Both coarse and final SS times remain traceable.
- `summary.json` and `complete.json`: counts and the frozen plan/task identity.

After the local run, `database/database_index.json` lists completed events,
their reference IDs, paths/counts and any pending tasks. Successful event
tasks are skipped on retry after identity checks. To refresh only the index:

```bash
python -m ss_precursor.event_database \
  --plan outputs/glad_m35_event_reference/events.json \
  --root outputs/glad_m35_event_reference/database --index-only
```

To process one full event for review first, use the same command with
`--task-index 0 --workers 1` instead of `--index-only`. The normal runner will
then resume and prepare the remaining events. Cache metadata is compatible
with later geographic/circular selection, but no binning or multi-event
stacking is invoked by this runner.

Validation includes known positive/negative fractional delays, whole-trace
precursor-time preservation, poor/ambiguous fits, event isolation, database
output identities and resume. An eight-station real GLAD sample yielded seven
accepted fits and one low-correlation flag. That sample is in `validation_sample`;
it is not a complete event cache, and its reference may differ from the one
chosen using the full station set. A single reference can mismatch some source
radiation/distance-dependent waveform shapes; flagged traces remain available
for review rather than being force-aligned. Event time origins are local to
their SS references; cross-event alignment will be decided separately.
