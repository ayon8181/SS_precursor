# Stack SY stations across all events

## Vista: submit the complete workflow in one node allocation

The supplied script follows `../hv_ratio_kernels/run_specfem.sh`: account
`EAR21003`, partition `gh`, exclusive allocation, and exported environment.
It requests one 72-core node and runs 16 Python workers by default. Vista's
scheduler does not accept `--mem`; see the
[Vista guide](https://docs.tacc.utexas.edu/hpc/vista/). Use this
single-node script to run event workers and bin workers within one allocation,
instead of assigning an entire node to every one-process array task.

From the transferred project directory on Vista, create `logs`, set your actual
Vista Python executable, and submit (replace the Python and data paths):

```bash
mkdir -p logs
export SS_WORKERS=16
sbatch scripts/vista_pipeline.slurm \
  /path/to/vista/environment/bin/python \
  '/path/to/proc/*.proc_prem_2.T010-250s.h5' \
  outputs/vista_sy sy_config.yaml
```

Keep the glob quoted. The Python environment must be installed for Vista's ARM
architecture and contain the project dependencies. Transfer source and data;
use a Vista-installed Python environment. No MPI launcher or GPU is needed.
The job creates its plans on the compute node, then preprocesses all event tasks,
indexes the geographic bins, and stacks them. Logs go to `logs/ss-vista-JOBID.log`;
final stacks go to `outputs/vista_sy/stacks/`. To retry, submit the same command;
completed tasks are reused. Choose a new output root when changing inputs or
settings. After a time-limit termination, check for stale per-task locks as
described below before retrying.

Monitor with `squeue -u "$USER"` and `tail -f logs/ss-vista-JOBID.log`.

The remaining instructions describe the individual stages and generic job arrays.

Run from this repository with a Python environment containing the dependencies
in `pyproject.toml`. The existing local environment is
`/home/ayon/miniconda3/envs/asdf/bin/python`; set `SS_PY` to an environment
available on the compute nodes. All paths in generated plans are absolute, so
create plans on the filesystem where the jobs will run.

The workflow uses **separate 5° × 5° bins at the 400 and 670 km underside
reflection points**, calculated from the reference rays. SS surface bounce
coordinates remain in the event metadata. Bins pool traces across earthquakes;
no single-event stacks are averaged together. The network selection is exactly
`SY`, excluding real networks and codes such as `SY1`.

## 1. Freeze the inputs and preprocessing settings

```bash
export SS_PY=/home/ayon/miniconda3/envs/asdf/bin/python
export MPLCONFIGDIR=/tmp/ss-precursor-mpl
mkdir -p logs

"$SS_PY" -m ss_precursor plan-events \
  '../SmKS_processing/seis/proc/*.proc_prem_2.T010-250s.h5' \
  --config sy_config.yaml --network SY --output outputs/sy_events.json
```

This reads event catalogs, without reading waveforms, and creates an explicit
task for every event in every matched file. Multi-event files are supported;
their waveforms still require unambiguous event associations. Plans snapshot
configuration, input paths, file sizes, and modification times. Use a new plan
filename/cache root when changing preprocessing settings or input files.

The batch defaults in `sy_config.yaml` retain T displacement in mm, a 25–100 s
band, distances 80–160°, and all source depths (0–6371 km). **All SY stations are considered; these filters and
the existing waveform/branch/alignment checks still determine acceptance.**
Change the YAML before planning if different selections are intended. The
minimum reliable synthetic period remains 10 s. Predicted interference is
flagged; `reject_predicted_interference: true` enables rejection.

## 2. Submit one preprocessing task per event

```bash
n_events=$("$SS_PY" -m ss_precursor task-count outputs/sy_events.json)
if [ "$n_events" -gt 0 ]; then
  sbatch --array="0-$((n_events-1))%16" \
    scripts/prepare_array.slurm outputs/sy_events.json outputs/sy_run
fi
```

Add your site's `--account`, `--partition`, resource limits, or environment
setup as appropriate. The array templates request one node and one CPU per task;
memory allocation follows site policy. Use the single-node script above on Vista.
The concurrency cap of 16 is an editable starting point. The templates require Slurm to
export `SS_PY` (the normal export-all behavior). They do not submit anything
until you run `sbatch`.

Each task selects SY station IDs **before reading their StationXML or waveforms**.
Only the selected event and waveform tag are read. Inputs stay read-only.
Station failures are logged and do not abort the event. Events with no SY
stations, or no accepted traces, complete with an empty cache and remain
visible in the coverage summary. Event-level failures cause a nonzero task exit.

Wait for all event tasks to finish successfully before the next stage. Repeating
the array command skips completed tasks with matching identities and retries
unfinished tasks. Per-task output is renamed into place only after completion;
different tasks never write into the same event directory. A hard-killed task
may leave an `event_XXXXXX.lock` directory: remove that lock only after confirming
the corresponding job is no longer active, then retry.

## 3. Build the cross-event geographic bin index

```bash
"$SS_PY" -m ss_precursor plan-bins \
  --plan outputs/sy_events.json --root outputs/sy_run \
  --output outputs/sy_bin_plan
```

This reads one event's metadata at a time and creates an indexed SQLite file;
it does not load the full waveform collection into memory. It rejects duplicate
event/trace pairs so repeated inputs cannot inflate hit counts. It also checks
cache identity, network, configuration, time grids, and row counts.

`outputs/sy_bin_plan/coverage.json` reports **every occupied phase/bin**, including
underfilled bins and reasons. Missing cells have no input traces. A bin must
meet all configured thresholds to receive a stacking task: currently 10 traces,
three independent earthquakes, and 10° of distance aperture. The summary of
accepted/rejected traces and rejection reasons is `selection_summary.json`,
including task indices with no selected SY stations or no accepted traces.
If there are no eligible bins, the plan contains zero stacking tasks; inspect
coverage and sampling before changing the thresholds. A 5° grid is not a claim
of 5° spatial resolution.

By default this command fails if any event task is missing. `--allow-incomplete`
explicitly allows a partial run and records missing task indices in the plan
and every resulting stack. A bin-plan directory is immutable: use a new output
directory to re-index after additional events finish.

To change bin size, thresholds, weights, or stack grids without repeating ASDF
preprocessing, supply `--stack-settings settings.yaml` to `plan-bins` and choose
a new bin-plan output directory. That YAML may override `bin_size_degree`,
`minimum_bin_traces`, `minimum_bin_events`, `minimum_distance_aperture_degree`,
`stack_weighting`, `slowness_s_degree`, `reference_distance_degree`, and
`precursor_stack_offsets_s`. Preprocessing changes such as a different period
band are rejected here and require new event caches.

## 4. Submit one stacking task per eligible phase/bin

```bash
n_bins=$("$SS_PY" -m ss_precursor task-count outputs/sy_bin_plan/plan.json)
if [ "$n_bins" -gt 0 ]; then
  sbatch --array="0-$((n_bins-1))%32" \
    scripts/stack_array.slurm outputs/sy_bin_plan/plan.json outputs/sy_stacks
fi
```

Each bin task memory-maps event caches and copies only its member rows. It
produces a conventional linear time–slowness vespagram plus a stack along the
event/station-specific predicted precursor-minus-SS times. The default is equal
weight per trace; `stack_weighting: event_equal` gives every earthquake equal
total weight within that bin. Semblance, contributing count, and valid weight
sums are saved separately. Fractional delays use interpolation with no wrapping
or extrapolation. These are time stacks, not recovered interface depths.

Outputs are named, for example, `S400S_lat20_lon30/`, and contain:

- `vespagram.npz`: `tau`, `slownesses`, `reference_distance`, `stack`,
  `semblance`, `count`, `weight_sum`.
- `predicted_stack.npz`: `offsets` relative to the predicted differential time,
  `stack`, `semblance`, `count`, `weight_sum`.
- `members.json`: exact events, station/trace IDs, event-cache task and row
  indices, distances, azimuths, reflection coordinates, predicted differential
  times, and interference flag counts.
- `provenance.json`: bin statistics, plan identities, configuration, weights,
  and any missing event tasks. Full event metadata and rejection reasons remain
  in `outputs/sy_run/events/event_XXXXXX/`.
- `stacks.png` when `--plots` is supplied (enabled in the Slurm template).
- `complete.json`: completion identity for resumable execution.

Use the same `--plots` setting when retrying into an existing stack directory;
use a new output root for a different plan or output setting. Event caches use
uncompressed `waveforms.npy` and `tau.npy` to avoid repeatedly decompressing
every event for each bin. Preserve these caches while bin tasks run.

## Local parallel execution instead of Slurm

After creating the event plan, the same two stages can run with local processes:

```bash
"$SS_PY" -m ss_precursor prepare-events \
  --plan outputs/sy_events.json --root outputs/sy_run --workers 4
# Run plan-bins as above after preprocessing completes.
"$SS_PY" -m ss_precursor stack-bins \
  --plan outputs/sy_bin_plan/plan.json --output outputs/sy_stacks --workers 4 --plots
```

For debugging, pass `--task-index 0` to either stage to run just that task. Use
one worker per scheduler array task to avoid nested parallelism. Limit BLAS/OpenMP
threads to one when using process parallelism, as the Slurm templates do.
