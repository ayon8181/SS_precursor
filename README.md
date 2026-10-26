# SS precursor recovery

For combined SS/S400S/S670S absolute-slowness contours, separate phase panels,
and distance-sorted wiggle sections from existing caches, see
[PAPER_PLOTS.md](PAPER_PLOTS.md).

A small Python package for inspecting ASDF synthetics, measuring SS alignment,
and producing conventional linear slant stacks. Original ASDF files are opened
read-only. Start with the first-event diagnostics before interpreting depths.

For all-event **SY-network midpoint-bin stacking**, including parallel job-array
templates and restart instructions, see [BATCH_STACKING.md](BATCH_STACKING.md).

## Run in the existing environment

The installed `asdf` environment contains the required dependencies; no install
or changes to the sibling project are needed. Run from this directory:

```bash
export MPLCONFIGDIR=/tmp/ss-precursor-mpl
SS_PY=/home/ayon/miniconda3/envs/asdf/bin/python

"$SS_PY" -m ss_precursor inspect \
  '../SmKS_processing/seis/proc/*.proc_prem_2.T010-250s.h5' \
  --stations 2 --output outputs/inspection.json

"$SS_PY" -m ss_precursor prepare \
  ../SmKS_processing/seis/proc/201801090005A.proc_prem_2.T010-250s.h5 \
  --config example_config.yaml --output outputs/prem_201801090005A

# Restart stacking/plotting from saved preprocessing without reopening ASDF.
"$SS_PY" -m ss_precursor stack outputs/prem_201801090005A

"$SS_PY" -m unittest discover -s tests -v

# Save controlled recovery measurements and a figure.
"$SS_PY" -m ss_precursor validate --output outputs/validation
```

`inspect --stations 0` inspects every station. The default inspects three per
file and reports the total station count. Inspection reports retain all event
origins and focal mechanisms, waveform event associations, tags, timing,
sampling, coordinates, units when embedded, and individual metadata errors.
Use repeated `prepare --station NETWORK.STATION` options for a small explicit
diagnostic subset; the selected IDs are saved in provenance. The first metadata
and numerical validation summaries are in `reports/metadata.md` and
`reports/validation.md`.

## Confirmed inputs and defaults

- User-confirmed displacement in **mm**, existing **T component**, minimum
  reliable period **10 s**, and PREM interface depths.
- The inspected tag is `proc_prem`; it is explicitly configured, never guessed.
- ObsPy's PREM has boundaries at **400 and 670 km**. These are verified against
  its velocity model before making predictions.
- Travel times and geometry use the explicitly saved preferred origin. In the
  initially inspected file this is the CMT centroid, distinct from the
  hypocentre. Waveform times are calculated from absolute start times, not from
  sample zero or an assumed origin offset.
- Existing RT components are preserved. Their relative timing, location, and
  channel prefixes must agree. Absolute transverse handedness is not inferred
  from the component letters. The SS-positive convention applies one sign to
  the entire trace, preserving precursor-to-SS polarity. NE input is optionally
  rotated with ObsPy's NE-to-RT convention after timing checks.
- Exploratory processing uses a 25–100 s, order-four, zero-phase Butterworth
  bandpass, linear detrend, 5% taper at each end, reflected filter padding, and
  an edge guard. All are configurable. No unit conversion is implicit.
- SS is the largest absolute filtered displacement in a ±30 s predicted
  window, refined by a three-sample parabola. Boundary maxima and unstable
  normalization amplitudes are rejected. Save measured and predicted times.
- Distance selection is an exploratory 80–160 degrees, with individual
  interference flags and predicted curves. Flagged traces remain visible by
  default. This range is not asserted to be universally free of interference.

## Conventions and outputs

Relative time is `tau = time - measured_SS_time`. Positive slowness means
precursor-minus-SS time increases with epicentral distance:

```
V(tau, p) = sum(w_i * u_i[tau + p*(distance_i-reference_distance)])
            / sum(valid w_i)
```

Interpolation never wraps or extrapolates. Each output sample has its own
valid-weight denominator, contributing count, weight sum, and weighted
semblance. Zero energy has undefined semblance. Sample-wise semblance needs
count/aperture context; one trace alone cannot establish coherence.

Phase-specific stacks sample `T_precursor - T_SS + offset` independently for
each event/station. Underside phases are expressed as `S^400S` and `S^670S`;
ray paths must contain exactly one reflection at the requested boundary and
have a unique minor-arc branch. Ambiguities are logged as rejections.

Each `prepare` directory contains:

- `provenance.json`: exact configuration, configuration hash, source file
  identity/size/modification time, software versions, selected event and origin,
  reference boundaries. Source identity uses stat metadata rather than a full
  file-content hash.
- `traces.json`: accepted metadata, predicted branches/reflection coordinates,
  SS measurements, interference flags, rejected stations and reasons.
- `preprocessed.npz`: normalized waveforms on a shared time grid (1 s default).
  Filtering precedes interpolation; choose the output sampling interval to
  resolve the configured bandwidth.
- `coverage.json`: phase-specific reflection bins, traces, independent events,
  distance aperture/spread, azimuths, coordinates, and eligibility flags.

`stack` adds vespagram arrays, phase-specific predicted-moveout stacks,
`record_section.png`, and `stacks.png`. These first-event plots combine accepted
stations for processing diagnostics; they are not geographic depth estimates.
The record-section plot shows absolute predicted arrivals relative to measured
SS. Moveout stacks use theoretical precursor-minus-SS differences, making any
remaining differential bias visible rather than absorbing the SS residual.

Bins use precursor reflection coordinates on spherical reference rays, with SS
surface bounce points stored separately. Cells wrap longitude into [-180,180),
include both poles, and are not equal area. Eligibility defaults to at least
10 traces, three earthquakes, and 10 degrees of distance aperture. A first-event
run is therefore intentionally ineligible for scientific bin interpretation.

## Scientific baseline and remaining milestones

The methods baseline is [Bai, Zhang & Ritsema (2012)](https://doi.org/10.1111/j.1365-246X.2011.05256.x),
particularly sections 2.2–2.3: measured SS peak alignment, whole-trace SS polarity
normalization, and phase-specific moveout stacks. This implementation differs
in its Butterworth bandpass, rectangular reflection bins, and initial linear
vespagrams. The paper's cosine-square filter and 10-degree circular caps are not
reproduced. Its discussion of anisotropic synthetic versus isotropic travel-time
bias motivates retaining the raw 1D recovery residual.

[Zheng, Ventosa & Romanowicz (2015)](https://doi.org/10.1093/gji/ggv118)
distinguishes local slant-stack filtering from conventional slant stacking.
Only the conventional linear baseline is implemented here.

`depth_recovery.depth_stack` implements sampling genuine trial-model
precursor-minus-SS predictions, optionally with trace-by-depth differential
mantle corrections. Positive corrections sample later relative times. Its sign
is verified using controlled pulses. It does not yet build trial-depth Earth
models or claim recovery from the real PREM files.

Next validation gates are multi-event bin selection, constructing and verifying
models that actually move the target boundary, real PREM depth recovery with a
documented tolerance, earthquake bootstrap, and paired 3D/correction experiments.
No 3D model or mantle corrections have been supplied. No global interpretation
is made from the diagnostic stacks.

## Hawaii comparison

See [HAWAII_COMPARISON.md](HAWAII_COMPARISON.md) for the PREM/GLAD SY-only versus all-station circular stacks and plots.
