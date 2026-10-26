# SS precursor recovery from 1D and 3D ASDF synthetics

## Objective

Build a Python workflow to recover mantle discontinuities from SS precursor
stacks using existing synthetic seismograms stored in ASDF files.

This project was agreed with Jeroen Ritsema. The experiments are:

1. Recover known discontinuities from 1D synthetics.
2. Apply identical processing to 3D synthetics.
3. Quantify apparent discontinuity depth variations and additional arrivals.
4. Apply ray-theoretical differential mantle corrections and assess residual bias.
5. Assess dependence on spatial sampling, period band, and processing choices.

Start with a few well-sampled geographic bins. Expand globally only after
validating the processing.

## Working instructions

- Inspect the repository and follow its existing conventions.
- Implement incrementally and run meaningful validation at each milestone.
- Continue through independent work without requesting approval at every stage.
- Never fabricate waveform metadata, model settings, or successful results.
- Preserve original ASDF files.
- Make routine processing choices configurable and document defaults.
- Distinguish manuscript-derived settings from implementation choices.
- If real inputs are unavailable, validate computational components with
  controlled examples and clearly report the remaining integration work.

## Inputs

Required:
- ASDF paths for each Earth model.
- Explicit waveform tag.
- Event metadata and station coordinates.
- Waveform units and component convention.
- Reference velocity model.
- Actual discontinuity depths in each simulation.
- Validated minimum synthetic period.

Optional:
- Ray-theoretical mantle corrections for each precursor relative to SS.
- Observed data for later comparison.

Do not assume that PREM contains interfaces at exactly 410 and 660 km:
inspect the simulation model and use its actual interface depths.

## Stage 1: Inspect ASDF structure

Use pyasdf to inspect representative files.

Report:
- Available waveform tags and station identifiers.
- Trace component codes, units if available, sampling intervals, and duration.
- Trace start times relative to event origin.
- Event origins, preferred origin, centroid information, and focal mechanism.
- Station latitude, longitude, and elevation.
- Whether horizontal components are NE or already RT.
- Missing or inconsistent metadata.

Do not assume one event per ASDF file. Require explicit event association
when the file structure is ambiguous.

Keep PDE/hypocentral and centroid coordinates/times distinguishable.
Document which origin defines travel-time predictions and rotation.

Deliverable: an inspection command and a concise metadata report.

## Stage 2: Reading, geometry, and selection

Use pyasdf for reading, ObsPy for waveform operations and reference arrivals,
and NumPy/SciPy for numerical processing.

Read stations incrementally rather than loading the full dataset into memory.

For each event–station pair:
1. Read the configured waveform tag.
2. Validate timing and horizontal components.
3. Rotate NE to RT if necessary.
4. Preserve already rotated RT data after checking their convention.
5. Analyse transverse waveforms initially.
6. Compute source–receiver distance and azimuths.
7. Predict SS and the required underside reflections, such as S400S/S670S.
8. Obtain reference underside reflection coordinates at the chosen interface.

Validate phase naming and the correct reflection branch in the travel-time
tool. Do not silently select the first returned arrival.

Assign geographic bins using precursor reflection locations, not station
locations. Record the SS surface bounce point separately.

Use 5° latitude × 5° longitude bins initially. Handle longitude wrapping,
the dateline, and polar boundaries explicitly. Such cells are not equal area.

For each bin report:
- Number of traces.
- Number of independent earthquakes.
- Distance range and spread.
- Azimuth distribution.
- Reference reflection locations.

A minimum hit count alone is insufficient: slant stacking also needs
distance aperture.

## Stage 3: Preprocessing and SS alignment

Make these settings configurable:
- Period band and filter design.
- Input/output waveform units.
- Detrend, taper, padding, and optional resampling.
- SS search window and precursor windows.
- Amplitude normalization and trace weights.
- Polarity convention.
- Distance and source-depth selection.
- Interference screening.
- Minimum event count and distance aperture.

Use a band within the validated synthetic bandwidth.

Align traces using measured SS arrival times, initialized by reference
predictions. Save both measured and predicted SS times and alignment residuals.

Define relative time as:
    tau = time - measured_SS_time

Normalize initially by a clearly defined SS-window amplitude.
Reject near-zero SS amplitudes rather than amplifying unstable traces.

Document polarity handling. Do not flip precursor signs independently to
make a desired arrival stack coherently.

Do not apply one blanket “safe distance range” without examining predicted
interference and the actual record sections.

Deliverable: plots of aligned, normalized record sections with predicted
precursor arrivals and interfering phases.

## Stage 4: Linear slant stacks

Implement conventional delay-and-sum stacks first.

For SS-aligned trace u_i, distance Delta_i in degrees, reference distance
Delta_0, and relative slowness p in seconds/degree:

    V(tau, p) =
        sum_i w_i u_i[tau + p(Delta_i - Delta_0)] / sum_i w_i

Here p is the precursor moveout relative to SS, not its absolute ray parameter.

Document and test the sign convention using artificial arrivals with known
moveout. Use interpolation for fractional-sample delays. Exclude samples
outside trace support rather than wrapping or padding them as valid data.

Output:
- Time–slowness vespagram.
- Stack along the predicted precursor moveout.
- Coherence or semblance as a separate diagnostic.
- Effective contributing trace count.

Make slowness range, sampling, and reference distance configurable.

Do not implement local slant-stack filtering, phase-weighted stacking, or
nonlinear stacks until the linear baseline is validated. These are extensions,
not interchangeable names for the same operation.

## Stage 5: Moveout correction and depth recovery

A linear slant stack is an arrival-identification tool. For depth recovery
across a broad distance/depth range, use phase-specific predicted moveout.

For each trial interface depth h, compute:
    tau_i(h) = T_precursor,i(h) - T_SS,i

Construct a depth stack by sampling SS-aligned traces at tau_i(h).

Account for event depth individually. Avoid treating all events as sharing
one travel-time curve.

Verify that the travel-time model actually supports varying the reflection
depth. Changing only a phase label is not an adequate implementation if the
underlying model does not contain the requested boundary.

Keep:
- Primary and secondary peaks.
- Peak amplitude and coherence.
- Peak width.
- Reference discontinuity depth.
- Recovered-minus-reference depth.

Do not force a peak to exist when no coherent arrival is detected.

## Stage 6: Ray-theoretical mantle corrections

Define the differential correction explicitly:

    c_i(h) =
        deltaT_precursor,i(h) - deltaT_SS,i

A positive c_i means that the precursor arrives later relative to SS in
the 3D model than in the reference model.

For correction during depth stacking, sample the 3D trace at:
    tau_i(h) + c_i(h)

Verify this convention with a controlled imposed shift.

Do not shift the entire seismogram by one mantle correction: a common shift
is removed by SS alignment and cannot correct precursor–SS differential times.

Keep separate results for:
- Uncorrected 1D.
- Uncorrected 3D.
- Ray-corrected 3D.

If corrections are not supplied, implement a documented input interface and
continue the uncorrected experiments. Do not invent corrections.

## Stage 7: Uncertainty and comparisons

Keep these quantities separate:
- Recovery bias relative to known interface depths.
- Sampling variability from bootstrap resampling.
- Sensitivity to processing choices.
- Frequency-dependent and finite-frequency resolution.

Bootstrap earthquakes initially, retaining their associated traces together.
Do not interpret bootstrap precision as proof of accurate recovery.

Use identical event–station subsets when comparing models.

Report:
- Apparent topography with fixed true discontinuities.
- Residual bias after ray corrections.
- Additional coherent peaks absent from the 1D baseline.
- Dependence on period, bin size, distance aperture, and event coverage.
- Missing or poorly constrained bins.

A 5° map grid does not imply 5° spatial resolution.

## Validation requirements

1. Recover the known moveout and intercept of artificial arrivals.
2. Recover an imposed differential time shift with the correct sign.
3. Recover known 1D interfaces within a documented tolerance.
4. Check depth-stack sensitivity to interpolation and depth-grid resolution.
5. Test longitude wrapping and geographic bin assignment.
6. Confirm paired comparisons use the same traces.
7. Verify failed stations do not terminate the dataset run.

## Suggested structure

- asdf_io.py
- geometry.py
- preprocessing.py
- arrivals.py
- stacking.py
- depth_recovery.py
- corrections.py
- diagnostics.py
- cli.py
- example_config.yaml
- README.md
- tests/

Use a small package or adapt an existing repository; avoid unnecessary
frameworks.

## Outputs

- Configuration and provenance.
- Per-trace metadata, selection flags, and rejection reasons.
- Saved vespagrams and depth stacks.
- Per-bin recovered depths and coverage statistics.
- Bootstrap results.
- Paired model comparisons.
- Diagnostic plots and a short validation report.

Support restarting from saved preprocessing results.

## First milestone

Process one event, inspect SS alignment, and validate the stacker on controlled
moveout. Then recover the known 1D interfaces in a few well-sampled bins.

Do not begin global interpretation before these checks pass.

## References to inspect

- Bai, Zhang and Ritsema (2012), An analysis of SS precursors using
  3D spectral-element method seismograms.
  https://doi.org/10.1111/j.1365-246X.2011.05256.x
- Do double “SS precursors” mean double discontinuities?
  https://doi.org/10.1111/j.1365-246X.2012.05683.x
- High resolution upper mantle discontinuity images across the Pacific Ocean
  from SS precursors using local slant stack filters:
  https://academic.oup.com/gji/article/202/1/175/583675

Read relevant methods before claiming exact reproduction. Identify which
paper defines the baseline and document deviations.
