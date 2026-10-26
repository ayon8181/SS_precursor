# Vespagrams for circular clusters

```bash
cd ~/WORK/SS_precursor/SS_precursor
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m ss_precursor.cap_vespagrams \
  --caps outputs/circular_caps_radius_comparison \
  --output outputs/circular_cap_vespagrams_repeat
```

Use your ASDF Python environment. The output folder must be new. By default,
this plots the three example centres (S511, S545, S579), both precursor phases,
and all four saved radii. Waveforms and exact cap memberships are reused. For
another saved centre or a subset of radii:

```bash
python -m ss_precursor.cap_vespagrams \
  --caps outputs/circular_caps_radius_comparison \
  --centers S511 --radii-km 900 \
  --output outputs/cap_S511_900km_vespagrams
```

The scanned residual slowness is -2 to +2 s/degree with a 0.025 step; time is
-40 to +40 seconds relative to each predicted precursor, at 0.5-second spacing.
Each query uses the cached, source-depth-specific PREM phase-minus-SS prediction
plus offset plus residual slowness times (distance minus reference distance).
The reference distance is 130 degrees for every radius and centre; use
`--reference-distance` to change it. The vertical axis is **residual slowness**,
not absolute ray parameter. Zero residual follows the exact model curve.

SS normalization/polarity are retained. Signed linear stacks are saved; contours
show their Hilbert envelope, normalized once per panel. No row normalization or
slowness smoothing is applied. Stars mark model predictions, not picked peaks.
Panels cannot be compared as precursor/SS amplitudes; use signed NPZ arrays for
that. Samples outside each trace's predicted phase +/-60-second input window are
excluded, and pixels with less than 80% of the cap's waveforms (or fewer than 3)
are masked. White areas can therefore mean insufficient support or values below
the 0.2 envelope display threshold, rather than absent signal.

Outputs include:

- `S511_vespagrams.png/pdf` (and S545/S579): two phases by four radii.
- `S511_semblance.png/pdf` (and S545/S579): corresponding semblance, all on 0–1.
- `reference_radius_comparison.png/pdf`: the selected centres together at the
  available radius nearest 900 km.
- One NPZ per centre/phase/radius: signed stack, semblance, support/weight sums,
  residual slowness and time coordinates. Arrays have slowness by time ordering.
- `provenance.json`: exact global member indices, selection identities, settings,
  aperture and amplitude scales for each panel. Member indices refer to the
  source caps' `waveform_members.json`.

The zero-residual row is the same model-aligned signed stack saved by the
circular-stack comparison, restricted to the plotted offsets. Cap radii overlap;
their panels are correlated. Broad distance apertures can produce beams narrow
in slowness. Shape and darkness alone do not establish a precursor detection.
