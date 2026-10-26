# Temporal-window semblance check

The original vespagram code computes instantaneous weighted semblance. It uses
identical valid samples and weights in numerator and denominator, with no epsilon
added to the denominator. Its energy mask is only denominator > 0; it does not
perform temporal averaging or explicitly mask negligible positive energy.

This diagnostic leaves all signed stacks, their envelopes, memberships, and
moveout corrections unchanged. It recomputes shifted per-trace samples and saves
the sufficient statistics A=sum(w*x), E=sum(w*x*x), W=sum(w), and support counts.
Finite-window semblance is

```
S(t,p) = sum_window A(t,p)^2 / sum_window [W(t,p)*E(t,p)]
```

This sums the numerator and denominator before division. It does not average
instantaneous ratios, smooth a final image, or square a time-integrated waveform.
At every time sample, both terms use the same finite trace mask and positive
weights. Every sample in a full window must satisfy the original minimum support
(80% of the cap, at least 3 traces). Different valid sets can occur at different
time samples, but the numerator/denominator masks agree at each sample. Missing
window edges are masked; extra waveform samples are computed outside the plotted
time range rather than padding/reflection of an existing semblance image.

Negligible-energy windows are masked if sum(E)/sum(W) is below max(1e-20,
1e-6 times the panel's maximum per-sample weighted mean trace-square amplitude).
The same energy reference/floor is used for every window length in that panel.
No stabilizing constant is added to either term. The threshold is configurable
using `--energy-floor-fraction`.

A common dominant period is estimated independently from SS: average weighted
power spectra of the selected reference cap's SS windows (-60 to +60 s), demeaned
and Hann tapered, with the spectral peak restricted to the 25–100 s processing
band. Zero padding locates the peak; it does not increase spectral resolution.
This is a representative SS spectral period, not a precise precursor-period
measurement. The reference is S511/S400S-selected, 900 km. Its peak is about
47.6 s. The instantaneous, 0.25T, 0.5T and 1T windows use 1, 25, 49 and 97 samples
at 0.5 s spacing (nominal averaging durations 0.5, 12.5, 24.5 and 48.5 s).
Full averaging duration is sample count times sample spacing; the first-to-last
sample span is one spacing shorter. The first case is instantaneous.

```bash
cd ~/WORK/SS_precursor/SS_precursor
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m ss_precursor.semblance_check \
  --vespagrams outputs/circular_cap_vespagrams \
  --output outputs/temporal_semblance_repeat
```

Use the ASDF environment and a new output folder. The diagnostic reuses the exact
24 panels from the source vespagrams. Outputs include complete comparisons by
centre, compact two-phase comparisons at 900 km, the reference SS spectrum,
NPZ arrays containing the summed terms/ratios/energy and masks at each window,
and provenance with all settings and values at the model prediction. The common
0–1 plotting scale is unchanged. The instantaneous column includes the new
energy mask, and is checked numerically against the original wherever valid.

Lobe merging or attenuation under temporal averaging demonstrates sensitivity
to the coherence estimator. It does not identify separate reflections, and a
longer window mixes signal and surrounding coda. These comparisons alone do not
establish an optimal window length or a detection significance.

The completed comparison checks all 24 original panels. Recomputed instantaneous
values match the originals wherever valid. At S511/S670S/900 km, semblance at
the model prediction is approximately 0.637, 0.532, 0.387 and 0.283 across the
four windows. Small lobes merge substantially under quarter/half-period averaging.
Increasing the averaging duration also includes less coherent waveform portions;
it need not increase semblance. Energy-floor sensitivity was checked separately
on the +/-10 s interior, where the saved sample terms allow complete windows:
fractions 1e-8, 1e-6 and 1e-4 give identical valid masks and values in all 96
panel/window comparisons. Details are in `energy_threshold_sensitivity.json`.
