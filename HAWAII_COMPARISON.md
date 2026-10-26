# Hawaii: SY versus all stations

Main code: `scripts/hawaii_circular_comparison.py` and `scripts/hawaii_red_blue.py`, using the numerical functions in `ss_precursor/`.

The center is 19.5 N, 155.5 W, radius 900 km. Both PREM S400S and S670S bounce points must be inside the cap; distances are 100–160 degrees. SY-only and all-network populations use matching PREM/GLAD records from the PREM event list.

Processing: 10–100 s, 2-corner forward–reverse filtering, theoretical PREM SS timing for both models, individual model SS amplitude normalization, and PREM polarity multipliers for both. Measured SS peaks set amplitude and sign only. Original waveform timestamps, including the +5 s start, are honored. Vespagrams use linear moveout without phase-specific curvature correction.

## Run

Install with `python -m pip install -e .` in an environment supporting the dependencies in `pyproject.toml`.

These scripts require the existing event plans and metadata under `outputs/prem_event_reference/` and `outputs/glad_m35_event_reference/`, plus the referenced ASDF inputs. Prepare those using `EVENT_REFERENCE_DATABASE.md`. Plans contain local paths; update or regenerate them after moving data. Input data and generated outputs are excluded from the repository.

```bash
python scripts/hawaii_circular_comparison.py
python scripts/hawaii_red_blue.py
```

Outputs: `outputs/Hawaii_900km_comparison/`. Six worker processes calculate independent ray geometries. Existing selection and waveform arrays are reused; use a fresh output directory when changing processing or selection settings.

Red–blue plots use fixed 1-degree distance means divided by trace support, with common amplitude scales. Empty bins are gray. Wiggle record sections plot individual waveforms without distance averaging.

Dense arrays may concentrate distances and overweight a few events in the equal-trace stack. More traces need not improve slowness resolution.
