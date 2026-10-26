"""Save controlled numerical recovery evidence independently of ASDF inputs."""
from pathlib import Path
import numpy as np
from .stacking import slant_stack
from .depth_recovery import depth_stack
from .pipeline import dump_json


def controlled_validation(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def pulse(t, center):
        x = (t-center)/3
        return (1-x*x)*np.exp(-x*x/2)
    time = np.arange(-240, -60, 0.5)
    distances = np.linspace(95, 155, 25)
    intercept, true_p = -150.25, -0.45
    traces = [pulse(time, intercept+true_p*(d-125)) for d in distances]
    tau = np.arange(-180, -120, 0.25)
    ps = np.arange(-0.8, -0.1, 0.025)
    result = slant_stack([time]*len(traces), traces, distances, tau, ps, 125)
    i, j = np.unravel_index(np.nanargmax(result['stack']), result['stack'].shape)
    depths = np.arange(350, 451, 0.5)
    predictions = -150-0.3*(depths[None, :]-400)+0.1*(distances[:, None]-125)
    shifted = [pulse(time, -150+0.1*(d-125)+6) for d in distances]
    raw = depth_stack([time]*len(traces), shifted, depths, predictions)
    corrected = depth_stack([time]*len(traces), shifted, depths, predictions, np.full_like(predictions, 6))
    recovered_raw = float(depths[np.argmax(raw['stack'])])
    recovered_corrected = float(depths[np.argmax(corrected['stack'])])
    report = {'true_moveout_s_degree': true_p, 'recovered_moveout_s_degree': float(ps[i]),
              'true_intercept_s': intercept, 'recovered_intercept_s': float(tau[j]),
              'moveout_tolerance_s_degree': 0.025, 'intercept_tolerance_s': 0.25,
              'imposed_differential_shift_s': 6, 'controlled_true_depth_km': 400,
              'uncorrected_controlled_depth_km': recovered_raw, 'corrected_controlled_depth_km': recovered_corrected,
              'depth_tolerance_km': 0.5,
              'note': 'Artificial Gaussian-derivative arrivals and analytic depth/time mapping; not real PREM interface recovery.'}
    report['passed'] = bool(abs(ps[i]-true_p) <= 0.025 and abs(tau[j]-intercept) <= 0.25 and abs(recovered_corrected-400) <= 0.5)
    dump_json(output/'controlled_validation.json', report)
    np.savez_compressed(output/'controlled_vespagram.npz', **result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].pcolormesh(tau, ps, result['stack'], shading='auto', cmap='RdBu_r')
    axes[0].plot(intercept, true_p, 'kx', label='Known pulse')
    axes[0].set(xlabel='Relative time (s)', ylabel='Relative moveout (s/degree)', title='Controlled linear stack')
    axes[0].legend()
    axes[1].plot(depths, raw['stack'], label='Imposed +6 s shift')
    axes[1].plot(depths, corrected['stack'], label='Sample prediction +6 s')
    axes[1].axvline(400, color='k', ls=':', label='Controlled true depth')
    axes[1].set(xlabel='Artificial depth coordinate (km)', ylabel='Amplitude', title='Correction sign validation')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'controlled_validation.png', dpi=160); plt.close(fig)
    return report
