"""Headless plots; diagnostic all-event stacks are not geographic depth estimates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_record_section(tau, data, rows, output):
    fig, ax = plt.subplots(figsize=(11, 8))
    order = np.argsort([r['distance_degree'] for r in rows])
    # Plot a deterministic subset for legibility; saved arrays contain every trace.
    order = order[np.linspace(0, len(order)-1, min(120, len(order)), dtype=int)]
    for i in order:
        row = rows[i]
        ax.plot(tau, row['distance_degree']+data[i]*0.7, color='0.3', lw=0.4)
    colors = ['C0', 'C1', 'C2']
    for k, phase in enumerate(rows[0]['arrivals']):
        x = [r['arrivals'][phase]['time_s']-r['measured_ss_s'] for r in rows]
        ax.scatter(x, [r['distance_degree'] for r in rows], s=9, marker='|', color=colors[k % 3], label=phase)
    phases = sorted({a['phase'] for r in rows for a in r['interference']})
    for phase in phases:
        xy = [(a['time_s']-r['measured_ss_s'], r['distance_degree']) for r in rows for a in r['interference'] if a['phase'] == phase]
        ax.scatter(*np.asarray(xy).T, s=5, marker='x', alpha=0.4, label=phase)
    ax.set(xlim=(tau[0], tau[-1]), xlabel='Time relative to measured SS (s)', ylabel='Epicentral distance (degrees)',
           title='Aligned T displacement / SS peak amplitude (all accepted stations)')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_stacks(result, predicted, offsets, output):
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), constrained_layout=True)
    mesh = axes[0].pcolormesh(result['tau'], result['slownesses'], result['stack'], shading='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    fig.colorbar(mesh, ax=axes[0], label='Normalized amplitude', extend='both')
    axes[0].set(ylabel='Relative moveout (s/degree)', title='Linear vespagram; color clipped at ±0.1 SS amplitude')
    mesh = axes[1].pcolormesh(result['tau'], result['slownesses'], result['count'], shading='auto')
    fig.colorbar(mesh, ax=axes[1], label='Trace count')
    axes[1].set(ylabel='Relative moveout (s/degree)', xlabel='Time relative to measured SS (s)', title='Contributing trace count')
    mesh = axes[2].pcolormesh(result['tau'], result['slownesses'], result['semblance'], shading='auto', vmin=0, vmax=1)
    fig.colorbar(mesh, ax=axes[2], label='Semblance')
    axes[2].set(ylabel='Relative moveout (s/degree)', xlabel='Time relative to measured SS (s)', title='Sample-wise weighted semblance; assess with trace count')
    for phase, stack in predicted.items():
        axes[3].plot(offsets, stack['stack'], label=phase)
    axes[3].set(xlabel='Time relative to predicted precursor-minus-SS time (s)', ylabel='Normalized amplitude', title='Phase-specific moveout stacks (diagnostic)')
    axes[3].legend()
    fig.savefig(output, dpi=160); plt.close(fig)
