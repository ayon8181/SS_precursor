"""Depth sampling consumes genuine phase-specific trial-model predictions.

The caller must generate predictions in models containing each trial interface;
this module deliberately does not approximate depth changes by phase relabeling.
"""
import numpy as np
from .stacking import sample_trace, weighted_samples


def depth_stack(times, traces, trial_depths, relative_predictions, differential_corrections=None, weights=None):
    """Sample tau_i(h)+c_i(h); positive c means later precursor relative to SS.

    Predictions and corrections are trace-by-depth matrices in seconds.
    Missing predictions/corrections are excluded, never replaced by zero.
    """
    depths = np.asarray(trial_depths, dtype=float)
    predicted = np.asarray(relative_predictions, dtype=float)
    if depths.ndim != 1 or not len(depths) or np.any(np.diff(depths) <= 0) or not np.all(np.isfinite(depths)):
        raise ValueError('Depth grid must be finite and strictly increasing')
    if predicted.shape != (len(traces), len(depths)) or len(times) != len(traces):
        raise ValueError('Predictions must have shape (trace_count, depth_count)')
    corrections = np.zeros_like(predicted) if differential_corrections is None else np.asarray(differential_corrections, dtype=float)
    if corrections.shape != predicted.shape:
        raise ValueError('Corrections must match prediction shape')
    samples = [sample_trace(t, u, q) for t, u, q in zip(times, traces, predicted+corrections)]
    return {'depths_km': depths, **weighted_samples(samples, weights)}
