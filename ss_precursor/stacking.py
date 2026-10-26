"""Linear stacks; p is precursor-minus-SS moveout in seconds/degree."""
import numpy as np


def sample_trace(time, data, query):
    """Linear interpolation without extrapolation or bridging missing samples."""
    time, data, query = map(np.asarray, (time, data, query))
    if time.ndim != 1 or data.shape != time.shape or len(time) < 2:
        raise ValueError("Trace needs matching one-dimensional time/data arrays")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Trace time must be finite and strictly increasing")
    return np.interp(query, time, data, left=np.nan, right=np.nan)


def weighted_samples(samples, weights=None):
    """Normalize each output sample by its own valid positive-weight support."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or not len(samples):
        raise ValueError("Expected nonempty trace-by-sample matrix")
    w = np.ones(len(samples)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(samples),) or np.any(~np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Weights must be finite, nonnegative, and not all zero")
    valid = np.isfinite(samples) & (w[:, None] > 0)
    effective = valid * w[:, None]
    x = np.where(valid, samples, 0)
    den = effective.sum(axis=0)
    num = (x * effective).sum(axis=0)
    energy = (x*x*effective).sum(axis=0)
    stack = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    semblance = np.divide(num*num, den*energy, out=np.full_like(num, np.nan), where=(den*energy) > 0)
    return {"stack": stack, "semblance": semblance, "count": valid.sum(axis=0),
            "weight_sum": den}


def slant_stack(times, traces, distances, tau, slownesses, reference_distance=None, weights=None):
    """V(tau,p)=sum w_i*u_i[tau+p*(Delta_i-Delta_0)] / sum(valid w_i)."""
    distances = np.asarray(distances, dtype=float)
    if len(traces) != len(times) or len(traces) != len(distances) or not len(traces):
        raise ValueError("Mismatched or empty traces, times, and distances")
    if not np.all(np.isfinite(distances)):
        raise ValueError("Distances must be finite")
    ref = float(np.mean(distances) if reference_distance is None else reference_distance)
    rows = []
    for p in slownesses:
        samples = [sample_trace(t, u, np.asarray(tau) + p*(d-ref))
                   for t, u, d in zip(times, traces, distances)]
        rows.append(weighted_samples(samples, weights))
    if not rows:
        raise ValueError("Slowness grid is empty")
    return {**{key: np.asarray([r[key] for r in rows]) for key in rows[0]},
            "tau": np.asarray(tau), "slownesses": np.asarray(slownesses),
            "reference_distance": ref}


def predicted_stack(times, traces, predicted_relative_times, offsets, weights=None):
    if not (len(times) == len(traces) == len(predicted_relative_times)):
        raise ValueError("Predictions must match traces")
    return weighted_samples([sample_trace(t, u, np.asarray(offsets)+arrival)
                             for t, u, arrival in zip(times, traces, predicted_relative_times)], weights)
