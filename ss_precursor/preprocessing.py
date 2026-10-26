"""Conservative configurable transverse processing, with explicit timing."""
import numpy as np
from scipy.signal import detrend, sosfiltfilt, butter
from scipy.signal.windows import tukey


def transverse(stream, backazimuth, convention):
    """Preserve existing T, or rotate matched NE using ObsPy's NE->RT convention."""
    if convention == "RT":
        t, r = stream.select(component="T"), stream.select(component="R")
        if len(t) != 1 or len(r) != 1:
            raise ValueError("Expected one R and one T trace for explicit RT convention")
        pair = (r[0], t[0])
    elif convention == "NE":
        n, e = stream.select(component="N"), stream.select(component="E")
        if len(n) != 1 or len(e) != 1:
            raise ValueError("Expected exactly one N and one E trace")
        pair = (n[0], e[0])
    else:
        raise ValueError("Configure horizontal_convention as RT or NE")
    a, b = pair
    if (a.stats.starttime != b.stats.starttime or a.stats.delta != b.stats.delta or len(a) != len(b)
            or a.stats.location != b.stats.location or a.stats.channel[:-1] != b.stats.channel[:-1]):
        raise ValueError("Horizontal traces have incompatible timing or channel identifiers")
    if convention == "RT":
        return b.copy()
    rotated = stream.__class__([a.copy(), b.copy()]).rotate("NE->RT", back_azimuth=backazimuth)
    return rotated.select(component="T")[0]


def align(trace, origin, predicted_ss, cfg):
    dt = trace.stats.delta
    periods = cfg['period_band_s']
    if not (cfg['minimum_synthetic_period_s'] <= periods[0] < periods[1]):
        raise ValueError("Period band exceeds validated synthetic bandwidth")
    if 1/periods[0] >= 0.5/dt:
        raise ValueError("Filter exceeds Nyquist frequency")
    u = np.asarray(trace.data, dtype=float)
    if not np.all(np.isfinite(u)) or len(u) < 10:
        raise ValueError("Trace has missing/nonfinite samples or insufficient data")
    u = detrend(u, type=cfg['detrend']) * tukey(len(u), alpha=2*cfg['taper_fraction_per_end'])
    sos = butter(cfg['filter_order'], [1/periods[1], 1/periods[0]], fs=1/dt, btype='bandpass', output='sos')
    pad = int(round(cfg['filter_padding_s']/dt))
    u = sosfiltfilt(sos, np.pad(u, pad, mode='reflect'))
    if pad:
        u = u[pad:-pad]
    time = float(trace.stats.starttime-origin.time) + np.arange(len(u))*dt
    lo, hi = predicted_ss + np.asarray(cfg['ss_search_window_s'])
    margin = cfg['edge_guard_s']
    if lo < time[0]+margin or hi > time[-1]-margin:
        raise ValueError("SS search overlaps trace boundary/filter edge guard")
    index = np.flatnonzero((time >= lo) & (time <= hi))
    k = int(index[np.argmax(np.abs(u[index]))])
    if k in (index[0], index[-1]):
        raise ValueError("SS maximum is at search boundary")
    measured = float(time[k])
    # Refine an absolute-amplitude extremum by a three-sample parabola.
    y = np.abs(u[k-1:k+2]); den = y[0]-2*y[1]+y[2]
    shift = 0 if den == 0 else float(np.clip(0.5*(y[0]-y[2])/den, -0.5, 0.5))
    measured += shift*dt
    amp = float(abs(u[k])-0.25*(y[0]-y[2])*shift)
    if amp <= max(cfg['minimum_ss_amplitude_mm'], cfg['minimum_ss_fraction_of_trace_peak']*np.max(np.abs(u))):
        raise ValueError("SS amplitude is too small for stable normalization")
    sign = float(np.sign(u[k])) if cfg['polarity'] == 'ss_positive' else 1.0
    if cfg['polarity'] not in ('ss_positive', 'preserve'):
        raise ValueError("Unsupported polarity convention")
    valid = (time >= time[0]+margin) & (time <= time[-1]-margin)
    return time[valid]-measured, (u/amp*sign)[valid], {
        "predicted_ss_s": predicted_ss, "measured_ss_s": measured,
        "alignment_residual_s": measured-predicted_ss, "ss_amplitude_mm": amp,
        "polarity_multiplier": sign, "input_delta_s": dt,
        "trace_start_from_origin_s": float(trace.stats.starttime-origin.time)}
