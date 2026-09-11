"""Lock-point from a *finished* error sweep (Linien simple autolock).

Port of ``get_lock_point`` in linien-org/linien
``linien-common/linien_common/common.py``: in a window that contains both
extrema of the target S-curve, lock at the mid-height sample between min
and max. We add an automatic window picker for the first S-curve from
the high-voltage end of a recorded trace (works for uphill or downhill).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def _odd_window(n, frac=0.03, minimum=7, maximum=51):
    w = int(frac * n)
    w = max(w, minimum)
    w = min(w, maximum, n if n % 2 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def _plateau_fraction(error, i_a, i_b, frac=0.15):
    """Fraction of samples between extrema that sit near zero.

    A real S-curve is a steep slope (only the zero-crossing is near
    baseline). The gap from one line's max to the next line's min is a
    long plateau, so this fraction is large.
    """
    a, b = sorted((int(i_a), int(i_b)))
    if b - a < 2:
        return 1.0
    seg = np.asarray(error[a : b + 1], dtype=float)
    amp = max(abs(float(seg[0])), abs(float(seg[-1])), 1e-12)
    return float(np.mean(np.abs(seg) < frac * amp))


def _pair_slope(voltage, error, i_a, i_b):
    dv = abs(float(voltage[i_b] - voltage[i_a]))
    if dv < 1e-12:
        return 0.0
    return abs(float(error[i_b] - error[i_a])) / dv


def get_lock_point(voltage, error, i0, i1):
    """Linien-style lock point inside ``error[i0:i1+1]``.

    Returns lock voltage, mid-height of the two extrema (Doppler offset),
    whether error *rises with voltage*, and the min/max/zero indices.
    """
    v = np.asarray(voltage, dtype=float)
    e = np.asarray(error, dtype=float)
    if len(v) != len(e):
        raise ValueError("voltage and error must have equal length.")
    lo, hi = (int(i0), int(i1)) if i0 <= i1 else (int(i1), int(i0))
    hi = min(hi, len(e) - 1)
    lo = max(lo, 0)
    if hi - lo < 3:
        raise ValueError("Lock window is too short.")

    crop = e[lo : hi + 1]
    min_local = int(np.argmin(crop))
    max_local = int(np.argmax(crop))
    mean_signal = 0.5 * (float(crop[min_local]) + float(crop[max_local]))
    a, b = sorted((min_local, max_local))
    slope = crop[a : b + 1] - mean_signal
    zero_local = a + int(np.argmin(np.abs(slope)))
    min_idx = lo + min_local
    max_idx = lo + max_local
    zero_idx = lo + zero_local
    slope_rising_vs_voltage = bool(v[max_idx] > v[min_idx])
    return {
        "lock_voltage": float(v[zero_idx]),
        "mean_error": float(mean_signal),
        "slope_rising_vs_voltage": slope_rising_vs_voltage,
        "min_idx": min_idx,
        "max_idx": max_idx,
        "zero_idx": zero_idx,
        "i0": lo,
        "i1": hi,
        "span_v": abs(float(v[max_idx] - v[min_idx])),
    }


def apply_lockpoint_correction(lp, correction_v=0.0, lock_min_v=None, lock_max_v=None):
    """Add a DC offset to a fitted lock voltage.

    Example: fit ``0.232`` V, ``correction_v=+0.005`` → park at ``0.237`` V.
    """
    out = dict(lp)
    fitted = float(out["lock_voltage"])
    corr = float(correction_v)
    parked = fitted + corr
    if lock_min_v is not None:
        parked = max(float(lock_min_v), parked)
    if lock_max_v is not None:
        parked = min(float(lock_max_v), parked)
    out["lock_voltage_fit"] = fitted
    out["lockpoint_correction_v"] = corr
    out["lock_voltage"] = parked
    return out


def _scurve_candidates(
    voltage,
    error,
    min_span_v=0.0015,
    max_span_v=0.018,
    rel_prominence=0.15,
    max_plateau=0.35,
    rel_slope=0.40,
):
    """Valid S-curve min/max pairs, high-V line first.

    Consecutive opposite extrema are *not* enough: the max of one line and
    the min of the next line are also opposite, and that inter-peak valley
    is wider and flatter than a real S-curve. Keep a pair only if its
    voltage span is a linewidth (not a line spacing) and the samples
    between the lobes are a steep slope rather than a plateau.
    """
    v = np.asarray(voltage, dtype=float)
    e = np.asarray(error, dtype=float)
    if len(v) != len(e) or len(e) < 20:
        raise ValueError("Need a longer error trace.")

    w = _odd_window(len(e))
    try:
        smooth = savgol_filter(e, window_length=w, polyorder=2)
    except ValueError:
        smooth = e.copy()

    ptp = float(np.ptp(smooth))
    if ptp < 1e-9:
        raise RuntimeError("Error trace is flat.")
    prom = rel_prominence * ptp
    span = abs(float(v[-1] - v[0])) or 1.0
    dx = span / max(len(v) - 1, 1)
    distance = max(3, int(round(min_span_v / dx)))

    hi, _ = find_peaks(smooth, prominence=prom, distance=distance)
    lo, _ = find_peaks(-smooth, prominence=prom, distance=distance)
    extrema = [(int(i), "max") for i in hi] + [(int(i), "min") for i in lo]
    extrema.sort(key=lambda t: float(v[t[0]]), reverse=True)
    if len(extrema) < 2:
        raise RuntimeError("No S-curve extrema found in the error trace.")

    pairs = []
    for (i_a, t_a), (i_b, t_b) in zip(extrema, extrema[1:]):
        if t_a == t_b:
            continue
        span_ab = abs(float(v[i_a] - v[i_b]))
        pairs.append(
            {
                "i0": int(i_a),
                "i1": int(i_b),
                "span_v": span_ab,
                "slope": _pair_slope(v, smooth, i_a, i_b),
                "plateau": _plateau_fraction(smooth, i_a, i_b),
                "v_center": 0.5 * (float(v[i_a]) + float(v[i_b])),
            }
        )
    if not pairs:
        raise RuntimeError("Could not pair min/max lobes for the first S-curve.")

    compact = [
        p
        for p in pairs
        if min_span_v <= p["span_v"] <= max_span_v and p["plateau"] <= max_plateau
    ]
    if compact:
        peak_slope = max(p["slope"] for p in compact)
        chosen = [p for p in compact if p["slope"] >= rel_slope * peak_slope]
        if chosen:
            return chosen

    # Span window missed every pair (unusual linewidth). Take the steepest
    # non-plateau pair so we still lock to a line, not a valley.
    steep = [p for p in pairs if p["plateau"] <= max_plateau]
    if steep:
        return [max(steep, key=lambda p: p["slope"])]
    return [min(pairs, key=lambda p: p["span_v"])]


def first_scurve_window(
    voltage,
    error,
    min_span_v=0.0015,
    max_span_v=0.018,
    rel_prominence=0.15,
    max_plateau=0.35,
    rel_slope=0.40,
):
    """Indices of the first S-curve from the high-voltage end of the trace."""
    cands = _scurve_candidates(
        voltage,
        error,
        min_span_v=min_span_v,
        max_span_v=max_span_v,
        rel_prominence=rel_prominence,
        max_plateau=max_plateau,
        rel_slope=rel_slope,
    )
    return cands[0]["i0"], cands[0]["i1"]


def lock_point_from_trace(
    voltage,
    error,
    hint_voltage=None,
    peak_choice=None,
    **window_kw,
):
    """Lock point on one S-curve of a finished error trace.

    Default (no hint, no choice): first S-curve from the high-V end
    (rightmost line). ``hint_voltage`` picks the S-curve whose min/max
    midpoint is closest to that voltage (use the survey target).
    ``peak_choice`` is 1-based left-to-right among the detected S-curves
    and is used only when ``hint_voltage`` is omitted.
    """
    cands = _scurve_candidates(voltage, error, **window_kw)
    if hint_voltage is not None:
        pick = min(cands, key=lambda p: abs(p["v_center"] - float(hint_voltage)))
        method = "hint"
    elif peak_choice is not None:
        ltr = sorted(cands, key=lambda p: p["v_center"])
        idx = min(max(int(peak_choice) - 1, 0), len(ltr) - 1)
        pick = ltr[idx]
        method = f"choice_{int(peak_choice)}"
    else:
        pick = cands[0]
        method = "first_scurve"
    result = get_lock_point(voltage, error, pick["i0"], pick["i1"])
    result["window"] = (pick["i0"], pick["i1"])
    result["method"] = method
    result["v_center"] = float(pick["v_center"])
    result["n_scurves"] = len(cands)
    result["scurve_centers"] = [float(p["v_center"]) for p in sorted(cands, key=lambda p: p["v_center"])]
    return result


def error_is_bipolar(error, frac=0.25):
    """True if the error has both a real max lobe and a real min lobe."""
    e = np.asarray(error, dtype=float)
    if e.size < 8:
        return False
    amp = max(abs(float(e.max())), abs(float(e.min())), 1e-12)
    return float(e.max()) >= frac * amp and float(e.min()) <= -frac * amp


def lock_point_from_single_line(
    voltage, error, max_span_v=0.018, hint_voltage=None
):
    """Linien mid-height on a trace that should contain one S-curve.

    Fine scans are already cropped around one line, so using the whole
    trace avoids a peak-picker miss. If min/max are farther apart than
    a linewidth (a neighbor leaked into the window), pick the S-curve
    nearest ``hint_voltage`` instead of the rightmost line.
    """
    e = np.asarray(error, dtype=float)
    lp = get_lock_point(voltage, e, 0, len(e) - 1)
    lp["window"] = (lp["i0"], lp["i1"])
    lp["method"] = "full_trace"
    if lp["span_v"] <= float(max_span_v):
        return lp
    try:
        lp2 = lock_point_from_trace(voltage, e, hint_voltage=hint_voltage)
        return lp2
    except Exception:
        return lp
