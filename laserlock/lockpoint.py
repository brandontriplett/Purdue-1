"""Lock-point from a *finished* error sweep (Linien simple autolock).

Port of ``get_lock_point`` in linien-org/linien
``linien-common/linien_common/common.py``: in a window that contains both
extrema of the target S-curve, lock at the mid-height sample between min
and max. We add an automatic window picker for the first S-curve from
the high-voltage end of a downhill trace.
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
    }


def first_scurve_window(
    voltage,
    error,
    min_span_v=0.004,
    max_span_v=0.040,
    rel_prominence=0.25,
):
    """Indices of the first S-curve from the high-voltage end of a downhill trace.

    Downhill recording: index 0 is the highest voltage. Extrema are scanned
    in that order so the rightmost spectroscopic line is first.
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
    extrema.sort(key=lambda t: t[0])
    if len(extrema) < 2:
        raise RuntimeError("No S-curve extrema found in the error trace.")

    for (i_a, t_a), (i_b, t_b) in zip(extrema, extrema[1:]):
        if t_a == t_b:
            continue
        span_ab = abs(float(v[i_a] - v[i_b]))
        if min_span_v <= span_ab <= max_span_v:
            return int(i_a), int(i_b)

    # Fall back to the first opposite pair even if slightly wide/narrow.
    for (i_a, t_a), (i_b, t_b) in zip(extrema, extrema[1:]):
        if t_a != t_b:
            return int(i_a), int(i_b)
    raise RuntimeError("Could not pair min/max lobes for the first S-curve.")


def lock_point_from_trace(voltage, error, **window_kw):
    """Automatic lock point: first S-curve from the high-V end, then Linien mid-height."""
    i0, i1 = first_scurve_window(voltage, error, **window_kw)
    result = get_lock_point(voltage, error, i0, i1)
    result["window"] = (i0, i1)
    return result
