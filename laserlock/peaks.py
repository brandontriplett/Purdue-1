"""Sub-Doppler peak finding for one Doppler well.

The well contains three peaks, left to right: small, taller, tallest.
Selection is by that left-to-right index (1, 2, or 3), not by raw prominence.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter


def _odd_window(value, n, minimum=5):
    value = int(value)
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    if value >= n:
        value = n - 1 if n % 2 == 0 else n
    if value < 5:
        raise ValueError("Trace is too short for the requested smoothing.")
    return value


def _refine_quadratic(x, y, peak_index, half_window=12):
    lo = max(0, peak_index - half_window)
    hi = min(len(x), peak_index + half_window + 1)
    xx = np.asarray(x[lo:hi], dtype=float)
    yy = np.asarray(y[lo:hi], dtype=float)
    if len(xx) < 5:
        return float(x[peak_index])
    try:
        a, b, _c = np.polyfit(xx, yy, 2)
        if a >= 0:
            return float(x[peak_index])
        vertex = -b / (2 * a)
        if xx.min() <= vertex <= xx.max():
            return float(vertex)
    except Exception:
        pass
    return float(x[peak_index])


def extract_residual(
    x,
    signal,
    median_size=21,
    background_frac=0.32,
    cleanup_frac=0.006,
    peak_polarity="positive",
    narrow_span_v=0.12,
):
    """Remove the Doppler envelope. Returns background, residual, search signal.

    A full Doppler well uses a wide Savitzky–Golay smoother. A narrow fine
    window is locally almost linear, so a 2nd-order smoother over 32 % of
    the trace would swallow the sub-Doppler line itself.
    """
    x = np.asarray(x, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(x) != len(signal):
        raise ValueError("x and signal must have equal lengths.")
    n = len(signal)
    if n < 50:
        raise ValueError("Signal is too short.")

    median_size = _odd_window(median_size, n, minimum=5)
    cleanup_window = _odd_window(max(11, int(round(cleanup_frac * n))), n, minimum=11)

    despiked = median_filter(signal, size=median_size)
    span = abs(float(x[-1] - x[0]))
    if span <= narrow_span_v:
        # Fine / cropped window: fit a line, not a Doppler-scale smoother.
        coeff = np.polyfit(x, despiked, 1)
        background = np.polyval(coeff, x)
    else:
        background_window = _odd_window(int(background_frac * n), n, minimum=101)
        background = savgol_filter(despiked, window_length=background_window, polyorder=2)
    residual = signal - background
    clean = savgol_filter(residual, window_length=cleanup_window, polyorder=2)
    search = -clean if peak_polarity == "negative" else clean
    return {
        "background": background,
        "residual": residual,
        "clean": clean,
        "search": search,
        "x": x,
        "signal": signal,
    }


def find_all_peaks(
    x,
    search,
    signal,
    min_distance_v=0.008,
    prominence_sigma=1.0,
    quadratic_half_window=12,
):
    """Peak list sorted by voltage (left to right)."""
    x = np.asarray(x, dtype=float)
    search = np.asarray(search, dtype=float)
    span = abs(float(x[-1] - x[0])) or 1.0
    dx = span / max(len(x) - 1, 1)
    distance = max(3, int(round(min_distance_v / dx)))
    noise = float(np.median(np.abs(search - np.median(search)))) + 1e-12
    prominence = prominence_sigma * 1.4826 * noise

    idx, props = find_peaks(search, prominence=prominence, distance=distance)
    peaks = []
    for i, prom in zip(idx, props["prominences"]):
        peaks.append({
            "index": int(i),
            "voltage": _refine_quadratic(x, search, int(i), quadratic_half_window),
            "prominence": float(prom),
            "height": float(search[i]),
            "raw_signal": float(signal[i]),
        })
    peaks.sort(key=lambda p: p["voltage"])
    return peaks


def select_triplet(
    peaks,
    rel_prominence=0.12,
    cluster_window_v=0.08,
    n_peaks=3,
):
    """Return n_peaks left-to-right in the cluster around the strongest line.

    None if a clear triplet is not present.
    """
    if len(peaks) < n_peaks:
        return None

    strongest = max(peaks, key=lambda p: p["prominence"])
    floor = rel_prominence * strongest["prominence"]
    nearby = [
        p for p in peaks
        if abs(p["voltage"] - strongest["voltage"]) <= cluster_window_v
        and p["prominence"] >= floor
    ]
    if len(nearby) < n_peaks:
        return None
    nearby.sort(key=lambda p: p["prominence"], reverse=True)
    triplet = sorted(nearby[:n_peaks], key=lambda p: p["voltage"])
    for i, p in enumerate(triplet, start=1):
        p["label"] = i
    return triplet


def choose_peak(triplet, choice=3):
    """choice is 1-based left-to-right (1=left, 2=middle, 3=right)."""
    if not triplet:
        raise RuntimeError("No peak triplet to choose from.")
    if choice < 1 or choice > len(triplet):
        raise ValueError(f"PEAK_CHOICE must be in 1..{len(triplet)}, got {choice}.")
    return triplet[choice - 1]


def significant_peaks(peaks, rel_prominence=0.12):
    """Peaks above a fraction of the strongest, left to right."""
    if not peaks:
        return []
    strongest = max(peaks, key=lambda p: p["prominence"])
    floor = rel_prominence * strongest["prominence"]
    vis = [p for p in peaks if p["prominence"] >= floor]
    vis.sort(key=lambda p: p["voltage"])
    for i, p in enumerate(vis, start=1):
        p["label"] = i
    return vis


def pick_visible_peak(peaks, peak_choice, rel_prominence=0.12, hint_voltage=None):
    """Select among whatever lines are in this trace (fine scan may crop the triplet).

    1 → leftmost, 2 → middle of the visible set, 3 → tallest remaining.
    If ``hint_voltage`` is set (the coarse lock coordinate), pick the
    significant line nearest that voltage. That is the right fallback
    when the fine window is centered on the chosen peak.
    """
    vis = significant_peaks(peaks, rel_prominence=rel_prominence)
    if not vis:
        return None, []
    if hint_voltage is not None:
        chosen = min(vis, key=lambda p: abs(p["voltage"] - float(hint_voltage)))
    elif peak_choice <= 1:
        chosen = vis[0]
    elif peak_choice >= 3:
        chosen = max(vis, key=lambda p: p["prominence"])
    else:
        chosen = vis[len(vis) // 2]
    return chosen, vis


def _triplet_is_plausible(triplet):
    """True if the cluster looks like small / taller / tallest, left to right."""
    if not triplet or len(triplet) != 3:
        return False
    proms = [p["prominence"] for p in triplet]
    return proms[-1] == max(proms)


def analyze_spectrum(
    x,
    signal,
    peak_choice=3,
    n_peaks=3,
    rel_prominence=0.12,
    cluster_window_v=0.08,
    min_distance_v=0.008,
    peak_polarity="positive",
    require_triplet=True,
    hint_voltage=None,
):
    """Full pipeline.

    Coarse scans should use require_triplet=True.
    Fine scans should use require_triplet=False so a cropped window still
    yields a lock coordinate from the visible line(s). Pass hint_voltage
    (the coarse target) so a cropped fine window locks to the same line.
    """
    extracted = extract_residual(x, signal, peak_polarity=peak_polarity)
    peaks = find_all_peaks(
        extracted["x"],
        extracted["search"],
        extracted["signal"],
        min_distance_v=min_distance_v,
    )
    triplet = select_triplet(
        peaks,
        rel_prominence=rel_prominence,
        cluster_window_v=cluster_window_v,
        n_peaks=n_peaks,
    )
    use_triplet = bool(triplet) and (
        require_triplet or _triplet_is_plausible(triplet)
    )
    visible = []
    if use_triplet:
        target = choose_peak(triplet, peak_choice)
        ok = True
        visible = triplet
    elif require_triplet:
        target = None
        ok = False
        triplet = None
    else:
        # Cropped window, or a 3-peak cluster that is not the spectroscopic
        # triplet (background artifacts). Do not use that cluster as 1/2/3.
        triplet = None
        target, visible = pick_visible_peak(
            peaks,
            peak_choice,
            rel_prominence=rel_prominence,
            hint_voltage=hint_voltage,
        )
        ok = target is not None
    return {
        **extracted,
        "peaks": peaks,
        "triplet": triplet,
        "visible": visible,
        "target": target,
        "ok": ok,
        "peak_choice": peak_choice,
        "require_triplet": require_triplet,
        "hint_voltage": hint_voltage,
    }


def plot_peak_analysis(x, signal, analysis, title="Peak extraction"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x, signal, label="Raw PD")
    axes[0].plot(x, analysis["background"], label="Doppler / background")
    axes[0].set_ylabel("PD (V)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, analysis["search"], label="Background-subtracted")
    axes[1].axhline(0, linestyle=":", linewidth=1)
    labeled = analysis.get("triplet") or analysis.get("visible") or []
    if labeled:
        px = [p["voltage"] for p in labeled]
        py = np.interp(px, x, analysis["search"])
        axes[1].plot(px, py, "o", markersize=8, label="Selected lines")
        for p in labeled:
            axes[1].annotate(
                str(p.get("label", "")),
                (p["voltage"], np.interp(p["voltage"], x, analysis["search"])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )
    axes[1].set_ylabel("Residual (V)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, signal, label="Raw PD")
    for p in labeled:
        axes[2].axvline(p["voltage"], linestyle=":", linewidth=1, alpha=0.7)
    target = analysis.get("target")
    if target is not None:
        axes[2].axvline(
            target["voltage"],
            linestyle="--",
            linewidth=1.6,
            label=f"Peak {analysis['peak_choice']} = {target['voltage']:+.6f} V",
        )
    axes[2].set_xlabel("Laser modulation voltage (V)")
    axes[2].set_ylabel("PD (V)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
