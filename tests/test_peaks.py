import numpy as np

from laserlock.peaks import analyze_spectrum


def _synthetic_well(n=4000):
    x = np.linspace(-0.4, 0.4, n)
    doppler = 0.45 - 0.12 * np.exp(-((x + 0.05) ** 2) / (2 * 0.12**2))
    peaks = (
        0.008 * np.exp(-((x + 0.15) ** 2) / (2 * 0.004**2))
        + 0.018 * np.exp(-((x + 0.12) ** 2) / (2 * 0.004**2))
        + 0.028 * np.exp(-((x + 0.105) ** 2) / (2 * 0.004**2))
    )
    rng = np.random.default_rng(0)
    y = doppler + peaks + 0.0015 * rng.normal(size=n)
    return x, y


def test_triplet_left_to_right():
    x, y = _synthetic_well()
    result = analyze_spectrum(x, y, peak_choice=3)
    assert result["ok"]
    volts = [p["voltage"] for p in result["triplet"]]
    assert volts[0] < volts[1] < volts[2]
    assert abs(volts[0] + 0.15) < 0.01
    assert abs(volts[2] + 0.105) < 0.01
    assert result["target"]["voltage"] == volts[2]


def test_peak_choice_selects_left_peak():
    x, y = _synthetic_well()
    result = analyze_spectrum(x, y, peak_choice=1)
    volts = [p["voltage"] for p in result["triplet"]]
    assert result["target"]["voltage"] == volts[0]


def test_fine_scan_without_full_triplet():
    """A tight window around the tallest line: no triplet, still lockable."""
    x, y = _synthetic_well()
    mask = (x > -0.113) & (x < -0.097)
    result = analyze_spectrum(
        x[mask], y[mask], peak_choice=3, require_triplet=False, hint_voltage=-0.105
    )
    assert result["ok"]
    assert result["triplet"] is None
    assert abs(result["target"]["voltage"] + 0.105) < 0.01


def test_fine_scan_two_peaks_choice_3_picks_tallest():
    x, y = _synthetic_well()
    mask = (x > -0.13) & (x < -0.09)
    result = analyze_spectrum(
        x[mask], y[mask], peak_choice=3, require_triplet=False
    )
    assert result["ok"]
    assert result["triplet"] is None
    assert abs(result["target"]["voltage"] + 0.105) < 0.01


def test_fine_scan_choice_1_with_only_left_line():
    x, y = _synthetic_well()
    mask = (x > -0.16) & (x < -0.14)
    result = analyze_spectrum(
        x[mask], y[mask], peak_choice=1, require_triplet=False
    )
    assert result["ok"]
    assert abs(result["target"]["voltage"] + 0.15) < 0.01


def test_fine_scan_false_triplet_does_not_steal_peak():
    """Cropping off the left of the well used to invent a fake 1/2/3 cluster."""
    x, y = _synthetic_well()
    mask = x > -0.11
    result = analyze_spectrum(
        x[mask],
        y[mask],
        peak_choice=3,
        require_triplet=False,
        hint_voltage=-0.105,
    )
    assert result["ok"]
    assert abs(result["target"]["voltage"] + 0.105) < 0.015


def test_fine_scan_with_full_triplet_still_uses_choice():
    x, y = _synthetic_well()
    result = analyze_spectrum(x, y, peak_choice=3, require_triplet=False)
    assert result["ok"]
    assert result["triplet"] is not None
    volts = [p["voltage"] for p in result["triplet"]]
    assert result["target"]["voltage"] == volts[2]


def test_missing_peaks_returns_not_ok():
    x = np.linspace(-0.4, 0.4, 2000)
    y = 0.4 + 0.001 * np.sin(40 * x)
    result = analyze_spectrum(x, y, peak_choice=3)
    assert result["ok"] is False
    assert result["triplet"] is None
