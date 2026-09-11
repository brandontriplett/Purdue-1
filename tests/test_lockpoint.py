import numpy as np

from laserlock.lockpoint import (
    apply_lockpoint_correction,
    error_is_bipolar,
    first_scurve_window,
    get_lock_point,
    lock_point_from_single_line,
    lock_point_from_trace,
)


def _scurve(v, v0, amp=0.008, width=0.004, offset=0.0):
    x = (v - v0) / width
    return offset + amp * (-2.0 * x) * np.exp(-0.5 * x * x)


def _downhill_trace(v_high, v_low, n, centers, amps, width=0.004):
    v = np.linspace(v_high, v_low, n)
    e = np.zeros_like(v)
    for c, a in zip(centers, amps):
        e += _scurve(v, c, amp=a, width=width)
    return v, e


def _uphill_trace(v_low, v_high, n, centers, amps, width=0.004):
    return _downhill_trace(v_low, v_high, n, centers, amps, width=width)


def test_lockpoint_correction_adds_to_the_fit():
    lp = apply_lockpoint_correction({"lock_voltage": 0.232}, correction_v=0.005)
    assert abs(lp["lock_voltage_fit"] - 0.232) < 1e-12
    assert abs(lp["lockpoint_correction_v"] - 0.005) < 1e-12
    assert abs(lp["lock_voltage"] - 0.237) < 1e-12


def test_lockpoint_correction_zero_is_a_no_op():
    lp = apply_lockpoint_correction({"lock_voltage": 0.232}, correction_v=0.0)
    assert abs(lp["lock_voltage"] - 0.232) < 1e-12
    assert abs(lp["lock_voltage_fit"] - 0.232) < 1e-12


def test_lock_point_is_mid_height_between_extrema():
    v = np.linspace(0.20, 0.10, 400)
    e = _scurve(v, 0.15, amp=0.01, offset=0.002)
    i0, i1 = 0, len(v) - 1
    lp = get_lock_point(v, e, i0, i1)
    assert abs(lp["lock_voltage"] - 0.15) < 0.002
    assert abs(lp["mean_error"] - 0.002) < 0.001
    # Downhill: index increases as V drops. Min of odd S is on the high-V side.
    assert lp["slope_rising_vs_voltage"] is False


def test_first_scurve_is_the_high_voltage_line():
    v, e = _downhill_trace(
        0.20,
        0.05,
        600,
        centers=(0.17, 0.13),
        amps=(0.007, 0.010),
    )
    i0, i1 = first_scurve_window(v, e)
    lp = get_lock_point(v, e, i0, i1)
    assert abs(lp["lock_voltage"] - 0.17) < 0.004
    assert lp["lock_voltage"] > 0.15


def test_lock_point_from_trace_picks_first_from_the_right():
    v, e = _downhill_trace(
        0.195,
        0.075,
        500,
        centers=(0.170, 0.130, 0.090),
        amps=(0.006, 0.009, 0.004),
    )
    lp = lock_point_from_trace(v, e)
    assert abs(lp["lock_voltage"] - 0.170) < 0.005


def test_doppler_offset_does_not_move_the_zero():
    v = np.linspace(0.12, 0.08, 300)
    e = _scurve(v, 0.10, amp=0.008, offset=0.015)
    lp = get_lock_point(v, e, 0, len(v) - 1)
    assert abs(lp["lock_voltage"] - 0.10) < 0.002
    assert abs(lp["mean_error"] - 0.015) < 0.002


def test_does_not_pair_max_of_one_line_with_min_of_next():
    """Lab failure: ~3 mV S-curves, ~30 mV spacing. Old 4–40 mV gate
    skipped the real first pair and glued max of line 3 to min of line 2."""
    v, e = _downhill_trace(
        0.194,
        0.034,
        2000,
        centers=(0.157, 0.125, 0.065),
        amps=(0.005, 0.007, 0.003),
        width=0.0015,
    )
    lp = lock_point_from_trace(v, e)
    vmin = v[lp["min_idx"]]
    vmax = v[lp["max_idx"]]
    assert abs(lp["lock_voltage"] - 0.157) < 0.004
    assert abs(vmin - vmax) < 0.010
    assert min(vmin, vmax) > 0.145
    assert lp["span_v"] < 0.010


def test_narrow_scurve_is_not_rejected_as_noise():
    v, e = _downhill_trace(
        0.18,
        0.10,
        800,
        centers=(0.150, 0.120),
        amps=(0.006, 0.006),
        width=0.0015,
    )
    i0, i1 = first_scurve_window(v, e)
    span = abs(v[i0] - v[i1])
    assert span < 0.008
    assert span > 0.001
    lp = get_lock_point(v, e, i0, i1)
    assert abs(lp["lock_voltage"] - 0.150) < 0.004


def test_single_line_uses_whole_trace_mid_height():
    v = np.linspace(0.22, 0.16, 400)
    e = _scurve(v, 0.19, amp=0.006, width=0.0015, offset=0.001)
    lp = lock_point_from_single_line(v, e)
    assert abs(lp["lock_voltage"] - 0.19) < 0.002
    assert lp["span_v"] < 0.010
    assert lp["method"] == "full_trace"


def test_error_is_bipolar_rejects_a_single_lobe():
    v = np.linspace(0.10, 0.14, 400)
    e = -0.03 * np.exp(-((v - 0.11) ** 2) / (2 * 0.004**2))
    assert error_is_bipolar(_scurve(v, 0.12, amp=0.01))
    assert not error_is_bipolar(e)


def test_single_line_hint_keeps_the_target_when_neighbor_leaks_in():
    """±25 mV around peak 2 also sees peak 3 (~40 mV away). Hint must win."""
    v, e = _uphill_trace(
        0.049,
        0.139,
        800,
        centers=(0.094, 0.134),
        amps=(0.009, 0.006),
        width=0.0015,
    )
    lp = lock_point_from_single_line(v, e, hint_voltage=0.094)
    assert abs(lp["lock_voltage"] - 0.094) < 0.006
    assert lp["span_v"] < 0.012


def test_single_line_on_a_cropped_window():
    v, e = _downhill_trace(
        0.20,
        0.05,
        1200,
        centers=(0.170, 0.130, 0.090),
        amps=(0.005, 0.007, 0.003),
        width=0.0015,
    )
    mask = (v <= 0.185) & (v >= 0.155)
    lp = lock_point_from_single_line(v[mask], e[mask])
    assert abs(lp["lock_voltage"] - 0.170) < 0.003
    assert lp["span_v"] < 0.010


def test_uphill_trace_still_picks_the_high_voltage_line():
    v, e = _uphill_trace(
        0.05,
        0.20,
        600,
        centers=(0.17, 0.13),
        amps=(0.007, 0.010),
    )
    lp = lock_point_from_trace(v, e)
    assert abs(lp["lock_voltage"] - 0.17) < 0.004
    assert lp["lock_voltage"] > 0.15


def test_uphill_three_lines_picks_rightmost():
    v, e = _uphill_trace(
        0.075,
        0.195,
        500,
        centers=(0.170, 0.130, 0.090),
        amps=(0.006, 0.009, 0.004),
    )
    lp = lock_point_from_trace(v, e)
    assert abs(lp["lock_voltage"] - 0.170) < 0.005


def test_hint_picks_the_middle_scurve():
    v, e = _uphill_trace(
        0.075,
        0.195,
        800,
        centers=(0.170, 0.130, 0.090),
        amps=(0.006, 0.009, 0.004),
    )
    lp = lock_point_from_trace(v, e, hint_voltage=0.130)
    assert abs(lp["lock_voltage"] - 0.130) < 0.005
    assert lp["method"] == "hint"


def test_peak_choice_2_picks_the_middle_scurve():
    v, e = _uphill_trace(
        0.075,
        0.195,
        800,
        centers=(0.170, 0.130, 0.090),
        amps=(0.006, 0.009, 0.004),
    )
    lp = lock_point_from_trace(v, e, peak_choice=2)
    assert abs(lp["lock_voltage"] - 0.130) < 0.005
    assert lp["n_scurves"] >= 2


def test_peak_choice_1_picks_the_left_scurve():
    v, e = _uphill_trace(
        0.075,
        0.195,
        800,
        centers=(0.170, 0.130, 0.090),
        amps=(0.006, 0.009, 0.004),
    )
    lp = lock_point_from_trace(v, e, peak_choice=1)
    assert abs(lp["lock_voltage"] - 0.090) < 0.005


def test_triplet_scan_extents_cover_neighbors():
    from laserlock.lockin import error_scan_window, triplet_scan_extents

    triplet = (
        {"voltage": -0.030},
        {"voltage": 0.037},
        {"voltage": 0.077},
    )
    below, above = triplet_scan_extents(triplet, 0.037, below_v=0.060, above_v=0.10)
    start, stop = error_scan_window(0.037, below, above)
    assert start < -0.030
    assert stop > 0.077
    assert below == 0.037 - (-0.030) + 0.060
    assert above == 0.077 - 0.037 + 0.10


def test_error_scan_window_is_uphill_around_peak():
    from laserlock.lockin import error_scan_window

    start, stop = error_scan_window(0.15, below_v=0.060, above_v=0.10)
    assert start == 0.09
    assert stop == 0.25
    assert start < 0.15 < stop


def test_approach_duration_matches_scan_rate():
    from laserlock.lockin import approach_duration

    # Full window in 0.5 s → lock at stop takes the full scan time.
    assert abs(approach_duration(0.00, 0.035, 0.035, 0.5) - 0.5) < 1e-12
    # Halfway across the window is half the scan time.
    assert abs(approach_duration(0.00, 0.035, 0.0175, 0.5) - 0.25) < 1e-12
    # Lock at start: floor at 20 ms (slew_to is a no-op if already there).
    assert approach_duration(0.00, 0.035, 0.00, 0.5) == 0.02


def test_inverted_polarity_still_picks_the_high_v_line():
    v, e = _downhill_trace(
        0.194,
        0.034,
        2000,
        centers=(0.157, 0.125, 0.065),
        amps=(-0.005, -0.007, -0.003),
        width=0.0015,
    )
    lp = lock_point_from_trace(v, e)
    assert abs(lp["lock_voltage"] - 0.157) < 0.004
    assert lp["span_v"] < 0.010
