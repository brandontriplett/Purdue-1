import numpy as np

from laserlock.lockpoint import first_scurve_window, get_lock_point, lock_point_from_trace


def _scurve(v, v0, amp=0.008, width=0.004, offset=0.0):
    x = (v - v0) / width
    return offset + amp * (-2.0 * x) * np.exp(-0.5 * x * x)


def _downhill_trace(v_high, v_low, n, centers, amps):
    v = np.linspace(v_high, v_low, n)
    e = np.zeros_like(v)
    for c, a in zip(centers, amps):
        e += _scurve(v, c, amp=a)
    return v, e


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
