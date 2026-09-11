import numpy as np

from laserlock.waveform import (
    TABLE_LENGTH,
    choose_scope_window,
    crop_linear_scan,
    make_dc_table,
    make_quick_scan_plan,
    make_scan_plan,
    make_slew_plan,
    scope_duration,
)


def test_slew_table_starts_and_ends_at_requested_volts():
    plan = make_slew_plan(-0.2, 0.4, duration=1.5)
    assert len(plan.table) == TABLE_LENGTH
    assert plan.table[0] == np.clip(-0.2, -1, 1)
    assert plan.table[-1] == np.clip(0.4, -1, 1)
    assert np.all(np.diff(plan.table[: plan.slices["slew"].stop]) >= -1e-12)


def test_scan_plan_fills_a_legal_scope_window():
    plan, decimation = make_scan_plan(
        v_current=0.0,
        v_start=-0.4,
        v_stop=0.4,
        scan_time=4.0,
        settle_time=1.5,
        approach_time=1.5,
    )
    assert decimation in (2**n for n in range(17))
    assert abs(plan.duration - scope_duration(decimation)) < 1e-9
    assert "approach" in plan.slices
    assert "settle" in plan.slices
    assert "scan" in plan.slices
    assert "hold" in plan.slices
    assert plan.table[0] == 0.0
    assert plan.table[-1] == 0.4
    scan = plan.table[plan.slices["scan"]]
    assert scan[0] == np.clip(scan[0], -1, 1)
    assert scan[-1] <= 0.4 + 1e-6


def test_scan_without_approach_when_already_at_start():
    plan, _ = make_scan_plan(
        v_current=0.25,
        v_start=0.25,
        v_stop=0.35,
        scan_time=3.0,
        settle_time=1.0,
        approach_time=1.5,
    )
    assert "approach" not in plan.slices
    assert plan.table[0] == 0.25
    assert plan.table[-1] == 0.35


def test_choose_scope_window_neighbors():
    dec, dur = choose_scope_window(5.0)
    assert dec == 65536
    assert abs(dur - 8.589934592) < 1e-6
    dec, dur = choose_scope_window(3.0)
    assert dec == 32768


def test_error_scan_plan_is_a_short_analog_ramp():
    from laserlock.waveform import make_error_scan_plan, scope_duration

    plan, dec = make_error_scan_plan(0.20, 0.08, scan_time=0.5)
    assert plan.table[0] == 0.20
    assert plan.table[-1] == 0.08
    assert "scan" in plan.slices
    assert scope_duration(dec) >= 0.5
    assert plan.duration < 4.0


def test_eight_second_quick_scan_fits_a_scope_window():
    plan, dec = make_quick_scan_plan(-0.50, -0.38, scan_time=8.0)
    assert plan.table[0] == -0.50
    assert plan.table[-1] == -0.38
    assert scope_duration(dec) >= 8.0
    assert "release" not in plan.slices


def test_short_quick_scan_holds_at_stop_not_idle():
    """0.15 s ramps must not put a 0 V tail in the middle of the table."""
    plan, dec = make_quick_scan_plan(-0.50, -0.38, scan_time=0.15)
    n = len(plan.table)
    # Still on the ramp / stop hold, not idle 0 V.
    assert plan.table[n // 4] < -0.35
    assert plan.table[-1] == -0.38
    assert "release" not in plan.slices
    assert scope_duration(dec) >= 0.15


def test_quick_scan_table_ends_at_zero():
    plan, scope_dec = make_quick_scan_plan(-0.4, 0.4, scan_time=1.0)
    assert plan.table[0] == -0.4
    assert plan.table[-1] == 0.0
    assert "scan" in plan.slices
    assert "release" in plan.slices
    # ASG period must outlive a ~2 s scope record.
    assert plan.duration > 4.0
    assert scope_duration(scope_dec) < plan.duration
    scan_points = plan.slices["scan"].stop - plan.slices["scan"].start
    assert scan_points < TABLE_LENGTH // 2


def test_crop_ignores_a_single_pd_glitch():
    n = 2000
    v = np.linspace(-0.10, 0.40, n)
    s = 0.22 + 0.01 * np.sin(np.linspace(0, 6, n))
    s[40] += 0.08  # cold-start PD spike
    cv, cs = crop_linear_scan(v, s, -0.10, 0.40, end_guard=0.04)
    assert len(cv) > 500


def test_crop_does_not_cut_survey_on_absorption_spikes():
    """PD spikes are spectrum. They must not shrink a −0.8→+0.8 V survey."""
    n = 7000
    start, stop = -0.8, 0.8
    v = np.linspace(start, stop, n)
    s = np.full(n, 0.263)
    # Sharp features like the notebook survey that used to crop at ~−0.63 V.
    for v0, amp in ((-0.74, -0.013), (-0.70, 0.015), (-0.64, -0.012), (0.10, 0.02)):
        i = int((v0 - start) / (stop - start) * (n - 1))
        s[i] += amp
    cv, cs = crop_linear_scan(v, s, start, stop, end_guard=0.04)
    assert cv[0] < -0.75
    assert cv[-1] > 0.70
    assert abs(cv[-1] - cv[0]) > 0.80 * abs(stop - start)


def test_crop_does_not_cut_at_midscan_pd_spike():
    """A 120 mV spike near 0 V must not keep only the first half of ±0.4 V."""
    n = 7000
    start, stop = -0.4, 0.4
    v = np.linspace(start, stop, n)
    s = 0.40 + 0.15 * np.sin(np.linspace(0, 3, n))
    i = int((0.01 - start) / (stop - start) * (n - 1))
    s[i] -= 0.12
    cv, _cs = crop_linear_scan(v, s, start, stop, end_guard=0.04)
    assert cv[-1] > 0.30
    assert abs(cv[-1] - cv[0]) > 0.70


def test_crop_linear_scan_drops_wrap_to_start():
    v = np.linspace(-0.4, 0.4, 1000)
    v[900:] = -0.4  # wrap to start
    s = np.ones(1000)
    cv, cs = crop_linear_scan(v, s, -0.4, 0.4, end_guard=0.04)
    assert cv[-1] < 0.4
    assert cv[-1] > 0.2
    assert np.all(np.diff(cv) > -1e-12)


def test_dc_table_is_constant():
    table = make_dc_table(0.123)
    assert len(table) == TABLE_LENGTH
    assert np.allclose(table, 0.123)
