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
