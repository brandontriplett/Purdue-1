"""IQ dither, recorded uphill error sweep, DC park, PI lock.

ASG holds DC via offset (amplitude = 0). IQ dithers and demodulates.
Lock voltage comes from a *finished* error trace (Linien min/max/mid),
not a live sample-by-sample catch.
"""

from __future__ import annotations

import time

import numpy as np

from .connection import silence_feedback
from .lockpoint import (
    apply_lockpoint_correction,
    lock_point_from_single_line,
    lock_point_from_trace,
)
from .waveform import DAC_MAX, DAC_MIN


def _hold_local(drive, voltage, max_jump_v=None, label="hold"):
    """hold_dc, but refuse a jump larger than ``max_jump_v`` (if set)."""
    target = float(voltage)
    jump = abs(target - float(drive.current_v))
    print(
        f"{label}: {drive.current_v:+.6f} -> {target:+.6f} V  "
        f"(Δ {1e3 * jump:.1f} mV)"
    )
    if max_jump_v is not None and jump > float(max_jump_v):
        raise RuntimeError(
            f"{label} would move {1e3 * jump:.1f} mV "
            f"(limit {1e3 * float(max_jump_v):.0f} mV). "
            "Refusing to leave the local window."
        )
    drive.hold_dc(target, verbose=False)


def move_to_scan_start(drive, v_start, scan_time=0.5, max_abs_jump_v=0.50):
    """Analog move to a scan's low-V start. No extra sit."""
    return _slew_to_scan_start(
        drive, v_start, scan_time=scan_time, max_abs_jump_v=max_abs_jump_v
    )


def _slew_to_scan_start(drive, v_start, scan_time=0.5, max_abs_jump_v=0.50):
    """Analog move to the next scan's low-V start. No extra sit.

    Triplet start → line start can be ~100–160 mV; that is an expected
    handoff, not a runaway jump. Refuse only a move larger than
    ``max_abs_jump_v``.
    """
    target = float(v_start)
    jump = abs(target - float(drive.current_v))
    print(
        f"Scan start: {drive.current_v:+.6f} -> {target:+.6f} V  "
        f"(Δ {1e3 * jump:.1f} mV)"
    )
    if jump > float(max_abs_jump_v):
        raise RuntimeError(
            f"Scan start would move {1e3 * jump:.1f} mV "
            f"(limit {1e3 * float(max_abs_jump_v):.0f} mV). "
            "Refusing to leave the local window."
        )
    if jump < 0.002:
        return
    dt = min(float(scan_time), max(0.05, jump / 0.25))
    drive.slew_to(target, duration=dt)


def _sampler_mean(rp, name, t=0.01):
    sampler = getattr(rp, "sampler", None)
    if sampler is None:
        raise RuntimeError("Red Pitaya sampler is not available.")
    mean, _std, _mx, _mn = sampler.stats(name, t=t)
    return float(mean)


def sample_lock_signals(rp, t=0.01):
    """Mean iq0, in1, asg0, and out1 over ``t`` seconds."""
    return {
        "error": _sampler_mean(rp, "iq0", t=t),
        "pd": _sampler_mean(rp, "in1", t=t),
        "asg": _sampler_mean(rp, "asg0", t=t),
        "out1": _sampler_mean(rp, "out1", t=t),
    }


def dither_status(rp):
    """IQ dither amplitude, routing, and enable flag."""
    iq = rp.iq0
    amp = float(getattr(iq, "amplitude", 0.0) or 0.0)
    out = str(getattr(iq, "output_direct", "?"))
    on = bool(getattr(iq, "on", True)) if hasattr(iq, "on") else True
    freq = float(getattr(iq, "frequency", 0.0) or 0.0)
    return {"amplitude": amp, "output_direct": out, "on": on, "frequency": freq}


def ensure_dither(rp):
    """Re-assert IQ dither onto OUT1 and print whether it looks live."""
    iq = rp.iq0
    if hasattr(iq, "on"):
        iq.on = True
    if getattr(iq, "output_direct", None) != "out1":
        iq.output_direct = "out1"
    st = dither_status(rp)
    print(
        f"Dither: {st['amplitude']:.4f} V @ {st['frequency']:.0f} Hz  "
        f"output_direct={st['output_direct']}  on={st['on']}"
    )
    if st["amplitude"] < 1e-5 or st["output_direct"] != "out1" or not st["on"]:
        print("WARNING: dither does not look routed to OUT1.")
        return st
    try:
        _mean, std, mx, mn = rp.sampler.stats("out1", t=0.02)
        ptp = float(mx) - float(mn)
        print(
            f"  OUT1 while parked: std {1e3 * float(std):.2f} mV,  "
            f"ptp {1e3 * ptp:.2f} mV  "
            f"(2 mV amp dither is ~4 mVpp if present)"
        )
        if ptp < 0.001:
            print(
                "WARNING: OUT1 ptp is < 1 mV. Dither may be missing on the "
                "analog output even if IQ registers look on."
            )
    except Exception as exc:
        print(f"  Could not sample OUT1 dither ({exc}).")
    return st


def sample_error_pd(rp, t=0.01):
    """Mean lock-in error (iq0) and photodiode (in1) over ``t`` seconds."""
    sig = sample_lock_signals(rp, t=t)
    return sig["error"], sig["pd"]


def configure_dither(
    rp,
    frequency_hz=10_000.0,
    amplitude_v=0.003,
    bandwidth_hz=2_000.0,
    acbandwidth_hz=1_000.0,
    phase_deg=0.0,
    quadrature_factor=10,
    pid_corr_min_v=-0.20,
    pid_corr_max_v=0.20,
):
    """Turn on IQ dither/demod. PID is routed but gains stay zero."""
    iq = rp.iq0
    pid = rp.pid0

    pid.input = "iq0"
    pid.output_direct = "off"
    pid.setpoint = 0.0
    pid.p = 0
    pid.i = 0
    pid.ival = 0
    pid.inputfilter = [0, 0, 0, 0]
    if hasattr(pid, "paused"):
        pid.paused = False
    if hasattr(pid, "max_voltage"):
        pid.max_voltage = float(pid_corr_max_v)
        pid.min_voltage = float(pid_corr_min_v)

    iq.setup(
        frequency=float(frequency_hz),
        bandwidth=[float(bandwidth_hz), float(bandwidth_hz)],
        gain=0.0,
        phase=float(phase_deg),
        acbandwidth=float(acbandwidth_hz),
        amplitude=float(amplitude_v),
        input="in1",
        output_direct="out1",
        output_signal="quadrature",
        quadrature_factor=float(quadrature_factor),
    )
    if hasattr(iq, "on"):
        iq.on = True

    print("Dither on, PID open.")
    print(f"  {amplitude_v:.4f} V @ {frequency_hz:.1f} Hz, phase {phase_deg:.1f} deg")
    return iq, pid


def diagnose_hold_points(drive, rp, voltages, settle_s=0.4, sample_t=0.05):
    """Park at each voltage and print ASG / OUT1 / PD / error.

    Use this to see whether the DAC is actually moving and whether the
    lock-in produces an error on the peak vs on the wing.
    """
    print()
    print("------------------------------------------------------------")
    print("HOLD-POINT DIAGNOSTIC")
    print("------------------------------------------------------------")
    print("  cmd is what we asked for. asg0 is the ASG module output.")
    print("  If asg0 does not follow cmd, the laser is not scanning.")
    rows = []
    for voltage in voltages:
        voltage = float(voltage)
        drive.hold_dc(voltage, verbose=True)
        time.sleep(settle_s)
        sig = sample_lock_signals(rp, t=sample_t)
        d_asg = sig["asg"] - voltage
        print(
            f"  cmd={voltage:+.4f}  asg0={sig['asg']:+.4f} (Δ{d_asg:+.4f})  "
            f"out1={sig['out1']:+.4f}  PD={sig['pd']:+.4f}  err={sig['error']:+.4f}"
        )
        rows.append({"cmd": voltage, **sig})
    return rows


def error_scan_window(v_peak, below_v, above_v, lock_min_v=DAC_MIN, lock_max_v=DAC_MAX):
    """Low-V start and high-V stop around ``v_peak`` (uphill ramp).

    ``below_v`` is how far below the peak the scan starts.
    ``above_v`` is how far above the peak it stops.
    """
    v_start = max(float(lock_min_v), float(v_peak) - float(below_v))
    v_stop = min(float(lock_max_v), float(v_peak) + float(above_v))
    if v_start >= v_stop:
        raise ValueError(
            f"Approach window is empty: start {v_start:.4f} V, stop {v_stop:.4f} V."
        )
    return v_start, v_stop


def triplet_scan_extents(triplet, target_v, below_v, above_v):
    """below/above so an uphill ramp around ``target_v`` covers the whole triplet.

    ``below_v`` pads below the leftmost line, ``above_v`` above the rightmost.
    The selected peak (1, 2, or 3) is ``target_v``; neighbors stay in view.
    """
    left = min(float(p["voltage"]) for p in triplet)
    right = max(float(p["voltage"]) for p in triplet)
    below = float(below_v) + max(0.0, float(target_v) - left)
    above = float(above_v) + max(0.0, right - float(target_v))
    return below, above


def record_dithered_sweep(
    drive,
    rp,
    v_peak,
    below_v=0.060,
    above_v=0.060,
    scan_time=0.5,
    pause_s=5.0,
    lock_min_v=DAC_MIN,
    lock_max_v=DAC_MAX,
    max_jump_v=None,
    name="ERROR SCAN",
    **_ignored,
):
    """Fast analog uphill ramp with dither on. Record IN1 and iq0.

    Sit at the low-voltage start, ramp toward high V, then jump back to
    start so we do not heat at the endpoint. ``pause_s`` is the sit at
    start before the ramp.
    """
    v_start, v_stop = error_scan_window(
        v_peak, below_v, above_v, lock_min_v, lock_max_v
    )

    print()
    print(f"Error-scan reference: {v_peak:+.6f} V")
    print(f"Window: {v_start:+.6f} -> {v_stop:+.6f} V")
    cap = 0.50 if max_jump_v is None else max(float(max_jump_v), 0.50)
    _slew_to_scan_start(
        drive, v_start, scan_time=float(scan_time), max_abs_jump_v=cap
    )
    voltage, pd, error, plan = drive.fast_error_scan(
        start_v=v_start,
        stop_v=v_stop,
        scan_time=float(scan_time),
        pause_s=float(pause_s),
        name=str(name),
    )
    log = {
        "voltage": np.asarray(voltage, dtype=float),
        "error": np.asarray(error, dtype=float),
        "pd": np.asarray(pd, dtype=float),
        "asg": np.asarray(voltage, dtype=float),
        "out1": np.asarray(voltage, dtype=float),
        "v_start": v_start,
        "v_stop": v_stop,
        "v_peak": v_peak,
        "pause_s": float(pause_s),
        "scan_time": float(scan_time),
        "below_v": float(below_v),
        "above_v": float(above_v),
        "plan": plan,
    }
    print(
        f"PD ptp {log['pd'].max()-log['pd'].min():.4f} V,  "
        f"|error| max {np.max(np.abs(log['error'])):.4f} V"
    )
    return log


def approach_duration(v_start, v_stop, lock_v, scan_time):
    """Seconds for start → lock at the same V/s as a ``scan_time`` ramp."""
    span = abs(float(v_stop) - float(v_start))
    dist = abs(float(lock_v) - float(v_start))
    dt = float(scan_time) * dist / span if span > 1e-9 else 0.05
    return max(0.02, min(dt, float(scan_time)))


def approach_lock_from_start(drive, log, lp, max_jump_v=None):
    """Analog ramp start → lock at the same V/s as the lock scan. Stay there.

    The laser must already be at ``log['v_start']`` (low side). No extra sit.
    """
    v_start = float(log["v_start"])
    v_stop = float(log["v_stop"])
    lock_v = float(lp["lock_voltage"])
    scan_time = float(log.get("scan_time", 0.5))
    dt = approach_duration(v_start, v_stop, lock_v, scan_time)
    if abs(float(drive.current_v) - v_start) > 0.002:
        _hold_local(
            drive, v_start, max_jump_v=max_jump_v, label="Return to lock-scan start"
        )
    print(
        f"Approach lock from start at scan rate: "
        f"{v_start:+.6f} -> {lock_v:+.6f} V in {dt:.3f} s"
    )
    drive.slew_to(lock_v, duration=dt)
    ensure_dither(drive.rp)
    print(
        f"Lock point {lp['lock_voltage']:+.6f} V  "
        f"(error mid-height {lp['mean_error']:+.4f} V, "
        f"span {1e3 * lp['span_v']:.1f} mV, "
        f"dE/dV {'rising' if lp['slope_rising_vs_voltage'] else 'falling'})"
    )
    if lp["slope_rising_vs_voltage"]:
        print(
            "Slope rises with voltage: if the PID runs away, "
            "add 180 deg to IQ phase (or use a negative I)."
        )
    return lp


def park_at_lock_point(drive, log, single_line=False, max_jump_v=None):
    """Compute Linien lock point and analog-ramp to it from the scan start.

    Start is the low-voltage end of an uphill scan. Always go start → lock
    at the scan V/s, never jump from the high-V stop.
    """
    if single_line:
        lp = lock_point_from_single_line(log["voltage"], log["error"])
    else:
        lp = lock_point_from_trace(log["voltage"], log["error"])
    return approach_lock_from_start(drive, log, lp, max_jump_v=max_jump_v)


def run_line_scan(
    drive,
    rp,
    v_guess,
    below_v=0.025,
    above_v=0.025,
    scan_time=0.5,
    line_pause_s=5.0,
    lock_below_v=None,
    lock_above_v=None,
    lock_min_v=DAC_MIN,
    lock_max_v=DAC_MAX,
    max_jump_v=0.12,
):
    """Uphill line scan. Returns center and jumps to the lock-scan start."""
    if lock_below_v is None:
        lock_below_v = 0.70 * float(below_v)
    if lock_above_v is None:
        lock_above_v = 0.70 * float(above_v)
    window = float(below_v) + float(above_v)
    print()
    print("------------------------------------------------------------")
    print("LINE SCAN — one peak, measure center")
    print("------------------------------------------------------------")
    line = record_dithered_sweep(
        drive,
        rp,
        v_peak=v_guess,
        below_v=below_v,
        above_v=above_v,
        scan_time=scan_time,
        pause_s=float(line_pause_s),
        lock_min_v=lock_min_v,
        lock_max_v=lock_max_v,
        max_jump_v=max_jump_v,
        name="LINE SCAN",
    )
    center_lp = lock_point_from_single_line(
        line["voltage"], line["error"], hint_voltage=v_guess
    )
    center = center_lp["lock_voltage"]
    print(
        f"Measured center {center:+.6f} V  "
        f"(span {1e3 * center_lp['span_v']:.1f} mV, "
        f"method {center_lp.get('method', '?')})"
    )
    shift = center - float(v_guess)
    print(f"Center vs guess: {1e3 * shift:+.1f} mV")
    if abs(shift) > 0.5 * window:
        print(
            "WARNING: center is near the edge of the line-scan window. "
            "The lock scan still uses the measured center."
        )

    lock_start, lock_stop = error_scan_window(
        center, lock_below_v, lock_above_v, lock_min_v, lock_max_v
    )
    _hold_local(
        drive, lock_start, max_jump_v=max_jump_v, label="Next start (lock scans)"
    )
    return {
        "line": line,
        "center": center_lp,
        "lock_start": lock_start,
        "lock_stop": lock_stop,
        "lock_below_v": float(lock_below_v),
        "lock_above_v": float(lock_above_v),
        "scan_time": float(scan_time),
    }


def run_paired_lock_scans(
    drive,
    rp,
    center,
    below_v=0.0175,
    above_v=0.0175,
    scan_time=0.5,
    lock_pause_s=5.0,
    repeat_pause_s=0.0,
    lock_min_v=DAC_MIN,
    lock_max_v=DAC_MAX,
    max_jump_v=0.12,
):
    """Two uphill lock scans back-to-back, then analog approach from start.

    Scan 1 sits ``lock_pause_s`` at the low start. Scan 2 repeats immediately
    (``repeat_pause_s``, default 0). After scan 2 we are at start; we then
    ramp start → lock at the same V/s as the scans and stay there.
    """
    print()
    print("------------------------------------------------------------")
    print("LOCK SCAN 1")
    print("------------------------------------------------------------")
    lock1 = record_dithered_sweep(
        drive,
        rp,
        v_peak=center,
        below_v=below_v,
        above_v=above_v,
        scan_time=scan_time,
        pause_s=float(lock_pause_s),
        lock_min_v=lock_min_v,
        lock_max_v=lock_max_v,
        max_jump_v=max_jump_v,
        name="LOCK SCAN 1",
    )

    print()
    print("------------------------------------------------------------")
    print("LOCK SCAN 2 — immediate repeat")
    print("------------------------------------------------------------")
    lock2 = record_dithered_sweep(
        drive,
        rp,
        v_peak=center,
        below_v=below_v,
        above_v=above_v,
        scan_time=scan_time,
        pause_s=float(repeat_pause_s),
        lock_min_v=lock_min_v,
        lock_max_v=lock_max_v,
        max_jump_v=max_jump_v,
        name="LOCK SCAN 2",
    )

    lp1 = lock_point_from_single_line(
        lock1["voltage"], lock1["error"], hint_voltage=center
    )
    lp2 = lock_point_from_single_line(
        lock2["voltage"], lock2["error"], hint_voltage=center
    )
    dv_mV = 1e3 * (lp2["lock_voltage"] - lp1["lock_voltage"])
    print(
        f"Lock scan 1: {lp1['lock_voltage']:+.6f} V  "
        f"span {1e3 * lp1['span_v']:.1f} mV"
    )
    print(
        f"Lock scan 2: {lp2['lock_voltage']:+.6f} V  "
        f"span {1e3 * lp2['span_v']:.1f} mV"
    )
    print(f"Delta {dv_mV:+.2f} mV  (scan 2 minus scan 1)")

    # Already at the shared start after scan 2. Ramp start → lock at scan V/s.
    approach_lock_from_start(drive, lock2, lp2, max_jump_v=max_jump_v)
    return {
        "lock_scan": lock1,
        "lock_scan_2": lock2,
        "lp1": lp1,
        "lock": lp2,
    }


def run_fine_scan_and_park(
    drive,
    rp,
    v_guess,
    below_v=0.025,
    above_v=0.025,
    scan_time=0.25,
    pause_s=5.0,
    lock_min_v=DAC_MIN,
    lock_max_v=DAC_MAX,
    max_jump_v=0.50,
    correction_v=0.0,
):
    """One uphill fine scan, then analog-ramp start → lock at scan rate.

    After the ramp we are already at start. Lock-point math is milliseconds;
    the approach runs before any plotting. ``correction_v`` is added to the
    fitted lock (fit 0.232 + 0.005 → park 0.237).
    """
    print()
    print("------------------------------------------------------------")
    print("FINE SCAN — then park at lock")
    print("------------------------------------------------------------")
    log = record_dithered_sweep(
        drive,
        rp,
        v_peak=v_guess,
        below_v=below_v,
        above_v=above_v,
        scan_time=scan_time,
        pause_s=float(pause_s),
        lock_min_v=lock_min_v,
        lock_max_v=lock_max_v,
        max_jump_v=max_jump_v,
        name="FINE SCAN",
    )
    lp = lock_point_from_single_line(
        log["voltage"], log["error"], hint_voltage=v_guess
    )
    print(
        f"Fine-scan lock {lp['lock_voltage']:+.6f} V  "
        f"(span {1e3 * lp['span_v']:.1f} mV, "
        f"method {lp.get('method', '?')})"
    )
    print(f"vs guess: {1e3 * (lp['lock_voltage'] - float(v_guess)):+.1f} mV")
    lp = apply_lockpoint_correction(
        lp,
        correction_v=correction_v,
        lock_min_v=lock_min_v,
        lock_max_v=lock_max_v,
    )
    if abs(float(correction_v)) > 1e-12:
        print(
            f"Lockpoint correction {float(correction_v):+.6f} V  "
            f"→ park {lp['lock_voltage']:+.6f} V"
        )
    approach_lock_from_start(drive, log, lp, max_jump_v=max_jump_v)
    return log, lp


def centered_fine_scans(*args, **kwargs):
    """Back-compat alias for ``run_line_scan``."""
    return run_line_scan(*args, **kwargs)


def _print_lock_sample(t_s, pid, sig):
    ival = float(pid.ival)
    print(
        f"  t={t_s:.1f}s  ival={ival:+.4f} V  "
        f"out1={sig['out1']:+.4f} V  asg0={sig['asg']:+.4f} V  "
        f"error={sig['error']:+.4f} V  PD={sig['pd']:+.4f} V"
    )
    return ival


def set_iq_phase(rp, phase_deg):
    """Set lock-in phase. Does not move the ASG park or PID gains."""
    iq = rp.iq0
    wrapped = ((float(phase_deg) + 180.0) % 360.0) - 180.0
    iq.phase = wrapped
    actual = float(getattr(iq, "phase", wrapped))
    print(f"IQ phase {actual:.1f} deg")
    return actual


def flip_iq_phase(rp):
    """Add 180°. Wrong-sign lock becomes right-sign (or the reverse)."""
    current = float(getattr(rp.iq0, "phase", 0.0) or 0.0)
    return set_iq_phase(rp, current + 180.0)


def open_pid(rp):
    """Drop PID from OUT1. Dither and ASG park stay as they are."""
    pid = rp.pid0
    pid.output_direct = "off"
    pid.p = 0
    pid.i = 0
    pid.ival = 0
    if hasattr(pid, "paused"):
        pid.paused = False
    print("PID open. Dither still on; ASG still at the park.")


def apply_pid(
    rp,
    p=0.0,
    i_hz=-3.0,
    corr_min_v=-0.20,
    corr_max_v=0.20,
    phase_deg=None,
    monitor_s=2.4,
):
    """Close PI immediately (no ramp). FPGA stays on after return.

    Loopback sat at P=0, I=−3 Hz. Watch out1 and ival, not asg0.
    """
    if phase_deg is not None:
        set_iq_phase(rp, float(phase_deg))
    ensure_dither(rp)
    pid = rp.pid0
    if hasattr(pid, "paused"):
        pid.paused = False
    pid.inputfilter = [0, 0, 0, 0]
    pid.input = "iq0"
    pid.setpoint = 0.0
    pid.ival = 0.0
    if hasattr(pid, "max_voltage"):
        pid.max_voltage = float(corr_max_v)
        pid.min_voltage = float(corr_min_v)
    pid.output_direct = "out1"
    pid.p = float(p)
    pid.i = float(i_hz)
    print(
        f"PID on  P={pid.p:g}  I={pid.i:g} Hz  "
        f"phase={float(getattr(rp.iq0, 'phase', 0.0) or 0.0):.1f} deg"
    )
    if monitor_s <= 0:
        return pid
    print("t     ival      out1      err       PD")
    n = max(1, int(round(float(monitor_s) / 0.4)))
    for k in range(n):
        time.sleep(0.4)
        sig = sample_lock_signals(rp, t=0.04)
        print(
            f"{0.4 * (k + 1):3.1f}  {float(pid.ival):+.4f}  {sig['out1']:+.4f}  "
            f"{sig['error']:+.4f}  {sig['pd']:+.4f}"
        )
    return pid


def ramp_pid_gains(pid, p_final, i_final, ramp_s, dt=0.25, on_step=None):
    """Ramp P and I from 0 to the finals. ``on_step`` returning False aborts."""
    p_final = float(p_final)
    i_final = float(i_final)
    ramp_s = max(0.0, float(ramp_s))
    dt = max(0.05, float(dt))
    if ramp_s < 1e-6:
        pid.p = p_final
        pid.i = i_final
        return True
    n = max(1, int(round(ramp_s / dt)))
    for k in range(1, n + 1):
        frac = k / float(n)
        pid.p = p_final * frac
        pid.i = i_final * frac
        if on_step is not None:
            if on_step(k, n, pid) is False:
                return False
        else:
            time.sleep(dt)
    return True


def engage_i_lock(
    rp,
    i_gain_hz=-3.0,
    p_gain=0.0,
    monitor_s=8.0,
    rail_v=0.20,
    pd_ref=None,
    max_error_v=0.003,
    max_pd_miss_v=0.005,
    ramp_s=4.0,
):
    """Close PI lock, ramping gains from 0, then judge the quiet traces.

    ASG holds the park. PID sums onto OUT1. ``asg0`` staying put is
    expected; ``out1`` should track ``ival``. Instant runaway is almost
    always the wrong IQ phase — ``open_pid()``, ``flip_iq_phase()``, retry.
    """
    pid = rp.pid0
    if hasattr(pid, "paused"):
        pid.paused = False
    pid.ival = 0.0
    pid.setpoint = 0.0
    pid.p = 0
    pid.i = 0
    pid.inputfilter = [0, 0, 0, 0]
    pid.output_direct = "out1"
    print("Closing PID loop (soft start)...")
    print("  asg0 is the DC park. out1 = asg0 + pid + dither. Watch out1, not asg0.")
    ensure_dither(rp)
    rail = 0.90 * abs(float(rail_v))
    aborted = []

    def _step(k, n, pid_mod):
        time.sleep(max(0.05, float(ramp_s) / max(n, 1)))
        try:
            sig = sample_lock_signals(rp, t=0.03)
            ival = float(pid_mod.ival)
            print(
                f"  ramp {k}/{n}  P={pid_mod.p:.4f}  I={pid_mod.i:.2f} Hz  "
                f"ival={ival:+.4f}  out1={sig['out1']:+.4f}  "
                f"err={sig['error']:+.4f}  PD={sig['pd']:+.4f}"
            )
        except Exception:
            ival = float(pid_mod.ival)
            print(
                f"  ramp {k}/{n}  P={pid_mod.p:.4f}  I={pid_mod.i:.2f} Hz  "
                f"ival={ival:+.4f}"
            )
        if abs(ival) >= rail:
            print(
                "ABORT: ival hit the rail during the ramp. Wrong sign. "
                "open_pid(rp); flip_iq_phase(rp); re-run engage."
            )
            pid_mod.p = 0
            pid_mod.i = 0
            pid_mod.output_direct = "off"
            aborted.append(True)
            return False
        return True

    print(
        f"Ramping P 0→{float(p_gain):g}, I 0→{float(i_gain_hz):g} Hz "
        f"in {float(ramp_s):.1f} s"
    )
    ok_ramp = ramp_pid_gains(
        pid, p_gain, i_gain_hz, ramp_s=ramp_s, on_step=_step
    )
    if not ok_ramp or aborted:
        return False
    print(f"PID engaged.  P={pid.p}  I={pid.i} Hz")
    print("If it runs away: open_pid(rp); flip_iq_phase(rp); retry this cell.")

    n = max(1, int(round(float(monitor_s) / 0.5)))
    print(f"Monitoring for {monitor_s:.1f} s...")
    ivals = []
    errors = []
    pds = []
    out1s = []
    asgs = []
    for k in range(n):
        time.sleep(0.5)
        try:
            sig = sample_lock_signals(rp, t=0.05)
            ival = _print_lock_sample(0.5 * (k + 1), pid, sig)
            errors.append(sig["error"])
            pds.append(sig["pd"])
            out1s.append(sig["out1"])
            asgs.append(sig["asg"])
        except Exception:
            ival = float(pid.ival)
            print(f"  t={0.5 * (k + 1):.1f}s  ival={ival:+.4f} V")
        ivals.append(ival)

    ok = True
    rail = 0.90 * abs(float(rail_v))
    if ivals and abs(ivals[-1]) >= rail:
        print(
            "FAIL: PID correction is against the rail. "
            "Wrong sign or missed the slope. "
            "open_pid(rp); flip_iq_phase(rp); re-run engage (stay parked)."
        )
        return False

    mean_err = float(np.mean(np.abs(errors))) if errors else float("nan")
    print(f"Mean |error| {1e3 * mean_err:.1f} mV")
    if errors and mean_err > float(max_error_v):
        print(
            f"FAIL: |error| is larger than {1e3 * float(max_error_v):.1f} mV. "
            "Not sitting on the lock point."
        )
        ok = False

    if pd_ref is not None and pds:
        dpd = float(np.mean(pds)) - float(pd_ref)
        print(f"PD vs lock-scan peak: {1e3 * dpd:+.1f} mV")
        if abs(dpd) > float(max_pd_miss_v):
            print(
                f"FAIL: PD is more than {1e3 * float(max_pd_miss_v):.1f} mV "
                "off the lock-scan peak."
            )
            ok = False

    if ivals and out1s and asgs:
        mean_delta = float(np.mean(np.asarray(out1s) - np.asarray(asgs)))
        mean_ival = float(np.mean(ivals))
        print(
            f"out1 − asg0 = {1e3 * mean_delta:+.1f} mV,  "
            f"mean ival = {1e3 * mean_ival:+.1f} mV"
        )
        if abs(mean_ival) > 0.002 and abs(mean_delta - mean_ival) > 0.5 * abs(mean_ival) + 0.004:
            print("FAIL: out1 is not following ival. PID may not be summing onto OUT1.")
            ok = False

    if ok:
        print("PASS: error near zero, PD on the peak, out1 follows ival, off the rails.")
    else:
        print("FAIL: PID does not appear to be locked.")
    return ok


def disengage_lock(drive, rp):
    """Zero PID/IQ and drop the ASG to idle."""
    silence_feedback(rp)
    drive.release_to_zero()
    print(f"Lock off. Drive at {drive.current_v:+.6f} V")
