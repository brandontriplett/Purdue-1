"""IQ dither, recorded downhill sweep, DC park, I-only PID.

ASG holds DC via offset (amplitude = 0). IQ dithers and demodulates.
Lock voltage comes from a *finished* error trace (Linien min/max/mid),
not a live sample-by-sample catch.
"""

from __future__ import annotations

import time

import numpy as np

from .connection import silence_feedback
from .lockpoint import lock_point_from_trace
from .waveform import DAC_MAX, DAC_MIN


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
    pid.inputfilter = []
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


def record_dithered_sweep(
    drive,
    rp,
    v_peak,
    margin_v=0.060,
    extra_v=0.060,
    scan_time=0.5,
    lock_min_v=DAC_MIN,
    lock_max_v=DAC_MAX,
    **_ignored,
):
    """Fast analog downhill ramp with dither on. Record IN1 and iq0.

    Same speed class as the coarse scan so thermal drift cannot shear the
    S-curve. Lock voltage is computed afterwards from the error array.
    """
    v_peak = float(v_peak)
    v_start = min(lock_max_v, v_peak + float(margin_v))
    v_stop = max(lock_min_v, v_peak - float(extra_v))
    if v_start <= v_stop:
        raise ValueError(
            f"Approach window is empty: start {v_start:.4f} V, stop {v_stop:.4f} V."
        )

    print()
    print(f"Coarse tallest peak: {v_peak:+.6f} V")
    voltage, pd, error, plan = drive.fast_error_scan(
        start_v=v_start,
        stop_v=v_stop,
        scan_time=float(scan_time),
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
        "plan": plan,
    }
    print(
        f"PD ptp {log['pd'].max()-log['pd'].min():.4f} V,  "
        f"|error| max {np.max(np.abs(log['error'])):.4f} V"
    )
    return log


def park_at_lock_point(drive, log):
    """Compute Linien lock point from a recorded sweep and hold_dc there."""
    lp = lock_point_from_trace(log["voltage"], log["error"])
    drive.hold_dc(lp["lock_voltage"], verbose=True)
    print(
        f"Lock point {lp['lock_voltage']:+.6f} V  "
        f"(error mid-height {lp['mean_error']:+.4f} V, "
        f"dE/dV {'rising' if lp['slope_rising_vs_voltage'] else 'falling'})"
    )
    if lp["slope_rising_vs_voltage"]:
        print(
            "Slope rises with voltage: if the PID runs away, "
            "add 180 deg to IQ phase (or use a negative I)."
        )
    return lp


def engage_i_lock(rp, i_gain_hz=1.0, p_gain=0.0, monitor_s=3.0, rail_v=0.20):
    """Close I-only PID. IQ/ASG must already be parked on the slope."""
    pid = rp.pid0
    if hasattr(pid, "paused"):
        pid.paused = False
    pid.ival = 0.0
    pid.setpoint = 0.0
    pid.p = 0
    pid.i = 0
    pid.output_direct = "out1"
    print("Closing PID loop (integral only)...")
    pid.i = float(i_gain_hz)
    pid.p = float(p_gain)
    print(f"PID engaged.  P={pid.p}  I={pid.i} Hz")
    print("If it runs away: disengage_lock(); set IQ phase += 180; retry the approach.")

    n = max(1, int(round(float(monitor_s) / 0.5)))
    print(f"Monitoring for {monitor_s:.1f} s...")
    ivals = []
    for k in range(n):
        time.sleep(0.5)
        ival = float(pid.ival)
        try:
            sig = sample_lock_signals(rp, t=0.05)
            print(
                f"  t={0.5 * (k + 1):.1f}s  ival={ival:+.4f} V  "
                f"error={sig['error']:+.4f} V  PD={sig['pd']:+.4f} V  "
                f"asg0={sig['asg']:+.4f} V"
            )
        except Exception:
            print(f"  t={0.5 * (k + 1):.1f}s  ival={ival:+.4f} V")
        ivals.append(ival)

    rail = 0.90 * abs(float(rail_v))
    if ivals and abs(ivals[-1]) >= rail:
        print(
            "WARNING: PID correction is against the rail. "
            "Wrong sign or missed the slope. "
            "disengage_lock(); add 180 deg to the IQ phase; re-approach."
        )
        return False
    print("PID correction stayed inside the rails.")
    return True


def disengage_lock(drive, rp):
    """Zero PID/IQ and drop the ASG to idle."""
    silence_feedback(rp)
    drive.release_to_zero()
    print(f"Lock off. Drive at {drive.current_v:+.6f} V")
