"""Slow downhill hunt: pass peak 3, catch peak 2, freeze ASG."""

from __future__ import annotations

import time

from .catch import SkipThenCatch
from .lockin import sample_lock_signals

# After walking slightly past the error zero, do not jump more than this
# back onto the line (keeps the correction a small step).
_MAX_STEP_BACK_V = 0.008
# Arm re-seed must not inherit Doppler error as "noise".
_ARM_ERR_THRESH_CAP_V = 0.008


def step_back_to_zero(v_overshoot, v_zero, max_back_v=_MAX_STEP_BACK_V):
    """From a sample past the error zero, step back (uphill) onto the zero."""
    v_now = float(v_overshoot)
    v_zero = float(v_zero)
    max_back_v = abs(float(max_back_v))
    # Downhill walk: zero is at higher V than the overshoot sample.
    if v_zero > v_now:
        return min(v_zero, v_now + max_back_v)
    if v_now > v_zero:
        return max(v_zero, v_now - max_back_v)
    return v_zero


def hunt_peak2_downhill(
    drive,
    rp,
    v_start,
    v_stop,
    step_v=0.0004,
    rate_v_s=0.001,
    sample_t=0.01,
    n_skip=1,
    gap_dv=0.010,
    seed_s=0.4,
    print_every=25,
    arm_below_v=None,
    max_after_landmark_v=None,
    catch_hint_v=None,
    catch_window_v=0.025,
    landmark_hint_v=None,
    min_pd_rise_v=0.020,
):
    """Walk high→low with ``hold_dc`` steps. Catch S-curve ``n_skip+1``.

    Returns ``(detector, log)``. ``log`` is a list of dicts (v, error, pd, state).
    On success ``detector.state == "caught"``, ``detector.v_catch`` is the
    interpolated error zero, and the ASG is stepped back onto that zero
    (``detector.v_lock``) rather than left past it.

    ``arm_below_v``: ignore S-curves while OUT1 is still above this
    (Doppler well). Re-seed noise thresholds when the walk reaches it.

    ``max_after_landmark_v``: once peak 3 is marked, do not walk more
    than this far past it (stop on / just after peak 2).

    ``catch_hint_v``: ignore a “peak 2” catch that is farther than
    ``catch_window_v`` from this voltage (survey peak 2). After peak 3
    is marked, the hint is shifted by the *live* landmark minus the
    survey p3−p2 spacing. Downhill peaks can sit tens of millivolts
    above the uphill survey (hysteresis); the first PD-peak S-curve
    is peak 3, the next is peak 2.
    """
    v_start = float(v_start)
    v_stop = float(v_stop)
    step_v = abs(float(step_v))
    rate_v_s = max(float(rate_v_s), 1e-6)
    dt_step = step_v / rate_v_s
    arm_below_v = None if arm_below_v is None else float(arm_below_v)
    max_after_landmark_v = (
        None if max_after_landmark_v is None else abs(float(max_after_landmark_v))
    )
    catch_hint_v = None if catch_hint_v is None else float(catch_hint_v)
    catch_window_v = abs(float(catch_window_v))
    landmark_hint_v = None if landmark_hint_v is None else float(landmark_hint_v)
    min_pd_rise_v = max(0.0, float(min_pd_rise_v))
    hint_retargeted = False

    det = SkipThenCatch(
        n_skip=n_skip,
        gap_dv=gap_dv,
        arm_below_v=arm_below_v,
        min_pd_rise_v=min_pd_rise_v,
    )
    drive.hold_dc(v_start, verbose=True)
    print(f"Hunt start {v_start:+.6f} V → stop {v_stop:+.6f} V  (OUT1 decreasing)")
    print("  order: start → skip peak 3 → catch peak 2 (lock target)")
    print(f"  step {1e3 * step_v:.2f} mV,  rate {1e3 * rate_v_s:.2f} mV/s,  {dt_step:.3f} s/step")
    if arm_below_v is not None:
        print(f"  ignore features until {arm_below_v:+.6f} V")
    if max_after_landmark_v is not None:
        print(
            f"  after peak 3, stop within {1e3 * max_after_landmark_v:.0f} mV "
            "(do not crawl through peak 1)"
        )

    n_seed = max(8, int(round(float(seed_s) / max(sample_t, 0.005))))
    pds, errs = [], []
    for _ in range(n_seed):
        sig = sample_lock_signals(rp, t=sample_t)
        pds.append(sig["pd"])
        errs.append(sig["error"])
    seed = det.seed_wing(pds, errs)
    print(
        f"  wing PD {seed['pd_baseline']:+.4f} V,  "
        f"err thresh {1e3 * seed['err_thresh']:.2f} mV"
    )

    log = []
    v = v_start
    n = 0
    last_state = det.state
    armed = arm_below_v is None
    t0 = time.time()
    while v >= v_stop - 1e-12:
        t_step = time.time()
        drive.hold_dc(v, verbose=False)
        sig = sample_lock_signals(rp, t=sample_t)
        if (
            not armed
            and arm_below_v is not None
            and v <= arm_below_v + 1e-12
        ):
            recent = log[-24:]
            pds = [row["pd"] for row in recent] or [sig["pd"]]
            errs = [row["error"] for row in recent] or [sig["error"]]
            quiet = [
                row
                for row in recent
                if abs(row["error"]) < 0.015
            ]
            if len(quiet) >= 6:
                pds = [row["pd"] for row in quiet]
                errs = [row["error"] for row in quiet]
            seed = det.seed_wing(pds, errs)
            det.inner.err_thresh = min(det.inner.err_thresh, _ARM_ERR_THRESH_CAP_V)
            det.reset()
            armed = True
            print(
                f"  armed      V={v:+.6f}  "
                f"err thresh {1e3 * det.inner.err_thresh:.2f} mV  "
                f"(was Doppler; now near peak 3)"
            )
        state = det.update(v, sig["error"], sig["pd"])
        if state == "hunt" and last_state == "gap":
            quiet = [
                row
                for row in log[-30:]
                if abs(row["error"]) < 0.008
            ]
            if len(quiet) >= 6:
                seed = det.seed_wing(
                    [row["pd"] for row in quiet],
                    [row["error"] for row in quiet],
                )
                det.inner.err_thresh = min(
                    det.inner.err_thresh, _ARM_ERR_THRESH_CAP_V
                )
                print(
                    f"  re-seed    V={v:+.6f}  "
                    f"err thresh {1e3 * det.inner.err_thresh:.2f} mV  "
                    f"(quiet wing between peak 3 and peak 2)"
                )
        log.append(
            {
                "voltage": v,
                "error": sig["error"],
                "pd": sig["pd"],
                "state": state,
            }
        )
        if state != last_state:
            print(
                f"  {state:10s}  V={v:+.6f}  err={sig['error']:+.4f}  "
                f"PD={sig['pd']:+.4f}"
            )
            last_state = state
        elif print_every and n % int(print_every) == 0:
            print(
                f"  {state:10s}  V={v:+.6f}  err={sig['error']:+.4f}  "
                f"PD={sig['pd']:+.4f}"
            )
        n += 1
        if (
            not hint_retargeted
            and state == "landmark"
            and catch_hint_v is not None
            and landmark_hint_v is not None
            and det.v_landmark is not None
        ):
            spacing = landmark_hint_v - catch_hint_v
            if spacing > 0.005:
                new_hint = float(det.v_landmark) - spacing
                print(
                    f"  peak-2 hint {catch_hint_v:+.6f} → {new_hint:+.6f} V  "
                    f"(live peak 3, survey spacing {1e3 * spacing:.0f} mV)"
                )
                catch_hint_v = new_hint
            hint_retargeted = True
        if state == "skip":
            print(
                f"  skip line  V={v:+.6f}  PD={sig['pd']:+.4f}  "
                f"(above survey peak 3 {landmark_hint_v:+.6f}; keep looking)"
            )
            last_state = "skip"
        if state == "caught":
            valley = (
                det.pd_landmark is not None
                and sig["pd"] < det.pd_landmark - 0.040
            )
            if valley:
                print(
                    f"  skip catch V={det.v_catch:+.6f}  "
                    f"PD={sig['pd']:+.4f}  (valley, not a peak; keep walking)"
                )
                det.v_catch = None
                det.v_overshoot = None
                det.passed = det.n_skip
                det.phase = "hunt"
                det.state = "hunt"
                det.inner.reset()
                det.inner.require_pd = True
                last_state = "hunt"
            elif (
                catch_hint_v is not None
                and abs(float(det.v_catch) - catch_hint_v) > catch_window_v
            ):
                # Too far from survey peak 2 (lab: Doppler / peak-3 wing).
                # Count it as the skipped landmark and gap before hunting again.
                print(
                    f"  skip catch V={det.v_catch:+.6f}  "
                    f"(not near survey peak 2 {catch_hint_v:+.6f}; "
                    "treat as peak 3, keep walking)"
                )
                det.v_landmark = float(det.v_catch)
                det.v_catch = None
                det.passed = det.n_skip
                det.phase = "gap"
                det.state = "gap"
                det.inner.reset()
                last_state = "gap"
            else:
                break
        if (
            max_after_landmark_v is not None
            and det.v_landmark is not None
            and (det.v_landmark - v) >= max_after_landmark_v
        ):
            print(
                f"  stop       V={v:+.6f}  "
                f"(peak 3 seen, {1e3 * (det.v_landmark - v):.1f} mV past landmark)"
            )
            break
        leftover = dt_step - (time.time() - t_step)
        if leftover > 0:
            time.sleep(leftover)
        v -= step_v

    elapsed = time.time() - t0
    v_now = float(log[-1]["voltage"]) if log else v_start
    if det.state == "caught" and det.v_catch is not None:
        v_overshoot = (
            float(det.v_overshoot)
            if det.v_overshoot is not None
            else v_now
        )
        freeze_v = step_back_to_zero(v_overshoot, det.v_catch)
        det.v_overshoot = v_overshoot
        det.v_lock = freeze_v
        print(
            f"  past zero at {v_overshoot:+.6f} V;  "
            f"step back to {freeze_v:+.6f} V (error zero)"
        )
    elif log:
        freeze_v = v_now
        det.v_lock = None
    else:
        freeze_v = v_start
        det.v_lock = None
    drive.hold_dc(freeze_v, verbose=True)
    if det.state == "caught":
        print(
            f"Caught peak 2 at {det.v_catch:+.6f} V  "
            f"(landmark {det.v_landmark:+.6f} V)  in {elapsed:.1f} s"
        )
        print(f"Frozen at {freeze_v:+.6f} V (step back from {v_overshoot:+.6f} V).")
    else:
        print(f"Hunt ended without catch ({elapsed:.1f} s, {len(log)} steps).")
    return det, log
