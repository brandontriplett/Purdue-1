"""Hunt walk: skip a peak-3-wing catch and freeze on peak 2."""

import math
from unittest.mock import patch

from laserlock.hunt import hunt_peak2_downhill, step_back_to_zero


class FakeDrive:
    def __init__(self):
        self.current_v = 0.0

    def hold_dc(self, v, verbose=False):
        self.current_v = float(v)


def _pd_err(v, peaks, w=0.004):
    pd = 0.60
    err = 0.0
    w = float(w)
    for vc, height, eamp in peaks:
        dv = v - vc
        g = math.exp(-(dv**2) / (2 * w**2))
        pd += height * g
        err += eamp * (-2.0 * dv / w) * g
    return pd, err


def _run(peaks, w=0.004, **kwargs):
    drive = FakeDrive()

    def sample(_rp, t=0.01):
        pd, err = _pd_err(drive.current_v, peaks, w=w)
        return {"pd": pd, "error": err}

    defaults = dict(
        v_start=0.30,
        v_stop=0.12,
        step_v=0.0005,
        rate_v_s=1.0,
        sample_t=0.01,
        seed_s=0.08,
        print_every=0,
        n_skip=1,
        gap_dv=0.010,
    )
    defaults.update(kwargs)
    with patch("laserlock.hunt.sample_lock_signals", side_effect=sample):
        with patch("laserlock.hunt.time.sleep", return_value=None):
            det, log = hunt_peak2_downhill(drive, rp=None, **defaults)
    return det, log, drive


# Last lab hunt: Doppler ~0.254, peak-3 wing ~0.217, survey peak 2 ~0.151.
_LAB = (
    (0.254, 0.04, 0.04),
    (0.217, 0.08, 0.05),
    (0.151, 0.12, 0.07),
)


def test_step_back_to_zero_moves_uphill():
    assert abs(step_back_to_zero(0.148, 0.151) - 0.151) < 1e-12
    # Cap at 8 mV.
    assert abs(step_back_to_zero(0.140, 0.151) - 0.148) < 1e-12


def test_old_arm_catches_peak3_wing():
    """Arming 5 mV below start treats Doppler as peak 3 and freezes on 0.217."""
    det, _log, _drive = _run(
        _LAB,
        arm_below_v=0.295,
        max_after_landmark_v=0.10,
        catch_hint_v=None,
    )
    assert det.state == "caught"
    assert abs(det.v_catch - 0.217) < 0.008


def test_hint_skips_peak3_wing_and_catches_peak2():
    """Catch far from survey peak 2 is the landmark; walk on to peak 2."""
    det, _log, drive = _run(
        _LAB,
        arm_below_v=0.295,
        max_after_landmark_v=0.10,
        catch_hint_v=0.151,
        catch_window_v=0.040,
    )
    assert det.state == "caught"
    assert abs(det.v_catch - 0.151) < 0.008
    assert abs(drive.current_v - det.v_catch) < 1e-9
    assert det.v_landmark is not None
    assert abs(det.v_landmark - 0.217) < 0.010


def test_arm_above_survey_p3_skips_doppler():
    """Arm ~40 mV above survey peak 3: Doppler is ignored, 0.217 is peak 3."""
    det, _log, _drive = _run(
        _LAB,
        arm_below_v=0.230,
        max_after_landmark_v=0.10,
        catch_hint_v=0.151,
        catch_window_v=0.040,
    )
    assert det.state == "caught"
    assert abs(det.v_landmark - 0.217) < 0.010
    assert abs(det.v_catch - 0.151) < 0.008


def test_step_back_from_past_zero_onto_the_line():
    """Walk slightly past the error zero, then freeze uphill on the zero."""
    det, log, drive = _run(
        _LAB,
        arm_below_v=0.230,
        max_after_landmark_v=0.10,
        catch_hint_v=0.151,
        catch_window_v=0.040,
    )
    assert det.state == "caught"
    assert det.v_overshoot is not None
    assert det.v_lock is not None
    assert det.v_overshoot <= det.v_lock + 1e-12
    assert abs(drive.current_v - det.v_lock) < 1e-9
    assert abs(det.v_lock - 0.151) < 0.008
    # Last walk sample is at or downhill of the freeze.
    assert log[-1]["voltage"] <= det.v_lock + 1e-12


def test_hysteresis_shifted_p3_is_still_the_landmark():
    """Uphill survey p3=+0.190 p2=+0.148; downhill live p3=+0.259 p2=+0.217.

    Last lab run skipped +0.259 as 'too high' and called peak 2 peak 3.
    """
    live = (
        (0.259, 0.15, 0.06),
        (0.217, 0.17, 0.07),
        (0.150, 0.06, 0.04),
    )
    det, _log, drive = _run(
        live,
        w=0.010,
        v_start=0.285,
        v_stop=0.128,
        arm_below_v=0.280,
        gap_dv=0.025,
        max_after_landmark_v=0.072,
        catch_hint_v=0.148,
        catch_window_v=0.040,
        landmark_hint_v=0.190,
    )
    assert det.state == "caught"
    assert abs(det.v_landmark - 0.259) < 0.012
    assert abs(det.v_lock - 0.217) < 0.012
    assert abs(drive.current_v - det.v_lock) < 1e-9


def test_start_closer_to_peak3_still_catches_peak2():
    """Start ~75 mV above peak 3 (survey 0.217 → start 0.292, use 0.26)."""
    det, _log, drive = _run(
        _LAB,
        v_start=0.260,
        v_stop=0.12,
        arm_below_v=0.230,
        max_after_landmark_v=0.10,
        catch_hint_v=0.151,
        catch_window_v=0.040,
    )
    assert det.state == "caught"
    assert abs(det.v_lock - 0.151) < 0.008
    assert abs(drive.current_v - det.v_lock) < 1e-9
