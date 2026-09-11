from laserlock.catch import CatchDetector, SkipThenCatch


def _s_curve_peak(v_center, v_start, v_stop, step, pd_wing, pd_height, err_amp, scale=1.0):
    """Downhill samples through one odd discriminant on a PD bump."""
    points = []
    v = v_start
    while v >= v_stop - 1e-12:
        dv = v - v_center
        # ~8 mV wide Gaussian-like bump and odd error
        pd = pd_wing + scale * pd_height * (2.718 ** (-(dv ** 2) / (2 * 0.004**2)))
        err = scale * err_amp * (-2.0 * dv / 0.004) * (2.718 ** (-(dv ** 2) / (2 * 0.004**2)))
        points.append((v, err, pd))
        v -= step
    return points


def _seed(det, pd_wing=0.236, err_noise=0.0002, n=30):
    pds = [pd_wing + ((i % 5) - 2) * err_noise * 0.5 for i in range(n)]
    errs = [((i % 7) - 3) * err_noise for i in range(n)]
    return det.seed_wing(pds, errs)


def test_wing_does_not_confirm():
    det = CatchDetector(confirm_dv=0.015)
    _seed(det)
    det.reset()
    states = []
    for i in range(40):
        v = 0.13 - 0.0005 * i
        states.append(det.update(v, 0.0002, 0.236))
    assert "confirmed" not in states
    assert states[-1] == "search"


def test_confirms_first_s_curve_not_the_next_line():
    det = CatchDetector(confirm_dv=0.015)
    _seed(det, pd_wing=0.236)
    det.reset()
    first = _s_curve_peak(
        v_center=0.113,
        v_start=0.126,
        v_stop=0.090,
        step=0.0005,
        pd_wing=0.236,
        pd_height=0.006,
        err_amp=0.007,
    )
    second = _s_curve_peak(
        v_center=0.044,
        v_start=0.0895,
        v_stop=0.030,
        step=0.0005,
        pd_wing=0.224,
        pd_height=0.004,
        err_amp=0.004,
    )
    states = []
    for v, e, pd in first + second:
        states.append(det.update(v, e, pd))
        if det.state == "confirmed":
            break
    assert det.state == "confirmed"
    assert abs(det.v_zero - 0.113) < 0.004


def test_noise_spike_without_opposite_lobe_is_rejected():
    det = CatchDetector(confirm_dv=0.010)
    _seed(det)
    det.reset()
    # PD bump and a one-sided error blip, then back to wing
    points = []
    for i in range(20):
        v = 0.12 - 0.0005 * i
        points.append((v, 0.0001, 0.236))
    # spike
    points += [
        (0.110, -0.006, 0.239),
        (0.1095, -0.007, 0.240),
        (0.1090, 0.001, 0.239),  # would look like a zero if we did not confirm
        (0.1085, 0.0002, 0.237),
        (0.1080, 0.0001, 0.236),
    ]
    for i in range(20):
        v = 0.1075 - 0.0005 * i
        points.append((v, 0.0001, 0.236))
    states = [det.update(v, e, pd) for v, e, pd in points]
    assert "confirmed" not in states


def test_thresholds_scale_with_the_wing():
    """Same shape, 3× larger signals, still confirms near the same voltage."""
    zeros = []
    for scale in (1.0, 3.0):
        det = CatchDetector(confirm_dv=0.015)
        _seed(det, pd_wing=0.236 * scale, err_noise=0.0002 * scale)
        det.reset()
        points = _s_curve_peak(
            v_center=0.100,
            v_start=0.120,
            v_stop=0.080,
            step=0.0005,
            pd_wing=0.236 * scale,
            pd_height=0.006,
            err_amp=0.007,
            scale=1.0 if scale == 1.0 else 3.0,
        )
        # When scale==3, pd_wing already *3 via argument; don't triple again
        if scale == 3.0:
            points = _s_curve_peak(
                v_center=0.100,
                v_start=0.120,
                v_stop=0.080,
                step=0.0005,
                pd_wing=0.236 * 3,
                pd_height=0.006 * 3,
                err_amp=0.007 * 3,
                scale=1.0,
            )
        for v, e, pd in points:
            det.update(v, e, pd)
            if det.state == "confirmed":
                zeros.append(det.v_zero)
                break
        else:
            zeros.append(None)
    assert zeros[0] is not None and zeros[1] is not None
    assert abs(zeros[0] - zeros[1]) < 0.003


def test_shoulder_start_with_weak_approaching_lobe():
    """Lab trace: start PD almost as high as peak 3; first lobe ~4 mV.

    Absolute 4σ vs the shoulder (~5.8 mV) missed that lobe and never confirmed.
    """
    det = CatchDetector(n_sigma=2.5, confirm_dv=0.015)
    pds = [0.2318 + ((i % 5) - 2) * 0.00025 for i in range(25)]
    errs = [((i % 7) - 3) * 0.0004 for i in range(25)]
    det.seed_wing(pds, errs)
    det.reset()
    first = _s_curve_peak(
        v_center=0.170,
        v_start=0.195,
        v_stop=0.150,
        step=0.0005,
        pd_wing=0.2318,
        pd_height=0.0032,
        err_amp=0.0045,
    )
    second = _s_curve_peak(
        v_center=0.130,
        v_start=0.1495,
        v_stop=0.110,
        step=0.0005,
        pd_wing=0.226,
        pd_height=0.006,
        err_amp=0.008,
    )
    for v, e, pd in first + second:
        det.update(v, e, pd)
        if det.state == "confirmed":
            break
    assert det.state == "confirmed"
    assert abs(det.v_zero - 0.170) < 0.005


def test_close_first_peak_is_not_eaten_by_a_blind_window():
    """Peak only ~11 mV below the start — the old ignore_dv=12 mV missed this."""
    det = CatchDetector(confirm_dv=0.015)
    _seed(det)
    det.reset()
    points = _s_curve_peak(
        v_center=0.115,
        v_start=0.126,
        v_stop=0.095,
        step=0.0005,
        pd_wing=0.236,
        pd_height=0.006,
        err_amp=0.007,
    )
    for v, e, pd in points:
        det.update(v, e, pd)
        if det.state == "confirmed":
            break
    assert det.state == "confirmed"
    assert abs(det.v_zero - 0.115) < 0.004


def test_skip_then_catch_locks_the_second_line():
    """Downhill: peak 3 then peak 2. Catch peak 2, not peak 3."""
    det = SkipThenCatch(n_skip=1, gap_dv=0.008, confirm_dv=0.015)
    _seed(det.inner)
    det.reset()
    p3 = _s_curve_peak(
        v_center=0.170,
        v_start=0.195,
        v_stop=0.150,
        step=0.0005,
        pd_wing=0.230,
        pd_height=0.005,
        err_amp=0.006,
    )
    p2 = _s_curve_peak(
        v_center=0.130,
        v_start=0.1495,
        v_stop=0.100,
        step=0.0005,
        pd_wing=0.224,
        pd_height=0.008,
        err_amp=0.009,
    )
    states = []
    for v, e, pd in p3 + p2:
        states.append(det.update(v, e, pd))
        if det.state == "caught":
            break
    assert det.state == "caught"
    assert det.v_landmark is not None
    assert abs(det.v_landmark - 0.170) < 0.006
    assert abs(det.v_catch - 0.130) < 0.006
    assert "landmark" in states


def test_dc_error_on_the_wing_does_not_inflate_threshold():
    """Park on a Doppler slope: error has a DC offset, noise is small."""
    det = CatchDetector(n_sigma=2.5, err_floor=0.0015)
    pds = [0.40 + ((i % 5) - 2) * 0.0003 for i in range(25)]
    errs = [0.076 + ((i % 7) - 3) * 0.0004 for i in range(25)]
    seed = det.seed_wing(pds, errs)
    assert seed["err_thresh"] < 0.020
    assert seed["err_thresh"] >= 0.0015


def test_skip_then_catch_ignores_doppler_above_arm():
    """A strong S-curve well above peak 3 is not the landmark."""
    det = SkipThenCatch(
        n_skip=1, gap_dv=0.008, confirm_dv=0.015, arm_below_v=0.180
    )
    _seed(det.inner)
    det.reset()
    doppler = _s_curve_peak(
        v_center=0.220,
        v_start=0.240,
        v_stop=0.200,
        step=0.0005,
        pd_wing=0.50,
        pd_height=0.08,
        err_amp=0.08,
    )
    p3 = _s_curve_peak(
        v_center=0.170,
        v_start=0.195,
        v_stop=0.150,
        step=0.0005,
        pd_wing=0.230,
        pd_height=0.005,
        err_amp=0.006,
    )
    p2 = _s_curve_peak(
        v_center=0.130,
        v_start=0.1495,
        v_stop=0.100,
        step=0.0005,
        pd_wing=0.224,
        pd_height=0.008,
        err_amp=0.009,
    )
    states = []
    for v, e, pd in doppler + p3 + p2:
        states.append(det.update(v, e, pd))
        if det.state == "caught":
            break
    assert "approach" in states
    assert det.state == "caught"
    assert abs(det.v_landmark - 0.170) < 0.006
    assert abs(det.v_catch - 0.130) < 0.006


def test_zero_is_interpolated_between_straddle_samples():
    det = CatchDetector(confirm_dv=0.015, past_zero_v=0.002, require_pd=False)
    _seed(det)
    det.reset()
    # Sign change between 0.1005 (neg) and 0.1000 (pos).
    det.update(0.1015, -0.006, 0.242)
    det.update(0.1010, -0.004, 0.243)
    det.update(0.1005, -0.002, 0.243)
    det.update(0.1000, +0.002, 0.243)
    assert det.state == "confirmed"
    assert 0.1000 < det.v_zero < 0.1005
    assert abs(det.v_overshoot - 0.1000) < 1e-12


def test_error_only_mode_keeps_a_weak_pd_lobe():
    """Peak 2 can be a small PD bump; PD flicker must not abort the error zero."""
    det = CatchDetector(require_pd=False, past_zero_v=0.002, confirm_dv=0.015)
    _seed(det)
    det.reset()
    points = _s_curve_peak(
        v_center=0.130,
        v_start=0.145,
        v_stop=0.115,
        step=0.0005,
        pd_wing=0.440,
        pd_height=0.001,
        err_amp=0.012,
    )
    for v, e, pd in points:
        det.update(v, e, pd)
        if det.state == "confirmed":
            break
    assert det.state == "confirmed"
    assert abs(det.v_zero - 0.130) < 0.004


def test_high_line_is_not_peak3_landmark():
    """A bright S-curve well above survey peak 3 is skipped, then p3 then p2."""
    det = SkipThenCatch(
        n_skip=1, gap_dv=0.008, confirm_dv=0.015, landmark_max_v=0.190
    )
    _seed(det.inner)
    det.reset()
    extra = _s_curve_peak(
        v_center=0.220,
        v_start=0.240,
        v_stop=0.200,
        step=0.0005,
        pd_wing=0.50,
        pd_height=0.15,
        err_amp=0.06,
    )
    p3 = _s_curve_peak(
        v_center=0.170,
        v_start=0.195,
        v_stop=0.150,
        step=0.0005,
        pd_wing=0.230,
        pd_height=0.005,
        err_amp=0.006,
    )
    p2 = _s_curve_peak(
        v_center=0.130,
        v_start=0.1495,
        v_stop=0.100,
        step=0.0005,
        pd_wing=0.224,
        pd_height=0.008,
        err_amp=0.009,
    )
    states = []
    for v, e, pd in extra + p3 + p2:
        states.append(det.update(v, e, pd))
        if det.state == "caught":
            break
    assert "skip" in states
    assert det.state == "caught"
    assert abs(det.v_landmark - 0.170) < 0.006
    assert abs(det.v_catch - 0.130) < 0.006


def test_skip_then_catch_does_not_freeze_on_the_landmark():
    det = SkipThenCatch(n_skip=1, gap_dv=0.008, confirm_dv=0.015)
    _seed(det.inner)
    det.reset()
    p3 = _s_curve_peak(
        v_center=0.170,
        v_start=0.195,
        v_stop=0.150,
        step=0.0005,
        pd_wing=0.230,
        pd_height=0.005,
        err_amp=0.006,
    )
    for v, e, pd in p3:
        det.update(v, e, pd)
    assert det.state != "caught"
    assert det.passed == 1
