from unittest.mock import patch

from laserlock.lockin import apply_pid, flip_iq_phase, open_pid, ramp_pid_gains, set_iq_phase


class FakePid:
    def __init__(self):
        self.p = 0.0
        self.i = 0.0
        self.ival = 0.0
        self.setpoint = 0.0
        self.output_direct = "off"
        self.paused = False
        self.ps = []
        self.is_ = []


class FakeIQ:
    def __init__(self, phase=0.0):
        self.phase = float(phase)


class FakeRP:
    def __init__(self):
        self.pid0 = FakePid()
        self.iq0 = FakeIQ()


def test_ramp_reaches_finals_from_zero():
    pid = FakePid()

    def step(k, n, p):
        p.ps.append(p.p)
        p.is_.append(p.i)

    assert ramp_pid_gains(pid, 0.1, 10.0, ramp_s=1.0, dt=0.25, on_step=step)
    assert pid.p == 0.1
    assert pid.i == 10.0
    assert pid.ps[0] < 0.1
    assert pid.is_[0] < 10.0
    assert pid.ps[-1] == 0.1


def test_ramp_abort_stops_below_final():
    pid = FakePid()

    def step(k, n, p):
        if k >= 2:
            return False
        return True

    assert ramp_pid_gains(pid, 1.0, 100.0, ramp_s=1.0, dt=0.25, on_step=step) is False
    assert pid.p < 1.0
    assert pid.i < 100.0


def test_open_pid_does_not_touch_iq_phase():
    rp = FakeRP()
    rp.iq0.phase = 33.0
    rp.pid0.p = 1.0
    rp.pid0.i = 10.0
    rp.pid0.output_direct = "out1"
    open_pid(rp)
    assert rp.pid0.output_direct == "off"
    assert rp.pid0.p == 0
    assert rp.pid0.i == 0
    assert rp.iq0.phase == 33.0


def test_apply_pid_closes_loop_immediately():
    rp = FakeRP()
    rp.iq0.phase = 0.0
    with patch("laserlock.lockin.ensure_dither"):
        pid = apply_pid(rp, p=0.0, i_hz=-3.0, monitor_s=0.0)
    assert pid.output_direct == "out1"
    assert pid.p == 0.0
    assert pid.i == -3.0
    assert pid.ival == 0.0
    assert pid.inputfilter == [0, 0, 0, 0]


def test_flip_iq_phase_adds_180():
    rp = FakeRP()
    rp.iq0.phase = 0.0
    flip_iq_phase(rp)
    assert abs(abs(rp.iq0.phase) - 180.0) < 1e-6
    set_iq_phase(rp, 0.0)
    assert abs(rp.iq0.phase) < 1e-6
