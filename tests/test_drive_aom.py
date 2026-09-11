from laserlock.drive import LaserDrive


class FakeAsg:
    def __init__(self):
        self.output_direct = "off"
        self.frequency = 0.0
        self.amplitude = 0.0
        self.offset = 0.0
        self.trigger_source = "off"
        self.waveform = None
        self.setup_calls = []

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeScope:
    data_length = 16384


class FakeRP:
    def __init__(self):
        self.asg0 = FakeAsg()
        self.asg1 = FakeAsg()
        self.scope = FakeScope()


def test_start_aom_routes_asg1_to_out2_not_out1():
    rp = FakeRP()
    drive = LaserDrive(rp)
    drive.start_aom(28.8313e6, 0.5)
    assert rp.asg1.output_direct == "out2"
    assert rp.asg1.waveform == "sin"
    assert rp.asg1.amplitude == 0.5
    assert abs(rp.asg1.frequency - 28.8313e6) < 1.0
    assert rp.asg0.output_direct == "off"
    assert rp.asg0.setup_calls == []


def test_stop_aom_disconnects_out2():
    rp = FakeRP()
    drive = LaserDrive(rp)
    drive.start_aom(28.8313e6, 0.5)
    drive.stop_aom()
    assert rp.asg1.output_direct == "off"
    assert rp.asg1.amplitude == 0.0
    assert rp.asg0.output_direct == "off"


def test_shutdown_stops_aom():
    rp = FakeRP()
    drive = LaserDrive(rp)
    drive.start_aom(28.8313e6, 0.5)
    drive.shutdown(slew_time=0.01)
    assert rp.asg1.output_direct == "off"
    assert rp.asg1.amplitude == 0.0
