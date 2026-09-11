from laserlock.koheron import (
    FakeTransport,
    KoheronLaser,
    LaserSettings,
    build_remote_query_script,
    parse_ctl200_reply,
)


def test_parse_strips_prompt_and_echo():
    raw = ">>\r\n>>rtset\r\n10493.000000\r\n>>"
    assert parse_ctl200_reply(raw, "rtset") == "10493.000000"


def test_parse_write_echo():
    raw = ">>rtset 12000\r\n12000.000000\r\n"
    assert parse_ctl200_reply(raw, "rtset 12000") == "12000.000000"


def test_on_enables_tec_then_lason_and_does_not_sleep(monkeypatch):
    slept = []
    import laserlock.koheron as koheron

    monkeypatch.setattr(koheron.time, "sleep", lambda s: slept.append(s))
    bus = FakeTransport(
        replies={
            "model": "CTL200-1-B-200",
            "version": "V0.21",
            "err": "0",
        }
    )
    laser = KoheronLaser(bus)
    laser.on()
    assert "tecon 1" in bus.sent
    assert "lason 1" in bus.sent
    assert bus.sent.index("tecon 1") < bus.sent.index("lason 1")
    assert "ilaser 250" in bus.sent
    assert "ilmax 250" in bus.sent
    assert "pgain 0.001" in bus.sent
    assert "igain 0.0005" in bus.sent
    assert "dgain 0.0001" in bus.sent
    # apply_settings / UART pacing uses sleep; on() itself must not wait warmup.
    assert 300.0 not in slept
    assert all(s < 1.0 for s in slept)


def test_set_rtset_does_not_touch_lason():
    bus = FakeTransport(replies={"rtset 10500.0": "10500.000000", "rtset": "10500.000000"})
    laser = KoheronLaser(bus)
    laser.set_rtset(10500)
    assert any(cmd.startswith("rtset 10500") for cmd in bus.sent)
    assert not any(cmd.startswith("lason") for cmd in bus.sent)
    assert laser.settings.rtset_ohm == 10500


def test_off_leaves_tec_on_by_default():
    bus = FakeTransport()
    laser = KoheronLaser(bus)
    laser.off()
    assert "lason 0" in bus.sent
    assert "tecon 0" not in bus.sent


def test_off_can_disable_tec():
    bus = FakeTransport()
    laser = KoheronLaser(bus)
    laser.off(tec_off=True)
    assert bus.sent == ["lason 0", "tecon 0"]


def test_apply_settings_does_not_change_lason():
    bus = FakeTransport()
    laser = KoheronLaser(bus, settings=LaserSettings(rtset_ohm=10493, ilaser_ma=250))
    laser.apply_settings()
    assert not any(cmd.startswith("lason") for cmd in bus.sent)
    assert not any(cmd.startswith("tecon") for cmd in bus.sent)
    assert "ilaser 250" in bus.sent
    assert "ilmax 250" in bus.sent
    assert any(cmd.startswith("rtset 10493") for cmd in bus.sent)


def test_wait_stable_uses_override_seconds(monkeypatch):
    slept = []
    import laserlock.koheron as koheron

    monkeypatch.setattr(koheron.time, "sleep", lambda s: slept.append(s))
    laser = KoheronLaser(FakeTransport(), settings=LaserSettings(warmup_s=300))
    laser.wait_stable(12)
    assert slept == [12]


def test_wait_stable_default_is_warmup_knob(monkeypatch):
    slept = []
    import laserlock.koheron as koheron

    monkeypatch.setattr(koheron.time, "sleep", lambda s: slept.append(s))
    laser = KoheronLaser(FakeTransport(), settings=LaserSettings(warmup_s=7.5))
    laser.wait_stable()
    assert slept == [7.5]


def test_connect_does_not_talk_to_the_ctl200():
    bus = FakeTransport(replies={"model": "", "version": ""}, default="")
    laser = KoheronLaser.connect(transport=bus)
    assert bus.sent == []
    assert laser.transport is bus


def test_remote_script_is_valid_python_and_embeds_cmds():
    script = build_remote_query_script("/dev/ttyPS1", 115200, 1.0, ["rtset 10493", "lason 1"])
    compile(script, "<remote>", "exec")
    assert "rtset 10493" in script
    assert "lason 1" in script


def test_close_does_not_send_lason():
    bus = FakeTransport()
    laser = KoheronLaser(bus)
    laser.close()
    assert bus.closed is True
    assert bus.sent == []
