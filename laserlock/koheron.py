"""Koheron CTL200 diode + TEC, over the Red Pitaya UART.

Does not import pyrpl and does not touch the FPGA. Closing this client
does not change lason — turn the diode off with ``off()``.

``on()`` never waits. Thermal settle is ``wait_stable()``, for headless
use; skip it when you want the notebooks immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Protocol


DEFAULT_PORT = "/dev/ttyPS1"
DEFAULT_HOSTNAME = "rp-f0c970.local"
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "root"
DEFAULT_SSH_PORT = 22
BAUD = 115200
QUERY_TIMEOUT_S = 1.0


@dataclass
class LaserSettings:
    rtset_ohm: float = 10493.0
    pgain: float = 0.001
    igain: float = 0.0005
    dgain: float = 0.0001
    ilaser_ma: float = 250.0
    ilmax_ma: float = 250.0
    lckon: int = 0
    tprot: int = 1
    rtmin_ohm: float = 10000.0
    rtmax_ohm: float = 11000.0
    vtmin_v: float = -3.0
    vtmax_v: float = 3.0
    warmup_s: float = 300.0


class Transport(Protocol):
    def query_many(self, cmds: list[str]) -> list[str]: ...
    def close(self) -> None: ...


def parse_ctl200_reply(raw: str, cmd: str) -> str:
    """Last payload line: drop prompts, blanks, and the echoed command."""
    cmd = (cmd or "").strip()
    verb = cmd.split()[0] if cmd else ""
    lines = []
    for line in raw.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line.startswith(">>"):
            line = line.lstrip(">").strip()
        if not line:
            continue
        if cmd and line == cmd:
            continue
        if verb and line == verb:
            continue
        lines.append(line)
    return lines[-1] if lines else ""


def _as_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _as_int(text: str):
    value = _as_float(text)
    if value is None:
        return None
    return int(value)


def _cmd_value(value) -> str:
    """Match the archived CTL200 style: ``ilaser 250`` not ``ilaser 250.0``."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class LocalSerialTransport:
    def __init__(self, port: str = DEFAULT_PORT, baud: int = BAUD, timeout: float = QUERY_TIMEOUT_S):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for a local CTL200 connection."
            ) from exc
        self._serial = serial.Serial(port, baudrate=int(baud), timeout=float(timeout))
        self.timeout = float(timeout)

    def query_many(self, cmds: list[str]) -> list[str]:
        return [_query_serial(self._serial, cmd) for cmd in cmds]

    def close(self) -> None:
        self._serial.close()


class SshSerialTransport:
    """SSH to the RP (same user/password as PyRPL), then UART one-shot."""

    def __init__(
        self,
        hostname: str = DEFAULT_HOSTNAME,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        ssh_port: int = DEFAULT_SSH_PORT,
        port: str = DEFAULT_PORT,
        baud: int = BAUD,
        timeout: float = QUERY_TIMEOUT_S,
    ):
        self.hostname = hostname
        self.user = user
        self.password = password
        self.ssh_port = int(ssh_port)
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self._client = None

    def _ssh(self):
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
        try:
            import paramiko
        except ImportError as exc:
            raise ImportError(
                "paramiko is required to reach the CTL200 UART from this machine."
            ) from exc
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                self.hostname,
                port=self.ssh_port,
                username=self.user,
                password=self.password,
                timeout=8,
                allow_agent=True,
                look_for_keys=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SSH to {self.user}@{self.hostname} failed: {exc}"
            ) from exc
        self._client = client
        return client

    def ping(self) -> str:
        """SSH only. Does not open the laser UART."""
        client = self._ssh()
        stdin, stdout, stderr = client.exec_command("uname -n", timeout=8)
        name = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(
                f"SSH to {self.user}@{self.hostname} failed: {err or name or status}"
            )
        return name or self.hostname

    def query_many(self, cmds: list[str]) -> list[str]:
        script = build_remote_query_script(
            port=self.port,
            baud=self.baud,
            timeout=self.timeout,
            cmds=list(cmds),
        )
        client = self._ssh()
        try:
            stdin, stdout, stderr = client.exec_command(
                "python3 -",
                timeout=max(20.0, 4.0 + 2.0 * len(cmds)),
            )
        except Exception as exc:
            raise TimeoutError(
                f"SSH to {self.user}@{self.hostname} timed out talking to {self.port}."
            ) from exc
        stdin.write(script)
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(
                f"SSH UART query failed (exit {status}): {(err or out).strip() or 'no stderr'}"
            )
        try:
            raws = json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(
                f"SSH UART returned unreadable output: {out!r} {err!r}"
            ) from exc
        if not isinstance(raws, list) or len(raws) != len(cmds):
            raise RuntimeError(f"SSH UART reply count mismatch: {raws!r}")
        return [str(item) for item in raws]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class FakeTransport:
    """In-memory CTL200 for tests. ``replies`` keys are full cmds or verbs."""

    def __init__(self, replies=None, default="0"):
        self.sent: list[str] = []
        self.replies = dict(replies or {})
        self.default = str(default)
        self.closed = False

    def query_many(self, cmds: list[str]) -> list[str]:
        out = []
        for cmd in cmds:
            self.sent.append(cmd)
            verb = cmd.split()[0]
            raw = self.replies.get(cmd, self.replies.get(verb, self.default))
            out.append(raw)
        return out

    def close(self) -> None:
        self.closed = True


def _query_serial(ser, cmd: str, delay: float = 0.1) -> str:
    """Archived TurnLaserOn send_cmd: write, short wait, drain in_waiting."""
    ser.reset_input_buffer()
    ser.write(f"{cmd}\r\n".encode("ascii"))
    ser.flush()
    time.sleep(delay)
    raw = b""
    while ser.in_waiting:
        raw += ser.read(ser.in_waiting)
        time.sleep(0.01)
    return raw.decode("ascii", errors="replace")


def build_remote_query_script(port: str, baud: int, timeout: float, cmds: list[str]) -> str:
    payload = json.dumps(
        {"port": port, "baud": int(baud), "timeout": float(timeout), "cmds": list(cmds)}
    )
    return _REMOTE_QUERY_SCRIPT.replace("__PAYLOAD__", payload)


_REMOTE_QUERY_SCRIPT = r"""
import json, sys, time
try:
    import serial
except ImportError:
    sys.stderr.write("pyserial is not installed on the Red Pitaya (pip3 install pyserial)\n")
    sys.exit(2)

cfg = json.loads(r'''__PAYLOAD__''')
ser = serial.Serial(cfg["port"], baudrate=int(cfg["baud"]), timeout=float(cfg["timeout"]))
timeout = float(cfg["timeout"])
out = []
try:
    for cmd in cfg["cmds"]:
        ser.reset_input_buffer()
        ser.write((cmd + "\r\n").encode("ascii"))
        ser.flush()
        time.sleep(0.1)
        raw = b""
        while ser.in_waiting:
            raw += ser.read(ser.in_waiting)
            time.sleep(0.01)
        out.append(raw.decode("ascii", errors="replace"))
finally:
    ser.close()
print(json.dumps(out))
"""


class KoheronLaser:
    def __init__(self, transport: Transport, settings: LaserSettings | None = None):
        self.transport = transport
        self.settings = settings or LaserSettings()

    @classmethod
    def connect(
        cls,
        hostname: str | None = DEFAULT_HOSTNAME,
        port: str = DEFAULT_PORT,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        ssh_port: int = DEFAULT_SSH_PORT,
        settings: LaserSettings | None = None,
        transport: Transport | None = None,
    ) -> "KoheronLaser":
        if transport is None:
            if os.path.exists(port):
                print(f"CTL200 local UART {port}")
                transport = LocalSerialTransport(port=port)
            elif hostname:
                print(f"CTL200 UART {port} via ssh {user}@{hostname}")
                transport = SshSerialTransport(
                    hostname=hostname,
                    user=user,
                    password=password,
                    ssh_port=ssh_port,
                    port=port,
                )
            else:
                raise FileNotFoundError(
                    f"No local serial {port} and no hostname for SSH."
                )
        laser = cls(transport, settings=settings)
        if isinstance(transport, SshSerialTransport):
            host = transport.ping()
            print(f"  RP ssh ok ({host}). UART {port} is not probed until On/Status.")
        elif isinstance(transport, LocalSerialTransport):
            print(f"  local serial {port} open. CTL200 is not probed until On/Status.")
        return laser

    def query(self, cmd: str) -> str:
        raw = self.transport.query_many([cmd])[0]
        return parse_ctl200_reply(raw, cmd)

    def query_many(self, cmds: list[str]) -> list[str]:
        raws = self.transport.query_many(cmds)
        return [parse_ctl200_reply(raw, cmd) for raw, cmd in zip(raws, cmds)]

    def get(self, name: str) -> str:
        return self.query(name)

    def set(self, name: str, value) -> str:
        return self.query(f"{name} {value}")

    def apply_settings(self, settings: LaserSettings | None = None) -> None:
        """Write PID, RTSET, and limits. Does not change lason."""
        if settings is not None:
            self.settings = settings
        s = self.settings
        cmds = [
            f"vtmax {_cmd_value(s.vtmax_v)}",
            f"vtmin {_cmd_value(s.vtmin_v)}",
            f"rtmin {_cmd_value(s.rtmin_ohm)}",
            f"rtmax {_cmd_value(s.rtmax_ohm)}",
            f"ilmax {_cmd_value(s.ilmax_ma)}",
            f"ilaser {_cmd_value(s.ilaser_ma)}",
            f"pgain {_cmd_value(s.pgain)}",
            f"igain {_cmd_value(s.igain)}",
            f"dgain {_cmd_value(s.dgain)}",
            f"rtset {_cmd_value(s.rtset_ohm)}",
            f"tprot {int(s.tprot)}",
            f"lckon {int(s.lckon)}",
        ]
        self.query_many(cmds)
        print(
            f"Applied CTL200 settings: rtset {s.rtset_ohm:.3f} Ω  "
            f"PID {s.pgain}/{s.igain}/{s.dgain}  "
            f"ilaser {_cmd_value(s.ilaser_ma)} mA  ilmax {_cmd_value(s.ilmax_ma)} mA"
        )

    def set_rtset(self, ohm: float) -> str:
        """Live thermistor setpoint. Does not cycle lason."""
        self.settings.rtset_ohm = float(ohm)
        reply = self.set("rtset", self.settings.rtset_ohm)
        print(f"rtset -> {reply or self.settings.rtset_ohm} Ω")
        return reply

    def on(self) -> None:
        """Enable TEC + diode current. Returns immediately; no warmup sleep."""
        s = self.settings
        self.apply_settings()
        self.query_many(
            [
                "tecon 1",
                f"ilaser {_cmd_value(s.ilaser_ma)}",
                "lason 1",
            ]
        )
        actual = self.get("ilaser")
        err = self.get("err")
        print(
            f"Laser ON  ilaser setpoint {_cmd_value(s.ilaser_ma)} mA  "
            f"readback {actual or '?'} mA  err {err or '0'}"
        )
        print("No warmup wait. Call wait_stable() only if you want to block.")

    def off(self, tec_off: bool = False) -> None:
        """Disable diode current. TEC stays on unless ``tec_off``."""
        cmds = ["lason 0"]
        if tec_off:
            cmds.append("tecon 0")
        self.query_many(cmds)
        if tec_off:
            print("Laser OFF, TEC OFF")
        else:
            print("Laser OFF (TEC still on)")

    def status(self) -> dict:
        names = ("lason", "ilaser", "vlaser", "rtset", "rtact", "tecon", "err")
        values = self.query_many(list(names))
        data = dict(zip(names, values))
        print(
            f"lason {data['lason']}  ilaser {data['ilaser']} mA  "
            f"vlaser {data['vlaser']} V  rtset {data['rtset']} Ω  "
            f"rtact {data['rtact']} Ω  tecon {data['tecon']}  err {data['err']}"
        )
        return {
            "lason": _as_int(data["lason"]),
            "ilaser": _as_float(data["ilaser"]),
            "vlaser": _as_float(data["vlaser"]),
            "rtset": _as_float(data["rtset"]),
            "rtact": _as_float(data["rtact"]),
            "tecon": _as_int(data["tecon"]),
            "err": data["err"],
            "raw": data,
        }

    def wait_stable(self, seconds: float | None = None) -> None:
        """Optional blocker. ``on()`` does not call this."""
        wait_s = self.settings.warmup_s if seconds is None else float(seconds)
        if wait_s < 0:
            raise ValueError("seconds must be >= 0")
        print(f"Waiting {wait_s:g} s for thermal settle (laser already on).")
        time.sleep(wait_s)

    def close(self) -> None:
        self.transport.close()
        print("CTL200 UART released (diode state unchanged).")
