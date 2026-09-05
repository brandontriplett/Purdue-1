"""PyRPL connection with a repo-local user_dir and safe analog outputs."""

from __future__ import annotations

import os
from pathlib import Path

from pyrpl import Pyrpl

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_USER_DIR = REPO_DIR / "pyrpl_data"

HOSTNAME = "rp-f0c970.local"
PYRPL_CONFIG = "my_config"

_OUTPUT_MODULES = (
    "asg0",
    "asg1",
    "pid0",
    "pid1",
    "pid2",
    "iq0",
    "iq1",
    "iq2",
)


def default_user_dir() -> str:
    return str(DEFAULT_USER_DIR)


def silence_feedback(rp) -> None:
    """Zero PID and IQ and disconnect them from the analog outputs.

    Leaves ASG routing alone so a live LaserDrive is not interrupted.
    Saved PyRPL configs often restore pid0 paused, on OUT1, with nonzero
    gains; if ``paused`` stays True the lock never actually closes.
    """
    for name in ("pid0", "pid1", "pid2"):
        pid = getattr(rp, name, None)
        if pid is None:
            continue
        try:
            pid.p = 0
            pid.i = 0
            pid.ival = 0
            pid.output_direct = "off"
            if hasattr(pid, "paused"):
                pid.paused = False
        except Exception:
            pass

    for name in ("iq0", "iq1", "iq2"):
        iq = getattr(rp, name, None)
        if iq is None:
            continue
        try:
            iq.amplitude = 0
            iq.output_direct = "off"
        except Exception:
            pass


def silence_other_outputs(rp) -> None:
    """Disconnect every DSP module from the analog outputs.

    Saved PyRPL configs often restore pid0 onto OUT1 with nonzero gains.
    Do this immediately after connect, before the laser drive enables OUT1.
    """
    for name in _OUTPUT_MODULES:
        module = getattr(rp, name, None)
        if module is None:
            continue
        try:
            module.output_direct = "off"
        except Exception:
            pass
    silence_feedback(rp)


def connect(
    hostname: str = HOSTNAME,
    config: str = PYRPL_CONFIG,
    user_dir: str | None = None,
    reloadfpga: bool = True,
    gui: bool = False,
):
    """Connect to the Red Pitaya and return (pyrpl, redpitaya)."""
    user_dir = user_dir or default_user_dir()
    os.makedirs(user_dir, exist_ok=True)
    os.environ["PYRPL_USER_DIR"] = user_dir

    print("Connecting to Red Pitaya...")
    print(f"  hostname:    {hostname}")
    print(f"  config:      {config}")
    print(f"  user_dir:    {user_dir}")
    print(f"  reloadfpga:  {reloadfpga}")

    p = Pyrpl(
        config=config,
        hostname=hostname,
        user_dir=user_dir,
        reloadfpga=reloadfpga,
        gui=gui,
    )
    rp = p.rp
    silence_other_outputs(rp)
    print("Connected. Analog outputs are disconnected until the laser drive enables OUT1.")
    return p, rp
