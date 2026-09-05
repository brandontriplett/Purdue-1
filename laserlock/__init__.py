"""Hardware helpers for the PyRPL laser lock.

The notebook is the test UI. A later headless script can import the same API.
"""

from .drive import LaserDrive
from .connection import (
    HOSTNAME,
    PYRPL_CONFIG,
    connect,
    default_user_dir,
    silence_feedback,
)
from .lockin import (
    configure_dither,
    diagnose_hold_points,
    disengage_lock,
    engage_i_lock,
    park_at_lock_point,
    record_dithered_sweep,
    sample_lock_signals,
)
from .lockpoint import first_scurve_window, get_lock_point, lock_point_from_trace
from .waveform import WaveformPlan, choose_scope_window, make_scan_plan, make_slew_plan

__all__ = [
    "LaserDrive",
    "HOSTNAME",
    "PYRPL_CONFIG",
    "WaveformPlan",
    "choose_scope_window",
    "configure_dither",
    "connect",
    "diagnose_hold_points",
    "default_user_dir",
    "disengage_lock",
    "engage_i_lock",
    "first_scurve_window",
    "get_lock_point",
    "lock_point_from_trace",
    "make_scan_plan",
    "make_slew_plan",
    "park_at_lock_point",
    "record_dithered_sweep",
    "sample_lock_signals",
    "silence_feedback",
]
