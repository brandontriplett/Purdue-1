"""Hardware helpers for the PyRPL laser lock.

The notebook is the test UI. A later headless script can import the same API.

``from laserlock.koheron import KoheronLaser`` does not import pyrpl.
Other names are loaded lazily so a power-only session never touches the FPGA.
"""

from .koheron import KoheronLaser, LaserSettings

__all__ = [
    "KoheronLaser",
    "LaserDrive",
    "LaserSettings",
    "HOSTNAME",
    "PYRPL_CONFIG",
    "SkipThenCatch",
    "WaveformPlan",
    "approach_duration",
    "apply_pid",
    "approach_lock_from_start",
    "centered_fine_scans",
    "CatchDetector",
    "choose_scope_window",
    "configure_dither",
    "connect",
    "diagnose_hold_points",
    "default_user_dir",
    "disengage_lock",
    "engage_i_lock",
    "ensure_dither",
    "flip_iq_phase",
    "open_pid",
    "set_iq_phase",
    "error_scan_window",
    "first_scurve_window",
    "get_lock_point",
    "hunt_peak2_downhill",
    "apply_lockpoint_correction",
    "lock_point_from_single_line",
    "lock_point_from_trace",
    "make_scan_plan",
    "make_slew_plan",
    "move_to_scan_start",
    "park_at_lock_point",
    "record_dithered_sweep",
    "run_fine_scan_and_park",
    "run_line_scan",
    "run_paired_lock_scans",
    "sample_lock_signals",
    "silence_feedback",
    "triplet_scan_extents",
]

_LAZY = {
    "LaserDrive": (".drive", "LaserDrive"),
    "HOSTNAME": (".connection", "HOSTNAME"),
    "PYRPL_CONFIG": (".connection", "PYRPL_CONFIG"),
    "connect": (".connection", "connect"),
    "default_user_dir": (".connection", "default_user_dir"),
    "silence_feedback": (".connection", "silence_feedback"),
    "CatchDetector": (".catch", "CatchDetector"),
    "SkipThenCatch": (".catch", "SkipThenCatch"),
    "hunt_peak2_downhill": (".hunt", "hunt_peak2_downhill"),
    "apply_pid": (".lockin", "apply_pid"),
    "approach_duration": (".lockin", "approach_duration"),
    "approach_lock_from_start": (".lockin", "approach_lock_from_start"),
    "centered_fine_scans": (".lockin", "centered_fine_scans"),
    "configure_dither": (".lockin", "configure_dither"),
    "diagnose_hold_points": (".lockin", "diagnose_hold_points"),
    "disengage_lock": (".lockin", "disengage_lock"),
    "engage_i_lock": (".lockin", "engage_i_lock"),
    "ensure_dither": (".lockin", "ensure_dither"),
    "flip_iq_phase": (".lockin", "flip_iq_phase"),
    "open_pid": (".lockin", "open_pid"),
    "set_iq_phase": (".lockin", "set_iq_phase"),
    "error_scan_window": (".lockin", "error_scan_window"),
    "move_to_scan_start": (".lockin", "move_to_scan_start"),
    "park_at_lock_point": (".lockin", "park_at_lock_point"),
    "record_dithered_sweep": (".lockin", "record_dithered_sweep"),
    "run_fine_scan_and_park": (".lockin", "run_fine_scan_and_park"),
    "run_line_scan": (".lockin", "run_line_scan"),
    "run_paired_lock_scans": (".lockin", "run_paired_lock_scans"),
    "sample_lock_signals": (".lockin", "sample_lock_signals"),
    "triplet_scan_extents": (".lockin", "triplet_scan_extents"),
    "first_scurve_window": (".lockpoint", "first_scurve_window"),
    "apply_lockpoint_correction": (".lockpoint", "apply_lockpoint_correction"),
    "get_lock_point": (".lockpoint", "get_lock_point"),
    "lock_point_from_single_line": (".lockpoint", "lock_point_from_single_line"),
    "lock_point_from_trace": (".lockpoint", "lock_point_from_trace"),
    "WaveformPlan": (".waveform", "WaveformPlan"),
    "choose_scope_window": (".waveform", "choose_scope_window"),
    "make_scan_plan": (".waveform", "make_scan_plan"),
    "make_slew_plan": (".waveform", "make_slew_plan"),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
