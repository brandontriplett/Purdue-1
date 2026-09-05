"""One-shot ASG trajectories for the laser current-modulation input.

Owns OUT1 for the life of the experiment. After a spectrum scan the
output slews back to idle (0 V) so the laser is not left at a high
current while you inspect plots.


Hard rules, all enforced here:

- Do not call ``asg.setup()`` or ``asg.trigger_source = ...`` after OUT1 is
  live. Both regenerate the waveform table and ``setup()`` zeros the DAC.
- Do not set ``asg.on = False`` (that is the FPGA zero bit).
- Do not set ``output_direct = "off"`` until an explicit shutdown that has
  already slewed to idle.
- After every trajectory, freeze by loading a DC table at the end voltage
  *before* resetting the sample pointer.
"""

from __future__ import annotations

import time

import numpy as np
from pyrpl.async_utils import wait

from .waveform import (
    DAC_MAX,
    DAC_MIN,
    TABLE_LENGTH,
    WaveformPlan,
    crop_linear_scan,
    make_dc_table,
    make_error_scan_plan,
    make_quick_scan_plan,
    make_scan_plan,
    make_slew_plan,
    scan_keep_slice,
)

IDLE_V = 0.0

# Control-register bits for ASG0 trigger source (bits 0-2).
# Using the Python property calls setup() and wipes custom tables.
_TRIG_MASK = 0x7
_TRIG_OFF = 0x0
_TRIG_IMMEDIATE = 0x1


def _extract_channels(raw):
    if raw is None:
        raise TimeoutError(
            "Scope wait returned None. In a Jupyter notebook PyRPL's wait() "
            "does that when the trigger never fires, instead of raising "
            "TimeoutError. Use scope_trigger='immediately'."
        )
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 0 or raw.size == 0:
        raise RuntimeError(
            f"Scope returned empty data (shape {raw.shape}). "
            "The trigger likely never fired."
        )
    if raw.ndim == 1:
        ch1 = raw.reshape(-1)
        return ch1, None
    if raw.ndim == 2:
        if raw.shape[0] == 2:
            return raw[0].reshape(-1), raw[1].reshape(-1)
        if raw.shape[1] == 2:
            return raw[:, 0].reshape(-1), raw[:, 1].reshape(-1)
        if raw.shape[0] == 1:
            return raw[0].reshape(-1), None
        if raw.shape[1] == 1:
            return raw[:, 0].reshape(-1), None
    raise RuntimeError(f"Unexpected scope data shape: {raw.shape}")


def _extract_ch1(raw) -> np.ndarray:
    ch1, _ch2 = _extract_channels(raw)
    return ch1


class LaserDrive:
    """Drives the laser current-modulation input on OUT1."""

    def __init__(self, rp, idle_v: float = IDLE_V):
        if not (DAC_MIN <= idle_v <= DAC_MAX):
            raise ValueError(f"idle_v {idle_v} V is outside ±1 V.")
        self.rp = rp
        self.asg = rp.asg0
        self.scope = rp.scope
        self.idle_v = float(idle_v)
        self.current_v = float(idle_v)
        self._out_enabled = False
        # "immediately" is the path that actually works with a custom ASG
        # table. "asg0" would be nicer alignment, but PyRPL only strobes that
        # trigger when asg.trigger_source is set through the Python property,
        # and that property calls setup() and wipes the table.
        self.scope_trigger = "immediately"

    # ------------------------------------------------------------------
    # Low-level ASG
    # ------------------------------------------------------------------

    def _set_trigger(self, source: str) -> None:
        control = int(self.asg._read(0x0))
        control &= ~_TRIG_MASK
        if source == "immediately":
            control |= _TRIG_IMMEDIATE
        elif source != "off":
            raise ValueError(f"Unsupported trigger source {source!r}")
        self.asg._write(0x0, control)

    def _prepare_oneshot_flags(self) -> None:
        self.asg.periodic = False
        self.asg._sm_wrappointer = False
        self.asg.cycles_per_burst = 1

    def _prepare_oneshot_mode(self) -> None:
        self._prepare_oneshot_flags()
        self.asg.amplitude = 1.0
        self.asg.offset = 0.0

    def _upload_table(self, table: np.ndarray) -> None:
        table = np.asarray(table, dtype=float).reshape(-1)
        if len(table) != TABLE_LENGTH:
            raise ValueError(f"ASG table must have {TABLE_LENGTH} points.")
        if np.any(table < DAC_MIN - 1e-6) or np.any(table > DAC_MAX + 1e-6):
            raise ValueError("ASG table exceeds ±1 V.")
        self.asg.data = np.clip(table, DAC_MIN, DAC_MAX)

    def _freeze_at(self, voltage: float) -> None:
        """Hold voltage. Uses ASG offset so a stopped state-machine still moves."""
        self.hold_dc(voltage, verbose=False)

    def _play(
        self,
        plan: WaveformPlan,
        record: bool = False,
        decimation: int | None = None,
        freeze: bool = True,
        record_iq: bool = False,
    ):
        """Upload and trigger a one-shot table. Optionally record IN1 (and iq0)."""
        if not self._out_enabled:
            raise RuntimeError("OUT1 is not enabled. Call enable_idle() first.")

        hold_v = self.current_v
        # Keep the DAC at hold_v via offset while the table is loaded.
        # A stopped ASG ignores waveform RAM until it is retriggered;
        # offset is applied continuously.
        self.asg.amplitude = 0.0
        self.asg.offset = hold_v
        self._prepare_oneshot_flags()
        actual_frequency = 1.0 / plan.duration
        self.asg.frequency = actual_frequency
        # FrequencyRegister snaps to a nearby legal value.
        played_duration = 1.0 / float(self.asg.frequency)

        if abs(plan.v_start - hold_v) > 2e-3:
            raise RuntimeError(
                f"Waveform starts at {plan.v_start:.4f} V but the drive is "
                f"holding {self.current_v:.4f} V. That would jump."
            )
        self._upload_table(plan.table)

        asg_lag = 0.0
        t_trigger = None
        iq_trace = None
        wait_s = max(played_duration, plan.duration) + 0.15
        try:
            if record:
                if decimation is None:
                    raise ValueError("decimation is required when record=True")
                self._arm_scope(decimation=decimation, record_iq=record_iq)
                t_arm = time.time()
                scope_future = self.scope.single_async()
                # immediately-mode starts recording at arm. Keep this short
                # so the waveform still fits in the scope window.
                if self.scope.trigger_source == "asg0":
                    time.sleep(0.15)
                else:
                    time.sleep(0.05)

            self.asg.sm_reset = True
            time.sleep(0.002)
            self.asg.sm_reset = False
            # Pointer is at sample 0. Switch from offset-hold to table
            # playback: table[0] is hold_v, so the DAC does not jump.
            self._prepare_oneshot_flags()
            self.asg.offset = 0.0
            self.asg.amplitude = 1.0
            t_trigger = time.time()
            if record:
                asg_lag = t_trigger - t_arm
            self._set_trigger("immediately")
            time.sleep(0.002)
            self._set_trigger("off")

            if record:
                raw = wait(scope_future, timeout=self.scope.duration + 5.0)
                measured, iq_trace = _extract_channels(raw)
                t = np.asarray(self.scope.times, dtype=float).reshape(-1)
                t = t - t[0]
                n = min(len(t), len(measured))
                t = t[:n]
                measured = measured[:n]
                if iq_trace is not None:
                    iq_trace = np.asarray(iq_trace, dtype=float).reshape(-1)[:n]
                # Scope t=0 is arm time; the ASG starts asg_lag later.
                table_t = np.linspace(0.0, played_duration, TABLE_LENGTH, endpoint=False)
                commanded = np.interp(
                    t - asg_lag,
                    table_t,
                    plan.table,
                    left=plan.table[0],
                    right=plan.table[-1],
                )
            else:
                time.sleep(wait_s)
                measured = None
                t = None
                commanded = None
                iq_trace = None
        finally:
            if t_trigger is not None:
                if freeze:
                    leftover = wait_s - (time.time() - t_trigger)
                    if leftover > 0:
                        time.sleep(leftover)
                    self._freeze_at(plan.v_stop)
                else:
                    self.current_v = float(plan.v_stop)

        return t, commanded, measured, played_duration, asg_lag, iq_trace

    def _arm_scope(self, decimation: int, record_iq: bool = False) -> None:
        scope = self.scope
        scope.input1 = "in1"
        scope.input2 = "iq0" if record_iq else "off"
        scope.ch1_active = True
        scope.ch2_active = bool(record_iq)
        scope.average = False
        scope.trace_average = 1
        scope.run_continuous = False
        scope.rolling_mode = False
        scope.decimation = decimation
        scope.trigger_source = self.scope_trigger
        if self.scope_trigger == "asg0":
            # delay = duration/2 → trace starts at the ASG trigger.
            scope.trigger_delay = scope.duration / 2.0
        else:
            scope.trigger_delay = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable_idle(self) -> None:
        """Configure a 0 V (or idle_v) DC table, then connect OUT1."""
        if self._out_enabled:
            print(f"OUT1 already enabled, holding {self.current_v:+.4f} V.")
            return
        asg = self.asg
        asg.output_direct = "off"
        asg.setup(
            waveform="dc",
            frequency=10.0,
            amplitude=1.0,
            offset=0.0,
            trigger_source="off",
            cycles_per_burst=1,
        )
        self._prepare_oneshot_mode()
        self._upload_table(make_dc_table(self.idle_v))
        asg.sm_reset = True
        time.sleep(0.002)
        asg.sm_reset = False
        asg.on = True
        self._set_trigger("off")
        asg.output_direct = "out1"
        self._out_enabled = True
        self.current_v = self.idle_v
        print(f"OUT1 enabled, parked at {self.current_v:+.4f} V.")

    def hold_dc(self, voltage: float, verbose: bool = False) -> None:
        """Hold OUT1 at ``voltage`` using ASG offset.

        Amplitude is set to 0 so the DAC follows ``offset`` even when the
        waveform state-machine has stopped (rewriting the table does not).
        """
        voltage = float(np.clip(voltage, DAC_MIN, DAC_MAX))
        self.asg.amplitude = 0.0
        self.asg.offset = voltage
        self._set_trigger("off")
        self.current_v = voltage
        if verbose:
            print(f"Holding {voltage:+.4f} V")

    def jump_to(self, voltage: float, verbose: bool = True) -> None:
        """Step OUT1 immediately. No ramp and no state-machine reset."""
        self.hold_dc(voltage, verbose=verbose)

    def release_to_zero(self) -> None:
        """Drop OUT1 to idle immediately. No slew, no state-machine reset."""
        self.hold_dc(self.idle_v, verbose=True)

    def slew_to(self, voltage: float, duration: float = 1.5) -> None:
        """Smoothly ramp OUT1 from the current voltage to ``voltage``."""
        voltage = float(voltage)
        if abs(voltage - self.current_v) < 1e-4:
            return
        plan = make_slew_plan(self.current_v, voltage, duration)
        print(
            f"Slew {self.current_v:+.4f} V -> {voltage:+.4f} V "
            f"in {duration:.3f} s"
        )
        self._play(plan, record=False)

    def scan(
        self,
        start_v: float,
        stop_v: float,
        scan_time: float = 4.0,
        settle_time: float = 3.0,
        approach_time: float = 1.5,
        return_to_idle: bool = True,
        return_slew_time: float = 2.0,
    ):
        """Slew to start, settle, scan, then return to idle.

        Returns
        -------
        voltage, photodiode : ndarray
            Only the linear scan segment, not the approach/settle/hold.
        plan : WaveformPlan
        full : dict with keys t, commanded, measured for the whole window
        """
        plan, decimation = make_scan_plan(
            v_current=self.current_v,
            v_start=start_v,
            v_stop=stop_v,
            scan_time=scan_time,
            settle_time=settle_time,
            approach_time=approach_time,
        )
        print()
        print("------------------------------------------------------------")
        print("ONE-SHOT SCAN")
        print("------------------------------------------------------------")
        print(f"From current: {self.current_v:+.6f} V")
        print(f"Scan:         {start_v:+.6f} -> {stop_v:+.6f} V")
        print(f"Approach:     {approach_time:.3f} s")
        print(f"Settle:       {settle_time:.3f} s")
        print(f"Scan time:    {scan_time:.3f} s")
        print(f"Window:       {plan.duration:.3f} s  (decimation {decimation})")

        t, commanded, measured, played, asg_lag, _iq = self._play(
            plan, record=True, decimation=decimation
        )
        scan_slice = plan.slices.get("scan")
        if t is None or measured is None or commanded is None:
            raise RuntimeError("Scope returned no data.")

        # Crop by time using the planned scan window. Scale by the actual
        # ASG period in case FrequencyRegister snapped, and shift by asg_lag
        # because immediately-mode records a few tens of ms before the ASG.
        time_scale = played / plan.duration
        elapsed = 0.0
        scan_t0 = 0.0
        scan_t1 = plan.duration
        for seg in plan.segments:
            if seg.name == "scan":
                scan_t0 = elapsed * time_scale + asg_lag
                scan_t1 = (elapsed + seg.duration) * time_scale + asg_lag
                break
            elapsed += seg.duration

        mask = (t >= scan_t0) & (t <= scan_t1)
        if not np.any(mask):
            raise RuntimeError("Scan time window is empty in the scope trace.")

        voltage = commanded[mask]
        photodiode = measured[mask]
        full = {
            "t": t,
            "commanded": commanded,
            "measured": measured,
            "played_duration": played,
            "scan_slice": scan_slice,
            "scan_t0": scan_t0,
            "scan_t1": scan_t1,
        }
        print(f"Scan samples: {len(voltage)}")
        if return_to_idle:
            print(f"Returning to idle {self.idle_v:+.4f} V")
            self.slew_to(self.idle_v, duration=return_slew_time)
        print(f"Now holding:  {self.current_v:+.6f} V")
        return voltage, photodiode, plan, full

    def quick_scan(
        self,
        start_v: float,
        stop_v: float,
        scan_time: float = 1.0,
        pause_s: float = 0.05,
    ):
        """Jump to start, pause briefly, 1 s ramp, drop to 0 V.

        No freeze-at-endpoint and no slow return slew. Those are the
        post-scan discontinuities on the regular scan() path.
        """
        print()
        print("------------------------------------------------------------")
        print("QUICK SCAN (jump / ramp / release)")
        print("------------------------------------------------------------")
        self.jump_to(start_v)
        time.sleep(pause_s)

        plan, decimation = make_quick_scan_plan(
            start_v=start_v,
            stop_v=stop_v,
            scan_time=scan_time,
        )
        print(f"Scan:      {start_v:+.6f} -> {stop_v:+.6f} V")
        print(f"Scan time: {scan_time:.3f} s")
        print(f"Window:    {plan.duration:.3f} s  (decimation {decimation})")
        print(f"Pause:     {pause_s:.3f} s")

        t, commanded, measured, played, asg_lag, _iq = self._play(
            plan, record=True, decimation=decimation, freeze=False
        )
        # Scope has returned but the ASG is still in the long 0 V tail.
        # Overwrite the whole table with zeros so a later wrap to data[0]
        # is also 0 V.
        self.release_to_zero()

        if t is None or measured is None or commanded is None:
            raise RuntimeError("Scope returned no data.")

        time_scale = played / plan.duration
        elapsed = 0.0
        scan_t0 = asg_lag
        scan_t1 = plan.duration + asg_lag
        for seg in plan.segments:
            if seg.name == "scan":
                scan_t0 = elapsed * time_scale + asg_lag
                scan_t1 = (elapsed + 0.92 * seg.duration) * time_scale + asg_lag
                break
            elapsed += seg.duration

        mask = (t >= scan_t0) & (t <= scan_t1)
        if not np.any(mask):
            raise RuntimeError("Scan time window is empty in the scope trace.")

        voltage, photodiode = crop_linear_scan(
            commanded[mask],
            measured[mask],
            start_v,
            stop_v,
        )
        full = {
            "t": t,
            "commanded": commanded,
            "measured": measured,
            "played_duration": played,
            "scan_t0": scan_t0,
            "scan_t1": scan_t1,
        }
        print(f"Scan samples: {len(voltage)}")
        print(f"Now holding:  {self.current_v:+.6f} V")
        return voltage, photodiode, plan, full

    def fast_error_scan(
        self,
        start_v: float,
        stop_v: float,
        scan_time: float = 0.5,
        pause_s: float = 0.05,
    ):
        """Analog ramp with dither already on; record IN1 and iq0.

        Freeze at stop so the lock point can be applied immediately.
        Fast enough (like the coarse scan) that thermal drift does not
        shear the S-curve.
        """
        print()
        print("------------------------------------------------------------")
        print("FAST ERROR SCAN (analog ramp, record IN1 + iq0)")
        print("------------------------------------------------------------")
        self.hold_dc(start_v, verbose=True)
        time.sleep(pause_s)
        plan, decimation = make_error_scan_plan(start_v, stop_v, scan_time)
        print(f"Scan:      {start_v:+.6f} -> {stop_v:+.6f} V")
        print(f"Scan time: {scan_time:.3f} s")
        print(f"Window:    {plan.duration:.3f} s  (decimation {decimation})")

        t, commanded, measured, played, asg_lag, iq_trace = self._play(
            plan, record=True, decimation=decimation, freeze=True, record_iq=True
        )
        if t is None or measured is None or commanded is None or iq_trace is None:
            raise RuntimeError("Scope returned no PD/error data.")

        time_scale = played / plan.duration
        elapsed = 0.0
        scan_t0 = asg_lag
        scan_t1 = plan.duration + asg_lag
        for seg in plan.segments:
            if seg.name == "scan":
                scan_t0 = elapsed * time_scale + asg_lag
                scan_t1 = (elapsed + 0.95 * seg.duration) * time_scale + asg_lag
                break
            elapsed += seg.duration
        mask = (t >= scan_t0) & (t <= scan_t1)
        if not np.any(mask):
            raise RuntimeError("Scan time window is empty in the scope trace.")

        sl = scan_keep_slice(
            commanded[mask], measured[mask], start_v, stop_v, end_guard=0.03
        )
        voltage = commanded[mask][sl]
        photodiode = measured[mask][sl]
        error = iq_trace[mask][sl]
        print(f"Scan samples: {len(voltage)}")
        print(f"Now holding:  {self.current_v:+.6f} V")
        return voltage, photodiode, error, plan

    def shutdown(self, slew_time: float = 1.5) -> None:
        """Slew to idle, then disconnect OUT1."""
        if self._out_enabled:
            try:
                self.slew_to(self.idle_v, duration=slew_time)
            except Exception as exc:
                print(f"Shutdown slew failed ({exc}); disconnecting anyway.")
        self.asg.output_direct = "off"
        self._out_enabled = False
        print("OUT1 disconnected after slew to idle.")
