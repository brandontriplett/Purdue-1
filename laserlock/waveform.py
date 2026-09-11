"""Pure waveform construction for one-shot PyRPL ASG tables.

The ASG table is 16384 samples. With amplitude=1 V and offset=0 V the
table values *are* output volts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TABLE_LENGTH = 2**14
FPGA_DT = 8e-9  # 125 MHz
DAC_MIN = -1.0
DAC_MAX = 1.0
SCOPE_MAX_DECIMATION_EXP = 16


def scope_duration(decimation: int) -> float:
    return FPGA_DT * float(decimation) * TABLE_LENGTH


def legal_scope_windows() -> list[tuple[int, float]]:
    return [(2**n, scope_duration(2**n)) for n in range(SCOPE_MAX_DECIMATION_EXP + 1)]


def choose_scope_window(needed_s: float) -> tuple[int, float]:
    """Smallest legal scope duration that can hold needed_s seconds."""
    if needed_s <= 0:
        raise ValueError("needed_s must be positive")
    for decimation, duration in legal_scope_windows():
        if duration >= needed_s - 1e-12:
            return decimation, duration
    longest = legal_scope_windows()[-1]
    raise ValueError(
        f"Need {needed_s:.3f} s but the longest PyRPL scope window is "
        f"{longest[1]:.3f} s (decimation {longest[0]})."
    )


def clip_voltage(voltage: float) -> float:
    if voltage < DAC_MIN - 1e-9 or voltage > DAC_MAX + 1e-9:
        raise ValueError(f"Voltage {voltage:.4f} V is outside the Red Pitaya ±1 V range.")
    return float(np.clip(voltage, DAC_MIN, DAC_MAX))


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    v_start: float
    v_stop: float


@dataclass
class WaveformPlan:
    table: np.ndarray
    duration: float
    segments: list[Segment]
    point_counts: list[int]
    slices: dict[str, slice] = field(default_factory=dict)

    @property
    def v_start(self) -> float:
        return float(self.table[0])

    @property
    def v_stop(self) -> float:
        return float(self.table[-1])

    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.duration, TABLE_LENGTH, endpoint=False)

    def commanded_voltage(self, t: np.ndarray) -> np.ndarray:
        """Interpolate the uploaded table onto an arbitrary time axis."""
        t = np.asarray(t, dtype=float)
        table_t = self.times()
        return np.interp(t, table_t, self.table, left=self.table[0], right=self.table[-1])


def _allocate_points(durations: list[float], n: int = TABLE_LENGTH) -> list[int]:
    weights = np.asarray(durations, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Waveform total duration must be positive.")

    raw = weights / total * n
    counts = np.floor(raw).astype(int)

    # Every positive-duration segment gets at least one sample.
    for i, duration in enumerate(durations):
        if duration > 0 and counts[i] < 1:
            counts[i] = 1

    # Ramps need two samples if they actually move.
    for i, duration in enumerate(durations):
        if duration > 0 and counts[i] < 2:
            counts[i] = 2

    deficit = int(counts.sum() - n)
    # Give leftover samples to the longest segments; steal from the longest
    # if we overshot because of the minimums.
    order = np.argsort(-weights)
    if deficit > 0:
        for i in order:
            steal = min(deficit, max(0, counts[i] - (2 if durations[i] > 0 else 0)))
            counts[i] -= steal
            deficit -= steal
            if deficit == 0:
                break
        if deficit != 0:
            raise RuntimeError("Could not allocate waveform samples.")
    elif deficit < 0:
        extra = -deficit
        for i in order:
            counts[i] += extra
            extra = 0
            break

    if int(counts.sum()) != n:
        counts[-1] += n - int(counts.sum())

    return [int(c) for c in counts]


def render_table(segments: list[Segment], duration: float) -> WaveformPlan:
    if not segments:
        raise ValueError("Need at least one waveform segment.")
    if duration <= 0:
        raise ValueError("Waveform duration must be positive.")

    counts = _allocate_points([seg.duration for seg in segments])
    parts = []
    slices: dict[str, slice] = {}
    index = 0
    for seg, count in zip(segments, counts):
        sl = slice(index, index + count)
        slices[seg.name] = sl
        if count == 0:
            index += count
            continue
        if count == 1 or abs(seg.v_stop - seg.v_start) < 1e-15:
            part = np.full(count, seg.v_stop, dtype=float)
        else:
            # endpoint=False so a following hold at v_stop does not double
            # the endpoint. The final sample of the whole table is forced
            # to the last segment's v_stop below.
            part = np.linspace(seg.v_start, seg.v_stop, count, endpoint=False)
        parts.append(part)
        index += count

    table = np.concatenate(parts) if parts else np.zeros(TABLE_LENGTH)
    if len(table) != TABLE_LENGTH:
        raise RuntimeError(f"Rendered table length {len(table)}, expected {TABLE_LENGTH}.")

    table[0] = segments[0].v_start
    table[-1] = segments[-1].v_stop
    table = np.clip(table, DAC_MIN, DAC_MAX)
    return WaveformPlan(
        table=table,
        duration=float(duration),
        segments=list(segments),
        point_counts=counts,
        slices=slices,
    )


def make_slew_plan(v_from: float, v_to: float, duration: float) -> WaveformPlan:
    v_from = clip_voltage(v_from)
    v_to = clip_voltage(v_to)
    if duration <= 0:
        raise ValueError("Slew duration must be positive.")
    segments = [
        Segment("slew", duration, v_from, v_to),
        Segment("hold", max(0.05, 0.05 * duration), v_to, v_to),
    ]
    return render_table(segments, duration=duration + segments[1].duration)


def make_scan_plan(
    v_current: float,
    v_start: float,
    v_stop: float,
    scan_time: float,
    settle_time: float,
    approach_time: float,
) -> tuple[WaveformPlan, int]:
    """Build a one-shot scan table that fills one legal scope window."""
    v_current = clip_voltage(v_current)
    v_start = clip_voltage(v_start)
    v_stop = clip_voltage(v_stop)
    if scan_time <= 0:
        raise ValueError("scan_time must be positive.")
    if settle_time < 0 or approach_time < 0:
        raise ValueError("settle_time and approach_time must be >= 0.")

    need_approach = abs(v_current - v_start) > 1e-4
    if need_approach:
        approach = max(float(approach_time), 1.0)
    else:
        approach = 0.0
    needed = approach + settle_time + scan_time
    # Leave a little room so the final hold is never empty.
    decimation, window = choose_scope_window(needed + 0.05)
    leftover = window - needed
    if leftover < 0:
        leftover = 0.0

    segments = []
    if approach > 0:
        segments.append(Segment("approach", approach, v_current, v_start))
    if settle_time > 0:
        segments.append(Segment("settle", settle_time, v_start, v_start))
    segments.append(Segment("scan", scan_time, v_start, v_stop))
    segments.append(Segment("hold", leftover if leftover > 0 else 0.05, v_stop, v_stop))

    plan = render_table(segments, duration=window)
    return plan, decimation


def make_quick_scan_plan(
    start_v: float,
    stop_v: float,
    scan_time: float = 1.0,
) -> tuple[WaveformPlan, int]:
    """PD ramp. Long 0 V tail only for slow (~1 s) surveys.

    A 0.15 s ramp in an 8 s table is ~2 % of the waveform. Scope/ASG lag
    is ~50 ms (a third of that ramp), so the recorded window includes the
    0 V tail and the plot looks like a wrap in the middle of the scan.
    Short ramps use the error-scan recipe: fill one legal scope window
    and hold at stop, not idle.
    """
    start_v = clip_voltage(start_v)
    stop_v = clip_voltage(stop_v)
    if scan_time <= 0:
        raise ValueError("scan_time must be positive.")
    if scan_time < 0.5:
        return make_error_scan_plan(start_v, stop_v, scan_time)
    try:
        _, asg_window = choose_scope_window(max(scan_time + 3.0, 8.0))
    except ValueError:
        # ~8 s is the longest legal scope window; no room for a 0 V tail.
        return make_error_scan_plan(start_v, stop_v, scan_time)
    scope_decimation, _ = choose_scope_window(scan_time + 0.5)
    leftover = asg_window - scan_time
    segments = [
        Segment("scan", scan_time, start_v, stop_v),
        Segment("release", leftover, 0.0, 0.0),
    ]
    plan = render_table(segments, duration=asg_window)
    return plan, scope_decimation


def scan_keep_slice(
    voltage,
    signal,
    start_v: float,
    stop_v: float,
    end_guard: float = 0.04,
    use_signal_jumps: bool = False,
):
    """Slice of the monotonic ramp; drop wrap-around and the last few percent.

    Cropping uses the *commanded voltage* axis. Photodiode spikes are real
    spectrum (or noise), not wrap-around.
    """
    voltage = np.asarray(voltage, dtype=float)
    signal = np.asarray(signal, dtype=float)
    n_all = len(voltage)
    if n_all != len(signal) or n_all < 20:
        return slice(0, n_all)

    span = float(stop_v - start_v)
    if abs(span) < 1e-6:
        return slice(0, n_all)
    direction = 1.0 if span > 0 else -1.0

    dv = np.diff(voltage, prepend=voltage[0])
    jump_back = direction * dv < -0.02 * abs(span) - 1e-4
    past_stop = direction * (voltage - stop_v) > 0.02 * abs(span)
    before_start = direction * (voltage - start_v) < -0.02 * abs(span)

    dsig = np.abs(np.diff(signal, prepend=signal[0]))
    med = float(np.median(dsig)) + 1e-9
    jump_signal = dsig > max(8.0 * med, 0.015)

    progressed = np.where(direction * (voltage - start_v) > 0.01 * abs(span))[0]
    start_i = int(progressed[0]) if len(progressed) else 0

    flags = jump_back | past_stop | before_start
    if use_signal_jumps:
        flags = flags | jump_signal
    bad = np.where(flags)[0]
    bad = bad[bad > start_i + 5]
    end_i = int(bad[0]) if len(bad) else n_all

    if end_i - start_i < 20:
        return slice(0, n_all)

    v = voltage[start_i:end_i]
    progress = direction * (v - start_v)
    cutoff = (1.0 - end_guard) * abs(span)
    keep = progress <= cutoff
    if not np.any(keep):
        return slice(start_i, end_i)
    n = int(np.where(keep)[0][-1] + 1)
    n = max(n, 20)
    sl = slice(start_i, start_i + n)

    kept_span = abs(float(voltage[sl.stop - 1]) - float(voltage[sl.start]))
    too_short = (sl.stop - sl.start) < max(80, int(0.25 * n_all))
    too_narrow = kept_span < 0.50 * abs(span)
    if use_signal_jumps and (too_short or too_narrow):
        return scan_keep_slice(
            voltage,
            signal,
            start_v,
            stop_v,
            end_guard=end_guard,
            use_signal_jumps=False,
        )
    return sl


def crop_linear_scan(
    voltage,
    signal,
    start_v: float,
    stop_v: float,
    end_guard: float = 0.04,
):
    """Keep the monotonic ramp; drop wrap-around and the last few percent.

    The ASG sometimes retriggers to data[0] (scan start) after a one-shot.
    Those samples show up as a discontinuity near the end of the voltage axis.
    """
    sl = scan_keep_slice(voltage, signal, start_v, stop_v, end_guard)
    return np.asarray(voltage)[sl], np.asarray(signal)[sl]


def make_error_scan_plan(start_v: float, stop_v: float, scan_time: float = 0.5):
    """Analog ramp filling one legal scope window, then hold at stop.

    Fast compared to thermal drift (like the coarse scan). Freeze at stop
    so the lock point can be applied immediately.
    """
    start_v = clip_voltage(start_v)
    stop_v = clip_voltage(stop_v)
    if scan_time <= 0:
        raise ValueError("scan_time must be positive.")
    decimation, window = choose_scope_window(scan_time + 0.08)
    hold = max(window - scan_time, 0.01)
    segments = [
        Segment("scan", scan_time, start_v, stop_v),
        Segment("hold", hold, stop_v, stop_v),
    ]
    plan = render_table(segments, duration=window)
    return plan, decimation


def make_dc_table(voltage: float) -> np.ndarray:
    voltage = clip_voltage(voltage)
    return np.full(TABLE_LENGTH, voltage, dtype=float)
