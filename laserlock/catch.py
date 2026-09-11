"""Catch the first sub-Doppler line while walking downhill.

The start of the walk is often the Doppler *shoulder*, only a few millivolts
below the first peak, so an absolute PD threshold vs the wing is too tight.
A peak is a *local PD rise* as voltage decreases. An S-curve is a significant
error lobe, a zero crossing, then the opposite lobe — with hysteresis so one
noisy PD sample cannot abort the candidate.
"""

from __future__ import annotations


def _median(values):
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _mad(values, center):
    return _median(abs(float(v) - center) for v in values)


class CatchDetector:
    """Scale-relative S-curve detector.

    Call ``seed_wing(pd_samples, err_samples)`` on the start park, then
    ``update(voltage, error, pd)`` each downhill step.

    States: search → lobe → crossed → confirmed.
    ``v_zero`` is the sign-change voltage once confirmed.
    """

    def __init__(
        self,
        n_sigma=2.5,
        pd_floor=0.0008,
        err_floor=0.0015,
        confirm_dv=0.015,
        lookback_dv=0.006,
        pd_hysteresis=0.0015,
        require_pd=True,
        past_zero_v=0.002,
    ):
        self.n_sigma = float(n_sigma)
        self.pd_floor = float(pd_floor)
        self.err_floor = float(err_floor)
        self.confirm_dv = float(confirm_dv)
        self.lookback_dv = float(lookback_dv)
        self.pd_hysteresis = float(pd_hysteresis)
        self.require_pd = bool(require_pd)
        self.past_zero_v = float(past_zero_v)
        self.pd_baseline = 0.0
        self.pd_rise_thresh = self.pd_floor
        self.err_thresh = self.err_floor
        self.reset()

    def reset(self, v_start=None):
        self.v_start = None if v_start is None else float(v_start)
        self.last_error = 0.0
        self.last_voltage = None
        self.lobe_sign = 0.0
        self.v_lobe = None
        self.v_zero = None
        self.v_overshoot = None
        self.pd_peak = None
        self.history = []
        self.state = "search"

    def seed_wing(self, pd_samples, err_samples):
        pd_samples = list(pd_samples)
        err_samples = list(err_samples)
        self.pd_baseline = _median(pd_samples)
        pd_mad = _mad(pd_samples, self.pd_baseline)
        # MAD around the median. median(|error|) treats a Doppler-slope DC
        # offset as noise and can set err_thresh to hundreds of millivolts.
        err_mad = _mad(err_samples, _median(err_samples))
        sigma = 1.4826
        self.pd_rise_thresh = max(self.n_sigma * sigma * pd_mad, self.pd_floor)
        self.err_thresh = max(self.n_sigma * sigma * err_mad, self.err_floor)
        return {
            "pd_baseline": self.pd_baseline,
            "pd_thresh": self.pd_baseline + self.pd_rise_thresh,
            "pd_rise_thresh": self.pd_rise_thresh,
            "err_thresh": self.err_thresh,
        }

    def _local_pd_rise(self, voltage, pd):
        target = voltage + self.lookback_dv
        for v, _e, p in reversed(self.history[:-1]):
            if v >= target - 1e-12:
                return pd - p
        return 0.0

    def _on_peak(self, voltage, pd):
        if not self.require_pd:
            return True
        if self.state in ("lobe", "crossed") and self.pd_peak is not None:
            if pd >= self.pd_peak - self.pd_hysteresis:
                return True
        return self._local_pd_rise(voltage, pd) >= self.pd_rise_thresh

    def _is_lobe(self, error, factor=1.0):
        return abs(float(error)) >= factor * self.err_thresh

    def _interpolate_zero(self, voltage, error):
        v_prev = self.last_voltage
        e_prev = self.last_error
        if v_prev is None or error == e_prev:
            return voltage
        frac = (0.0 - e_prev) / (error - e_prev)
        frac = min(1.0, max(0.0, frac))
        return v_prev + frac * (voltage - v_prev)

    def _past_zero(self, error):
        """Opposite sign and a few millivolts past, not a full opposite lobe."""
        need = max(self.past_zero_v, 0.25 * self.err_thresh)
        return error * self.lobe_sign < 0.0 and abs(float(error)) >= need

    def _confirm(self, voltage, error, pd):
        # Valley zeros sit after PD has already fallen off the line.
        if self.pd_peak is not None and pd < self.pd_peak - max(
            self.pd_hysteresis, 0.008
        ):
            self.state = "search"
            self.lobe_sign = 0.0
            self.v_zero = None
            self.v_overshoot = None
            self.pd_peak = None
            return
        self.state = "confirmed"
        self.v_overshoot = voltage
        if self.v_zero is None:
            self.v_zero = self._interpolate_zero(voltage, error)

    def update(self, voltage, error, pd=None):
        if self.state == "confirmed":
            return self.state
        voltage = float(voltage)
        error = float(error)
        pd = 0.0 if pd is None else float(pd)
        self.history.append((voltage, error, pd))
        on_peak = self._on_peak(voltage, pd)
        crossed = self.last_error * error < 0.0

        if self.state == "search":
            # PD-gated peak 3: first lobe may be weak. Error-only peak 2:
            # ignore leftover millivolts between the lines.
            enter = 0.5 if self.require_pd else 1.0
            if on_peak and self._is_lobe(error, factor=enter):
                self.state = "lobe"
                self.lobe_sign = 1.0 if error >= 0 else -1.0
                self.v_lobe = voltage
                self.pd_peak = pd

        elif self.state == "lobe":
            self.pd_peak = max(self.pd_peak or pd, pd)
            if crossed:
                self.v_zero = self._interpolate_zero(voltage, error)
                if self._past_zero(error):
                    self._confirm(voltage, error, pd)
                else:
                    self.state = "crossed"
            elif not on_peak:
                self.state = "search"
                self.lobe_sign = 0.0
                self.pd_peak = None

        elif self.state == "crossed":
            self.pd_peak = max(self.pd_peak or pd, pd)
            if self.v_zero is not None and abs(voltage - self.v_zero) > self.confirm_dv:
                self.state = "search"
                self.lobe_sign = 0.0
                self.v_zero = None
                self.pd_peak = None
            elif self._past_zero(error) or (
                self._is_lobe(error) and error * self.lobe_sign < 0.0
            ):
                self._confirm(voltage, error, pd)
            elif self.require_pd and not on_peak and not self._is_lobe(error):
                self.state = "search"
                self.lobe_sign = 0.0
                self.v_zero = None
                self.pd_peak = None

        self.last_voltage = voltage
        self.last_error = error
        return self.state


class SkipThenCatch:
    """Downhill: finish ``n_skip`` S-curves as landmarks, catch the next.

    Peak 3 then peak 2: ``n_skip=1``. Do not freeze on the landmark zero.
    After a landmark, wait until error is quiet and we have moved
    ``gap_dv`` further downhill before hunting the target.
    """

    def __init__(
        self,
        n_skip=1,
        gap_dv=0.010,
        arm_below_v=None,
        landmark_max_v=None,
        min_pd_rise_v=0.0,
        **catch_kw,
    ):
        self.n_skip = int(n_skip)
        self.gap_dv = float(gap_dv)
        self.arm_below_v = None if arm_below_v is None else float(arm_below_v)
        self.landmark_max_v = (
            None if landmark_max_v is None else float(landmark_max_v)
        )
        self.min_pd_rise_v = max(0.0, float(min_pd_rise_v))
        self.inner = CatchDetector(**catch_kw)
        self.reset()

    def reset(self, v_start=None):
        self.inner.reset(v_start=v_start)
        self.passed = 0
        self.phase = "hunt"
        self.v_landmark = None
        self.v_catch = None
        self.v_overshoot = None
        self.v_lock = None
        self.v_gap_from = None
        self.pd_landmark = None
        self.state = "hunt"

    def seed_wing(self, pd_samples, err_samples):
        return self.inner.seed_wing(pd_samples, err_samples)

    def update(self, voltage, error, pd=None):
        voltage = float(voltage)
        error = float(error)
        if self.phase == "caught":
            self.state = "caught"
            return self.state

        # Doppler well sits above peak 3. Do not treat it as the landmark.
        if (
            self.arm_below_v is not None
            and voltage > self.arm_below_v + 1e-12
            and self.phase in ("hunt", "approach")
        ):
            self.inner.reset()
            self.phase = "approach"
            self.state = "approach"
            return self.state

        if self.phase == "gap":
            anchor = (
                self.v_gap_from
                if self.v_gap_from is not None
                else self.v_landmark
            )
            far = (
                anchor is not None and (anchor - voltage) >= self.gap_dv
            )
            drop = max(
                0.003,
                0.25
                * max(
                    0.0,
                    (self.pd_landmark or 0.0) - self.inner.pd_baseline,
                ),
            )
            off_peak = self.pd_landmark is None or pd <= self.pd_landmark - drop
            # Do not wait for quiet error: peak 2's approaching lobe may
            # already be starting. PD off the landmark is enough.
            if far and off_peak:
                self.inner.reset()
                self.inner.require_pd = True
                if self.min_pd_rise_v > 0.0:
                    self.inner.pd_rise_thresh = max(
                        self.inner.pd_rise_thresh, self.min_pd_rise_v
                    )
                self.phase = "hunt"
                self.state = "hunt"
                self.v_gap_from = None
                if self.passed == 0:
                    self.pd_landmark = None
            else:
                self.state = "gap"
            return self.state

        inner_state = self.inner.update(voltage, error, pd)
        if inner_state == "confirmed":
            v_zero = self.inner.v_zero
            if self.passed < self.n_skip:
                if (
                    self.landmark_max_v is not None
                    and v_zero is not None
                    and v_zero > self.landmark_max_v + 1e-12
                ):
                    # Extra line / Doppler above survey peak 3.
                    self.v_gap_from = v_zero
                    self.pd_landmark = pd if pd is not None else 0.0
                    self.phase = "gap"
                    self.state = "skip"
                    self.inner.reset()
                    return self.state
                self.passed += 1
                self.v_landmark = v_zero
                self.pd_landmark = pd if pd is not None else 0.0
                self.v_gap_from = v_zero
                self.phase = "gap"
                self.state = "landmark"
                return self.state
            self.passed += 1
            self.v_catch = v_zero
            self.v_overshoot = self.inner.v_overshoot
            self.phase = "caught"
            self.state = "caught"
            return self.state

        self.state = inner_state
        return self.state
