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
    ):
        self.n_sigma = float(n_sigma)
        self.pd_floor = float(pd_floor)
        self.err_floor = float(err_floor)
        self.confirm_dv = float(confirm_dv)
        self.lookback_dv = float(lookback_dv)
        self.pd_hysteresis = float(pd_hysteresis)
        self.pd_baseline = 0.0
        self.pd_rise_thresh = self.pd_floor
        self.err_thresh = self.err_floor
        self.reset()

    def reset(self, v_start=None):
        self.v_start = None if v_start is None else float(v_start)
        self.last_error = 0.0
        self.lobe_sign = 0.0
        self.v_lobe = None
        self.v_zero = None
        self.pd_peak = None
        self.history = []
        self.state = "search"

    def seed_wing(self, pd_samples, err_samples):
        pd_samples = list(pd_samples)
        err_samples = list(err_samples)
        self.pd_baseline = _median(pd_samples)
        pd_mad = _mad(pd_samples, self.pd_baseline)
        err_mad = _median(abs(float(e)) for e in err_samples)
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
        if self.state in ("lobe", "crossed") and self.pd_peak is not None:
            if pd >= self.pd_peak - self.pd_hysteresis:
                return True
        return self._local_pd_rise(voltage, pd) >= self.pd_rise_thresh

    def _is_lobe(self, error, factor=1.0):
        return abs(float(error)) >= factor * self.err_thresh

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
            # First lobe may be weaker than the departing one (asymmetric S).
            if on_peak and self._is_lobe(error, factor=0.5):
                self.state = "lobe"
                self.lobe_sign = 1.0 if error >= 0 else -1.0
                self.v_lobe = voltage
                self.pd_peak = pd

        elif self.state == "lobe":
            self.pd_peak = max(self.pd_peak or pd, pd)
            if crossed:
                self.state = "crossed"
                self.v_zero = voltage
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
            elif self._is_lobe(error) and error * self.lobe_sign < 0.0:
                self.state = "confirmed"
            elif not on_peak and not self._is_lobe(error):
                self.state = "search"
                self.lobe_sign = 0.0
                self.v_zero = None
                self.pd_peak = None

        self.last_error = error
        return self.state
