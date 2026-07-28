"""Unit tests for :mod:`geopulse.metrics.gic`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.exceptions import DataError, ShapeMismatchError
from geopulse.metrics.gic import GICStats, exceedance_curve, summary_stats


class TestSummaryStats:
    def test_sine_wave_peak_and_rms_match_analytic(self):
        t = np.linspace(0.0, 100.0, 100_001)
        amp = 20.0
        x = amp * np.sin(2 * np.pi * t / 5.0)
        s = summary_stats(x, dt_s=1e-3)
        assert s.peak_abs == pytest.approx(amp, rel=1e-6)
        assert s.rms == pytest.approx(amp / np.sqrt(2.0), rel=1e-4)
        # p50 of |amp * sin| over a whole number of periods = amp * sin(pi/4).
        assert s.p50 == pytest.approx(amp * np.sin(np.pi / 4), rel=5e-3)
        assert s.n_samples == x.size

    def test_step_signal_thresholds(self):
        # 10 samples at 3 A, 5 samples at 12 A, 5 samples at 60 A.
        x = np.concatenate([np.full(10, 3.0), np.full(5, 12.0), np.full(5, 60.0)])
        s = summary_stats(x, dt_s=1.0, thresholds=(5.0, 10.0, 50.0))
        assert s.peak_abs == 60.0
        assert s.duration_above[5.0] == 10.0
        assert s.duration_above[10.0] == 10.0
        assert s.duration_above[50.0] == 5.0

    def test_returns_gicstats_dataclass(self):
        s = summary_stats(np.arange(100.0), dt_s=1.0)
        assert isinstance(s, GICStats)

    def test_nan_and_inf_are_dropped(self):
        x = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0, 4.0, 5.0])
        s = summary_stats(x, dt_s=1.0, thresholds=(2.5,))
        assert s.n_samples == 5
        assert s.peak_abs == 5.0
        assert s.duration_above[2.5] == 3.0  # 3, 4, 5 → 3 samples

    def test_signed_input_uses_abs_for_percentiles(self):
        x = np.array([-100.0, 0.0, 0.0, 0.0])
        s = summary_stats(x, dt_s=1.0)
        assert s.peak_abs == 100.0
        assert s.p50 == pytest.approx(0.0)

    def test_rejects_non_1d(self):
        with pytest.raises(ShapeMismatchError, match="1-D"):
            summary_stats(np.zeros((3, 3)), dt_s=1.0)

    def test_rejects_bad_dt(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(DataError, match="dt_s"):
                summary_stats(np.arange(10.0), dt_s=bad)

    def test_rejects_all_nonfinite(self):
        with pytest.raises(DataError, match="no finite"):
            summary_stats(np.array([np.nan, np.inf, -np.inf]), dt_s=1.0)


class TestExceedanceCurve:
    def test_monotone_nonincreasing_and_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(scale=2.0, size=5000)
        levels, prob = exceedance_curve(x, n_points=64)
        assert levels.shape == (64,)
        assert prob.shape == (64,)
        assert np.all(np.diff(levels) > 0)
        assert np.all(np.diff(prob) <= 0)
        assert prob[0] <= 1.0
        assert prob[-1] > 0.0

    def test_gaussian_survival_tail_reasonable(self):
        # For X ~ N(0, 1), P(|X| >= 2) ≈ 0.0455. Empirical estimator
        # with 20k samples should land in a wide but definite band.
        rng = np.random.default_rng(0)
        x = rng.normal(size=20_000)
        levels, prob = exceedance_curve(x, n_points=200)
        i = int(np.argmin(np.abs(levels - 2.0)))
        assert 0.02 < prob[i] < 0.08

    def test_narrow_range_uses_linear_grid(self):
        x = np.linspace(9.5, 10.5, 500)
        levels, _ = exceedance_curve(x, n_points=10)
        dl = np.diff(levels)
        assert np.allclose(dl, dl[0])

    def test_wide_range_uses_log_grid(self):
        x = np.logspace(-2, 3, 500)
        levels, _ = exceedance_curve(x, n_points=8)
        # Log-spaced → constant ratio between neighbours.
        r = levels[1:] / levels[:-1]
        assert np.allclose(r, r[0])

    def test_rejects_non_1d(self):
        with pytest.raises(ShapeMismatchError, match="1-D"):
            exceedance_curve(np.zeros((3, 3)))

    def test_rejects_bad_n_points(self):
        for bad in (0, -5):
            with pytest.raises(DataError, match="n_points"):
                exceedance_curve(np.arange(10.0), n_points=bad)

    def test_rejects_all_zero(self):
        with pytest.raises(DataError, match="non-zero"):
            exceedance_curve(np.zeros(100))
