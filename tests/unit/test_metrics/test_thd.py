"""Unit tests for :mod:`geopulse.metrics.thd`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.devices.harmonics import HarmonicSpectrum, extract_harmonics
from geopulse.exceptions import DataError
from geopulse.metrics.thd import compute_thd


def _spec(amps: list[float], dc: float = 0.0) -> HarmonicSpectrum:
    return HarmonicSpectrum(
        fundamental_Hz=60.0,
        orders=np.arange(1, len(amps) + 1, dtype=np.int64),
        amplitudes_A=np.asarray(amps, dtype=np.float64),
        dc_A=dc,
    )


class TestComputeThd:
    def test_pure_fundamental_is_zero(self):
        assert compute_thd(_spec([100.0, 0.0, 0.0, 0.0, 0.0])) == 0.0

    def test_known_3_4_5_percent(self):
        # h2/h1 = 3 %, h3/h1 = 4 %.
        # THD = sqrt(3^2 + 4^2) / 100 = 5 %.
        s = _spec([100.0, 3.0, 4.0])
        assert compute_thd(s) == pytest.approx(0.05, rel=1e-9)

    def test_ieee_519_5_percent_edge(self):
        # A commonly cited IEEE 519 utility bus limit is 5% THD.
        # Sanity: a spectrum right at that threshold rounds correctly.
        s = _spec([100.0, 5.0])  # sqrt(5^2)/100 = 5 %
        assert round(compute_thd(s) * 100, 6) == 5.0

    def test_dc_excluded_by_default(self):
        s = _spec([100.0, 5.0], dc=200.0)
        assert compute_thd(s) == pytest.approx(0.05, rel=1e-9)

    def test_dc_included_when_requested(self):
        s = _spec([100.0, 3.0, 4.0], dc=12.0)
        # sqrt(12^2 + 3^2 + 4^2) / 100 = sqrt(144 + 9 + 16) / 100
        expected = float(np.sqrt(12.0**2 + 3.0**2 + 4.0**2)) / 100.0
        assert compute_thd(s, include_dc=True) == pytest.approx(expected, rel=1e-9)

    def test_end_to_end_via_extract_harmonics(self):
        # Build a waveform with known THD, run it through extract → compute_thd.
        f0 = 60.0
        fs = 6000.0
        n = int(fs)
        t = np.arange(n) / fs
        x = (
            100.0 * np.sin(2 * np.pi * f0 * t)
            + 3.0 * np.sin(2 * np.pi * 2 * f0 * t)
            + 4.0 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        s = extract_harmonics(t, x, fundamental_Hz=f0, n_harmonics=5)
        # THD ~ 5 % — pipe fully through.
        assert compute_thd(s) == pytest.approx(0.05, rel=5e-3)

    def test_rejects_missing_fundamental(self):
        # Orders start at 2 → no fundamental at index 0.
        s = HarmonicSpectrum(
            fundamental_Hz=60.0,
            orders=np.array([2, 3, 4], dtype=np.int64),
            amplitudes_A=np.array([1.0, 1.0, 1.0]),
            dc_A=0.0,
        )
        with pytest.raises(DataError, match="fundamental"):
            compute_thd(s)

    def test_rejects_zero_fundamental(self):
        with pytest.raises(DataError, match="undefined"):
            compute_thd(_spec([0.0, 1.0, 1.0]))
