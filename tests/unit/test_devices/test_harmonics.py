"""Unit tests for :mod:`geopulse.devices.harmonics`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.devices.harmonics import HarmonicSpectrum, extract_harmonics
from geopulse.exceptions import DataError, ShapeMismatchError


def _make_time(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / fs


class TestExtractHarmonics:
    def test_pure_sine_recovers_amplitude_rms(self):
        # 100 A peak, 60 Hz, exactly N whole cycles → no leakage.
        f0 = 60.0
        fs = 6000.0
        n = int(fs)  # 1 second → 60 whole cycles
        t = _make_time(n, fs)
        x = 100.0 * np.sin(2 * np.pi * f0 * t)
        s = extract_harmonics(t, x, fundamental_Hz=f0, n_harmonics=8)
        assert isinstance(s, HarmonicSpectrum)
        assert s.orders.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
        assert s.amplitudes_A[0] == pytest.approx(100.0 / np.sqrt(2.0), rel=1e-6)
        # No leakage into higher orders when the input is exactly at the fundamental.
        assert np.all(s.amplitudes_A[1:] < 1e-6)

    def test_dc_component_reported(self):
        f0 = 60.0
        fs = 6000.0
        t = _make_time(int(fs), fs)
        x = 42.0 + 10.0 * np.sin(2 * np.pi * f0 * t)
        s = extract_harmonics(t, x, fundamental_Hz=f0, n_harmonics=3)
        assert s.dc_A == pytest.approx(42.0, rel=1e-6)
        assert s.amplitudes_A[0] == pytest.approx(10.0 / np.sqrt(2.0), rel=1e-4)

    def test_h1_h3_h5_mix_amplitudes(self):
        f0 = 60.0
        fs = 6000.0
        t = _make_time(int(fs), fs)
        x = (
            100.0 * np.sin(2 * np.pi * f0 * t)
            + 20.0 * np.sin(2 * np.pi * 3 * f0 * t)
            + 5.0 * np.sin(2 * np.pi * 5 * f0 * t)
        )
        s = extract_harmonics(t, x, fundamental_Hz=f0, n_harmonics=6)
        assert s.amplitudes_A[0] == pytest.approx(100.0 / np.sqrt(2.0), rel=1e-4)
        assert s.amplitudes_A[2] == pytest.approx(20.0 / np.sqrt(2.0), rel=1e-4)
        assert s.amplitudes_A[4] == pytest.approx(5.0 / np.sqrt(2.0), rel=1e-4)
        for k in (1, 3, 5):  # h2, h4, h6 should be ≈ 0
            assert s.amplitudes_A[k] < 1e-6

    def test_parseval_rms_sums(self):
        f0 = 60.0
        fs = 6000.0
        t = _make_time(int(fs), fs)
        x = 100.0 * np.sin(2 * np.pi * f0 * t) + 30.0 * np.sin(2 * np.pi * 2 * f0 * t)
        s = extract_harmonics(t, x, fundamental_Hz=f0, n_harmonics=5)
        # AC RMS from spectrum vs from waveform.
        rms_from_spec = float(np.sqrt(np.sum(s.amplitudes_A**2)))
        rms_from_time = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
        assert rms_from_spec == pytest.approx(rms_from_time, rel=1e-3)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ShapeMismatchError, match="does not match"):
            extract_harmonics(np.arange(5.0), np.arange(6.0), fundamental_Hz=60.0)

    def test_rejects_multidim(self):
        with pytest.raises(ShapeMismatchError, match="1-D"):
            extract_harmonics(np.zeros((2, 3)), np.zeros((2, 3)), fundamental_Hz=60.0)

    def test_rejects_bad_fundamental(self):
        t = _make_time(100, 1000.0)
        x = np.zeros_like(t)
        for bad in (0.0, -60.0, float("nan"), float("inf")):
            with pytest.raises(DataError, match="fundamental_Hz"):
                extract_harmonics(t, x, fundamental_Hz=bad)

    def test_rejects_bad_n_harmonics(self):
        t = _make_time(100, 1000.0)
        x = np.zeros_like(t)
        for bad in (0, -5):
            with pytest.raises(DataError, match="n_harmonics"):
                extract_harmonics(t, x, fundamental_Hz=60.0, n_harmonics=bad)

    def test_rejects_nonuniform_time(self):
        t = np.array([0.0, 0.001, 0.005, 0.010])  # non-uniform
        x = np.zeros_like(t)
        with pytest.raises(DataError, match="uniformly"):
            extract_harmonics(t, x, fundamental_Hz=60.0)

    def test_rejects_above_nyquist(self):
        # fs = 1200 Hz, Nyquist = 600. Ask for 20 harmonics of 60 Hz → 1200 Hz > Nyquist.
        t = _make_time(1200, 1200.0)
        x = np.zeros_like(t)
        with pytest.raises(DataError, match="Nyquist"):
            extract_harmonics(t, x, fundamental_Hz=60.0, n_harmonics=20)
