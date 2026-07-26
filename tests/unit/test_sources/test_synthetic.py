"""Tests for :class:`geopulse.sources.synthetic.SyntheticSource`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.constants import NT_TO_T
from geopulse.exceptions import DataError
from geopulse.sources.synthetic import SyntheticSource


def test_gaussian_pulse_shape_and_peak():
    src = SyntheticSource("gaussian_pulse", amplitude_nT=500.0, t0_s=1800.0, sigma_s=300.0)
    b = src.load(start_s=0.0, end_s=3600.0, dt_s=1.0)

    assert b.bx_T.shape == (3600,)
    assert b.by_T.shape == (3600,)
    assert b.bz_T.shape == (3600,)
    assert b.bx_T.dtype == np.float64

    peak_idx = np.argmax(b.bx_T)
    assert peak_idx == 1800
    np.testing.assert_allclose(b.bx_T[peak_idx], 500.0 * NT_TO_T)

    np.testing.assert_array_equal(b.by_T, 0.0)
    np.testing.assert_array_equal(b.bz_T, 0.0)

    assert b.sampling_rate_Hz == pytest.approx(1.0)


def test_step_jumps_at_t0():
    src = SyntheticSource("step", amplitude_nT=100.0, t0_s=500.0)
    b = src.load(start_s=0.0, end_s=1000.0, dt_s=1.0)
    assert b.bx_T[0] == 0.0
    assert b.bx_T[-1] == pytest.approx(100.0 * NT_TO_T)


def test_sinusoid_amplitude_and_frequency():
    freq = 0.1
    src = SyntheticSource("sinusoid", amplitude_nT=200.0, frequency_Hz=freq)
    b = src.load(start_s=0.0, end_s=100.0, dt_s=0.1)
    np.testing.assert_allclose(np.abs(b.bx_T).max(), 200.0 * NT_TO_T, rtol=1e-6)
    zero_crossings = int(np.sum(np.diff(np.signbit(b.bx_T))))
    assert abs(zero_crossings - 2 * freq * 100.0) <= 1


def test_invalid_waveform_raises():
    with pytest.raises(DataError, match="Unknown waveform"):
        SyntheticSource("triangle")


def test_bad_time_window_raises():
    src = SyntheticSource("gaussian_pulse")
    with pytest.raises(DataError, match="end_s"):
        src.load(start_s=100.0, end_s=50.0)
