"""Unit tests for :mod:`geopulse.devices.blocker`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.devices.blocker import ResistiveBlocker
from geopulse.exceptions import DataError, ShapeMismatchError


class TestConstructor:
    def test_valid_resistance(self):
        b = ResistiveBlocker(resistance_Ohm=10.0)
        assert b.resistance_Ohm == 10.0

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
    def test_rejects_bad_resistance(self, bad):
        with pytest.raises(DataError, match="resistance_Ohm"):
            ResistiveBlocker(resistance_Ohm=bad)


class TestOhmsLaw:
    def test_constant_current_voltage_and_power(self):
        # 5 A through 10 Ω → 50 V drop, 250 W dissipation, sustained.
        b = ResistiveBlocker(resistance_Ohm=10.0)
        t = np.linspace(0.0, 60.0, 601)
        i = 5.0 * np.ones_like(t)
        r = b.inject_gic(t, i)

        assert np.allclose(r.response_current_A, i)
        assert r.metadata["blocker_resistance_Ohm"] == 10.0
        assert np.allclose(r.metadata["blocker_voltage_V"], 50.0)
        assert r.metadata["blocker_peak_voltage_V"] == pytest.approx(50.0)
        assert r.metadata["blocker_peak_power_W"] == pytest.approx(250.0)
        # Energy = P * duration = 250 * 60 = 15 000 J.
        assert r.metadata["blocker_dissipated_energy_J"] == pytest.approx(15_000.0)

    def test_zero_current_zero_everything(self):
        b = ResistiveBlocker(resistance_Ohm=42.0)
        t = np.linspace(0.0, 10.0, 101)
        i = np.zeros_like(t)
        r = b.inject_gic(t, i)
        assert r.metadata["blocker_peak_voltage_V"] == 0.0
        assert r.metadata["blocker_peak_power_W"] == 0.0
        assert r.metadata["blocker_dissipated_energy_J"] == 0.0

    def test_sign_invariance_of_energy(self):
        # Energy depends on I², so negative-sign current gives the
        # same total dissipation as its positive counterpart.
        b = ResistiveBlocker(resistance_Ohm=5.0)
        t = np.linspace(0.0, 10.0, 101)
        i_pos = 3.0 * np.ones_like(t)
        i_neg = -3.0 * np.ones_like(t)
        r_pos = b.inject_gic(t, i_pos)
        r_neg = b.inject_gic(t, i_neg)
        assert r_pos.metadata["blocker_dissipated_energy_J"] == pytest.approx(
            r_neg.metadata["blocker_dissipated_energy_J"]
        )
        assert r_pos.metadata["blocker_peak_voltage_V"] == pytest.approx(
            r_neg.metadata["blocker_peak_voltage_V"]
        )

    def test_large_resistance_effective_open(self):
        # 10^12 Ω models an ideal DC-blocking capacitor. 1 A through it
        # should give 10^12 V drop — dimensionally huge but mathematically
        # right; caller is expected to sanity-check voltage rating.
        b = ResistiveBlocker(resistance_Ohm=1e12)
        t = np.linspace(0.0, 1.0, 11)
        i = np.ones_like(t)
        r = b.inject_gic(t, i)
        assert r.metadata["blocker_peak_voltage_V"] == pytest.approx(1e12)


class TestABCContract:
    def test_returns_zero_thd_and_empty_harmonics(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        t = np.linspace(0.0, 1.0, 11)
        i = np.sin(2 * np.pi * t)
        r = b.inject_gic(t, i)
        assert r.thd == 0.0
        assert r.harmonics.shape == (0,)

    def test_no_thermal_fields(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        t = np.linspace(0.0, 1.0, 11)
        r = b.inject_gic(t, np.ones_like(t))
        assert r.top_oil_C is None
        assert r.hotspot_C is None

    def test_response_time_and_current_are_passthrough(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        t = np.arange(5, dtype=np.float64)
        i = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = b.inject_gic(t, i)
        assert np.array_equal(r.time_s, t)
        assert np.array_equal(r.response_current_A, i)


class TestEnergyIntegration:
    def test_trapezoidal_matches_analytic_for_ramp(self):
        # I(t) = t (0 to 10 s), R = 2 Ω. p(t) = 2 t².
        # ∫₀¹⁰ 2 t² dt = 2 · 1000/3 ≈ 666.667 J.
        b = ResistiveBlocker(resistance_Ohm=2.0)
        t = np.linspace(0.0, 10.0, 1001)  # dense grid → tiny trapezoidal error
        i = t.copy()
        r = b.inject_gic(t, i)
        assert r.metadata["blocker_dissipated_energy_J"] == pytest.approx(2000.0 / 3.0, rel=1e-3)


class TestErrors:
    def test_shape_mismatch(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        with pytest.raises(ShapeMismatchError, match="same length"):
            b.inject_gic(np.arange(5.0), np.arange(6.0))

    def test_non_1d(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        with pytest.raises(ShapeMismatchError, match="1-D"):
            b.inject_gic(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_too_few_samples(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        with pytest.raises(DataError, match="at least 2 samples"):
            b.inject_gic(np.array([0.0]), np.array([1.0]))

    def test_non_monotone_time(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        t = np.array([0.0, 2.0, 1.0, 3.0])  # goes backwards
        i = np.ones_like(t)
        with pytest.raises(DataError, match="strictly increasing"):
            b.inject_gic(t, i)


class TestABCParityArgs:
    def test_ac_kwargs_accepted_and_ignored(self):
        b = ResistiveBlocker(resistance_Ohm=1.0)
        t = np.linspace(0.0, 1.0, 11)
        i = np.ones_like(t)
        # These kwargs are accepted for ABC parity with TransformerModel
        # but must be inert — the response should be identical to the
        # default-kwarg call.
        r_default = b.inject_gic(t, i)
        r_with_ac = b.inject_gic(t, i, ac_voltage_V=345_000.0, ac_frequency_Hz=60.0)
        assert (
            r_default.metadata["blocker_dissipated_energy_J"]
            == r_with_ac.metadata["blocker_dissipated_energy_J"]
        )
