"""Tests for :class:`geopulse.devices.transformer.TransformerModel`.

Cross-checked against a hand port of GIC_HSR_Model/src/thermal.py which
itself implements Mate 2021 eqns 5-9 verbatim. Two independent
implementations must agree to floating-point precision on identical inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.devices.transformer import ThermalParams, TransformerModel
from geopulse.exceptions import DataError


def _reference_hotspot(time_s, k_load, i_gic_abs, params: ThermalParams):
    """Standalone bilinear top-oil + hot-spot solver — the "second opinion"."""
    n = time_s.size
    dt_min = float(time_s[1] - time_s[0]) / 60.0
    delta_u = params.to_rated_rise_C * np.asarray(k_load) ** 2
    zeta = 2.0 * params.to_time_constant_min / dt_min
    a = 1.0 / (1.0 + zeta)
    b = (1.0 - zeta) / (1.0 + zeta)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = a * (delta_u[i] + delta_u[i - 1]) - b * delta[i - 1]
    eta = params.hs_coeff_C_per_A * np.abs(i_gic_abs)
    return params.ambient_C + delta, params.ambient_C + delta + eta


def test_matches_reference_implementation():
    t_s = np.arange(0.0, 4.0 * 3600.0, 60.0)  # 4 h at 1-min sampling
    gic = 5.0 * np.sin(2 * np.pi * t_s / 3600.0)  # oscillating GIC
    k = np.full(t_s.shape, 0.63)
    p = ThermalParams()
    resp = TransformerModel(params=p, k_load=k).inject_gic(t_s, gic)
    top_ref, hs_ref = _reference_hotspot(t_s, k, gic, p)
    np.testing.assert_allclose(resp.top_oil_C, top_ref, rtol=1e-12)
    np.testing.assert_allclose(resp.hotspot_C, hs_ref, rtol=1e-12)


def test_steady_state_top_oil_matches_rated_rise():
    """For constant k, δ_e → δ_r · k² as t >> τ_e (Mate eqn 5 asymptote)."""
    t_s = np.arange(0.0, 12.0 * 3600.0, 60.0)  # 12 h >> τ_e = 71 min
    k = 0.8
    p = ThermalParams()
    resp = TransformerModel(params=p, k_load=k).inject_gic(t_s, np.zeros_like(t_s))
    expected_ss_top_oil = p.ambient_C + p.to_rated_rise_C * k**2
    # Final sample should be within 0.1 °C of the analytic steady state.
    assert abs(resp.top_oil_C[-1] - expected_ss_top_oil) < 0.1


def test_hotspot_scales_linearly_with_gic():
    """η = R_e · |I_e| — doubling GIC doubles the hot-spot rise above top oil."""
    t_s = np.arange(0.0, 3600.0, 60.0)
    resp1 = TransformerModel().inject_gic(t_s, np.full_like(t_s, 10.0))
    resp2 = TransformerModel().inject_gic(t_s, np.full_like(t_s, 20.0))
    eta1 = resp1.hotspot_C - resp1.top_oil_C
    eta2 = resp2.hotspot_C - resp2.top_oil_C
    np.testing.assert_allclose(eta2, 2.0 * eta1, rtol=1e-12)


def test_zero_gic_gives_hotspot_equals_top_oil():
    t_s = np.arange(0.0, 3600.0, 60.0)
    resp = TransformerModel().inject_gic(t_s, np.zeros_like(t_s))
    np.testing.assert_array_equal(resp.hotspot_C, resp.top_oil_C)


def test_metadata_flags_limit_exceedance():
    t_s = np.arange(0.0, 3600.0, 60.0)
    # Massive GIC to push past the 240 °C 8-h limit.
    gic = np.full_like(t_s, 500.0)
    resp = TransformerModel().inject_gic(t_s, gic)
    assert resp.metadata["avg_limit_exceeded"] is True


def test_rejects_nonuniform_sampling():
    t_s = np.array([0.0, 60.0, 120.0, 240.0, 300.0])  # gap at index 2→3
    with pytest.raises(DataError, match="uniformly-sampled"):
        TransformerModel().inject_gic(t_s, np.zeros_like(t_s))


def test_rejects_mismatched_shapes():
    with pytest.raises(DataError, match="shapes must match"):
        TransformerModel().inject_gic(np.arange(10.0), np.arange(5.0))


def test_thd_is_nan_and_harmonics_empty():
    """The thermal model does not compute harmonics; those live in devices/harmonics."""
    t_s = np.arange(0.0, 600.0, 60.0)
    resp = TransformerModel().inject_gic(t_s, np.zeros_like(t_s))
    assert np.isnan(resp.thd)
    assert resp.harmonics.size == 0
