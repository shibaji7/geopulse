"""Tests for :class:`geopulse.earth.layered_1d.Layered1D`.

The uniform half-space identity is the critical Phase-1 acceptance test:
Wait recursion with a single layer must reproduce
``Z(ω) = sqrt(i·ω·μ₀/σ)`` at ``rtol=1e-12``.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.constants import MU_0
from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.exceptions import DataError


def test_uniform_halfspace_matches_analytic():
    sigma = 0.01
    freqs = np.logspace(-4, 0, 100)
    omega = 2.0 * np.pi * freqs

    Z_analytic = np.sqrt(1j * omega * MU_0 / sigma)

    model = Layered1D([ConductivityLayer(np.inf, sigma)])
    impedance = model.compute_impedance(freqs)

    np.testing.assert_allclose(
        impedance.Z_values,
        Z_analytic,
        rtol=1e-12,
        err_msg="Wait recursion must reproduce uniform half-space exactly",
    )


def test_uniform_stack_reduces_to_halfspace():
    """A multi-layer stack of identical σ must equal the half-space."""
    sigma = 0.005
    freqs = np.logspace(-3, -1, 20)

    ref = Layered1D([ConductivityLayer(np.inf, sigma)]).compute_impedance(freqs)
    multi = Layered1D(
        [
            ConductivityLayer(10_000.0, sigma),
            ConductivityLayer(20_000.0, sigma),
            ConductivityLayer(50_000.0, sigma),
            ConductivityLayer(np.inf, sigma),
        ]
    ).compute_impedance(freqs)

    np.testing.assert_allclose(multi.Z_values, ref.Z_values, rtol=1e-10)


def test_dc_returns_zero():
    model = Layered1D([ConductivityLayer(np.inf, 0.01)])
    imp = model.compute_impedance(np.array([0.0, 1e-3]))
    assert imp.Z_values[0] == 0.0
    assert imp.Z_values[1] != 0.0


def test_layered_impedance_is_scalar_impedance():
    from geopulse.earth.impedance import ScalarImpedance

    model = Layered1D([ConductivityLayer(np.inf, 0.01)])
    imp = model.compute_impedance(np.array([0.01, 0.1]))
    assert isinstance(imp, ScalarImpedance)
    assert imp.rank == 0


def test_missing_halfspace_raises():
    with pytest.raises(DataError, match="terminating half-space"):
        Layered1D([ConductivityLayer(1000.0, 0.01)])


def test_empty_layers_raises():
    with pytest.raises(DataError, match="at least one layer"):
        Layered1D([])


def test_negative_frequency_raises():
    model = Layered1D([ConductivityLayer(np.inf, 0.01)])
    with pytest.raises(DataError, match="non-negative"):
        model.compute_impedance(np.array([-1.0, 0.1]))
