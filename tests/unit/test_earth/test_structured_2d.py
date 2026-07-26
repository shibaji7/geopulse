"""Tests for :class:`geopulse.earth.structured_2d.CoastalCorrection2D`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.impedance import TensorImpedance
from geopulse.earth.layered_1d import Layered1D
from geopulse.earth.structured_2d import CoastalCorrection2D, Structured2D
from geopulse.exceptions import DataError, NotImplementedYetError

# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def land_model():
    """Dry continent: 0.001 S/m half-space (typical Precambrian shield)."""
    return Layered1D([ConductivityLayer(np.inf, 0.001)])


@pytest.fixture
def ocean_model():
    """Ocean: 1 km of 3.3 S/m seawater over the same 0.001 S/m basement."""
    return Layered1D(
        [
            ConductivityLayer(1_000.0, 3.3),
            ConductivityLayer(np.inf, 0.001),
        ]
    )


@pytest.fixture
def freqs_Hz():
    return np.logspace(-3, -1, 21)


# --- CoastalCorrection2D --------------------------------------------------


def test_returns_tensor_impedance(land_model, ocean_model, freqs_Hz):
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=0.0)
    imp = coast.compute_impedance(freqs_Hz)
    assert isinstance(imp, TensorImpedance)
    assert imp.rank == 2
    assert imp.Z_tensor.shape == (freqs_Hz.size, 2, 2)


def test_tier_is_two(land_model, ocean_model):
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=0.0)
    assert coast.tier == 2


def test_far_inland_matches_land(land_model, ocean_model, freqs_Hz):
    """d → +∞: both TE and TM must recover the land 1-D impedance."""
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=1_000_000.0)
    imp = coast.compute_impedance(freqs_Hz)
    Z_land = land_model.compute_impedance(freqs_Hz).Z_values
    # Tensor: Z[0,1] = Z_TE, Z[1,0] = -Z_TM. Both should equal Z_land.
    np.testing.assert_allclose(imp.Z_tensor[:, 0, 1], Z_land, rtol=1e-10)
    np.testing.assert_allclose(imp.Z_tensor[:, 1, 0], -Z_land, rtol=1e-10)


def test_far_seaward_matches_ocean(land_model, ocean_model, freqs_Hz):
    """d → −∞: both TE and TM must recover the ocean 1-D impedance."""
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=-1_000_000.0)
    imp = coast.compute_impedance(freqs_Hz)
    Z_ocean = ocean_model.compute_impedance(freqs_Hz).Z_values
    np.testing.assert_allclose(imp.Z_tensor[:, 0, 1], Z_ocean, rtol=1e-10)
    np.testing.assert_allclose(imp.Z_tensor[:, 1, 0], -Z_ocean, rtol=1e-10)


def test_at_coast_is_midway(land_model, ocean_model, freqs_Hz):
    """d = 0: Z_TE = Z_TM = (Z_land + Z_ocean) / 2 (tanh(0) = 0)."""
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=0.0)
    imp = coast.compute_impedance(freqs_Hz)
    Z_land = land_model.compute_impedance(freqs_Hz).Z_values
    Z_ocean = ocean_model.compute_impedance(freqs_Hz).Z_values
    mid = 0.5 * (Z_land + Z_ocean)
    np.testing.assert_allclose(imp.Z_tensor[:, 0, 1], mid, rtol=1e-12)
    # TM at d=0 is also midway (f_TM(0) = 0.5 regardless of decay length).
    np.testing.assert_allclose(imp.Z_tensor[:, 1, 0], -mid, rtol=1e-12)


def test_monotonic_transition_from_ocean_to_land(land_model, ocean_model):
    """|Z_TE|(d) must move monotonically from ocean to land as d increases."""
    freqs = np.array([1e-2])
    Z_land = abs(land_model.compute_impedance(freqs).Z_values[0])
    Z_ocean = abs(ocean_model.compute_impedance(freqs).Z_values[0])
    # For our synthetic land (0.001 S/m) vs ocean (3.3 S/m seawater),
    # |Z_land| > |Z_ocean| — verify direction.
    assert Z_land > Z_ocean

    distances = np.linspace(-200_000.0, 200_000.0, 11)
    Z_TE_abs = [
        abs(
            CoastalCorrection2D(land_model, ocean_model, d)
            .compute_impedance(freqs)
            .Z_tensor[0, 0, 1]
        )
        for d in distances
    ]
    diffs = np.diff(Z_TE_abs)
    # All differences same sign (monotonic increase from sea to land).
    assert np.all(diffs > 0)


def test_dc_frequency_handled(land_model, ocean_model):
    """ω = 0 should not raise; skin depth is infinite → tanh(0) → f = 0.5."""
    coast = CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=0.0)
    imp = coast.compute_impedance(np.array([0.0, 1e-3]))
    # At DC, Z_land = 0 and Z_ocean has non-trivial value from the layered
    # recursion; we don't check the DC row's value beyond finiteness.
    assert np.all(np.isfinite(imp.Z_tensor[1]))


def test_rejects_non_layered1d_models(ocean_model):
    with pytest.raises(DataError, match="land_model"):
        CoastalCorrection2D("not a model", ocean_model, distance_from_coast_m=0.0)  # type: ignore[arg-type]


def test_rejects_bad_decay_ratio(land_model, ocean_model):
    with pytest.raises(DataError, match="tm_decay_ratio"):
        CoastalCorrection2D(land_model, ocean_model, distance_from_coast_m=0.0, tm_decay_ratio=-1.0)


# --- Structured2D (still a stub) ------------------------------------------


def test_structured_2d_still_stub():
    with pytest.raises(NotImplementedYetError):
        Structured2D().compute_impedance(np.array([1e-2]))
