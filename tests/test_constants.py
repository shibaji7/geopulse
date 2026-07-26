"""Tests for :mod:`geopulse.constants`."""

from __future__ import annotations

import math

import numpy as np

from geopulse.constants import (
    DEG_TO_RAD,
    EPSILON_0,
    KM_TO_M,
    MU_0,
    NT_TO_T,
    R_EARTH_M,
)


def test_mu0_matches_definition():
    """μ₀ must equal 4π × 10⁻⁷ H/m exactly (pre-2019 SI still holds numerically)."""
    assert MU_0 == 4.0 * math.pi * 1e-7


def test_epsilon0_positive_and_finite():
    assert EPSILON_0 > 0
    assert math.isfinite(EPSILON_0)


def test_earth_radius_ballpark():
    assert 6_300_000 < R_EARTH_M < 6_400_000


def test_unit_conversions_roundtrip():
    assert 1.0 * NT_TO_T == 1e-9
    assert 1.0 * KM_TO_M == 1000.0
    np.testing.assert_allclose(180.0 * DEG_TO_RAD, math.pi)
