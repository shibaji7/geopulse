"""Tests for :mod:`geopulse.geo` projections."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.geo import (
    latlon_to_local_xy_spherical_m,
    latlon_to_local_xy_wgs84_m,
    meridian_radius_m,
    prime_vertical_radius_m,
)


def test_spherical_projection_origin_zero():
    x, y = latlon_to_local_xy_spherical_m(45.0, -75.0, 45.0, -75.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_wgs84_projection_origin_zero():
    x, y = latlon_to_local_xy_wgs84_m(45.0, -75.0, 45.0, -75.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_wgs84_one_degree_latitude_matches_horton_appendix():
    """
    Horton (2012) eqn A3: L_N ≈ (111.133 - 0.56·cos 2φ)·Δlat, km.
    At φ = 45°, cos 2φ = 0. So 1° latitude ≈ 111.133 km.
    """
    _, y = latlon_to_local_xy_wgs84_m(46.0, -75.0, 45.0, -75.0)
    np.testing.assert_allclose(y / 1000.0, 111.133, rtol=1e-3)


def test_wgs84_one_degree_longitude_at_45N():
    """
    Horton (2012) eqn A7: L_E ≈ (111.5065 − 0.1872·cos 2φ)·cos φ·Δlon.
    At φ = 45°, cos 2φ = 0, cos φ = √2/2. So 1° lon ≈ 111.5065·0.7071 ≈ 78.85 km.
    """
    x, _ = latlon_to_local_xy_wgs84_m(45.0, -74.0, 45.0, -75.0)
    np.testing.assert_allclose(x / 1000.0, 111.5065 * np.sqrt(2) / 2, rtol=1e-3)


def test_wgs84_agrees_with_spherical_at_short_range():
    """~1 % agreement at ~100 km. The two diverge more at continent scale
    because spherical uses cos(lat0) while WGS84 uses cos(phi_mid); that
    difference is exactly why we switched the networks to WGS84."""
    x_sph, y_sph = latlon_to_local_xy_spherical_m(45.5, -74.5, 45.0, -75.0)
    x_wgs, y_wgs = latlon_to_local_xy_wgs84_m(45.5, -74.5, 45.0, -75.0)
    assert abs(x_sph - x_wgs) / abs(x_wgs) < 0.01
    assert abs(y_sph - y_wgs) / abs(y_wgs) < 0.01


def test_curvature_radii_bracket_mean_radius():
    """M(φ), N(φ) must sit near 6371 km (mean Earth radius) at mid-lat."""
    M = meridian_radius_m(45.0)
    N = prime_vertical_radius_m(45.0)
    assert 6_300_000 < M < 6_400_000
    assert 6_300_000 < N < 6_400_000
