r"""Geographic projections used for line integrals of the geoelectric field.

Two projections are provided:

* :func:`latlon_to_local_xy_spherical_m` — mean-radius spherical, simple and
  cheap, accurate to ~0.3 % for continent-scale networks.
* :func:`latlon_to_local_xy_wgs84_m` — WGS84 ellipsoid, matches Horton
  (2012) Appendix eqns A3, A5, A6 exactly.

Both return ``(x_east_m, y_north_m)`` offsets in metres from a projection
origin ``(lat0_deg, lon0_deg)``.

The WGS84 form is the default for :class:`~geopulse.network.powergrid.PowerGridNetwork`
and :class:`~geopulse.network.pipeline.PipelineNetwork` so that GIC benchmark
results match published papers to sub-percent precision.

Notes
-----
Horton (2012) Appendix — the north-south and east-west local distances
between two lat/lon points ``A``, ``B`` in kilometres are:

.. math::

    \phi &= (\mathrm{lat}_A + \mathrm{lat}_B) / 2                    \\
    L_N &= (111.133 - 0.56\,\cos 2\phi)\,\Delta\mathrm{lat}          \\
    L_E &= (111.5065 - 0.1872\,\cos 2\phi)\,\cos\phi\,\Delta\mathrm{lon}

with ``Δlat``, ``Δlon`` in degrees. This is the compact form of the exact
meridian- and prime-vertical-radius-of-curvature expressions:

.. math::

    M(\phi) &= \frac{a(1 - e^2)}{(1 - e^2 \sin^2\phi)^{3/2}}         \\
    N(\phi) &= \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}                    \\
    L_N &= \frac{\pi}{180}\,M(\phi)\,\Delta\mathrm{lat}              \\
    L_E &= \frac{\pi}{180}\,N(\phi)\,\cos\phi\,\Delta\mathrm{lon}

The implementation uses the exact ``M``, ``N`` form so we stay accurate at
polar latitudes where the truncated series in the paper's Appendix starts
to lose digits.

References
----------
.. [1] Horton, R., et al. (2012). A Test Case for the Calculation of
   Geomagnetically Induced Currents. IEEE Trans. Power Delivery, 27(4),
   Appendix.
"""

from __future__ import annotations

import numpy as np

from geopulse.constants import DEG_TO_RAD, R_EARTH_M, WGS84_A_M, WGS84_E2

__all__ = [
    "latlon_to_local_xy_spherical_m",
    "latlon_to_local_xy_wgs84_m",
    "meridian_radius_m",
    "prime_vertical_radius_m",
]


def meridian_radius_m(lat_deg: float) -> float:
    r"""WGS84 meridional radius of curvature ``M(φ)`` in metres.

    ``M(φ) = a(1 − e²) / (1 − e² sin²φ)^{3/2}`` — the radius of curvature
    of the ellipsoid along a meridian (north-south).
    """
    sin_phi = np.sin(lat_deg * DEG_TO_RAD)
    return WGS84_A_M * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin_phi * sin_phi) ** 1.5


def prime_vertical_radius_m(lat_deg: float) -> float:
    r"""WGS84 prime-vertical radius of curvature ``N(φ)`` in metres.

    ``N(φ) = a / √(1 − e² sin²φ)`` — the radius of curvature in the plane
    perpendicular to a meridian, at latitude ``φ``.
    """
    sin_phi = np.sin(lat_deg * DEG_TO_RAD)
    return WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)


def latlon_to_local_xy_spherical_m(
    lat_deg: float,
    lon_deg: float,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[float, float]:
    r"""Equirectangular projection about ``(lat0, lon0)`` — spherical Earth.

    Uses :data:`~geopulse.constants.R_EARTH_M` (mean radius 6371 km). Cheap
    and adequate to ~0.3 % for a few hundred kilometres, but drifts at
    continent scale. Prefer :func:`latlon_to_local_xy_wgs84_m` for anything
    published.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Point coordinates in degrees.
    lat0_deg, lon0_deg : float
        Projection origin in degrees.

    Returns
    -------
    x_east_m, y_north_m : float
        Eastward and northward offsets in metres.
    """
    dlat = (lat_deg - lat0_deg) * DEG_TO_RAD
    dlon = (lon_deg - lon0_deg) * DEG_TO_RAD
    y_m = R_EARTH_M * dlat
    x_m = R_EARTH_M * dlon * float(np.cos(lat0_deg * DEG_TO_RAD))
    return float(x_m), float(y_m)


def latlon_to_local_xy_wgs84_m(
    lat_deg: float,
    lon_deg: float,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[float, float]:
    r"""Equirectangular projection about ``(lat0, lon0)`` — WGS84 ellipsoid.

    Uses meridian- and prime-vertical-radii-of-curvature at the mean
    latitude ``φ = (lat + lat0) / 2`` to convert degree offsets into
    east/north offsets in metres. Matches Horton (2012) Appendix A3/A7
    exactly (the ``111.133 − 0.56 cos 2φ`` etc. truncated forms are the
    small-``e²`` series expansion of this same expression).

    Parameters
    ----------
    lat_deg, lon_deg : float
        Point coordinates in degrees.
    lat0_deg, lon0_deg : float
        Projection origin in degrees.

    Returns
    -------
    x_east_m, y_north_m : float
        Eastward and northward offsets in metres.

    Examples
    --------
    >>> # 1° latitude at 45° N ≈ 111.13 km per WGS84.
    >>> x, y = latlon_to_local_xy_wgs84_m(46.0, -75.0, 45.0, -75.0)
    >>> 111_000 < y < 112_000
    True
    """
    phi_mid_deg = 0.5 * (lat_deg + lat0_deg)
    M = meridian_radius_m(phi_mid_deg)
    N = prime_vertical_radius_m(phi_mid_deg)
    dlat_rad = (lat_deg - lat0_deg) * DEG_TO_RAD
    dlon_rad = (lon_deg - lon0_deg) * DEG_TO_RAD
    y_m = M * dlat_rad
    x_m = N * dlon_rad * float(np.cos(phi_mid_deg * DEG_TO_RAD))
    return float(x_m), float(y_m)
