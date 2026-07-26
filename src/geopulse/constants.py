"""Physical constants used throughout GeoPulse.

All values in SI units. Sources cited for every constant. These are EXACT or
CODATA-recommended values — do not modify without updating the source citation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MU_0",
    "EPSILON_0",
    "R_EARTH_M",
    "WGS84_A_M",
    "WGS84_B_M",
    "WGS84_E2",
    "NT_TO_T",
    "KM_TO_M",
    "DEG_TO_RAD",
]

# --- Electromagnetic constants ---
MU_0: float = 4.0 * np.pi * 1e-7
"""Vacuum permeability μ₀ in H/m (henries per meter). Exact (SI 2019)."""

EPSILON_0: float = 8.8541878128e-12
"""Vacuum permittivity ε₀ in F/m. CODATA 2018."""

# --- Earth geometry ---
R_EARTH_M: float = 6_371_000.0
"""Mean Earth radius in meters. IUGG reference (volumetric mean)."""

WGS84_A_M: float = 6_378_137.0
"""WGS84 equatorial radius (semi-major axis) in metres. Exact by definition."""

WGS84_B_M: float = 6_356_752.314245
"""WGS84 polar radius (semi-minor axis) in metres. Derived from a and f."""

WGS84_E2: float = 6.694379990141316e-3
"""WGS84 first eccentricity squared, e² = (a² − b²)/a². Horton (2012) Table IX."""

# --- Conversion factors ---
NT_TO_T: float = 1e-9
"""Convert nanotesla to Tesla."""

KM_TO_M: float = 1_000.0
"""Convert kilometers to meters."""

DEG_TO_RAD: float = np.pi / 180.0
"""Convert degrees to radians."""
