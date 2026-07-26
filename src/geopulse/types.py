"""Type aliases and custom types used across GeoPulse.

Centralising types here avoids circular imports and provides a single
reference for all type contracts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from geopulse.uq.uncertain import Uncertain

__all__ = [
    "FloatArray",
    "ComplexArray",
    "LatLon",
    "Uncertain",
]

# --- Array type aliases ---
FloatArray = npt.NDArray[np.float64]
"""1-D or N-D array of float64. The workhorse type."""

ComplexArray = npt.NDArray[np.complex128]
"""1-D or N-D array of complex128. For frequency-domain quantities."""

# --- Coordinate types ---
LatLon = tuple[float, float]
"""(latitude_deg, longitude_deg) pair, both in degrees."""
