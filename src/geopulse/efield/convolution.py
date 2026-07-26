"""Time-domain causal convolution E = z * b — STUB (WP1)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError

__all__ = ["compute_efield_convolution"]


def compute_efield_convolution(
    time_s: np.ndarray,
    b_field: np.ndarray,
    impulse_response: np.ndarray,
) -> np.ndarray:
    """Convolve a B-field time series with a causal impulse response (WP1)."""
    raise NotImplementedYetError("compute_efield_convolution", "WP1")
