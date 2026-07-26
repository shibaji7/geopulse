"""Thévenin voltage V_th = ∫E·dl for a conductor branch — STUB (WP2)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError

__all__ = ["compute_thevenin"]


def compute_thevenin(
    ex_Vm: np.ndarray,
    ey_Vm: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    """Compute V_th per branch from an E-field field (WP2)."""
    raise NotImplementedYetError("compute_thevenin", "WP2")
