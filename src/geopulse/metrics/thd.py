"""Total Harmonic Distortion metric — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError

__all__ = ["compute_thd"]


def compute_thd(current_A: np.ndarray, fundamental_Hz: float) -> float:
    """Compute THD of a current waveform (WP3)."""
    raise NotImplementedYetError("thd.compute_thd", "WP3")
