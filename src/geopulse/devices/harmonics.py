"""FFT-based harmonic + THD extraction utilities — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError

__all__ = ["extract_harmonics", "compute_thd"]


def extract_harmonics(
    time_s: np.ndarray,
    current_A: np.ndarray,
    fundamental_Hz: float,
    n_harmonics: int = 40,
) -> np.ndarray:
    """Extract the first ``n_harmonics`` harmonic amplitudes (WP3)."""
    raise NotImplementedYetError("extract_harmonics", "WP3")


def compute_thd(harmonics: np.ndarray) -> float:
    """Compute Total Harmonic Distortion from a harmonic-amplitude array (WP3)."""
    raise NotImplementedYetError("compute_thd", "WP3")
