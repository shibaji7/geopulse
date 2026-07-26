"""GIC summary statistics — STUB (WP2)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError

__all__ = ["summary_stats"]


def summary_stats(gic_A: np.ndarray) -> dict[str, float]:
    """Return peak, RMS, mean, percentile stats for a GIC time series (WP2)."""
    raise NotImplementedYetError("gic.summary_stats", "WP2")
