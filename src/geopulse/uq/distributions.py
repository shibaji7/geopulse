"""Distribution representations for uncertainty quantification — STUB (WP2)."""

from __future__ import annotations

from geopulse.exceptions import NotImplementedYetError

__all__ = ["Distribution"]


class Distribution:
    """Base class for named distributions (WP2 — replaces the string tag)."""

    def sample(self, n: int) -> list:
        raise NotImplementedYetError("Distribution.sample", "WP2")
