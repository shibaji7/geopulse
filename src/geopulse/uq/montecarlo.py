"""Monte-Carlo propagation engine — STUB (WP2).

The Phase-0 :func:`geopulse.uq.uncertain.propagate_uncertainty` covers the
basic case; WP2 adds parallel execution, adaptive sampling, and QMC.
"""

from __future__ import annotations

from geopulse.exceptions import NotImplementedYetError

__all__ = ["monte_carlo"]


def monte_carlo(*args, **kwargs):
    """Parallel/adaptive Monte-Carlo propagation engine (WP2)."""
    raise NotImplementedYetError("monte_carlo", "WP2")
