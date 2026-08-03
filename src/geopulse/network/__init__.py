"""Grounded-conductor network models.

Cable, power grid, pipeline, and railway networks all implement the same
:class:`~geopulse.network.base.ConductorNetwork` ABC.
"""

from __future__ import annotations

from geopulse.network.base import Branch, ConductorNetwork, Node
from geopulse.network.helpers import (
    add_tie,
    apply_resistive_blocker,
    evaluate_field_at_branch_midpoints,
    open_line,
)

__all__ = [
    "Branch",
    "ConductorNetwork",
    "Node",
    "add_tie",
    "apply_resistive_blocker",
    "evaluate_field_at_branch_midpoints",
    "open_line",
]
