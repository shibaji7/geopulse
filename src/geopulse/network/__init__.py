"""Grounded-conductor network models.

Cable, power grid, pipeline, and railway networks all implement the same
:class:`~geopulse.network.base.ConductorNetwork` ABC.
"""

from __future__ import annotations

from geopulse.network.base import Branch, ConductorNetwork, Node
from geopulse.network.railway import (
    RailwayNetwork,
    RailwayParameters,
    rail_to_earth_voltage_analytic,
)

__all__ = [
    "Branch",
    "ConductorNetwork",
    "Node",
    "RailwayNetwork",
    "RailwayParameters",
    "rail_to_earth_voltage_analytic",
]
