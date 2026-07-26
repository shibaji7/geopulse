"""Railway traction / track-circuit network — STUB (WP4)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geopulse.exceptions import NotImplementedYetError
from geopulse.network.base import Branch, ConductorNetwork, Node

__all__ = ["RailwayNetwork"]


class RailwayNetwork(ConductorNetwork):
    """Electrified railway traction network (WP4)."""

    def get_nodes(self) -> Sequence[Node]:
        raise NotImplementedYetError("RailwayNetwork.get_nodes", "WP4")

    def get_branches(self) -> Sequence[Branch]:
        raise NotImplementedYetError("RailwayNetwork.get_branches", "WP4")

    def assemble_network_admittance(self) -> np.ndarray:
        raise NotImplementedYetError("RailwayNetwork.assemble_network_admittance", "WP4")

    def assemble_earthing_impedance(self) -> np.ndarray:
        raise NotImplementedYetError("RailwayNetwork.assemble_earthing_impedance", "WP4")

    def compute_thevenin_voltages(self, ex_Vm, ey_Vm, lat_deg, lon_deg):
        raise NotImplementedYetError("RailwayNetwork.compute_thevenin_voltages", "WP4")

    @classmethod
    def from_file(cls, path: str) -> "RailwayNetwork":
        raise NotImplementedYetError("RailwayNetwork.from_file", "WP4")
