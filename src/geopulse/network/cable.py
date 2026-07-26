"""Submarine cable network — refactor of SCUBAS — STUB (WP2)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geopulse.exceptions import NotImplementedYetError
from geopulse.network.base import Branch, ConductorNetwork, Node

__all__ = ["CableNetwork"]


class CableNetwork(ConductorNetwork):
    """Submarine cable network (WP2)."""

    def get_nodes(self) -> Sequence[Node]:
        raise NotImplementedYetError("CableNetwork.get_nodes", "WP2")

    def get_branches(self) -> Sequence[Branch]:
        raise NotImplementedYetError("CableNetwork.get_branches", "WP2")

    def assemble_network_admittance(self) -> np.ndarray:
        raise NotImplementedYetError("CableNetwork.assemble_network_admittance", "WP2")

    def assemble_earthing_impedance(self) -> np.ndarray:
        raise NotImplementedYetError("CableNetwork.assemble_earthing_impedance", "WP2")

    def compute_thevenin_voltages(self, ex_Vm, ey_Vm, lat_deg, lon_deg):
        raise NotImplementedYetError("CableNetwork.compute_thevenin_voltages", "WP2")

    @classmethod
    def from_file(cls, path: str) -> "CableNetwork":
        raise NotImplementedYetError("CableNetwork.from_file", "WP2")
