"""Abstract base class for grounded conductor networks.

Infrastructure-agnostic interface: a submarine cable, a power grid, a
pipeline, and a railway all implement the same ABC. The
:mod:`~geopulse.solver` module never knows which kind of infrastructure it is
solving — only nodes, branches, admittances, and Thévenin voltages.

References
----------
.. [1] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Branch", "ConductorNetwork", "Node"]


@dataclass(frozen=True)
class Node:
    """A node (grounding point) in the conductor network.

    Attributes
    ----------
    node_id : str
        Unique identifier (e.g. ``"SUB_01"`` for a substation, ``"KP_100"``
        for a kilometer post on a pipeline).
    latitude_deg : float
        Geographic latitude, degrees north.
    longitude_deg : float
        Geographic longitude, degrees east.
    earthing_impedance_Ohm : float, optional
        Impedance to remote earth in Ohms. Default: 0.0 (perfectly grounded).
        Use ``numpy.inf`` for ungrounded (floating) nodes.
    """

    node_id: str
    latitude_deg: float
    longitude_deg: float
    earthing_impedance_Ohm: float = 0.0


@dataclass(frozen=True)
class Branch:
    """A branch (conductor segment) between two nodes.

    Attributes
    ----------
    branch_id : str
        Unique identifier.
    from_node : str
        ``node_id`` of the start node.
    to_node : str
        ``node_id`` of the end node.
    resistance_Ohm : float
        Total DC resistance of the branch in Ohms.
    length_m : float
        Physical length of the branch in meters.
    """

    branch_id: str
    from_node: str
    to_node: str
    resistance_Ohm: float
    length_m: float


class ConductorNetwork(abc.ABC):
    """Abstract base class for grounded conductor networks.

    Subclasses (all deferred to later phases):

    * ``CableNetwork`` — submarine cables (SCUBAS refactor).
    * ``PowerGridNetwork`` — substations + transmission lines.
    * ``PipelineNetwork`` — DSTL (Boteler 1997).
    * ``RailwayNetwork`` — track circuits + traction substations.
    """

    @abc.abstractmethod
    def get_nodes(self) -> Sequence[Node]:
        """Return all nodes in the network."""

    @abc.abstractmethod
    def get_branches(self) -> Sequence[Branch]:
        """Return all branches in the network."""

    @abc.abstractmethod
    def assemble_network_admittance(self) -> np.ndarray:
        """Assemble the network admittance matrix ``Y_n``.

        Returns
        -------
        numpy.ndarray
            Symmetric, real-valued matrix. Shape ``(n_nodes, n_nodes)``.
            Units: Siemens.

        Notes
        -----
        ``Y_n[i, i]`` = sum of admittances of branches connected to node ``i``.
        ``Y_n[i, j]`` = ``-admittance`` of the branch between ``i`` and ``j``.
        """

    @abc.abstractmethod
    def assemble_earthing_impedance(self) -> np.ndarray:
        """Assemble the diagonal earthing-impedance matrix ``Z_e``.

        Returns
        -------
        numpy.ndarray
            Diagonal matrix. Shape ``(n_nodes, n_nodes)``. Units: Ohms.
        """

    @abc.abstractmethod
    def compute_thevenin_voltages(
        self,
        ex_Vm: np.ndarray,
        ey_Vm: np.ndarray,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
    ) -> np.ndarray:
        """Compute the Thévenin equivalent voltage for each branch.

        ``V_th(ab) = ∫_a^b E(r) · dl``. For a uniform E-field over the branch
        length this reduces to ``V_th = Ex · Δx + Ey · Δy``.

        Parameters
        ----------
        ex_Vm, ey_Vm : numpy.ndarray
            North-south and east-west E-field components in V/m.
        lat_deg, lon_deg : numpy.ndarray
            Coordinates of the E-field evaluation points.

        Returns
        -------
        numpy.ndarray
            Thévenin voltage per branch. Shape ``(n_branches,)``. Units: V.
        """

    @classmethod
    @abc.abstractmethod
    def from_file(cls, path: str) -> "ConductorNetwork":
        """Load a network from disk (JSON, GeoJSON, or YAML).

        Parameters
        ----------
        path : str
            Path to the network definition file.

        Returns
        -------
        ConductorNetwork
            The loaded network.
        """
