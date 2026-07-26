"""Abstract base class for circuit solvers.

The solver takes a :class:`~geopulse.network.base.ConductorNetwork` (which
provides ``Y_n``, ``Z_e``, and ``V_th``) and returns the GIC in every branch
and the voltage at every node.

References
----------
.. [1] Lehtinen, M., & Pirjola, R. (1985). Currents produced in earthed
   conductor networks by geomagnetically-induced electric fields. Annales
   Geophysicae, 3, 479-484.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from geopulse.network.base import ConductorNetwork

__all__ = ["Solver", "SolverResult"]


@dataclass(frozen=True)
class SolverResult:
    """Result container from a circuit solve.

    Attributes
    ----------
    node_voltages_V : numpy.ndarray
        Voltage at each node relative to remote earth. Shape ``(n_nodes,)``
        or ``(n_times, n_nodes)`` for a time series.
    branch_currents_A : numpy.ndarray
        Current in each branch, in Amperes. Shape ``(n_branches,)`` or
        ``(n_times, n_branches)``.
    node_ids : list of str
        Ordered node IDs matching the columns of :attr:`node_voltages_V`.
    branch_ids : list of str
        Ordered branch IDs matching the columns of :attr:`branch_currents_A`.
    """

    node_voltages_V: np.ndarray
    branch_currents_A: np.ndarray
    node_ids: list[str]
    branch_ids: list[str]


class Solver(abc.ABC):
    """Abstract base class for circuit solvers.

    Subclasses:

    * ``LPMSolver`` — Lehtinen-Pirjola matrix (resistive, single-phase).
    * ``MNASolver`` — Modified Nodal Analysis (reactive, multi-phase).
    * ``PySpiceSolver`` — nonlinear transient via SPICE backend.
    """

    @abc.abstractmethod
    def solve(
        self,
        network: "ConductorNetwork",
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
    ) -> SolverResult:
        """Solve the circuit for GIC.

        Parameters
        ----------
        network : ConductorNetwork
            The network the matrices were assembled from. Solvers use it to
            pull metadata that the numeric matrices do not carry — branch
            endpoints, node/branch IDs for :class:`SolverResult`, and any
            solver-specific data (transformer types for MNA, netlist for
            PySpiceSolver, ...).
        network_admittance : numpy.ndarray
            ``Y_n`` matrix. Shape ``(n_nodes, n_nodes)``. Units: Siemens.
        earthing_impedance : numpy.ndarray
            ``Z_e`` diagonal matrix. Shape ``(n_nodes, n_nodes)``. Units: Ohms.
        thevenin_voltages : numpy.ndarray
            ``V_th`` per branch. Shape ``(n_branches,)`` or
            ``(n_times, n_branches)``. Units: Volts.

        Returns
        -------
        SolverResult
            Node voltages and branch currents.
        """
