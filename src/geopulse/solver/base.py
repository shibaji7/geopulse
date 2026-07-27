"""Abstract base class for circuit solvers.

The solver takes a :class:`~geopulse.network.base.ConductorNetwork` (which
provides ``[Y^n]``, earthing impedance, and ``V_th``) and returns the GIC in
every branch and the voltage at every node.

References
----------
.. [1] Pirjola, R. J., Boteler, D. H., Tuck, L., & Marsal, S. (2022).
   The Lehtinen-Pirjola method modified for efficient modelling of GIC in
   multiple voltage levels of a power network. Ann. Geophys., 40, 205-215.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
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
    metadata : dict
        Solver-specific auxiliary output. Populated by concrete solvers with
        e.g. ``{"solver_method": "NAM"}`` so downstream reports can attribute
        results to the correct algorithm. Mutable; the frozen-dataclass
        guarantee only covers rebinding the field, not its contents.
    """

    node_voltages_V: np.ndarray
    branch_currents_A: np.ndarray
    node_ids: list[str]
    branch_ids: list[str]
    metadata: dict = field(default_factory=dict)


class Solver(abc.ABC):
    """Abstract base class for circuit solvers.

    Subclasses:

    * ``NAMSolver`` — Nodal Admittance Matrix / Lehtinen-Pirjola modified
      (resistive, single-phase).
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
