"""Lehtinen-Pirjola matrix method for DC/quasi-DC GIC in earthed networks.

Solves the nodal ground-potential problem::

    Y_n · V = J_e

where

*   ``Y_n`` is the total nodal admittance matrix (network conductances plus
    grounding conductances on the diagonal),
*   ``J_e`` is the injected-current vector — for every branch ``(i, j)`` with
    conductance ``g_ij = 1/R_ij`` and Thévenin voltage ``V_th,ij``, add
    ``+g_ij · V_th,ij`` at node ``j`` and ``-g_ij · V_th,ij`` at node ``i``,
*   ``V`` is the ground-potential rise at every node.

The GIC to ground at node ``i`` is then ``I_gnd,i = g_gnd,i · V_i``, and the
branch current is ``I_ij = g_ij · (V_i − V_j + V_th,ij)`` following the
sign convention used above.

This is mathematically equivalent to the ``I = (1 + Y_n · Z_e)^{-1} J_e``
form quoted in the handoff spec: both formulations produce identical
per-branch currents; the ``Y·V = J`` form is preferred here because it
handles ungrounded (``g_gnd = 0``) delta-winding nodes gracefully by
excluding them from the linear solve.

References
----------
.. [1] Lehtinen, M., & Pirjola, R. (1985). Currents produced in earthed
   conductor networks by geomagnetically-induced electric fields. Annales
   Geophysicae, 3, 479-484.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from geopulse.exceptions import ConvergenceError, ShapeMismatchError
from geopulse.solver.base import Solver, SolverResult

__all__ = ["LPMSolver", "solve_lpm"]


def solve_lpm(
    network_admittance: np.ndarray,
    earthing_impedance: np.ndarray,
    thevenin_voltages: np.ndarray,
    branch_endpoints: np.ndarray,
    branch_conductances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Functional core of the LPM solve.

    Parameters
    ----------
    network_admittance : numpy.ndarray
        ``Y_n`` matrix (line conductances only, no grounding). Shape
        ``(n_nodes, n_nodes)``. Symmetric, Siemens.
    earthing_impedance : numpy.ndarray
        ``Z_e`` diagonal (Ohms). Shape ``(n_nodes, n_nodes)``. Ungrounded
        nodes carry ``0``; those are excluded from the solve automatically.
    thevenin_voltages : numpy.ndarray
        ``V_th`` per branch, Volts. Shape ``(n_branches,)``.
    branch_endpoints : numpy.ndarray of int
        ``(n_branches, 2)`` array of ``(from_index, to_index)`` matrix rows.
    branch_conductances : numpy.ndarray
        ``g_ij = 1/R_ij`` per branch, Siemens. Shape ``(n_branches,)``.

    Returns
    -------
    node_voltages_V : numpy.ndarray
        Ground-potential rise at every node, Volts. Shape ``(n_nodes,)``.
    branch_currents_A : numpy.ndarray
        Current in every branch, Amperes. Shape ``(n_branches,)``.

    Raises
    ------
    ConvergenceError
        If the reduced linear system is singular even after removing
        ungrounded nodes.
    """
    n_nodes = network_admittance.shape[0]
    n_branches = thevenin_voltages.shape[0]
    if branch_endpoints.shape != (n_branches, 2):
        raise ShapeMismatchError(
            f"branch_endpoints shape {branch_endpoints.shape} inconsistent with "
            f"{n_branches} branches"
        )
    if branch_conductances.shape != (n_branches,):
        raise ShapeMismatchError(
            f"branch_conductances shape {branch_conductances.shape} inconsistent with "
            f"{n_branches} branches"
        )

    # Assemble injected-current vector J.
    J = np.zeros(n_nodes, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        gV = branch_conductances[k] * thevenin_voltages[k]
        # Current source is oriented such that positive V_th pushes current
        # from node i toward node j (see module docstring).
        J[i] -= gV
        J[j] += gV

    # Total nodal admittance: Y_total = Y_network + diag(1/Z_e) where Z_e>0.
    Y_total = network_admittance.copy()
    Ze_diag = np.diag(earthing_impedance)
    grounded = Ze_diag > 0.0
    Y_total[np.arange(n_nodes), np.arange(n_nodes)] += np.where(
        grounded, 1.0 / np.where(grounded, Ze_diag, 1.0), 0.0
    )

    # Ungrounded nodes with no branch contribution (delta-winding artefacts)
    # produce zero diagonals and would make the system singular. Solve only
    # on the "active" subspace where the diagonal is non-zero; leave the
    # remaining voltages at zero (physically: no ground current, no branch
    # contribution → ground-potential rise is undefined and irrelevant).
    active = np.diag(Y_total) != 0.0
    V = np.zeros(n_nodes, dtype=np.float64)
    if active.any():
        Y_sub = Y_total[np.ix_(active, active)]
        J_sub = J[active]
        try:
            V[active] = np.linalg.solve(Y_sub, J_sub)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(f"LPM linear solve failed: {exc}") from exc

    # Recover per-branch currents from the solved voltages.
    branch_currents = np.zeros(n_branches, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        branch_currents[k] = branch_conductances[k] * (V[i] - V[j] + thevenin_voltages[k])
    return V, branch_currents


class LPMSolver(Solver):
    """Lehtinen-Pirjola matrix solver.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> from geopulse.network.powergrid import PowerGridNetwork  # doctest: +SKIP
    >>> net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")  # doctest: +SKIP
    >>> Y = net.assemble_network_admittance()  # doctest: +SKIP
    >>> Z = net.assemble_earthing_impedance()  # doctest: +SKIP
    >>> Vth = net.compute_thevenin_voltages(1e-3, 0.0)  # 1 mV/m eastward  # doctest: +SKIP
    >>> result = LPMSolver().solve(net, Y, Z, Vth)  # doctest: +SKIP
    """

    def solve(
        self,
        network,
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
    ) -> SolverResult:
        """Solve for GIC given a :class:`ConductorNetwork` and its matrices.

        Parameters
        ----------
        network : ConductorNetwork
            The network object; used to pull node ids, branch ids, and the
            branch endpoint / conductance metadata that the raw matrices do
            not carry.
        network_admittance : numpy.ndarray
            ``Y_n`` matrix, Siemens.
        earthing_impedance : numpy.ndarray
            ``Z_e`` diagonal matrix, Ohms.
        thevenin_voltages : numpy.ndarray
            ``V_th`` per branch, Volts. Length must equal
            ``len(network.get_branches())``.

        Returns
        -------
        SolverResult
            Node voltages and branch currents in canonical network order.
        """
        nodes = list(network.get_nodes())
        branches = list(network.get_branches())
        node_ids = [n.node_id for n in nodes]
        branch_ids = [b.branch_id for b in branches]
        node_index = {n.node_id: i for i, n in enumerate(nodes)}

        if thevenin_voltages.shape != (len(branches),):
            raise ShapeMismatchError(
                f"thevenin_voltages length {thevenin_voltages.shape} does not "
                f"match {len(branches)} branches"
            )

        endpoints = np.array(
            [(node_index[b.from_node], node_index[b.to_node]) for b in branches],
            dtype=np.int64,
        )
        conductances = np.array([1.0 / b.resistance_Ohm for b in branches], dtype=np.float64)

        logger.debug(
            "LPMSolver: n_nodes={}, n_branches={}, |V_th|_max={:.3g} V",
            len(nodes),
            len(branches),
            float(np.max(np.abs(thevenin_voltages))) if len(branches) else 0.0,
        )

        V, I_branch = solve_lpm(
            network_admittance,
            earthing_impedance,
            thevenin_voltages,
            endpoints,
            conductances,
        )
        return SolverResult(
            node_voltages_V=V,
            branch_currents_A=I_branch,
            node_ids=node_ids,
            branch_ids=branch_ids,
        )
