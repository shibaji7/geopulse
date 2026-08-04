r"""Nodal Admittance Matrix (NAM) solver for DC/quasi-DC GIC in earthed networks.

The Nodal Admittance Matrix (NAM) method is the long-established standard in
the power industry for GIC calculations. It solves the nodal ground-potential
problem directly::

    ([Y^n] + [Y^e]) · V = J^e

where

* ``[Y^n]`` is the network admittance matrix (line conductances between
  nodes, with row sums on the diagonal),
* ``[Y^e]`` is the diagonal earthing-admittance matrix (``g_gnd`` on the
  diagonal; **zero** for ungrounded nodes — no fake-infinity workaround
  needed),
* ``[J^e]`` is the injected-current vector: for every branch ``(i, j)``
  with conductance ``g_ij = 1/R_ij`` and Thévenin voltage ``V_th,ij``,
  add ``+g_ij · V_th,ij`` at node ``j`` and ``−g_ij · V_th,ij`` at node
  ``i`` (Pirjola et al. 2022 Eq. 5),
* ``V`` is the ground-potential rise at every node.

The GIC to ground at node ``i`` is then ``I_gnd,i = g_gnd,i · V_i`` and the
branch current is ``I_ij = g_ij · (V_i − V_j + V_th,ij)``.

The matrix ``[Y^n] + [Y^e]`` is symmetric positive definite, admitting
efficient Cholesky decomposition and sparse-matrix techniques for
large-scale grids (Pirjola et al. 2022 § 3).

Naming — NAM vs. LPm
--------------------
Pirjola, Boteler, Tuck & Marsal (2022) introduce the "Lehtinen-Pirjola
modified" (LPm) method as a re-derivation of the original 1985 LP method
that avoids the fake-infinity earthing impedance for ungrounded nodes.
The LPm and NAM formulations are **mathematically identical** — they
solve the same symmetric-positive-definite system — but the NAM name is
what the power industry has used for decades (Boteler 2014).

This solver defaults to the NAM label but exposes a ``method_label``
parameter so publications and reports can cite either name without
switching implementations. The math never differs.

References
----------
.. [1] Pirjola, R. J., Boteler, D. H., Tuck, L., & Marsal, S. (2022).
   The Lehtinen-Pirjola method modified for efficient modelling of
   geomagnetically induced currents in multiple voltage levels of a
   power network. Annales Geophysicae, 40, 205-215.
   https://doi.org/10.5194/angeo-40-205-2022
.. [2] Boteler, D. H. (2014). Methodology for simulation of
   geomagnetically induced currents in power systems. J. Space Weather
   Space Clim., 4, A21.
.. [3] Lehtinen, M., & Pirjola, R. (1985). Currents produced in earthed
   conductor networks by geomagnetically-induced electric fields.
   Annales Geophysicae, 3, 479-484. (Original LP method — historical.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from geopulse.exceptions import ConvergenceError, DataError, ShapeMismatchError
from geopulse.solver.base import Solver, SolverResult

if TYPE_CHECKING:  # pragma: no cover
    from geopulse.network.base import ConductorNetwork

__all__ = ["NAMSolver", "solve_nam"]

_VALID_LABELS = ("NAM", "LPm")


def solve_nam(
    network_admittance: np.ndarray,
    earthing_impedance: np.ndarray,
    thevenin_voltages: np.ndarray,
    branch_endpoints: np.ndarray,
    branch_conductances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Functional core of the NAM (=LPm) solve.

    Solves ``([Y^n] + [Y^e]) V = J^e`` on the active (grounded or
    branch-connected) subspace, then reconstructs the per-branch currents
    from the node voltages.

    Parameters
    ----------
    network_admittance : numpy.ndarray
        ``[Y^n]`` — network admittance matrix, line conductances only.
        Shape ``(n_nodes, n_nodes)``. Symmetric, Siemens.
    earthing_impedance : numpy.ndarray
        Diagonal earthing-impedance matrix (Ohms). Shape
        ``(n_nodes, n_nodes)``. Ungrounded nodes carry ``0`` and are
        automatically excluded from the linear solve.
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
        If the reduced linear system is singular after removing
        ungrounded nodes.
    ShapeMismatchError
        If ``branch_endpoints`` or ``branch_conductances`` is inconsistent
        with the number of branches implied by ``thevenin_voltages``.
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

    # Assemble the driving-current vector J^e (Pirjola et al. 2022 Eq. 5).
    J = np.zeros(n_nodes, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        gV = branch_conductances[k] * thevenin_voltages[k]
        # Positive V_th pushes current from node i toward node j.
        J[i] -= gV
        J[j] += gV

    # Assemble [Y^n] + [Y^e]. Ungrounded nodes contribute 0 on the diagonal
    # of [Y^e] — no fake-infinity Z_e workaround, matching Pirjola et al.
    # 2022 Eq. 22.
    Y_total = network_admittance.copy()
    Ze_diag = np.diag(earthing_impedance)
    grounded = Ze_diag > 0.0
    Y_total[np.arange(n_nodes), np.arange(n_nodes)] += np.where(
        grounded, 1.0 / np.where(grounded, Ze_diag, 1.0), 0.0
    )

    # Ungrounded nodes with no branch contribution (delta-winding artefacts)
    # produce zero diagonals and would make the system singular. Solve on
    # the "active" subspace where the diagonal is non-zero and leave the
    # rest at zero (physically: no ground current, no branch contribution
    # → the ground-potential rise is undefined and irrelevant).
    active = np.diag(Y_total) != 0.0
    V = np.zeros(n_nodes, dtype=np.float64)
    if active.any():
        Y_sub = Y_total[np.ix_(active, active)]
        J_sub = J[active]
        try:
            V[active] = np.linalg.solve(Y_sub, J_sub)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(f"NAM linear solve failed: {exc}") from exc

    # Recover per-branch currents from the solved voltages.
    branch_currents = np.zeros(n_branches, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        branch_currents[k] = branch_conductances[k] * (V[i] - V[j] + thevenin_voltages[k])
    return V, branch_currents


class NAMSolver(Solver):
    r"""Nodal Admittance Matrix (NAM) / Lehtinen-Pirjola modified (LPm) solver.

    Both names refer to the same algorithm: solve
    ``([Y^n] + [Y^e]) V = J^e``. NAM is the power-industry standard label
    (Boteler 2014); LPm is the derivation and label from Pirjola et al.
    (2022) [1]_. This class uses the NAM label by default; set
    ``method_label="LPm"`` if you want that string echoed in the
    :class:`SolverResult` metadata for a publication that cites the
    Pirjola paper directly.

    Parameters
    ----------
    method_label : {"NAM", "LPm"}, optional
        Cosmetic label written into ``SolverResult.metadata`` and the
        debug log so reports can cite either name without ambiguity. Both
        settings execute **identical numerics**. Default: ``"NAM"``.

    Raises
    ------
    DataError
        If ``method_label`` is not one of the accepted strings.

    Examples
    --------
    >>> from geopulse.network.powergrid import PowerGridNetwork  # doctest: +SKIP
    >>> net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")  # doctest: +SKIP
    >>> Y = net.assemble_network_admittance()  # doctest: +SKIP
    >>> Z = net.assemble_earthing_impedance()  # doctest: +SKIP
    >>> Vth = net.compute_thevenin_voltages(1e-3, 0.0)  # doctest: +SKIP
    >>> result = NAMSolver().solve(net, Y, Z, Vth)  # doctest: +SKIP
    >>> NAMSolver(method_label="LPm")   # doctest: +SKIP
    """

    def __init__(self, method_label: Literal["NAM", "LPm"] = "NAM") -> None:
        if method_label not in _VALID_LABELS:
            raise DataError(f"method_label must be one of {_VALID_LABELS}, got {method_label!r}")
        self.method_label = method_label

    def solve(
        self,
        network: ConductorNetwork,
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
    ) -> SolverResult:
        """Solve for GIC given a :class:`ConductorNetwork` and its matrices.

        Parameters
        ----------
        network : ConductorNetwork
            Provides node/branch ids and the endpoint/conductance metadata.
        network_admittance : numpy.ndarray
            ``[Y^n]``, Siemens.
        earthing_impedance : numpy.ndarray
            Diagonal earthing-impedance matrix, Ohms.
        thevenin_voltages : numpy.ndarray
            ``V_th`` per branch, Volts.

        Returns
        -------
        SolverResult
            Node voltages and branch currents. ``metadata['solver_method']``
            carries the chosen label (``"NAM"`` or ``"LPm"``).
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
            "NAMSolver(method_label={!r}): n_nodes={}, n_branches={}, |V_th|_max={:.3g} V",
            self.method_label,
            len(nodes),
            len(branches),
            float(np.max(np.abs(thevenin_voltages))) if len(branches) else 0.0,
        )

        V, I_branch = solve_nam(
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
            metadata={"solver_method": self.method_label},
        )
