r"""Modified Nodal Analysis (MNA) solver — DC with ideal voltage sources.

MNA (Ho, Ruehli, Brennan 1975) [1]_ extends the plain Nodal Admittance
Matrix (NAM) approach used by :class:`~geopulse.solver.nam.NAMSolver` to
handle **ideal voltage sources** — branches with a prescribed
``V_from − V_to = V_s`` and unknown current. Pure NAM cannot represent
zero-impedance connections or forced potentials because the branch
admittance matrix would be singular.

The augmented linear system is::

    ⎡ G   Bᵀ ⎤ ⎡ V     ⎤   ⎡ J   ⎤
    ⎢       ⎥ ⎢       ⎥ = ⎢     ⎥
    ⎣ B   0  ⎦ ⎣ I_vs  ⎦   ⎣ V_s ⎦

where

* ``G = [Y^n] + [Y^e]`` is the same nodal-conductance matrix that NAM
  solves (network admittance plus diagonal grounding admittances),
* ``J`` is the same driving-current vector NAM builds from per-branch
  ``V_th * g_ij`` (Pirjola et al. 2022 Eq. 5),
* ``B`` is the ``(m_vs × n_nodes)`` incidence matrix of the ``m_vs``
  ideal voltage-source branches (``+1`` at the ``from`` node, ``-1`` at
  the ``to`` node),
* ``V_s`` is the vector of prescribed source voltages,
* ``V`` is the vector of unknown node potentials,
* ``I_vs`` is the vector of unknown voltage-source currents.

When ``voltage_sources`` is empty the augmentation vanishes and MNA
reduces to NAM exactly — verified in the test suite.

Use cases in GIC context
------------------------
* **Zero-impedance grounding switches**: force a specific substation to
  V = 0 (or to a fixed potential imposed by a common ground bus) with
  a single voltage source between the substation node and a dedicated
  reference node.
* **Calibrated GIC injectors**: set up controlled bench tests where the
  measurement rig imposes a known potential.
* **Protective-relay-forced potentials**: model relay operation that
  clamps a node to a specific voltage during a fault sequence.

Explicitly not modelled here
----------------------------
* **Reactive elements** (L, C) — companion models + trapezoidal
  integration are a v0.3 item.
* **Nonlinear elements** (transformer magnetisation curves, diode
  drops) — needs Newton-Raphson iteration; slated for the
  :mod:`geopulse.solver.pyspice_bridge` module.
* **Time-domain integration** — this solver is DC / quasi-DC only, one
  static solve at a time. Callers who need a time series call it in a
  loop the same way case-07 does with NAM.

References
----------
.. [1] Ho, C.-W., Ruehli, A. E., Brennan, P. A. (1975). *The modified
   nodal approach to network analysis.* IEEE Trans. Circuits and
   Systems, CAS-22(6), 504-509. https://doi.org/10.1109/TCS.1975.1084079
.. [2] Vlach, J., Singhal, K. (1994). *Computer Methods for Circuit
   Analysis and Design*, 2nd ed. Van Nostrand Reinhold.
.. [3] Chua, L. O., Lin, P.-M. (1975). *Computer-Aided Analysis of
   Electronic Circuits: Algorithms and Computational Techniques.*
   Prentice-Hall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger

from geopulse.exceptions import ConvergenceError, DataError, ShapeMismatchError
from geopulse.solver.base import Solver, SolverResult

__all__ = ["MNASolver", "VoltageSource", "solve_mna"]


@dataclass(frozen=True)
class VoltageSource:
    """Ideal voltage source between two network nodes.

    Attributes
    ----------
    source_id : str
        Unique identifier surfaced in ``SolverResult.metadata``.
    from_node : str
        Node id at the ``+`` terminal.
    to_node : str
        Node id at the ``-`` terminal.
    voltage_V : float
        Prescribed potential difference in Volts:
        ``V_from − V_to = voltage_V``.
    """

    source_id: str
    from_node: str
    to_node: str
    voltage_V: float


def solve_mna(
    network_admittance: np.ndarray,
    earthing_impedance: np.ndarray,
    thevenin_voltages: np.ndarray,
    branch_endpoints: np.ndarray,
    branch_conductances: np.ndarray,
    vs_endpoints: np.ndarray,
    vs_voltages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Functional core of the MNA solve.

    Solves the augmented system ``[G Bᵀ; B 0] [V; I_vs] = [J; V_s]``.
    Reduces to :func:`~geopulse.solver.nam.solve_nam` when ``vs_endpoints``
    is empty.

    Parameters
    ----------
    network_admittance : numpy.ndarray
        ``[Y^n]`` matrix, shape ``(n_nodes, n_nodes)``, Siemens.
    earthing_impedance : numpy.ndarray
        Diagonal earthing-impedance matrix, shape ``(n_nodes, n_nodes)``,
        Ohms. Ungrounded nodes carry 0 and are automatically excluded
        from the linear solve (same convention as NAM).
    thevenin_voltages : numpy.ndarray
        ``V_th`` per resistive branch, Volts. Shape ``(n_branches,)``.
    branch_endpoints : numpy.ndarray of int
        ``(n_branches, 2)`` array of ``(from_index, to_index)`` matrix rows.
    branch_conductances : numpy.ndarray
        ``g_ij = 1/R_ij`` per resistive branch, Siemens. Shape
        ``(n_branches,)``.
    vs_endpoints : numpy.ndarray of int
        ``(n_vs, 2)`` array of voltage-source ``(from_index, to_index)``.
        Empty ``(0, 2)`` is allowed and degenerates to NAM.
    vs_voltages : numpy.ndarray
        Prescribed source voltages ``V_s``, shape ``(n_vs,)``, Volts.

    Returns
    -------
    node_voltages_V : numpy.ndarray
        ``V``, shape ``(n_nodes,)``.
    branch_currents_A : numpy.ndarray
        Currents in the resistive branches, shape ``(n_branches,)``.
    vs_currents_A : numpy.ndarray
        Currents flowing through each ideal voltage source, shape
        ``(n_vs,)``. Sign convention: positive I_vs means current
        **leaves the** ``from_node`` **through the source** (and
        equivalently, enters ``to_node``). Follows the standard MNA
        derivation where the incidence matrix has ``+1`` at ``from`` and
        ``-1`` at ``to``; positive I_vs contributes ``+I_vs`` to the
        current-leaving row of ``from_node``.

    Raises
    ------
    ConvergenceError
        If the augmented linear system is singular (typically a
        redundant loop of voltage sources).
    ShapeMismatchError
        If any of the endpoint / conductance / voltage arrays have
        shapes inconsistent with the declared branch / source counts.
    """
    n_nodes = int(network_admittance.shape[0])
    n_branches = int(thevenin_voltages.shape[0])
    n_vs = int(vs_endpoints.shape[0])
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
    if vs_endpoints.shape != (n_vs, 2):
        raise ShapeMismatchError(
            f"vs_endpoints shape {vs_endpoints.shape} inconsistent with n_vs={n_vs}"
        )
    if vs_voltages.shape != (n_vs,):
        raise ShapeMismatchError(
            f"vs_voltages shape {vs_voltages.shape} inconsistent with n_vs={n_vs}"
        )

    # Driving-current vector J^e (Pirjola et al. 2022 Eq. 5).
    J = np.zeros(n_nodes, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        gV = branch_conductances[k] * thevenin_voltages[k]
        J[i] -= gV
        J[j] += gV

    # G = [Y^n] + diag(1/Z_e).
    G = network_admittance.copy()
    Ze_diag = np.diag(earthing_impedance)
    grounded = Ze_diag > 0.0
    G[np.arange(n_nodes), np.arange(n_nodes)] += np.where(
        grounded, 1.0 / np.where(grounded, Ze_diag, 1.0), 0.0
    )

    if n_vs == 0:
        # Degenerates to NAM: solve on the active subspace.
        active = np.diag(G) != 0.0
        V = np.zeros(n_nodes, dtype=np.float64)
        if active.any():
            try:
                V[active] = np.linalg.solve(G[np.ix_(active, active)], J[active])
            except np.linalg.LinAlgError as exc:
                raise ConvergenceError(f"MNA linear solve failed: {exc}") from exc
        branch_currents = np.zeros(n_branches, dtype=np.float64)
        for k in range(n_branches):
            i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
            branch_currents[k] = branch_conductances[k] * (V[i] - V[j] + thevenin_voltages[k])
        return V, branch_currents, np.zeros(0, dtype=np.float64)

    # Build the incidence matrix B (n_vs × n_nodes): +1 at from, -1 at to.
    B = np.zeros((n_vs, n_nodes), dtype=np.float64)
    for m in range(n_vs):
        i, j = int(vs_endpoints[m, 0]), int(vs_endpoints[m, 1])
        B[m, i] += 1.0
        B[m, j] -= 1.0

    # Any node touched by a V-source is "active" even if it has zero G
    # diagonal (an ideal source can hold an otherwise-floating node to a
    # prescribed potential). Any node with a non-zero G diagonal is also
    # active.
    active_mask = np.diag(G) != 0.0
    active_mask = active_mask | (np.abs(B).sum(axis=0) > 0)
    n_act = int(active_mask.sum())
    if n_act == 0:
        return (
            np.zeros(n_nodes, dtype=np.float64),
            np.zeros(n_branches, dtype=np.float64),
            np.zeros(n_vs, dtype=np.float64),
        )

    G_sub = G[np.ix_(active_mask, active_mask)]
    J_sub = J[active_mask]
    B_sub = B[:, active_mask]

    # Assemble the (n_act + n_vs) × (n_act + n_vs) augmented matrix.
    A = np.zeros((n_act + n_vs, n_act + n_vs), dtype=np.float64)
    A[:n_act, :n_act] = G_sub
    A[:n_act, n_act:] = B_sub.T
    A[n_act:, :n_act] = B_sub
    rhs = np.concatenate([J_sub, vs_voltages])

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceError(
            f"MNA augmented solve failed (singular — likely redundant V-source loop): {exc}"
        ) from exc

    V = np.zeros(n_nodes, dtype=np.float64)
    V[active_mask] = sol[:n_act]
    vs_currents = sol[n_act:]

    branch_currents = np.zeros(n_branches, dtype=np.float64)
    for k in range(n_branches):
        i, j = int(branch_endpoints[k, 0]), int(branch_endpoints[k, 1])
        branch_currents[k] = branch_conductances[k] * (V[i] - V[j] + thevenin_voltages[k])

    return V, branch_currents, vs_currents


class MNASolver(Solver):
    r"""Modified Nodal Analysis solver — DC with ideal voltage sources.

    Extends :class:`~geopulse.solver.nam.NAMSolver` to accept a list of
    ideal voltage sources between named nodes. When ``voltage_sources``
    is empty the two solvers produce numerically identical results
    (verified in the test suite).

    Examples
    --------
    >>> from geopulse.solver.mna import MNASolver, VoltageSource   # doctest: +SKIP
    >>> vs = [VoltageSource("gnd_switch", "sub_A", "ground_ref", 0.0)]
    >>> result = MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)  # doctest: +SKIP
    >>> result.metadata["voltage_source_currents_A"]                     # doctest: +SKIP
    {'gnd_switch': 12.34}
    """

    def solve(
        self,
        network,
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
        voltage_sources: Sequence[VoltageSource] = (),
    ) -> SolverResult:
        """Solve the MNA-augmented DC system.

        Parameters
        ----------
        network : ConductorNetwork
            Same object handed to :class:`NAMSolver.solve` — used to look
            up node ids and branch endpoints.
        network_admittance, earthing_impedance, thevenin_voltages :
            As :class:`NAMSolver.solve`.
        voltage_sources : Sequence[VoltageSource], optional
            Ideal voltage sources to add. Default: empty tuple, which
            degenerates to a NAM solve.

        Returns
        -------
        SolverResult
            ``node_voltages_V`` and ``branch_currents_A`` as usual.
            ``metadata`` carries ``"solver_method": "MNA"`` always, plus
            ``"voltage_source_currents_A"`` (``{source_id: current_A}``
            dict) when ``voltage_sources`` is non-empty.

        Raises
        ------
        DataError
            If a :class:`VoltageSource` references an unknown node id, or
            two sources share the same ``source_id``.
        ShapeMismatchError
            If ``thevenin_voltages`` length does not match branch count.
        ConvergenceError
            If the augmented system is singular (typically a redundant
            V-source loop).
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

        vs_list = list(voltage_sources)
        seen_ids: set[str] = set()
        for vs in vs_list:
            if vs.source_id in seen_ids:
                raise DataError(f"duplicate voltage-source id {vs.source_id!r}")
            seen_ids.add(vs.source_id)
            if vs.from_node not in node_index:
                raise DataError(
                    f"voltage source {vs.source_id!r} references unknown from_node {vs.from_node!r}"
                )
            if vs.to_node not in node_index:
                raise DataError(
                    f"voltage source {vs.source_id!r} references unknown to_node {vs.to_node!r}"
                )

        n_vs = len(vs_list)
        if n_vs > 0:
            vs_endpoints = np.array(
                [(node_index[vs.from_node], node_index[vs.to_node]) for vs in vs_list],
                dtype=np.int64,
            )
            vs_voltages = np.array([vs.voltage_V for vs in vs_list], dtype=np.float64)
        else:
            vs_endpoints = np.zeros((0, 2), dtype=np.int64)
            vs_voltages = np.zeros(0, dtype=np.float64)

        logger.debug(
            "MNASolver: n_nodes={}, n_branches={}, n_voltage_sources={}",
            len(nodes),
            len(branches),
            n_vs,
        )

        V, I_branch, I_vs = solve_mna(
            network_admittance,
            earthing_impedance,
            thevenin_voltages,
            endpoints,
            conductances,
            vs_endpoints,
            vs_voltages,
        )
        metadata: dict = {"solver_method": "MNA"}
        if n_vs > 0:
            metadata["voltage_source_currents_A"] = {
                vs_list[m].source_id: float(I_vs[m]) for m in range(n_vs)
            }
        return SolverResult(
            node_voltages_V=V,
            branch_currents_A=I_branch,
            node_ids=node_ids,
            branch_ids=branch_ids,
            metadata=metadata,
        )
