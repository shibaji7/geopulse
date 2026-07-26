"""Tests for :class:`geopulse.solver.lpm.LPMSolver`.

Includes a 2-node hand-calculable case that exercises the LPM assembly
without any benchmark data files.
"""

from __future__ import annotations

import numpy as np

from geopulse.solver.lpm import solve_lpm


def test_two_node_hand_calculation():
    """
    Two nodes joined by one 1 Ω branch, each grounded through 10 Ω.
    Drive branch with V_th = 100 V (from node 0 to node 1).

    Hand solution (Y·V = J):
        g_branch = 1 S, g_gnd = 0.1 S each
        Y_total  = [[1.1, -1.0], [-1.0, 1.1]]
        J        = [-100, +100]    (V_th pushes +100 V into node 1, −100 V out of 0)
        V        = solve(Y, J) → V0 ≈ -47.62 V, V1 ≈ +47.62 V (antisymmetric)
        I_branch = 1 · (V0 - V1 + V_th) = 1·(-47.62 - 47.62 + 100) = 4.76 A
        I_gnd,0  = 0.1 · V0 = -4.76 A
        I_gnd,1  = 0.1 · V1 = +4.76 A   (equal magnitudes; ground current returns)
    """
    Y_net = np.array([[1.0, -1.0], [-1.0, 1.0]])
    Z_e = np.diag([10.0, 10.0])
    V_th = np.array([100.0])
    endpoints = np.array([[0, 1]])
    conductances = np.array([1.0])

    V, I = solve_lpm(Y_net, Z_e, V_th, endpoints, conductances)

    # Symmetric voltage split. Closed form:
    #   Y_tot = [[1.1, -1], [-1, 1.1]], J = [-100, +100]
    #   V = Y_tot^{-1} J → V0 = -100/2.1 ≈ -47.619, V1 = +47.619.
    np.testing.assert_allclose(V[0], -V[1], atol=1e-9)
    np.testing.assert_allclose(V[0], -100.0 / 2.1, rtol=1e-6)

    # Branch current from hand calc: g · (V_i - V_j + V_th) with V_th input orientation
    np.testing.assert_allclose(I[0], 1.0 * (V[0] - V[1] + V_th[0]))

    # Ground currents balance
    I_gnd = np.diag(1.0 / np.diag(Z_e)) @ V
    np.testing.assert_allclose(I_gnd[0], -I_gnd[1], atol=1e-9)


def test_ungrounded_node_no_crash():
    """A node with g_gnd == 0 must not make the linear system singular."""
    Y_net = np.array([[1.0, -1.0], [-1.0, 1.0]])
    Z_e = np.diag([10.0, 0.0])  # node 1 ungrounded (Z_e[1,1] = 0 → skipped)
    V_th = np.array([50.0])
    endpoints = np.array([[0, 1]])
    conductances = np.array([1.0])

    V, I = solve_lpm(Y_net, Z_e, V_th, endpoints, conductances)
    # Both nodes have branch contribution → both active; must be finite.
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(I))


def test_solver_from_network_wires_metadata():
    """LPMSolver.solve must accept a ConductorNetwork and return SolverResult."""
    from geopulse.network.base import Branch, ConductorNetwork, Node
    from geopulse.solver.lpm import LPMSolver

    class _MiniNet(ConductorNetwork):
        def get_nodes(self):
            return [
                Node("A", 0.0, 0.0, earthing_impedance_Ohm=10.0),
                Node("B", 0.1, 0.0, earthing_impedance_Ohm=10.0),
            ]

        def get_branches(self):
            return [Branch("br_AB", "A", "B", resistance_Ohm=1.0, length_m=11_000.0)]

        def assemble_network_admittance(self):
            return np.array([[1.0, -1.0], [-1.0, 1.0]])

        def assemble_earthing_impedance(self):
            return np.diag([10.0, 10.0])

        def compute_thevenin_voltages(self, ex_Vm, ey_Vm, lat_deg=None, lon_deg=None):
            return np.array([100.0])

        @classmethod
        def from_file(cls, path):
            raise NotImplementedError

    net = _MiniNet()
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(0.0, 0.0)

    result = LPMSolver().solve(net, Y, Z, V_th)
    assert result.node_ids == ["A", "B"]
    assert result.branch_ids == ["br_AB"]
    assert result.node_voltages_V.shape == (2,)
    assert result.branch_currents_A.shape == (1,)
