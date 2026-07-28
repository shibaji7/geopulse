"""Unit tests for :mod:`geopulse.solver.mna`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.exceptions import ConvergenceError, DataError, ShapeMismatchError
from geopulse.network.base import Branch, ConductorNetwork, Node
from geopulse.solver.mna import MNASolver, VoltageSource, solve_mna
from geopulse.solver.nam import NAMSolver


class _TinyNetwork(ConductorNetwork):
    """Three-node toy network: A─(1Ω)─B─(1Ω)─C, all grounded through 10Ω."""

    def get_nodes(self):
        return [
            Node("A", latitude_deg=0.0, longitude_deg=0.0, earthing_impedance_Ohm=10.0),
            Node("B", latitude_deg=0.0, longitude_deg=0.0, earthing_impedance_Ohm=10.0),
            Node("C", latitude_deg=0.0, longitude_deg=0.0, earthing_impedance_Ohm=10.0),
        ]

    def get_branches(self):
        return [
            Branch("br_AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=0.0),
            Branch("br_BC", from_node="B", to_node="C", resistance_Ohm=1.0, length_m=0.0),
        ]

    def assemble_network_admittance(self) -> np.ndarray:
        g = 1.0
        Y = np.array(
            [
                [g, -g, 0.0],
                [-g, 2 * g, -g],
                [0.0, -g, g],
            ],
            dtype=np.float64,
        )
        return Y

    def assemble_earthing_impedance(self) -> np.ndarray:
        return np.diag([10.0, 10.0, 10.0]).astype(np.float64)

    def compute_thevenin_voltages(self, ex_Vm, ey_Vm, **kwargs):
        # Not used in these tests — pass V_th directly.
        return np.zeros(2, dtype=np.float64)

    @classmethod
    def from_file(cls, path):
        raise NotImplementedError


@pytest.fixture
def net():
    return _TinyNetwork()


@pytest.fixture
def matrices(net):
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    return Y, Z


class TestEmptySourcesMatchesNAM:
    def test_no_vsource_matches_nam_bitwise(self, net, matrices):
        Y, Z = matrices
        V_th = np.array([1.0, 0.5], dtype=np.float64)  # arbitrary drive
        nam = NAMSolver().solve(net, Y, Z, V_th)
        mna = MNASolver().solve(net, Y, Z, V_th)
        assert np.allclose(mna.node_voltages_V, nam.node_voltages_V, atol=1e-12)
        assert np.allclose(mna.branch_currents_A, nam.branch_currents_A, atol=1e-12)

    def test_metadata_reports_mna(self, net, matrices):
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        r = MNASolver().solve(net, Y, Z, V_th)
        assert r.metadata["solver_method"] == "MNA"
        assert "voltage_source_currents_A" not in r.metadata


class TestVoltageSourceForcesPotential:
    def test_single_source_pins_node_potential(self, net, matrices):
        # Force V[A] − V[C] = 5. With Y-network and 10Ω grounds, MNA
        # should return exactly V_A − V_C = 5 regardless of driver.
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        vs = [VoltageSource("test_vs", "A", "C", 5.0)]
        r = MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)
        idx = {n: i for i, n in enumerate(r.node_ids)}
        assert r.node_voltages_V[idx["A"]] - r.node_voltages_V[idx["C"]] == pytest.approx(
            5.0, abs=1e-12
        )

    def test_ideal_short_equalises_two_nodes(self, net, matrices):
        # 0 V source between A and C — should force V[A] == V[C].
        Y, Z = matrices
        V_th = np.array([1.0, -0.3], dtype=np.float64)  # asymmetric drive
        vs = [VoltageSource("short_ac", "A", "C", 0.0)]
        r = MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)
        idx = {n: i for i, n in enumerate(r.node_ids)}
        assert r.node_voltages_V[idx["A"]] == pytest.approx(r.node_voltages_V[idx["C"]], abs=1e-12)

    def test_vs_current_reported_in_metadata(self, net, matrices):
        Y, Z = matrices
        V_th = np.array([1.0, 0.0], dtype=np.float64)
        vs = [
            VoltageSource("vs1", "A", "C", 2.0),
            VoltageSource("vs2", "B", "C", 1.0),
        ]
        r = MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)
        currents = r.metadata["voltage_source_currents_A"]
        assert set(currents) == {"vs1", "vs2"}
        assert all(np.isfinite(v) for v in currents.values())


class TestConservation:
    def test_kirchhoff_current_law_at_node(self, net, matrices):
        # Sum of currents leaving a node (branch + ground + V-source) = 0.
        Y, Z = matrices
        V_th = np.array([2.0, -1.0], dtype=np.float64)
        vs = [VoltageSource("gnd_switch", "B", "C", 0.5)]
        r = MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)
        idx = {n: i for i, n in enumerate(r.node_ids)}
        Ze = np.diag(Z)
        i_vs = r.metadata["voltage_source_currents_A"]["gnd_switch"]

        # KCL at node B, sum of currents LEAVING = 0:
        #   -I_AB (branch A->B enters B, so leaves is negative)
        #   +I_BC (branch B->C leaves B)
        #   +I_gnd_B (to earth via 10 Ω, positive = out)
        #   +I_vs   (positive I_vs leaves vs.from_node = B by MNA sign convention)
        i_ab = r.branch_currents_A[0]
        i_bc = r.branch_currents_A[1]
        i_gnd_B = r.node_voltages_V[idx["B"]] / Ze[idx["B"]]
        net_out = -i_ab + i_bc + i_gnd_B + i_vs
        assert abs(net_out) < 1e-10


class TestErrors:
    def test_rejects_unknown_from_node(self, net, matrices):
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        vs = [VoltageSource("bad", "GHOST", "A", 1.0)]
        with pytest.raises(DataError, match="unknown from_node"):
            MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)

    def test_rejects_unknown_to_node(self, net, matrices):
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        vs = [VoltageSource("bad", "A", "GHOST", 1.0)]
        with pytest.raises(DataError, match="unknown to_node"):
            MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)

    def test_rejects_duplicate_source_ids(self, net, matrices):
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        vs = [
            VoltageSource("dup", "A", "B", 1.0),
            VoltageSource("dup", "B", "C", 2.0),
        ]
        with pytest.raises(DataError, match="duplicate"):
            MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)

    def test_rejects_thevenin_length_mismatch(self, net, matrices):
        Y, Z = matrices
        V_th = np.zeros(5, dtype=np.float64)  # network has 2 branches
        with pytest.raises(ShapeMismatchError, match="does not match"):
            MNASolver().solve(net, Y, Z, V_th)

    def test_singular_redundant_loop_raises_convergence_error(self, net, matrices):
        # Two V-sources between the same node pair with different voltages
        # → over-determined, singular augmented system.
        Y, Z = matrices
        V_th = np.zeros(2, dtype=np.float64)
        vs = [
            VoltageSource("vs1", "A", "B", 1.0),
            VoltageSource("vs2", "A", "B", 2.0),
        ]
        with pytest.raises(ConvergenceError, match="singular"):
            MNASolver().solve(net, Y, Z, V_th, voltage_sources=vs)


class TestFunctionalCore:
    def test_solve_mna_matches_class_solver(self, net, matrices):
        # Directly call the functional core; verify identical to the class API.
        Y, Z = matrices
        V_th = np.array([0.3, 0.7], dtype=np.float64)
        endpoints = np.array([[0, 1], [1, 2]], dtype=np.int64)
        conductances = np.array([1.0, 1.0], dtype=np.float64)
        vs_endpoints = np.array([[0, 2]], dtype=np.int64)  # A -> C
        vs_voltages = np.array([2.5], dtype=np.float64)

        V, I_br, I_vs = solve_mna(Y, Z, V_th, endpoints, conductances, vs_endpoints, vs_voltages)
        r = MNASolver().solve(net, Y, Z, V_th, voltage_sources=[VoltageSource("x", "A", "C", 2.5)])
        assert np.allclose(V, r.node_voltages_V, atol=1e-12)
        assert np.allclose(I_br, r.branch_currents_A, atol=1e-12)
        assert I_vs[0] == pytest.approx(r.metadata["voltage_source_currents_A"]["x"])
