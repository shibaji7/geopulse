"""Tests for :class:`geopulse.network.powergrid.PowerGridNetwork`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geopulse.network.base import Branch, Node
from geopulse.network.powergrid import PowerGridNetwork

EPRI21 = Path(__file__).parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

pytestmark = pytest.mark.skipif(not EPRI21.is_file(), reason="epri21.m benchmark data not present")


def test_from_file_loads_epri21():
    net = PowerGridNetwork.from_file(str(EPRI21))
    nodes = list(net.get_nodes())
    branches = list(net.get_branches())
    assert len(nodes) >= 20
    assert len(branches) >= 20
    assert all(isinstance(n, Node) for n in nodes)
    assert all(isinstance(b, Branch) for b in branches)


def test_admittance_symmetric_and_correctly_signed():
    net = PowerGridNetwork.from_file(str(EPRI21))
    Y = net.assemble_network_admittance()
    n = Y.shape[0]

    # Symmetric
    np.testing.assert_allclose(Y, Y.T)

    # Off-diagonal is negative sum of branch conductances between nodes;
    # each row's off-diagonal sum equals the negative of the diagonal
    # BEFORE grounding is added (which the network method omits).
    row_sums = Y.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)
    assert Y.shape == (n, n)


def test_earthing_impedance_diagonal_only():
    net = PowerGridNetwork.from_file(str(EPRI21))
    Z = net.assemble_earthing_impedance()
    # Zero off-diagonal
    np.testing.assert_array_equal(Z - np.diag(np.diag(Z)), 0.0)
    # Grounded nodes have positive z; ungrounded stay at zero.
    diag = np.diag(Z)
    assert np.all(diag >= 0.0)


def test_thevenin_uniform_eastward_field():
    net = PowerGridNetwork.from_file(str(EPRI21))
    branches = list(net.get_branches())
    # 1 V/km east = 1e-3 V/m east.
    Vth = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
    assert Vth.shape == (len(branches),)
    # At least some branches must have non-zero V_th (extended geographically).
    assert np.count_nonzero(Vth) > 0
    # For the Horton grid (~500 km east-west span) with 1 V/km field,
    # branch voltages should be < 1 kV in magnitude.
    assert np.all(np.abs(Vth) < 1_000.0)
