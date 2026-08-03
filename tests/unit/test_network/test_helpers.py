"""Unit tests for :mod:`geopulse.network.helpers`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.exceptions import DataError, ShapeMismatchError
from geopulse.network.base import Branch, ConductorNetwork, Node
from geopulse.network.helpers import (
    add_tie,
    apply_resistive_blocker,
    evaluate_field_at_branch_midpoints,
    open_line,
)

# ---------------------------------------------------------------------------
# apply_resistive_blocker
# ---------------------------------------------------------------------------


class TestApplyResistiveBlocker:
    def test_single_node_override(self):
        z = np.diag([1.0, 2.0, 3.0])
        out = apply_resistive_blocker(z, [1], r_blocker_Ohm=1e12)
        assert out[1, 1] == pytest.approx(1e12)
        # Others untouched.
        assert out[0, 0] == 1.0
        assert out[2, 2] == 3.0

    def test_multi_node_and_dedup(self):
        z = np.diag([1.0, 2.0, 3.0, 4.0])
        out = apply_resistive_blocker(z, [0, 2, 2, 0], r_blocker_Ohm=50.0)
        assert out[0, 0] == 50.0
        assert out[2, 2] == 50.0
        assert out[1, 1] == 2.0
        assert out[3, 3] == 4.0

    def test_returns_copy_not_view(self):
        z = np.diag([1.0, 2.0])
        out = apply_resistive_blocker(z, [0], r_blocker_Ohm=99.0)
        # Mutating output must not affect input.
        out[1, 1] = 999.0
        assert z[1, 1] == 2.0

    def test_rejects_non_square(self):
        with pytest.raises(ShapeMismatchError, match="square"):
            apply_resistive_blocker(np.zeros((3, 4)), [0], 100.0)

    def test_rejects_bad_resistance(self):
        z = np.diag([1.0])
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with pytest.raises(DataError, match="r_blocker_Ohm"):
                apply_resistive_blocker(z, [0], bad)

    def test_rejects_out_of_range_index(self):
        z = np.diag([1.0, 2.0, 3.0])
        with pytest.raises(DataError, match="out of range"):
            apply_resistive_blocker(z, [3], 100.0)
        with pytest.raises(DataError, match="out of range"):
            apply_resistive_blocker(z, [-1], 100.0)


# ---------------------------------------------------------------------------
# open_line
# ---------------------------------------------------------------------------


class TestOpenLine:
    def _line_admittance(self, g: float) -> np.ndarray:
        # Standard 3-node admittance for a single line 0-1 of conductance g.
        return np.array([[g, -g, 0.0], [-g, g, 0.0], [0.0, 0.0, 0.0]])

    def test_removes_line_cleanly(self):
        Y = self._line_admittance(2.0)
        out = open_line(Y, 0, 1, line_conductance_S=2.0)
        assert np.allclose(out, np.zeros((3, 3)))

    def test_composed_network_untouched_elsewhere(self):
        Y = np.array(
            [
                [3.0, -2.0, -1.0],
                [-2.0, 3.0, -1.0],
                [-1.0, -1.0, 2.0],
            ]
        )
        out = open_line(Y, 0, 1, line_conductance_S=2.0)
        # Only the (0,1)/(1,0) off-diag and diag contributions should change.
        assert out[0, 2] == -1.0
        assert out[1, 2] == -1.0
        assert out[2, 2] == 2.0
        assert out[0, 0] == pytest.approx(1.0)
        assert out[1, 1] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(0.0)
        assert out[1, 0] == pytest.approx(0.0)

    def test_returns_copy(self):
        Y = self._line_admittance(2.0)
        out = open_line(Y, 0, 1, line_conductance_S=2.0)
        out[0, 0] = 99.0
        assert Y[0, 0] == 2.0

    def test_rejects_same_endpoints(self):
        Y = self._line_admittance(1.0)
        with pytest.raises(DataError, match="differ"):
            open_line(Y, 1, 1, 1.0)

    def test_rejects_out_of_range(self):
        Y = self._line_admittance(1.0)
        with pytest.raises(DataError, match="out of range"):
            open_line(Y, 0, 5, 1.0)

    def test_rejects_bad_conductance(self):
        Y = self._line_admittance(1.0)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(DataError, match="line_conductance_S"):
                open_line(Y, 0, 1, bad)

    def test_rejects_non_square(self):
        with pytest.raises(ShapeMismatchError, match="square"):
            open_line(np.zeros((2, 3)), 0, 1, 1.0)


# ---------------------------------------------------------------------------
# add_tie
# ---------------------------------------------------------------------------


class TestAddTie:
    def test_adds_conductance_correctly(self):
        Y = np.zeros((3, 3))
        out = add_tie(Y, 0, 2, resistance_Ohm=0.5)
        # g = 1/0.5 = 2
        expected = np.array([[2.0, 0.0, -2.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 2.0]])
        assert np.allclose(out, expected)

    def test_add_then_open_roundtrip_is_identity(self):
        Y = np.diag([1.0, 2.0, 3.0])
        R = 0.25
        Y2 = add_tie(Y, 0, 2, resistance_Ohm=R)
        Y3 = open_line(Y2, 0, 2, line_conductance_S=1.0 / R)
        assert np.allclose(Y3, Y)

    def test_rejects_bad_resistance(self):
        Y = np.zeros((2, 2))
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(DataError, match="resistance_Ohm"):
                add_tie(Y, 0, 1, bad)


# ---------------------------------------------------------------------------
# evaluate_field_at_branch_midpoints
# ---------------------------------------------------------------------------


class _StubNet(ConductorNetwork):
    """Minimal 3-node line network used for field-sampling tests."""

    def __init__(self, nodes: list[Node], branches: list[Branch]) -> None:
        self._nodes = nodes
        self._branches = branches

    def get_nodes(self):
        return self._nodes

    def get_branches(self):
        return self._branches

    def assemble_network_admittance(self) -> np.ndarray:
        raise NotImplementedError

    def assemble_earthing_impedance(self) -> np.ndarray:
        raise NotImplementedError

    def compute_thevenin_voltages(self, ex_Vm, ey_Vm, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str):
        raise NotImplementedError


@pytest.fixture
def three_node_east_west_net():
    # Three nodes spaced ~50 km east-west at latitude 40 N.
    nodes = [
        Node("A", latitude_deg=40.0, longitude_deg=-90.0, earthing_impedance_Ohm=10.0),
        Node("B", latitude_deg=40.0, longitude_deg=-89.5, earthing_impedance_Ohm=10.0),
        Node("C", latitude_deg=40.0, longitude_deg=-89.0, earthing_impedance_Ohm=10.0),
    ]
    branches = [
        Branch("AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=1.0),
        Branch("BC", from_node="B", to_node="C", resistance_Ohm=1.0, length_m=1.0),
    ]
    return _StubNet(nodes, branches)


class TestEvaluateFieldAtBranchMidpoints:
    def test_uniform_field_returns_constant(self, three_node_east_west_net):
        uniform = lambda x, y: (1e-3, 2e-3)  # noqa: E731
        ex, ey = evaluate_field_at_branch_midpoints(three_node_east_west_net, uniform)
        assert ex.shape == (2,)
        assert ey.shape == (2,)
        assert np.allclose(ex, 1e-3)
        assert np.allclose(ey, 2e-3)

    def test_gradient_field_varies_across_branches(self, three_node_east_west_net):
        # Ex = 1e-3 * x_km (positive east). Branches AB and BC have different
        # midpoint x_km so should produce different Ex values.
        grad = lambda x, y: (1e-3 * x, 0.0)  # noqa: E731
        ex, _ = evaluate_field_at_branch_midpoints(three_node_east_west_net, grad)
        assert ex[0] != ex[1]

    def test_shape_matches_branch_count(self, three_node_east_west_net):
        ex, ey = evaluate_field_at_branch_midpoints(
            three_node_east_west_net, lambda x, y: (0.0, 0.0)
        )
        assert ex.shape == (2,)
        assert ey.shape == (2,)

    def test_rejects_non_callable(self, three_node_east_west_net):
        with pytest.raises(DataError, match="callable"):
            evaluate_field_at_branch_midpoints(three_node_east_west_net, 42.0)

    def test_rejects_empty_branches(self):
        nodes = [Node("A", 0.0, 0.0), Node("B", 0.0, 0.5)]
        net = _StubNet(nodes, [])
        with pytest.raises(DataError, match="zero branches"):
            evaluate_field_at_branch_midpoints(net, lambda x, y: (0.0, 0.0))

    def test_nan_coord_node_does_not_poison_finite_branches(self):
        # Regression for issue #23: a NaN-coord node elsewhere in the
        # network must NOT poison the local-projection origin, so a
        # branch between two finite nodes still gets sampled correctly.
        nodes = [
            Node("A", latitude_deg=40.0, longitude_deg=-90.0),
            Node("B", latitude_deg=40.0, longitude_deg=-89.5),
            Node("C", latitude_deg=float("nan"), longitude_deg=float("nan")),  # NaN elsewhere
        ]
        branches = [
            Branch("AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=1.0),
        ]
        net = _StubNet(nodes, branches)
        ex, ey = evaluate_field_at_branch_midpoints(net, lambda x, y: (1e-3, 2e-3))
        assert np.isfinite(ex).all()
        assert np.isfinite(ey).all()
        assert ex[0] == pytest.approx(1e-3)
        assert ey[0] == pytest.approx(2e-3)

    def test_branch_touching_nan_endpoint_returns_zero(self):
        # A branch whose endpoint has NaN coords is a zero-length
        # degenerate (co-located with parent bus); the right physics
        # is zero induced voltage, so the sample must be (0, 0) and
        # never NaN.
        nodes = [
            Node("A", latitude_deg=40.0, longitude_deg=-90.0),
            Node("B", latitude_deg=40.0, longitude_deg=-89.5),
            Node("C", latitude_deg=float("nan"), longitude_deg=float("nan")),
        ]
        branches = [
            Branch("AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=1.0),
            Branch("BC", from_node="B", to_node="C", resistance_Ohm=1.0, length_m=1.0),
        ]
        net = _StubNet(nodes, branches)
        ex, ey = evaluate_field_at_branch_midpoints(net, lambda x, y: (5.0, 5.0))
        assert np.isfinite(ex).all()
        assert np.isfinite(ey).all()
        # Finite-endpoint branch samples the field; NaN-endpoint branch is 0.
        assert ex[0] == pytest.approx(5.0)
        assert ex[1] == 0.0
        assert ey[1] == 0.0

    def test_all_nan_network_raises(self):
        # If every node has NaN coords the origin cannot be built.
        nodes = [
            Node("A", latitude_deg=float("nan"), longitude_deg=float("nan")),
            Node("B", latitude_deg=float("nan"), longitude_deg=float("nan")),
        ]
        branches = [Branch("AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=1.0)]
        net = _StubNet(nodes, branches)
        with pytest.raises(DataError, match="finite"):
            evaluate_field_at_branch_midpoints(net, lambda x, y: (0.0, 0.0))
