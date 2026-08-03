"""Unit tests for :mod:`geopulse.viz.network_map`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — force headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from geopulse.exceptions import DataError, ShapeMismatchError  # noqa: E402
from geopulse.network.base import Branch, ConductorNetwork, Node  # noqa: E402
from geopulse.viz.network_map import plot_network_map  # noqa: E402


class _StubNet(ConductorNetwork):
    """Minimal 4-node ring network used across the tests."""

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
def ring_net():
    nodes = [
        Node("A", latitude_deg=40.0, longitude_deg=-90.0, earthing_impedance_Ohm=10.0),
        Node("B", latitude_deg=40.5, longitude_deg=-89.5, earthing_impedance_Ohm=10.0),
        Node("C", latitude_deg=40.0, longitude_deg=-89.0, earthing_impedance_Ohm=10.0),
        Node("D", latitude_deg=39.5, longitude_deg=-89.5, earthing_impedance_Ohm=10.0),
    ]
    branches = [
        Branch("AB", from_node="A", to_node="B", resistance_Ohm=1.0, length_m=1.0),
        Branch("BC", from_node="B", to_node="C", resistance_Ohm=1.0, length_m=1.0),
        Branch("CD", from_node="C", to_node="D", resistance_Ohm=1.0, length_m=1.0),
        Branch("DA", from_node="D", to_node="A", resistance_Ohm=1.0, length_m=1.0),
    ]
    return _StubNet(nodes, branches)


class TestBasicPlotting:
    def test_no_values_topology_only(self, ring_net):
        fig = plot_network_map(ring_net)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # 4 line segments + 1 scatter collection.
        assert len(ax.lines) == 4
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_axes_labels_and_aspect(self, ring_net):
        fig = plot_network_map(ring_net, title="ring network")
        ax = fig.axes[0]
        assert ax.get_title() == "ring network"
        assert ax.get_xlabel() == "Longitude (°E)"
        assert ax.get_ylabel() == "Latitude (°N)"
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_returns_figure(self, ring_net):
        from matplotlib.figure import Figure

        fig = plot_network_map(ring_net)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestNodeValues:
    def test_array_input_adds_colorbar(self, ring_net):
        vals = np.array([1.0, 5.0, 2.0, 8.0])
        fig = plot_network_map(ring_net, vals)
        # Colour-bar creates a second axes on the figure.
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_dict_input_maps_by_node_id(self, ring_net):
        vals = {"A": 10.0, "C": 20.0}  # B and D missing → default 0
        fig = plot_network_map(ring_net, vals)
        # Should not raise; colour-bar should still be added.
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_wrong_length_array_raises(self, ring_net):
        with pytest.raises(ShapeMismatchError, match="does not match n_nodes"):
            plot_network_map(ring_net, np.array([1.0, 2.0]))

    def test_log_scale_requires_non_negative(self, ring_net):
        with pytest.raises(DataError, match="non-negative"):
            plot_network_map(ring_net, np.array([1.0, -1.0, 2.0, 3.0]), log_scale=True)

    def test_log_scale_ok(self, ring_net):
        vals = np.array([0.1, 1.0, 10.0, 100.0])  # 3-decade span
        fig = plot_network_map(ring_net, vals, log_scale=True)
        assert len(fig.axes) == 2  # scatter + colour-bar
        plt.close(fig)


class TestAxInjection:
    def test_reuses_supplied_axes(self, ring_net):
        outer_fig, ax = plt.subplots(figsize=(6, 4))
        fig = plot_network_map(ring_net, ax=ax)
        assert fig is outer_fig
        plt.close(outer_fig)

    def test_compose_two_panels(self, ring_net):
        vals_a = np.array([1.0, 2.0, 3.0, 4.0])
        vals_b = np.array([4.0, 3.0, 2.0, 1.0])
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
        plot_network_map(ring_net, vals_a, ax=ax_l, title="a")
        plot_network_map(ring_net, vals_b, ax=ax_r, title="b")
        assert ax_l.get_title() == "a"
        assert ax_r.get_title() == "b"
        plt.close(fig)


class TestSavepath:
    def test_savepath_writes_file(self, tmp_path, ring_net):
        out = tmp_path / "map.png"
        plot_network_map(ring_net, savepath=out)
        assert out.is_file()
        assert out.stat().st_size > 0
        plt.close("all")


class TestShowLabels:
    def test_labels_off_by_default(self, ring_net):
        fig = plot_network_map(ring_net)
        # No text annotations should be present other than axis text.
        annotations = [t for t in fig.axes[0].texts if t.get_text() in {"A", "B", "C", "D"}]
        assert annotations == []
        plt.close(fig)

    def test_labels_on_when_requested(self, ring_net):
        fig = plot_network_map(ring_net, show_labels=True)
        annotations = {t.get_text() for t in fig.axes[0].texts}
        assert annotations == {"A", "B", "C", "D"}
        plt.close(fig)


class TestErrors:
    def test_empty_network_raises(self):
        net = _StubNet([], [])
        with pytest.raises(DataError, match="zero nodes"):
            plot_network_map(net)


class TestBranchHandling:
    def test_branch_referencing_missing_node_is_skipped(self, ring_net):
        # Add a rogue branch that references a non-existent node "Z".
        bad_branch = Branch("ZZ", from_node="Z", to_node="A", resistance_Ohm=1.0, length_m=1.0)
        ring_net._branches.append(bad_branch)
        fig = plot_network_map(ring_net)
        # Should still succeed; the bad branch is quietly skipped (4 valid
        # branches drawn, not 5).
        assert len(fig.axes[0].lines) == 4
        plt.close(fig)
