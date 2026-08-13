"""Unit tests for :mod:`geopulse.viz.network_schematic`."""

from __future__ import annotations

import pytest

pytest.importorskip("schemdraw")

from geopulse.exceptions import DataError
from geopulse.viz.network_schematic import plot_transformer_dc_equivalent


class TestPlotTransformerDcEquivalent:
    @pytest.mark.parametrize("config", ["gywe-delta", "gywe-gywe", "gywe-gywe-auto"])
    def test_renders_without_error(self, config, tmp_path):
        # Smoke test: the three recognised winding configs each produce a
        # figure and can round-trip through savefig without raising.
        out = tmp_path / f"xf_{config}.png"
        fig = plot_transformer_dc_equivalent(config, savepath=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
        assert fig is not None

    def test_rejects_unknown_config(self):
        with pytest.raises(DataError, match="config must be one of"):
            plot_transformer_dc_equivalent("not-a-config")  # type: ignore[arg-type]

    def test_arrows_toggle_produces_different_output(self, tmp_path):
        # Compositional sanity: turning arrows off changes the figure
        # (different byte count is a weak but sufficient signal).
        with_arrows = tmp_path / "with.png"
        no_arrows = tmp_path / "without.png"
        plot_transformer_dc_equivalent("gywe-delta", savepath=str(with_arrows))
        plot_transformer_dc_equivalent(
            "gywe-delta", savepath=str(no_arrows), show_current_arrows=False
        )
        assert with_arrows.stat().st_size != no_arrows.stat().st_size

    def test_missing_schemdraw_raises_with_install_hint(self, monkeypatch):
        # Simulate schemdraw being uninstalled — the DataError should point
        # the user at the [viz] extra rather than the raw ImportError.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("schemdraw"):
                raise ImportError(f"mocked absence of {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DataError, match=r"pip install geopulse\[viz\]"):
            plot_transformer_dc_equivalent("gywe-delta")


class TestPlotSubstationDcEquivalent:
    def test_renders_sub6_from_horton(self, tmp_path):
        from geopulse.network.powergrid import PowerGridNetwork
        from geopulse.solver.nam import NAMSolver
        from geopulse.viz.network_schematic import plot_substation_dc_equivalent

        net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
        Y = net.assemble_network_admittance()
        Z = net.assemble_earthing_impedance()
        V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
        r = NAMSolver().solve(net, Y, Z, V_th)

        out = tmp_path / "sub6.png"
        fig = plot_substation_dc_equivalent(
            net,
            "dc_sub6",
            node_voltages_V=r.node_voltages_V,
            field_source_ex_Vm=1e-3,
            savepath=str(out),
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert fig is not None

    def test_rejects_unknown_sub(self):
        from geopulse.network.powergrid import PowerGridNetwork
        from geopulse.viz.network_schematic import plot_substation_dc_equivalent

        net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
        with pytest.raises(DataError, match="not found in network"):
            plot_substation_dc_equivalent(net, "dc_sub42")

    def test_rejects_isolated_sub(self):
        # dc_sub1 has no transformer branches — rendering it as a circuit
        # would be a floating icon; the function raises with an explanation.
        from geopulse.network.powergrid import PowerGridNetwork
        from geopulse.viz.network_schematic import plot_substation_dc_equivalent

        net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
        with pytest.raises(DataError, match="no transformer branches"):
            plot_substation_dc_equivalent(net, "dc_sub1")


class TestPlotNetworkDcEquivalent:
    def test_renders_horton_end_to_end(self, tmp_path):
        from geopulse.network.powergrid import PowerGridNetwork
        from geopulse.solver.nam import NAMSolver
        from geopulse.viz.network_schematic import plot_network_dc_equivalent

        net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
        Y = net.assemble_network_admittance()
        Z = net.assemble_earthing_impedance()
        V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
        r = NAMSolver().solve(net, Y, Z, V_th)

        out = tmp_path / "network.png"
        fig = plot_network_dc_equivalent(
            net,
            node_voltages_V=r.node_voltages_V,
            savepath=str(out),
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert fig is not None

    def test_works_without_solve_result(self, tmp_path):
        # No node_voltages_V → no live GIC labels, but figure still renders.
        from geopulse.network.powergrid import PowerGridNetwork
        from geopulse.viz.network_schematic import plot_network_dc_equivalent

        net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
        out = tmp_path / "network_no_gic.png"
        fig = plot_network_dc_equivalent(net, savepath=str(out))
        assert out.exists()
        assert fig is not None
