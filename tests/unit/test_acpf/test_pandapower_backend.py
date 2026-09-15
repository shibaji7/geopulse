"""Unit tests for :mod:`geopulse.acpf.pandapower_backend`."""

from __future__ import annotations

import pytest

pytest.importorskip("pandapower")

import pandapower as pp

from geopulse.acpf.base import ACPFResult
from geopulse.acpf.pandapower_backend import PandapowerBackend
from geopulse.exceptions import DataError, NotImplementedYetError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_bus_net():
    """A minimal 2-bus pandapower net: slack + load bus over a 10 km line."""
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=138.0, name="slack")
    b2 = pp.create_bus(net, vn_kv=138.0, name="load_bus")
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.4,
        c_nf_per_km=10.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, b2, p_mw=10.0, q_mvar=2.0)
    return net


# ---------------------------------------------------------------------------
# Construction / lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_can_be_instantiated_without_build(self):
        b = PandapowerBackend()
        assert b.net is None

    def test_solve_before_build_raises(self):
        b = PandapowerBackend()
        with pytest.raises(DataError, match="build.*must be called"):
            b.solve()

    def test_inject_before_build_raises(self):
        b = PandapowerBackend()
        with pytest.raises(DataError, match="build.*must be called"):
            b.inject_reactive({"a": 1.0})

    def test_missing_pandapower_raises_with_install_hint(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandapower" or name.startswith("pandapower."):
                raise ImportError(f"mocked absence of {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DataError, match=r"pip install geopulse\[acpf\]"):
            PandapowerBackend()


class TestBuild:
    def test_from_pandapower_net(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        assert b.net is not None
        assert len(b.net.bus) == 2

    def test_deep_copies_caller_net(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        # Mutate the backend's net; caller's copy must stay intact.
        b.net.bus.loc[0, "vn_kv"] = 999.0
        assert two_bus_net.bus.loc[0, "vn_kv"] == 138.0

    def test_rejects_unknown_ac_case_type(self):
        b = PandapowerBackend()
        with pytest.raises(DataError, match="MATPOWER .m file path or a pandapower net"):
            b.build(network=None, ac_case=42)

    def test_from_matpower_path_delegates_to_pp_converter(self, monkeypatch):
        # The MATPOWER-path branch of build() delegates to
        # pandapower.converter.matpower.from_mpc. Monkeypatch that
        # module-level import site so we don't need a real .m file
        # loadable by pandapower's converter (which has its own
        # per-version quirks).
        sentinel_net = pp.create_empty_network()
        pp.create_bus(sentinel_net, vn_kv=138.0, name="sentinel")

        import pandapower.converter.matpower as mpc_mod

        def fake_from_mpc(path):
            assert path == "some/path.m"
            return sentinel_net

        monkeypatch.setattr(mpc_mod, "from_mpc", fake_from_mpc)
        b = PandapowerBackend()
        b.build(network=None, ac_case="some/path.m")
        assert b.net is not None
        assert "sentinel" in list(b.net.bus["name"])

    def test_matpower_path_wraps_converter_exception(self, monkeypatch):
        # A malformed / unreadable file raises DataError with a helpful
        # message rather than passing pandapower's raw exception through.
        import pandapower.converter.matpower as mpc_mod

        def boom(path):
            raise ValueError("mocked parse error")

        monkeypatch.setattr(mpc_mod, "from_mpc", boom)
        b = PandapowerBackend()
        with pytest.raises(DataError, match="MATPOWER case"):
            b.build(network=None, ac_case="broken.m")


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


class TestSolve:
    def test_baseline_converges_and_reports_v(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        r = b.solve()
        assert isinstance(r, ACPFResult)
        assert r.converged is True
        assert "load_bus" in r.v_bus_pu
        # 10 MW / 2 MVAr load over a short line: V drop ~0.001 pu.
        assert 0.99 < r.v_bus_pu["load_bus"] < 1.0

    def test_result_has_backend_metadata(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        r = b.solve()
        assert r.metadata.get("backend") == "pandapower"

    def test_extreme_load_returns_non_converged_result_not_exception(self, two_bus_net):
        # Spec §7 / §8 item 6: collapse is a result, not a crash.
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        b.inject_reactive({"load_bus": 50_000.0})  # unphysically large
        r = b.solve()
        assert r.converged is False
        assert r.v_bus_pu == {}
        assert r.metadata["reason"] == "LoadflowNotConverged"


# ---------------------------------------------------------------------------
# Inject reactive absorption
# ---------------------------------------------------------------------------


class TestInjectReactive:
    def test_injection_lowers_bus_voltage(self, two_bus_net):
        # Spec §8 item 4 sanity: increasing ΔQ decreases V.
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        r0 = b.solve()
        b.inject_reactive({"load_bus": 5.0})
        r1 = b.solve()
        assert r1.converged is True
        assert r1.v_bus_pu["load_bus"] < r0.v_bus_pu["load_bus"]

    def test_injection_is_idempotent(self, two_bus_net):
        # Successive inject_reactive() calls replace, not accumulate.
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        b.inject_reactive({"load_bus": 5.0})
        v_once = b.solve().v_bus_pu["load_bus"]
        b.inject_reactive({"load_bus": 5.0})
        v_twice = b.solve().v_bus_pu["load_bus"]
        assert v_once == pytest.approx(v_twice, rel=1e-9)

    def test_replacing_injection_with_smaller_dq_restores_voltage(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        b.inject_reactive({"load_bus": 5.0})
        v_5 = b.solve().v_bus_pu["load_bus"]
        b.inject_reactive({"load_bus": 1.0})
        v_1 = b.solve().v_bus_pu["load_bus"]
        assert v_1 > v_5

    def test_zero_delta_q_reproduces_base_case(self, two_bus_net):
        # Spec §8 item 3: ΔQ = 0 should reproduce the base case exactly.
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        r0 = b.solve()
        b.inject_reactive({"load_bus": 0.0})
        r_zero = b.solve()
        assert r_zero.v_bus_pu["load_bus"] == pytest.approx(
            r0.v_bus_pu["load_bus"],
            rel=1e-9,
        )

    def test_unknown_bus_raises(self, two_bus_net):
        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        with pytest.raises(DataError, match="not found in pandapower net"):
            b.inject_reactive({"no_such_bus": 1.0})


# ---------------------------------------------------------------------------
# Continuation power flow — stubbed for now
# ---------------------------------------------------------------------------


class TestContinuation:
    def test_is_stubbed_with_wp_reference(self, two_bus_net):
        from geopulse.acpf.base import LoadDirection

        b = PandapowerBackend()
        b.build(network=None, ac_case=two_bus_net)
        with pytest.raises(NotImplementedYetError, match="acpf-continuation-pf"):
            b.continuation(LoadDirection())
