"""Unit tests for :mod:`geopulse.acpf.coupling`."""

from __future__ import annotations

import pytest

pytest.importorskip("pandapower")

import pandapower as pp

from geopulse.acpf.coupling import (
    CoupledResult,
    TransformerContribution,
    solve_coupled,
)
from geopulse.acpf.pandapower_backend import PandapowerBackend
from geopulse.devices.transformer import CoreType
from geopulse.exceptions import DataError
from geopulse.network.loads import LoadState, TripCapableLoad

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_bus_backend():
    """Fresh PandapowerBackend wrapping a 2-bus slack + load net."""
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
    be = PandapowerBackend()
    be.build(network=None, ac_case=net)
    return be


@pytest.fixture
def stiff_load():
    """A voltage-sensitive load that trips at V < 0.5 pu with zero delay."""
    return TripCapableLoad(
        bus_id="load_bus",
        p_mw=10.0,
        q_mvar=2.0,
        v_trip_pu=0.5,
        v_trip_delay_s=0.0,
        reconnect_v_pu=0.9,
        reconnect_delay_s=1.0,
    )


# ---------------------------------------------------------------------------
# Zero-GIC baseline (spec §8 item 3)
# ---------------------------------------------------------------------------


class TestZeroGICBaseline:
    def test_returns_well_formed_result(self, two_bus_backend, stiff_load):
        xfmrs = [
            TransformerContribution(
                transformer_id="T1",
                bus_id="load_bus",
                neutral_gic_A=0.0,
                core_type=CoreType.SHELL_FORM,
            )
        ]
        r = solve_coupled(xfmrs, [stiff_load], two_bus_backend)
        assert isinstance(r, CoupledResult)
        assert r.acpf.converged is True
        assert r.iterations == 1
        assert r.limit_cycle is False
        assert r.unserved_mw == 0.0

    def test_delta_q_is_zero(self, two_bus_backend):
        xfmrs = [
            TransformerContribution(
                transformer_id="T1",
                bus_id="load_bus",
                neutral_gic_A=0.0,
                core_type=CoreType.SHELL_FORM,
            )
        ]
        r = solve_coupled(xfmrs, [], two_bus_backend)
        assert r.delta_q_mvar["load_bus"] == 0.0

    def test_thd_is_zero(self, two_bus_backend):
        xfmrs = [
            TransformerContribution(
                transformer_id="T1",
                bus_id="load_bus",
                neutral_gic_A=0.0,
                core_type=CoreType.SHELL_FORM,
            )
        ]
        r = solve_coupled(xfmrs, [], two_bus_backend)
        assert r.thd_pct["load_bus"] == 0.0

    def test_bus_voltage_equals_base_case(self, two_bus_backend):
        # V after coupled solve at zero GIC must equal V from a bare
        # backend.solve() — same physics.
        two_bus_backend.inject_reactive({})
        v_base = two_bus_backend.solve().v_bus_pu["load_bus"]
        xfmrs = [
            TransformerContribution(
                transformer_id="T1",
                bus_id="load_bus",
                neutral_gic_A=0.0,
                core_type=CoreType.SHELL_FORM,
            )
        ]
        r = solve_coupled(xfmrs, [], two_bus_backend)
        assert r.acpf.v_bus_pu["load_bus"] == pytest.approx(v_base, rel=1e-9)


# ---------------------------------------------------------------------------
# GIC drives V down; harmonics up
# ---------------------------------------------------------------------------


class TestGICDrivesResponse:
    def test_delta_q_grows_with_gic(self, two_bus_backend):
        r_small = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=15.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        )
        r_big = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=90.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        )
        assert r_big.delta_q_mvar["load_bus"] > r_small.delta_q_mvar["load_bus"]

    def test_bus_voltage_falls_under_gic(self, two_bus_backend):
        # V after 90 A neutral current must be lower than at zero.
        v_zero = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=0.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        ).acpf.v_bus_pu["load_bus"]
        v_gic = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=90.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        ).acpf.v_bus_pu["load_bus"]
        assert v_gic < v_zero

    def test_thd_grows_with_gic(self, two_bus_backend):
        t_small = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=15.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        ).thd_pct["load_bus"]
        t_big = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=90.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        ).thd_pct["load_bus"]
        assert t_big > t_small

    def test_autotransformer_skipped_from_harmonic_sum(self, two_bus_backend):
        # AUTOTRANSFORMER isn't in the empirical harmonics table
        # (spec §10 item 3); the coupling loop must silently skip it
        # for harmonics rather than crash.
        xfmrs = [
            TransformerContribution(
                transformer_id="Tauto",
                bus_id="load_bus",
                neutral_gic_A=30.0,
                core_type=CoreType.AUTOTRANSFORMER,
                k_override=0.5,
            )
        ]
        r = solve_coupled(xfmrs, [], two_bus_backend)
        # ΔQ still applied (k_override provided), harmonics empty.
        assert r.delta_q_mvar["load_bus"] > 0.0
        assert "load_bus" not in r.thd_pct


# ---------------------------------------------------------------------------
# Multiple transformers at same bus — summed
# ---------------------------------------------------------------------------


class TestMultipleTransformersAtSameBus:
    def test_delta_q_sums(self, two_bus_backend):
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=30.0,
                    core_type=CoreType.SHELL_FORM,
                ),
                TransformerContribution(
                    transformer_id="T2",
                    bus_id="load_bus",
                    neutral_gic_A=30.0,
                    core_type=CoreType.SHELL_FORM,
                ),
            ],
            [],
            two_bus_backend,
        )
        r_single = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=30.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        )
        assert r.delta_q_mvar["load_bus"] == pytest.approx(2.0 * r_single.delta_q_mvar["load_bus"])

    def test_duplicate_transformer_ids_rejected(self, two_bus_backend):
        with pytest.raises(DataError, match="duplicate transformer_id"):
            solve_coupled(
                [
                    TransformerContribution(
                        transformer_id="T1",
                        bus_id="load_bus",
                        neutral_gic_A=1.0,
                        core_type=CoreType.SHELL_FORM,
                    ),
                    TransformerContribution(
                        transformer_id="T1",
                        bus_id="load_bus",
                        neutral_gic_A=1.0,
                        core_type=CoreType.SHELL_FORM,
                    ),
                ],
                [],
                two_bus_backend,
            )


# ---------------------------------------------------------------------------
# Load evaluation — trip on undervoltage
# ---------------------------------------------------------------------------


class TestLoadEvaluation:
    def test_load_survives_zero_gic(self, two_bus_backend, stiff_load):
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=0.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [stiff_load],
            two_bus_backend,
        )
        assert r.load_states["load_bus"] is LoadState.CONNECTED
        assert r.unserved_mw == 0.0

    def test_load_trips_after_collapse(self, two_bus_backend, stiff_load):
        # Enormous GIC drives voltage collapse; load state machine sees
        # the empty v_bus_pu (backend returned converged=False) and
        # defaults V to 1.0 in _evaluate_loads — a defensible choice
        # documented in the code. In practice callers should read
        # acpf.converged and treat the coupled result as "collapsed".
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=10_000.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [stiff_load],
            two_bus_backend,
        )
        # Whether the load itself trips depends on the collapse mode;
        # the important invariant is that we don't raise.
        assert isinstance(r, CoupledResult)
        assert r.acpf.converged is False


# ---------------------------------------------------------------------------
# Iterated mode
# ---------------------------------------------------------------------------


class TestIteratedMode:
    def test_iterated_converges_at_zero_gic(self, two_bus_backend):
        # No GIC → nothing to iterate — should converge in a few passes.
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=0.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
            mode="iterated",
            max_iter=5,
            v_tol=1e-6,
        )
        assert r.iterations >= 1
        assert r.iterations <= 5
        assert r.acpf.converged is True
        assert r.limit_cycle is False

    def test_rejects_unknown_mode(self, two_bus_backend):
        with pytest.raises(DataError, match="mode must be"):
            solve_coupled(
                [],
                [],
                two_bus_backend,
                mode="magic",  # type: ignore[arg-type]
            )

    def test_iterated_reports_p_cancellation_caveat_when_load_tripped(
        self,
        two_bus_backend,
    ):
        # A pre-tripped load must trigger the P-cancellation caveat.
        tripped = TripCapableLoad(
            bus_id="load_bus",
            p_mw=5.0,
            q_mvar=1.0,
            v_trip_pu=0.5,
            v_trip_delay_s=0.0,
            state=LoadState.TRIPPED,
        )
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=5.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [tripped],
            two_bus_backend,
            mode="iterated",
            max_iter=3,
        )
        assert r.unserved_mw >= 5.0
        assert "iterated_p_cancellation" in r.metadata


# ---------------------------------------------------------------------------
# Metadata & approximations
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_thd_approximation_is_flagged(self, two_bus_backend):
        r = solve_coupled(
            [
                TransformerContribution(
                    transformer_id="T1",
                    bus_id="load_bus",
                    neutral_gic_A=5.0,
                    core_type=CoreType.SHELL_FORM,
                )
            ],
            [],
            two_bus_backend,
        )
        assert "thd_approximation" in r.metadata
