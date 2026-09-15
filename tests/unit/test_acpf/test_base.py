"""Unit tests for :mod:`geopulse.acpf.base` — ABC and result dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.acpf.base import ACPFBackend, ACPFResult, LoadDirection, PVCurve


class TestACPFBackendABC:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            ACPFBackend()  # type: ignore[abstract]

    def test_missing_method_still_abstract(self):
        # A partial subclass is still abstract until every method is provided.
        class Partial(ACPFBackend):
            def build(self, network, ac_case): ...

            # missing inject_reactive, solve, continuation

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        class Complete(ACPFBackend):
            def build(self, network, ac_case): ...
            def inject_reactive(self, delta_q): ...
            def solve(self, init="auto"):
                return ACPFResult(
                    converged=True,
                    v_bus_pu={},
                    v_bus_angle_deg={},
                    p_bus_mw={},
                    q_bus_mvar={},
                )

            def continuation(self, direction):
                return PVCurve(
                    lambda_values=np.array([0.0]),
                    v_bus_pu={},
                    lambda_max=0.0,
                    converged=True,
                )

        b = Complete()
        assert b.solve().converged is True


class TestACPFResult:
    def test_default_metadata_is_empty_dict(self):
        r = ACPFResult(
            converged=True,
            v_bus_pu={"a": 1.0},
            v_bus_angle_deg={"a": 0.0},
            p_bus_mw={"a": 0.0},
            q_bus_mvar={"a": 0.0},
        )
        assert r.metadata == {}

    def test_frozen_disallows_mutation(self):
        r = ACPFResult(
            converged=True,
            v_bus_pu={},
            v_bus_angle_deg={},
            p_bus_mw={},
            q_bus_mvar={},
        )
        with pytest.raises((AttributeError, Exception)):
            r.converged = False  # type: ignore[misc]

    def test_non_converged_can_have_empty_bus_dicts(self):
        # Non-convergence is a physical result — the empty-per-bus form is
        # valid (spec §7).
        r = ACPFResult(
            converged=False,
            v_bus_pu={},
            v_bus_angle_deg={},
            p_bus_mw={},
            q_bus_mvar={},
            metadata={"reason": "LoadflowNotConverged"},
        )
        assert r.converged is False
        assert r.v_bus_pu == {}


class TestLoadDirection:
    def test_defaults_to_empty_maps(self):
        d = LoadDirection()
        assert d.p_direction_mw == {}
        assert d.q_direction_mvar == {}

    def test_carries_partial_direction(self):
        d = LoadDirection(p_direction_mw={"a": 1.0})
        assert d.p_direction_mw == {"a": 1.0}
        assert d.q_direction_mvar == {}


class TestPVCurve:
    def test_holds_a_trace(self):
        pv = PVCurve(
            lambda_values=np.linspace(0.0, 1.5, 10),
            v_bus_pu={"a": np.linspace(1.0, 0.9, 10)},
            lambda_max=1.5,
            converged=True,
        )
        assert pv.lambda_values.shape == (10,)
        assert pv.v_bus_pu["a"].shape == (10,)
        assert pv.lambda_max == 1.5
