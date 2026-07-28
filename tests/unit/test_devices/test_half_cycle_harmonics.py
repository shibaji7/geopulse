"""Stub-level tests for :mod:`geopulse.devices.half_cycle_harmonics`.

These tests confirm the module presents its API and refuses to compute
until the empirical curves are provided. When a contributor lands the
Walling & Khan / Girgis & Vedante coefficients (see
``benchmarks/walling_khan_1991/README.md``), replace the "raises"
assertions with numerical checks against paper spot values.
"""

from __future__ import annotations

import pytest

from geopulse.devices.half_cycle_harmonics import half_cycle_harmonics
from geopulse.exceptions import NotImplementedYetError


class TestHalfCycleHarmonicsStub:
    def test_raises_notimplemented_with_helpful_message(self):
        with pytest.raises(NotImplementedYetError) as excinfo:
            half_cycle_harmonics(50.0)
        msg = str(excinfo.value)
        # The error message must name the paper needed and point at the
        # placeholder benchmark directory so a future contributor knows
        # exactly what to provide.
        assert "Walling" in msg or "walling_khan" in msg
        assert "benchmarks/walling_khan_1991" in msg

    def test_raises_regardless_of_kwargs(self):
        # Every combination of documented arguments should still raise
        # until the curves are transcribed — the stub reserves the API
        # surface, it does not partially implement.
        for kw in (
            {},
            {"fundamental_Hz": 50.0},
            {"n_orders": 5},
            {"transformer_model": "girgis_vedante"},
            {"rated_exciting_current_A": 12.5},
        ):
            with pytest.raises(NotImplementedYetError):
                half_cycle_harmonics(50.0, **kw)

    def test_public_api_surface(self):
        # The presence of the function + its inclusion in the module's
        # __all__ is the API-reservation contract that other code plans
        # against.
        from geopulse.devices import half_cycle_harmonics as mod

        assert "half_cycle_harmonics" in mod.__all__
        assert callable(mod.half_cycle_harmonics)
