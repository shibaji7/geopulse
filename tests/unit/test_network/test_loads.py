"""Unit tests for :mod:`geopulse.network.loads`."""

from __future__ import annotations

import pytest

from geopulse.exceptions import DataError
from geopulse.network.loads import LoadState, TripCapableLoad

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_to_immortal_connected(self):
        # No thresholds configured → load never trips.
        load = TripCapableLoad(bus_id="b", p_mw=10.0, q_mvar=2.0)
        assert load.state is LoadState.CONNECTED
        for _ in range(100):
            load.evaluate(v_pu=0.10, thd_pct=99.0, dt_s=60.0)
        assert load.state is LoadState.CONNECTED

    def test_zip_fractions_must_sum_to_one(self):
        with pytest.raises(DataError, match="sum to 1.0"):
            TripCapableLoad(
                bus_id="b",
                p_mw=1.0,
                q_mvar=0.0,
                constant_power_fraction=0.5,
                constant_current_fraction=0.5,
                constant_impedance_fraction=0.5,
            )

    def test_zip_fractions_must_be_non_negative(self):
        with pytest.raises(DataError, match="non-negative"):
            TripCapableLoad(
                bus_id="b",
                p_mw=1.0,
                q_mvar=0.0,
                constant_power_fraction=1.2,
                constant_current_fraction=-0.2,
                constant_impedance_fraction=0.0,
            )

    def test_threshold_requires_matching_delay(self):
        with pytest.raises(DataError, match="v_trip_pu.*v_trip_delay_s"):
            TripCapableLoad(bus_id="b", p_mw=1.0, q_mvar=0.0, v_trip_pu=0.85)

    def test_delay_requires_matching_threshold(self):
        with pytest.raises(DataError, match="thd_trip_pct.*thd_trip_delay_s"):
            TripCapableLoad(
                bus_id="b",
                p_mw=1.0,
                q_mvar=0.0,
                thd_trip_delay_s=1.0,
            )

    def test_negative_delay_rejected(self):
        with pytest.raises(DataError, match="must be non-negative"):
            TripCapableLoad(
                bus_id="b",
                p_mw=1.0,
                q_mvar=0.0,
                v_trip_pu=0.85,
                v_trip_delay_s=-1.0,
            )


# ---------------------------------------------------------------------------
# ZIP decomposition (spec §6.4, §8 item 8)
# ---------------------------------------------------------------------------


class TestZipDecomposition:
    def test_constant_power_current_rises_as_voltage_falls(self):
        # Spec §8 item 8: constant-P load draws MORE current as V falls.
        load = TripCapableLoad(bus_id="b", p_mw=10.0, q_mvar=0.0)
        i_hi = load.current_magnitude_pu(v_pu=1.0)
        i_lo = load.current_magnitude_pu(v_pu=0.5)
        assert i_lo > i_hi
        assert i_lo == pytest.approx(2.0 * i_hi)

    def test_constant_impedance_current_falls_as_voltage_falls(self):
        # Spec §8 item 8: constant-Z load draws less current at low V.
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=0.0,
            constant_power_fraction=0.0,
            constant_current_fraction=0.0,
            constant_impedance_fraction=1.0,
        )
        i_hi = load.current_magnitude_pu(v_pu=1.0)
        i_lo = load.current_magnitude_pu(v_pu=0.5)
        assert i_lo < i_hi
        assert i_lo == pytest.approx(0.5 * i_hi)

    def test_constant_current_stays_flat(self):
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=0.0,
            constant_power_fraction=0.0,
            constant_current_fraction=1.0,
            constant_impedance_fraction=0.0,
        )
        i_hi = load.current_magnitude_pu(v_pu=1.0)
        i_lo = load.current_magnitude_pu(v_pu=0.5)
        assert i_lo == pytest.approx(i_hi)

    def test_current_draw_zero_when_not_connected(self):
        load = TripCapableLoad(bus_id="b", p_mw=10.0, q_mvar=5.0)
        load.state = LoadState.TRIPPED
        assert load.current_draw(1.0) == (0.0, 0.0)
        assert load.current_magnitude_pu(1.0) == 0.0

    def test_current_magnitude_rejects_nonpositive_voltage(self):
        load = TripCapableLoad(bus_id="b", p_mw=1.0, q_mvar=0.0)
        with pytest.raises(DataError, match="v_pu must be strictly positive"):
            load.current_magnitude_pu(v_pu=0.0)


# ---------------------------------------------------------------------------
# Trip state machine (spec §6.4, §8 item 8)
# ---------------------------------------------------------------------------


class TestVoltageTrip:
    def _make(self, **overrides):
        base = dict(
            bus_id="b",
            p_mw=10.0,
            q_mvar=2.0,
            v_trip_pu=0.85,
            v_trip_delay_s=1.0,
        )
        base.update(overrides)
        return TripCapableLoad(**base)

    def test_trip_requires_threshold_and_delay(self):
        # Spec §8 item 8: trip fires only after BOTH threshold AND delay.
        load = self._make()
        # Below threshold but delay not yet elapsed → still connected.
        assert load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.5) is LoadState.CONNECTED
        # Cumulative time now 1.0 s → trip.
        assert load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.5) is LoadState.TRIPPED

    def test_transient_dip_that_recovers_does_not_trip(self):
        # Voltage drops for 0.5 s (< delay), then recovers → no trip,
        # and the timer resets so a later 0.9 s dip also doesn't trip.
        load = self._make()
        load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.5)
        load.evaluate(v_pu=0.95, thd_pct=0.0, dt_s=0.5)  # recovered
        # Now 0.9 s below threshold — cumulative from this stretch alone
        # is only 0.9 s, so still no trip.
        assert load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.9) is LoadState.CONNECTED

    def test_at_threshold_is_not_below_threshold(self):
        # v_pu == v_trip_pu is NOT strictly below → no accumulation.
        load = self._make()
        for _ in range(10):
            assert load.evaluate(v_pu=0.85, thd_pct=0.0, dt_s=1.0) is LoadState.CONNECTED

    def test_zero_delay_trips_immediately(self):
        load = self._make(v_trip_delay_s=0.0)
        assert load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.0) is LoadState.TRIPPED


class TestHarmonicTrip:
    def test_thd_trip_fires_after_threshold_and_delay(self):
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=2.0,
            thd_trip_pct=8.0,
            thd_trip_delay_s=0.5,
        )
        assert load.evaluate(v_pu=1.0, thd_pct=10.0, dt_s=0.25) is LoadState.CONNECTED
        assert load.evaluate(v_pu=1.0, thd_pct=10.0, dt_s=0.25) is LoadState.TRIPPED

    def test_thd_recovery_resets_timer(self):
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=2.0,
            thd_trip_pct=8.0,
            thd_trip_delay_s=1.0,
        )
        load.evaluate(v_pu=1.0, thd_pct=10.0, dt_s=0.6)
        load.evaluate(v_pu=1.0, thd_pct=5.0, dt_s=0.1)  # clean again
        # Only 0.9 s of THD above threshold in one stretch → still connected.
        assert load.evaluate(v_pu=1.0, thd_pct=10.0, dt_s=0.9) is LoadState.CONNECTED


class TestReconnect:
    def _tripped_load(self):
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=2.0,
            v_trip_pu=0.85,
            v_trip_delay_s=0.0,
            reconnect_v_pu=0.95,
            reconnect_delay_s=2.0,
        )
        load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.0)  # trip
        assert load.state is LoadState.TRIPPED
        return load

    def test_reconnect_requires_sustained_voltage(self):
        load = self._tripped_load()
        # Voltage insufficient → stays tripped forever.
        for _ in range(10):
            load.evaluate(v_pu=0.90, thd_pct=0.0, dt_s=1.0)
        assert load.state is LoadState.TRIPPED

    def test_reconnect_after_sustained_voltage(self):
        load = self._tripped_load()
        load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=1.0)
        assert load.state is LoadState.TRIPPED  # not yet 2.0 s
        load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=1.0)
        assert load.state is LoadState.RECONNECTING
        # One more call transitions to CONNECTED.
        load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=1.0)
        assert load.state is LoadState.CONNECTED

    def test_no_auto_reconnect_when_config_absent(self):
        # A load with no reconnect settings stays tripped forever.
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=2.0,
            v_trip_pu=0.85,
            v_trip_delay_s=0.0,
        )
        load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.0)  # trip
        for _ in range(100):
            load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=60.0)
        assert load.state is LoadState.TRIPPED

    def test_reconnect_can_trip_again_immediately(self):
        # Voltage recovers, load reconnects, then dips below trip again
        # in the same timestep — must trip again, not stay CONNECTED.
        load = self._tripped_load()
        load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=2.0)  # → RECONNECTING
        assert load.state is LoadState.RECONNECTING
        # Now voltage collapses again on the reconnecting → connected
        # step; because trip delay is 0.0, we should trip again.
        assert load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.0) is LoadState.TRIPPED


class TestEvaluateInputValidation:
    def test_negative_dt_rejected(self):
        load = TripCapableLoad(bus_id="b", p_mw=1.0, q_mvar=0.0)
        with pytest.raises(DataError, match="dt_s must be non-negative"):
            load.evaluate(v_pu=1.0, thd_pct=0.0, dt_s=-0.1)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restores_connected_state(self):
        load = TripCapableLoad(
            bus_id="b",
            p_mw=10.0,
            q_mvar=0.0,
            v_trip_pu=0.85,
            v_trip_delay_s=0.0,
        )
        load.evaluate(v_pu=0.5, thd_pct=0.0, dt_s=0.0)
        assert load.state is LoadState.TRIPPED
        load.reset()
        assert load.state is LoadState.CONNECTED
        # Timers cleared so a brand-new dip needs the full delay again.
        assert load.evaluate(v_pu=0.9, thd_pct=0.0, dt_s=0.0) is LoadState.CONNECTED
