"""Voltage- and harmonic-sensitive load primitive.

Subsystem (4) of the AC-power-flow coupling programme laid out in
``scratchpad/geopulse-acpf-coupling-handoff.md``. Provides
:class:`TripCapableLoad` — a mechanism-only load model with:

* **ZIP decomposition** of the pre-trip power draw. Constant-power,
  constant-current, and constant-impedance fractions can be mixed at
  arbitrary ratios. The physically-critical property (spec §6.4) is
  that constant-power load draws *more* current as voltage falls —
  destabilising — while constant-impedance load draws less.
* **Threshold-plus-time-delay trip** on undervoltage and / or on
  harmonic distortion (THD). A dip below the voltage threshold must
  be sustained for the delay before the trip fires; a brief transient
  that recovers before the delay expires leaves the load online.
* **Explicit reconnection** with its own threshold and delay. Spec
  §6.4 notes that uncontrolled mass reconnection is itself a hazard,
  so the transition from ``TRIPPED`` back to ``CONNECTED`` is a
  first-class state machine step, not automatic.

Scope discipline
----------------
This module contains **mechanism only**. Application-specific
parameterisations — data-centre trip thresholds, aluminium-smelter
harmonic sensitivity, hospital reconnection delays — belong in a
downstream study repository, not here. The dataclass defaults are
deliberately null (``None``) for every trip / reconnect threshold so
the load defaults to "immortal" until a caller supplies the physics.

References
----------
- Handoff spec §4 (outputs), §5 (signature), §6.4 (algorithm),
  §8 item 8 (acceptance criteria).
- Kundur (1994) *Power System Stability and Control*, §7.2 — ZIP load
  model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from geopulse.exceptions import DataError

__all__ = ["LoadState", "TripCapableLoad"]


class LoadState(Enum):
    """Discrete state of a :class:`TripCapableLoad`.

    * ``CONNECTED`` — drawing power from the bus.
    * ``TRIPPED`` — offline; draws no power.
    * ``RECONNECTING`` — transient state entered from ``TRIPPED`` after
      the reconnect voltage threshold has been satisfied for the
      reconnect delay. A single ``evaluate()`` call in this state
      transitions the load back to ``CONNECTED`` unless conditions
      have already deteriorated again. Callers wanting to model
      inrush should apply their own current multiplier for one
      timestep when they see this state.
    """

    CONNECTED = auto()
    TRIPPED = auto()
    RECONNECTING = auto()


@dataclass
class TripCapableLoad:
    """Voltage- and harmonic-sensitive load with a trip state machine.

    Parameters
    ----------
    bus_id : str
        Identifier of the bus this load is attached to.
    p_mw : float
        Rated real power at ``V = 1.0 pu``, MW. Must be non-negative.
    q_mvar : float
        Rated reactive power at ``V = 1.0 pu``, MVAr. Signed
        (positive = absorbed, negative = injected).
    constant_power_fraction : float, optional
        ZIP weight on the constant-power component. Default: ``1.0``
        (worst case for voltage stability). See :meth:`current_draw`.
    constant_current_fraction : float, optional
        ZIP weight on the constant-current component. Default: ``0.0``.
    constant_impedance_fraction : float, optional
        ZIP weight on the constant-impedance component. Default:
        ``0.0``. The three fractions must sum to ``1.0`` to numerical
        tolerance.
    v_trip_pu : float or None, optional
        Undervoltage trip threshold in per-unit. ``None`` disables the
        undervoltage trip. Default: ``None``.
    v_trip_delay_s : float or None, optional
        Undervoltage trip delay in seconds. Required when
        ``v_trip_pu`` is set. Voltage must remain below ``v_trip_pu``
        for a *cumulative* time of at least ``v_trip_delay_s`` before
        the trip fires — a brief dip that recovers resets the timer.
    thd_trip_pct : float or None, optional
        Total-harmonic-distortion trip threshold in percent. ``None``
        disables the harmonic trip. Default: ``None``.
    thd_trip_delay_s : float or None, optional
        Harmonic trip delay in seconds. Required when ``thd_trip_pct``
        is set.
    reconnect_v_pu : float or None, optional
        Voltage that must be restored (and sustained for
        ``reconnect_delay_s``) before a tripped load reconnects.
        ``None`` means "no automatic reconnect".
    reconnect_delay_s : float or None, optional
        Reconnect delay in seconds. Required when ``reconnect_v_pu``
        is set.
    state : LoadState, optional
        Initial state. Default: ``LoadState.CONNECTED``.

    Attributes
    ----------
    state : LoadState
        Current state — mutated by :meth:`evaluate`.

    Notes
    -----
    The trip / reconnect timers are cumulative time below (or above)
    the relevant threshold. They reset the moment the condition
    reverses, which is the standard undervoltage-relay convention.

    Time-domain accuracy is limited by the ``dt_s`` the caller passes
    to :meth:`evaluate`: the state machine is a quasi-static snapshot
    at each call (spec §10 item 6). Sub-second trip logic on a 10 s
    ``dt_s`` will fire late by up to one step.

    Examples
    --------
    A voltage-sensitive load that trips below 0.85 pu after 100 ms and
    won't come back until 0.95 pu is sustained for 5 s:

    >>> load = TripCapableLoad(
    ...     bus_id="bus_42", p_mw=50.0, q_mvar=10.0,
    ...     v_trip_pu=0.85, v_trip_delay_s=0.1,
    ...     reconnect_v_pu=0.95, reconnect_delay_s=5.0,
    ... )
    >>> load.evaluate(v_pu=0.80, thd_pct=0.0, dt_s=0.2).name
    'TRIPPED'
    """

    bus_id: str
    p_mw: float
    q_mvar: float
    constant_power_fraction: float = 1.0
    constant_current_fraction: float = 0.0
    constant_impedance_fraction: float = 0.0
    v_trip_pu: float | None = None
    v_trip_delay_s: float | None = None
    thd_trip_pct: float | None = None
    thd_trip_delay_s: float | None = None
    reconnect_v_pu: float | None = None
    reconnect_delay_s: float | None = None
    state: LoadState = LoadState.CONNECTED

    # Cumulative timers driving the state machine. Not part of the
    # public dataclass surface — set via `evaluate()`.
    _v_below_time_s: float = field(default=0.0, repr=False, compare=False)
    _thd_above_time_s: float = field(default=0.0, repr=False, compare=False)
    _v_above_reconn_time_s: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate ZIP fractions and threshold / delay pairing."""
        self._validate_zip()
        self._validate_thresholds()

    def _validate_zip(self) -> None:
        """ZIP weights must be non-negative and sum to 1.0."""
        z, i, p = (
            self.constant_impedance_fraction,
            self.constant_current_fraction,
            self.constant_power_fraction,
        )
        if min(z, i, p) < 0.0:
            raise DataError(f"ZIP fractions must be non-negative, got Z={z}, I={i}, P={p}")
        s = z + i + p
        if abs(s - 1.0) > 1e-6:
            raise DataError(f"ZIP fractions must sum to 1.0, got Z+I+P = {s} (Z={z}, I={i}, P={p})")

    def _validate_thresholds(self) -> None:
        """Every threshold must be paired with a delay, and vice versa."""
        pairs = [
            ("v_trip_pu", self.v_trip_pu, "v_trip_delay_s", self.v_trip_delay_s),
            ("thd_trip_pct", self.thd_trip_pct, "thd_trip_delay_s", self.thd_trip_delay_s),
            ("reconnect_v_pu", self.reconnect_v_pu, "reconnect_delay_s", self.reconnect_delay_s),
        ]
        for th_name, th, dt_name, dt in pairs:
            if (th is None) != (dt is None):
                raise DataError(
                    f"{th_name} and {dt_name} must be set together "
                    f"(got {th_name}={th!r}, {dt_name}={dt!r})"
                )
            if dt is not None and dt < 0.0:
                raise DataError(f"{dt_name} must be non-negative, got {dt}")

    def current_draw(self, v_pu: float) -> tuple[float, float]:
        r"""Return the (P, Q) power currently drawn from the bus at ``v_pu``.

        Implements the ZIP decomposition:

        .. math::

            P(V) &= P_0 \, [\, f_Z V^2 + f_I V + f_P \,] \\
            Q(V) &= Q_0 \, [\, f_Z V^2 + f_I V + f_P \,]

        where ``V`` is per-unit voltage magnitude and
        ``f_Z + f_I + f_P = 1``.

        Returns ``(0.0, 0.0)`` when the load is not ``CONNECTED``
        (:class:`LoadState` ``TRIPPED`` and ``RECONNECTING`` both draw
        no power; callers modelling inrush can override for one
        timestep on seeing ``RECONNECTING``).

        Parameters
        ----------
        v_pu : float
            Bus voltage magnitude in per-unit.

        Returns
        -------
        tuple[float, float]
            ``(p_mw, q_mvar)`` currently drawn.
        """
        if self.state is not LoadState.CONNECTED:
            return (0.0, 0.0)
        z = self.constant_impedance_fraction
        i = self.constant_current_fraction
        p = self.constant_power_fraction
        scale = z * v_pu * v_pu + i * v_pu + p
        return (self.p_mw * scale, self.q_mvar * scale)

    def current_magnitude_pu(self, v_pu: float) -> float:
        r"""Return the per-unit current magnitude for a given voltage.

        For an active-power-only draw, per-unit apparent current at
        unit power base is ``|S(V)| / V``, so:

        * pure constant-Z: :math:`I \propto V` (falls with V)
        * pure constant-I: :math:`I = \text{const}` (flat)
        * pure constant-P: :math:`I \propto 1/V` (rises as V falls —
          spec §6.4, §8 item 8)

        Included so callers can verify the destabilising property of
        constant-power load without duplicating the ZIP arithmetic.

        Returns 0.0 when the load is not ``CONNECTED``.

        Parameters
        ----------
        v_pu : float
            Bus voltage magnitude in per-unit. Must be strictly positive.

        Returns
        -------
        float
            Apparent current magnitude in per-unit.

        Raises
        ------
        DataError
            If ``v_pu <= 0``.
        """
        if v_pu <= 0.0:
            raise DataError(f"v_pu must be strictly positive, got {v_pu}")
        if self.state is not LoadState.CONNECTED:
            return 0.0
        p_mw, q_mvar = self.current_draw(v_pu)
        s_mva = (p_mw * p_mw + q_mvar * q_mvar) ** 0.5
        return float(s_mva / v_pu)

    def evaluate(self, v_pu: float, thd_pct: float, dt_s: float) -> LoadState:
        """Advance the state machine one timestep and return the new state.

        Called once per quasi-static power-flow snapshot. The
        cumulative timers driving the state machine are updated based
        on whether ``v_pu`` / ``thd_pct`` currently satisfy the
        undervoltage / harmonic / reconnect conditions.

        Parameters
        ----------
        v_pu : float
            Bus voltage magnitude in per-unit at this timestep.
        thd_pct : float
            Bus total harmonic distortion in percent at this timestep.
        dt_s : float
            Time elapsed since the previous ``evaluate()`` call, in
            seconds. Must be non-negative.

        Returns
        -------
        LoadState
            The (possibly updated) state.

        Raises
        ------
        DataError
            If ``dt_s < 0``.
        """
        if dt_s < 0.0:
            raise DataError(f"dt_s must be non-negative, got {dt_s}")
        if self.state is LoadState.CONNECTED:
            self._tick_connected(v_pu, thd_pct, dt_s)
        elif self.state is LoadState.TRIPPED:
            self._tick_tripped(v_pu, dt_s)
        else:  # RECONNECTING → CONNECTED unless immediately tripping again
            self.state = LoadState.CONNECTED
            self._reset_timers()
            self._tick_connected(v_pu, thd_pct, dt_s)
        return self.state

    def _tick_connected(self, v_pu: float, thd_pct: float, dt_s: float) -> None:
        """Update timers and possibly trip while CONNECTED."""
        if self._undervoltage(v_pu):
            self._v_below_time_s += dt_s
            if self.v_trip_delay_s is not None and self._v_below_time_s >= self.v_trip_delay_s:
                self._trip()
                return
        else:
            self._v_below_time_s = 0.0
        if self._over_thd(thd_pct):
            self._thd_above_time_s += dt_s
            if (
                self.thd_trip_delay_s is not None
                and self._thd_above_time_s >= self.thd_trip_delay_s
            ):
                self._trip()
                return
        else:
            self._thd_above_time_s = 0.0

    def _tick_tripped(self, v_pu: float, dt_s: float) -> None:
        """Update timers and possibly reconnect while TRIPPED."""
        if self.reconnect_v_pu is None or self.reconnect_delay_s is None:
            return  # no auto-reconnect configured; stay TRIPPED forever
        if v_pu >= self.reconnect_v_pu:
            self._v_above_reconn_time_s += dt_s
            if self._v_above_reconn_time_s >= self.reconnect_delay_s:
                self.state = LoadState.RECONNECTING
                self._reset_timers()
        else:
            self._v_above_reconn_time_s = 0.0

    def _undervoltage(self, v_pu: float) -> bool:
        """Whether the voltage is currently below the trip threshold."""
        return self.v_trip_pu is not None and v_pu < self.v_trip_pu

    def _over_thd(self, thd_pct: float) -> bool:
        """Whether the THD is currently above the trip threshold."""
        return self.thd_trip_pct is not None and thd_pct > self.thd_trip_pct

    def _trip(self) -> None:
        """Transition to TRIPPED and clear connected-state timers."""
        self.state = LoadState.TRIPPED
        self._reset_timers()

    def _reset_timers(self) -> None:
        """Clear all cumulative timers driving the state machine."""
        self._v_below_time_s = 0.0
        self._thd_above_time_s = 0.0
        self._v_above_reconn_time_s = 0.0

    def reset(self) -> None:
        """Restore the load to CONNECTED with all timers cleared."""
        self.state = LoadState.CONNECTED
        self._reset_timers()
