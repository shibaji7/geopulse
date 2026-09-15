"""End-to-end DC-GIC → AC-power-flow coupling loop.

Subsystem (5) of the AC-power-flow coupling programme laid out in
``scratchpad/geopulse-acpf-coupling-handoff.md``. Ties every earlier
subsystem together:

1. **GIC input** — per-transformer neutral current from the caller
   (produced upstream by :class:`geopulse.solver.nam.NAMSolver`).
2. **Reactive absorption** — :func:`geopulse.devices.transformer.gic_to_reactive`
   maps each transformer's effective per-phase current to a
   MVAr load at its terminal bus.
3. **Harmonic injection** — :func:`geopulse.devices.transformer.saturation_harmonics`
   lookup gives per-order injection percentages; bus THD is the
   RSS aggregate.
4. **AC power flow** — the caller-supplied
   :class:`geopulse.acpf.base.ACPFBackend` (usually
   :class:`geopulse.acpf.pandapower_backend.PandapowerBackend`)
   solves the network with the injected ΔQ.
5. **Load evaluation** — each caller-supplied
   :class:`geopulse.network.loads.TripCapableLoad` reads its bus V and
   THD and updates its state via
   :meth:`~geopulse.network.loads.TripCapableLoad.evaluate`.

The result is a :class:`CoupledResult` carrying the AC solution,
per-bus ΔQ / harmonics / THD, and the load state after evaluation.
Non-convergence propagates through unchanged — spec §7 makes
voltage collapse a first-class physical result.

Iteration modes (spec §6.1)
---------------------------
* ``"single_pass"`` (default) — one flow of information from GIC
  through to loads, no feedback.
* ``"iterated"`` — after each solve, tripped-load Q is compensated
  via an extra negative reactive injection so the next solve sees
  the load as effectively off, and the loop re-runs until bus V
  changes by less than ``v_tol`` between iterations, ``max_iter``
  is reached, or a limit cycle is detected. **A limit cycle
  (tripped-load set repeats) is reported explicitly rather than
  suppressed** (spec §6.1, §7).

Approximation notes flagged to callers (spec §10)
-------------------------------------------------
* THD propagation uses the *injection-percentage as terminal-voltage-
  distortion* approximation. Proper harmonic power flow is a larger
  work package. Documented on every ``CoupledResult`` via
  ``metadata["thd_approximation"]``.
* Iterated mode compensates the reactive side of tripped loads but
  cannot cancel their real-power draw through the
  :class:`ACPFBackend` interface as it stands — the pandapower net
  still holds their P constant. In practice this means iterated
  results overestimate remaining real-power demand by the tripped
  loads' pre-trip P. Flagged in every ``CoupledResult`` with tripped
  loads via ``metadata["iterated_p_cancellation"] = "unmodelled"``.
* The state-machine time step (``dt_s``) is a quasi-static snapshot
  per spec §10 item 6 — sub-second trip logic on a 10 s ``dt_s``
  fires late by up to one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from geopulse.acpf.base import ACPFBackend, ACPFResult
from geopulse.devices.transformer import (
    CoreType,
    gic_to_reactive,
    saturation_harmonics,
)
from geopulse.exceptions import DataError
from geopulse.network.loads import LoadState, TripCapableLoad
from geopulse.uq.uncertain import Uncertain

__all__ = [
    "CoupledResult",
    "TransformerContribution",
    "solve_coupled",
]


@dataclass(frozen=True)
class TransformerContribution:
    """One transformer's contribution to the coupled solve.

    Captures the caller-supplied bookkeeping every transformer needs
    to enter the coupling loop: which bus its ΔQ lands on, what its
    core type is (for the K-factor and harmonic-table lookup), how
    much DC neutral current it carries out of the GIC solve, and — if
    it's an autotransformer or otherwise mis-approximated by the
    placeholder K-factor — an explicit override.

    Attributes
    ----------
    transformer_id : str
        Identifier for reporting; propagated through metadata.
    bus_id : str
        Bus in the ACPF backend's net where the ΔQ load is injected
        and whose V / THD is read to evaluate downstream loads.
    neutral_gic_A : float or Uncertain
        DC current through the transformer neutral, from the NAM
        solve. Signed; magnitude enters saturation depth.
    core_type : CoreType
        Transformer core construction (spec §6.2).
    v_pu_nominal : float, optional
        Terminal voltage in per-unit for the K-factor pre-solve
        scaling. Refined to the actual bus V during iterated mode.
        Default: ``1.0``.
    n_phases : int, optional
        Number of phases to divide the neutral current by when
        computing effective per-phase current. Default: ``3``.
        Autotransformers should combine series and common winding
        currents externally and pass a single effective per-phase
        value here.
    k_override : Uncertain or float or None, optional
        Override the placeholder K-factor (spec §6.2). Required when
        ``core_type is CoreType.AUTOTRANSFORMER``.
    """

    transformer_id: str
    bus_id: str
    neutral_gic_A: Uncertain | float
    core_type: CoreType
    v_pu_nominal: float = 1.0
    n_phases: int = 3
    k_override: Uncertain | float | None = None


@dataclass(frozen=True)
class CoupledResult:
    """End-to-end result of one GIC → grid-impact coupling run.

    A :class:`CoupledResult` with ``acpf.converged=False`` means
    voltage collapse under the imposed GIC — a valid physical result
    per spec §7. In that case ``load_states`` reflects whatever the
    loads' state machines saw before the collapsed solve, and
    ``unserved_mw`` and per-bus dicts may be empty.

    Attributes
    ----------
    acpf : ACPFResult
        The AC power-flow solution after ΔQ injection. Includes the
        ``converged`` flag.
    delta_q_mvar : dict[str, float]
        Per-bus reactive absorption applied to the backend, MVAr.
    harmonic_pct : dict[str, dict[int, float]]
        Per-bus harmonic-current injection as `%` of fundamental,
        keyed first by bus, then by harmonic order (fundamental omitted).
    thd_pct : dict[str, float]
        Per-bus total harmonic distortion, %, RSS-aggregated across
        the tabulated orders.
    load_states : dict[str, LoadState]
        Post-evaluation state of every :class:`TripCapableLoad`,
        keyed by ``bus_id``. Empty if no loads were supplied.
    unserved_mw : float
        Total pre-trip real-power demand of every load currently in
        ``TRIPPED`` or ``RECONNECTING``.
    iterations : int
        For ``mode="iterated"``, number of ACPF solves executed. For
        ``single_pass``, always ``1``.
    limit_cycle : bool
        ``True`` if iterated mode terminated because the tripped-load
        set repeated across iterations. A caller should report this
        as a physical result — the system oscillates between
        configurations — not as a solver failure.
    metadata : dict
        Backend/approximation notes (see module docstring).
    """

    acpf: ACPFResult
    delta_q_mvar: dict[str, float]
    harmonic_pct: dict[str, dict[int, float]]
    thd_pct: dict[str, float]
    load_states: dict[str, LoadState]
    unserved_mw: float
    iterations: int
    limit_cycle: bool
    metadata: dict = field(default_factory=dict)


def solve_coupled(
    transformers: list[TransformerContribution],
    loads: list[TripCapableLoad],
    backend: ACPFBackend,
    *,
    mode: Literal["single_pass", "iterated"] = "single_pass",
    max_iter: int = 20,
    v_tol: float = 1e-3,
    dt_s: float = 1.0,
    harmonic_max_order: int = 5,
) -> CoupledResult:
    r"""End-to-end coupled solve: GIC → ΔQ → AC power flow → load trip.

    Parameters
    ----------
    transformers : list[TransformerContribution]
        Every transformer contributing GIC → ΔQ. Multiple transformers
        at the same bus are summed. Every entry must reference a
        ``bus_id`` that already exists in the backend's built net.
    loads : list[TripCapableLoad]
        Voltage- / harmonic-sensitive loads to evaluate against the
        post-solve bus conditions. Empty list is fine; the solve
        still returns a well-formed :class:`CoupledResult` with
        ``load_states = {}``.
    backend : ACPFBackend
        Pre-built AC power-flow backend (``.build()`` already called).
    mode : {"single_pass", "iterated"}, optional
        Iteration strategy — see module docstring.
    max_iter : int, optional
        Cap on iterations in ``"iterated"`` mode. Default: ``20``.
    v_tol : float, optional
        Bus-voltage convergence tolerance in per-unit for
        ``"iterated"`` mode. Default: ``1e-3``.
    dt_s : float, optional
        Time-step in seconds passed to
        :meth:`TripCapableLoad.evaluate` — sets how quickly cumulative
        trip / reconnect timers advance per iteration. Default:
        ``1.0``.
    harmonic_max_order : int, optional
        Highest harmonic order to include in the injection lookup and
        the RSS-THD aggregate. Default: ``5`` (per spec §6.3).

    Returns
    -------
    CoupledResult
        Fully populated result. ``acpf.converged=False`` reports
        voltage collapse without raising.

    Raises
    ------
    DataError
        If ``mode`` is not one of the recognised strings, or if the
        backend has not been built, or any transformer's ``bus_id``
        cannot be resolved.
    """
    if mode not in ("single_pass", "iterated"):
        raise DataError(f"mode must be 'single_pass' or 'iterated', got {mode!r}")
    _check_transformers_have_unique_ids(transformers)

    # Step 1: transformer-level derived quantities (invariant under iteration).
    delta_q_per_bus = _sum_delta_q_by_bus(transformers)
    harmonic_pct = _sum_harmonics_by_bus(transformers, harmonic_max_order)
    thd_pct = {bus: _rss_thd(row) for bus, row in harmonic_pct.items()}

    if mode == "single_pass":
        return _solve_single_pass(
            backend=backend,
            loads=loads,
            dt_s=dt_s,
            delta_q_per_bus=delta_q_per_bus,
            harmonic_pct=harmonic_pct,
            thd_pct=thd_pct,
        )
    return _solve_iterated(
        backend=backend,
        loads=loads,
        dt_s=dt_s,
        delta_q_per_bus=delta_q_per_bus,
        harmonic_pct=harmonic_pct,
        thd_pct=thd_pct,
        max_iter=max_iter,
        v_tol=v_tol,
    )


# ---------------------------------------------------------------------------
# Coupling steps
# ---------------------------------------------------------------------------


def _check_transformers_have_unique_ids(
    transformers: list[TransformerContribution],
) -> None:
    """Reject duplicate transformer_ids so the caller can trace them back."""
    seen: set[str] = set()
    for t in transformers:
        if t.transformer_id in seen:
            raise DataError(f"duplicate transformer_id {t.transformer_id!r} in coupling input")
        seen.add(t.transformer_id)


def _sum_delta_q_by_bus(
    transformers: list[TransformerContribution],
) -> dict[str, float]:
    """Sum per-transformer ΔQ contributions grouped by bus."""
    out: dict[str, float] = {}
    for t in transformers:
        i_eff = _effective_per_phase(t.neutral_gic_A, t.n_phases)
        dq = gic_to_reactive(
            i_eff,
            t.core_type,
            v_pu=t.v_pu_nominal,
            k_override=t.k_override,
        )
        out[t.bus_id] = out.get(t.bus_id, 0.0) + float(dq.nominal)
    return out


def _effective_per_phase(
    neutral: Uncertain | float,
    n_phases: int,
) -> Uncertain | float:
    """Divide the neutral current by the phase count."""
    if n_phases <= 0:
        raise DataError(f"n_phases must be positive, got {n_phases}")
    if isinstance(neutral, Uncertain):
        return Uncertain(
            nominal=float(neutral.nominal) / n_phases,
            samples=(
                [s / n_phases for s in neutral.samples] if neutral.samples is not None else None
            ),
            distribution=neutral.distribution,
            params=neutral.params,
        )
    return float(neutral) / n_phases


def _sum_harmonics_by_bus(
    transformers: list[TransformerContribution],
    max_order: int,
) -> dict[str, dict[int, float]]:
    """Sum per-transformer harmonic-injection contributions by bus."""
    out: dict[str, dict[int, float]] = {}
    for t in transformers:
        # AUTOTRANSFORMER is not covered by the empirical harmonics
        # table; skip it silently so the coupling loop still runs for
        # networks with mixed core types.
        if t.core_type is CoreType.AUTOTRANSFORMER:
            continue
        i_eff = _effective_per_phase(t.neutral_gic_A, t.n_phases)
        i_val = float(i_eff.nominal if isinstance(i_eff, Uncertain) else i_eff)
        h = saturation_harmonics(i_val, t.core_type, max_order=max_order)
        row = out.setdefault(t.bus_id, {})
        for order in range(2, max_order + 1):
            row[order] = row.get(order, 0.0) + float(h[order - 1])
    return out


def _rss_thd(harmonics_row: dict[int, float]) -> float:
    """Root-sum-square THD approximation across the tabulated orders."""
    total_sq = sum(v * v for v in harmonics_row.values())
    return float(np.sqrt(total_sq))


def _solve_single_pass(
    *,
    backend: ACPFBackend,
    loads: list[TripCapableLoad],
    dt_s: float,
    delta_q_per_bus: dict[str, float],
    harmonic_pct: dict[str, dict[int, float]],
    thd_pct: dict[str, float],
) -> CoupledResult:
    """One pass: inject ΔQ, solve, evaluate loads."""
    backend.inject_reactive(delta_q_per_bus)
    acpf = backend.solve()
    load_states, unserved = _evaluate_loads(loads, acpf, thd_pct, dt_s)
    return CoupledResult(
        acpf=acpf,
        delta_q_mvar=dict(delta_q_per_bus),
        harmonic_pct={b: dict(r) for b, r in harmonic_pct.items()},
        thd_pct=dict(thd_pct),
        load_states=load_states,
        unserved_mw=unserved,
        iterations=1,
        limit_cycle=False,
        metadata=_build_metadata(load_tripped=unserved > 0.0, iterated=False),
    )


def _solve_iterated(
    *,
    backend: ACPFBackend,
    loads: list[TripCapableLoad],
    dt_s: float,
    delta_q_per_bus: dict[str, float],
    harmonic_pct: dict[str, dict[int, float]],
    thd_pct: dict[str, float],
    max_iter: int,
    v_tol: float,
) -> CoupledResult:
    """Cancel tripped-load Q and re-solve until the iterated loop converges."""
    prev_v: dict[str, float] = {}
    tripped_history: list[frozenset[str]] = []
    load_states: dict[str, LoadState] = {}
    unserved = 0.0
    acpf: ACPFResult | None = None

    for iteration in range(1, max_iter + 1):
        # Compensate tripped loads' Q by adding a negative reactive
        # injection at their bus (equal and opposite to their nominal
        # q_mvar) on top of the ΔQ injection.
        injection = dict(delta_q_per_bus)
        for load in loads:
            if load.state is LoadState.TRIPPED:
                injection[load.bus_id] = injection.get(load.bus_id, 0.0) - load.q_mvar
        backend.inject_reactive(injection)
        acpf = backend.solve()
        load_states, unserved = _evaluate_loads(loads, acpf, thd_pct, dt_s)

        # Limit-cycle detection: hash the tripped-load id-set. An empty
        # tripped set is not a limit cycle — it's the trivial no-drama
        # state that legitimately recurs across iterations. Only flag
        # when the same *non-empty* trip set has repeated.
        tripped_key = frozenset(
            f"{lo.bus_id}::{lo.p_mw}::{lo.q_mvar}"
            for lo in loads
            if lo.state is not LoadState.CONNECTED
        )
        if tripped_key and tripped_key in tripped_history:
            return CoupledResult(
                acpf=acpf,
                delta_q_mvar=dict(delta_q_per_bus),
                harmonic_pct={b: dict(r) for b, r in harmonic_pct.items()},
                thd_pct=dict(thd_pct),
                load_states=load_states,
                unserved_mw=unserved,
                iterations=iteration,
                limit_cycle=True,
                metadata=_build_metadata(
                    load_tripped=unserved > 0.0,
                    iterated=True,
                    limit_cycle=True,
                ),
            )
        tripped_history.append(tripped_key)

        # Voltage-convergence check.
        if _v_converged(prev_v, acpf.v_bus_pu, v_tol):
            return CoupledResult(
                acpf=acpf,
                delta_q_mvar=dict(delta_q_per_bus),
                harmonic_pct={b: dict(r) for b, r in harmonic_pct.items()},
                thd_pct=dict(thd_pct),
                load_states=load_states,
                unserved_mw=unserved,
                iterations=iteration,
                limit_cycle=False,
                metadata=_build_metadata(
                    load_tripped=unserved > 0.0,
                    iterated=True,
                ),
            )
        prev_v = dict(acpf.v_bus_pu)
    # max_iter exhausted.
    assert acpf is not None  # noqa: S101 - defensive; loop guarantees it
    return CoupledResult(
        acpf=acpf,
        delta_q_mvar=dict(delta_q_per_bus),
        harmonic_pct={b: dict(r) for b, r in harmonic_pct.items()},
        thd_pct=dict(thd_pct),
        load_states=load_states,
        unserved_mw=unserved,
        iterations=max_iter,
        limit_cycle=False,
        metadata=_build_metadata(
            load_tripped=unserved > 0.0,
            iterated=True,
            max_iter_exhausted=True,
        ),
    )


def _evaluate_loads(
    loads: list[TripCapableLoad],
    acpf: ACPFResult,
    thd_pct: dict[str, float],
    dt_s: float,
) -> tuple[dict[str, LoadState], float]:
    """Tick each load once against the current V / THD; sum unserved P."""
    states: dict[str, LoadState] = {}
    unserved = 0.0
    for load in loads:
        v_pu = float(acpf.v_bus_pu.get(load.bus_id, 1.0))
        thd = float(thd_pct.get(load.bus_id, 0.0))
        state = load.evaluate(v_pu=v_pu, thd_pct=thd, dt_s=dt_s)
        states[load.bus_id] = state
        if state is not LoadState.CONNECTED:
            unserved += float(load.p_mw)
    return states, unserved


def _v_converged(
    prev: dict[str, float],
    cur: dict[str, float],
    tol: float,
) -> bool:
    """max(|Δv|) over shared buses is below tol — treated as converged."""
    if not prev:
        return False
    shared = set(prev) & set(cur)
    if not shared:
        return False
    return max(abs(cur[b] - prev[b]) for b in shared) < tol


def _build_metadata(
    *,
    load_tripped: bool,
    iterated: bool,
    limit_cycle: bool = False,
    max_iter_exhausted: bool = False,
) -> dict:
    """Assemble the flags-and-approximations metadata dict."""
    md: dict = {
        "thd_approximation": (
            "RSS of injection percentages; proper harmonic power flow deferred "
            "to a follow-up work package (spec §6.3, §10 item 2)"
        ),
    }
    if iterated and load_tripped:
        md["iterated_p_cancellation"] = (
            "unmodelled — tripped loads' P draw is held constant by the "
            "underlying ACPF net; only their Q is compensated"
        )
    if limit_cycle:
        md["termination"] = "limit-cycle detected — tripped-load set repeated"
    if max_iter_exhausted:
        md["termination"] = "max_iter reached without voltage convergence"
    return md
