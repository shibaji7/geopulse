"""Abstract AC power-flow backend contract.

Every concrete backend (pandapower today, PSS/E or MATPOWER-native
tomorrow) implements this ABC so the coupling loop can be written
once. The ABC follows the same pattern as :mod:`geopulse.solver` /
:mod:`geopulse.earth` / :mod:`geopulse.sources` — an abstract class
plus small frozen result dataclasses — and its concrete
implementations are discoverable via Python entry points for
third-party backends.

Non-convergence policy
----------------------
Spec §7 is explicit: **non-convergence is a physical result
(voltage collapse), not an error**. Every :meth:`ACPFBackend.solve`
implementation MUST return an :class:`ACPFResult` with
``converged=False`` rather than raise, so the coupling loop can
report the collapse cleanly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ACPFBackend", "ACPFResult", "LoadDirection", "PVCurve"]


@dataclass(frozen=True)
class ACPFResult:
    """Result of one AC power-flow snapshot.

    A :class:`ACPFResult` with ``converged=False`` is a valid, meaningful
    physical result — the system has been driven past its voltage-stability
    limit. The other fields may be empty in that case.

    Attributes
    ----------
    converged : bool
        Whether Newton-Raphson (or the backend's chosen algorithm) found
        a solution.
    v_bus_pu : dict[str, float]
        Post-solve bus voltage magnitude in per-unit, keyed by bus name.
    v_bus_angle_deg : dict[str, float]
        Post-solve bus voltage angle in degrees, keyed by bus name.
    p_bus_mw : dict[str, float]
        Net real-power injection at each bus (positive = source), MW.
    q_bus_mvar : dict[str, float]
        Net reactive-power injection at each bus (positive = source),
        MVAr.
    iterations : int, optional
        Number of Newton iterations to reach convergence.
    metadata : dict
        Backend-specific extras (solver name, wall-clock, etc.).
    """

    converged: bool
    v_bus_pu: dict[str, float]
    v_bus_angle_deg: dict[str, float]
    p_bus_mw: dict[str, float]
    q_bus_mvar: dict[str, float]
    iterations: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LoadDirection:
    """Direction vector for continuation power flow.

    Each entry says how many MW / MVAr to add at that bus per unit
    increment of the CPF parameter ``λ``. Buses not listed are held
    fixed. All entries at a single scale factor together define one
    'direction' along the PV surface.

    Attributes
    ----------
    p_direction_mw : dict[str, float]
        Per-bus real-power increment, MW per unit ``λ``.
    q_direction_mvar : dict[str, float]
        Per-bus reactive-power increment, MVAr per unit ``λ``.
    """

    p_direction_mw: dict[str, float] = field(default_factory=dict)
    q_direction_mvar: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PVCurve:
    """Continuation power flow trace.

    Attributes
    ----------
    lambda_values : numpy.ndarray
        Continuation parameter samples, ``0`` at base case up to
        ``lambda_max`` at the nose point (or wherever the trace stops).
    v_bus_pu : dict[str, numpy.ndarray]
        Per-bus voltage magnitude in per-unit, one array of the same
        length as ``lambda_values`` per bus.
    lambda_max : float
        Value of ``λ`` at the nose point (voltage stability limit).
    converged : bool
        Whether a nose point was located within the caller's iteration
        budget. ``False`` means the trace stopped early; ``lambda_max``
        may be a lower bound rather than the true limit.
    metadata : dict
        Backend-specific extras.
    """

    lambda_values: np.ndarray
    v_bus_pu: dict[str, np.ndarray]
    lambda_max: float
    converged: bool
    metadata: dict = field(default_factory=dict)


class ACPFBackend(ABC):
    """Abstract AC power-flow backend.

    Life cycle:

    1. :meth:`build` — translate a GeoPulse network + external AC case
       (MATPOWER file, pandapower net, PSS/E RAW, ...) into the
       backend's internal representation.
    2. Optionally :meth:`inject_reactive` — apply the ``ΔQ`` map
       produced by :func:`geopulse.devices.transformer.gic_to_reactive`
       as additional bus loads.
    3. :meth:`solve` — run the AC power flow and return an
       :class:`ACPFResult`. **Must return ``converged=False`` rather
       than raise on non-convergence.**
    4. Optionally :meth:`continuation` — trace the PV curve to locate
       the nose point.
    """

    @abstractmethod
    def build(self, network: Any, ac_case: Any) -> None:
        """Translate the GeoPulse canonical network + AC case into backend form.

        Parameters
        ----------
        network : geopulse.network.powergrid.PowerGridNetwork
            The parsed GeoPulse DC-side network (buses, transformer
            neutrals, grounding impedances). Used to validate bus /
            transformer identifiers when :meth:`inject_reactive` is
            called later.
        ac_case : object
            An AC-side description. The exact type is backend-specific;
            common accepted forms include a path to a MATPOWER ``.m``
            file, a pandapower ``net`` object, or a PSS/E RAW file.
        """

    @abstractmethod
    def inject_reactive(self, delta_q: dict[str, float]) -> None:
        """Apply per-transformer reactive absorption as extra bus load (MVAr).

        Idempotent — successive calls replace any previously-injected
        ΔQ rather than accumulating.

        Parameters
        ----------
        delta_q : dict[str, float]
            Map from bus identifier (backend-specific spelling) to
            additional reactive absorption in MVAr.
        """

    @abstractmethod
    def solve(self, init: str = "auto") -> ACPFResult:
        """Run one AC power-flow snapshot.

        MUST return an :class:`ACPFResult` with ``converged=False``
        rather than raise if the solve fails (spec §7). Non-convergence
        is a physical result, not an error.

        Parameters
        ----------
        init : str, optional
            Initial-guess strategy. ``"auto"`` = backend default;
            other values are backend-specific (e.g. pandapower accepts
            ``"flat"``, ``"dc"``, ``"results"``). Default: ``"auto"``.

        Returns
        -------
        ACPFResult
            The (possibly non-converged) result.
        """

    @abstractmethod
    def continuation(self, direction: LoadDirection) -> PVCurve:
        """Trace the PV curve along ``direction`` to the nose point.

        Parameters
        ----------
        direction : LoadDirection
            Per-bus MW / MVAr increment per unit continuation
            parameter ``λ``.

        Returns
        -------
        PVCurve
            The trace, including ``lambda_max`` at the nose point.
        """
