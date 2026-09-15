"""Pandapower-backed AC power-flow backend.

Concrete :class:`~geopulse.acpf.base.ACPFBackend` implementation using
`pandapower <https://www.pandapower.org/>`_. Sits behind the
``[acpf]`` optional-dependency extra — importing this module does not
import pandapower; only instantiating :class:`PandapowerBackend` does.
That way ``import geopulse`` continues to work in a lean install, and
the AC pathway raises a clear, actionable error at the point the user
actually reaches for it.

Continuation power flow is stubbed with
:class:`~geopulse.exceptions.NotImplementedYetError` (work package
``acpf-continuation-pf``); the ABC still requires the method so the
coupling loop can call it once the implementation lands.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from geopulse.acpf.base import ACPFBackend, ACPFResult, LoadDirection, PVCurve
from geopulse.exceptions import DataError, NotImplementedYetError

__all__ = ["PandapowerBackend"]


def _require_pandapower() -> Any:
    """Import and return pandapower, raising with an install hint on failure."""
    try:
        import pandapower as pp
    except ImportError as exc:
        raise DataError(
            "pandapower is required for the AC power-flow backend; install "
            "with `pip install geopulse[acpf]` (or `pip install pandapower`)."
        ) from exc
    return pp


class PandapowerBackend(ACPFBackend):
    """AC power-flow backend implemented on pandapower's Newton-Raphson.

    Life cycle mirrors the ABC contract — :meth:`build`,
    :meth:`inject_reactive`, :meth:`solve`, :meth:`continuation` (stub).

    Attributes
    ----------
    net : pandapower.pandapowerNet
        The underlying network object after :meth:`build`. Exposed
        read-only for inspection; downstream mutations should go
        through the ABC methods so this backend can bookkeep its
        injected loads correctly.

    Examples
    --------
    Build from a MATPOWER file, inject ΔQ at one bus, solve:

    >>> from geopulse.acpf import PandapowerBackend                              # doctest: +SKIP
    >>> from geopulse.network.powergrid import PowerGridNetwork                  # doctest: +SKIP
    >>> net = PowerGridNetwork.from_file("horton_epri21.m")                      # doctest: +SKIP
    >>> backend = PandapowerBackend()                                            # doctest: +SKIP
    >>> backend.build(net, ac_case="horton_epri21.m")                            # doctest: +SKIP
    >>> backend.inject_reactive({"bus_6": 30.0})                                 # doctest: +SKIP
    >>> result = backend.solve()                                                 # doctest: +SKIP
    >>> result.converged, result.v_bus_pu["bus_6"]                               # doctest: +SKIP
    (True, 0.98...)
    """

    def __init__(self) -> None:
        """Verify pandapower is importable; otherwise raise with an install hint."""
        _require_pandapower()  # raise early so downstream methods can trust availability
        self._net: Any = None
        self._network: Any = None
        self._injected_load_idx: list[int] = []

    @property
    def net(self) -> Any:
        """The underlying pandapower ``net``, or ``None`` before :meth:`build`."""
        return self._net

    def build(self, network: Any, ac_case: Any) -> None:
        """Set up the pandapower network from an external AC case.

        Accepts either a path to a MATPOWER ``.m`` file (delegating to
        pandapower's own converter) or a pre-built pandapower ``net``
        object (deep-copied so subsequent :meth:`inject_reactive`
        mutations do not leak back to the caller's copy).

        Parameters
        ----------
        network : object
            The GeoPulse canonical network. Kept for later validation
            of bus / transformer identifiers.
        ac_case : str, pathlib.Path, or pandapower.pandapowerNet
            The AC-side description.

        Raises
        ------
        DataError
            If ``ac_case`` is neither a valid MATPOWER path nor a
            pandapower-net-like object.
        """
        _require_pandapower()
        if isinstance(ac_case, (str, Path)):
            path = str(ac_case)
            try:
                from pandapower.converter.matpower import from_mpc
            except ImportError as exc:  # pragma: no cover - pp ships this
                raise DataError(
                    "pandapower.converter.matpower.from_mpc is unavailable — "
                    "the installed pandapower build appears to be missing its "
                    "MATPOWER converter."
                ) from exc
            try:
                self._net = from_mpc(path)
            except Exception as exc:
                raise DataError(
                    f"Could not read {path!r} as a MATPOWER case via "
                    f"pandapower.converter.matpower.from_mpc: {exc}"
                ) from exc
        elif hasattr(ac_case, "bus") and hasattr(ac_case, "line"):
            # pandapower-net-shaped: deep-copy so we never mutate caller state.
            self._net = copy.deepcopy(ac_case)
        else:
            raise DataError(
                "ac_case must be a MATPOWER .m file path or a pandapower net; "
                f"got {type(ac_case).__name__}"
            )
        self._network = network
        self._injected_load_idx = []

    def inject_reactive(self, delta_q: dict[str, float]) -> None:
        """Apply per-bus reactive absorption as extra bus loads.

        Any previously-injected ΔQ loads are removed first — this
        method is idempotent.

        Parameters
        ----------
        delta_q : dict[str, float]
            Map from bus name (as stored in ``net.bus['name']``) to
            reactive absorption in MVAr. Positive values are absorbed
            (destabilising); negative values are injected.

        Raises
        ------
        DataError
            If :meth:`build` has not been called, or any bus name is
            not found in the pandapower net.
        """
        pp = _require_pandapower()
        self._check_built()
        # Remove the previous injection, if any.
        for load_idx in self._injected_load_idx:
            if load_idx in self._net.load.index:
                self._net.load.drop(load_idx, inplace=True)
        self._injected_load_idx = []
        # Add the new injection.
        for bus_key, dq_mvar in delta_q.items():
            bus_idx = self._resolve_bus(bus_key)
            new_load = pp.create_load(
                self._net,
                bus=bus_idx,
                p_mw=0.0,
                q_mvar=float(dq_mvar),
                name=f"acpf_delta_q_{bus_key}",
            )
            self._injected_load_idx.append(int(new_load))

    def solve(self, init: str = "auto") -> ACPFResult:
        """Run Newton-Raphson power flow.

        Returns a non-converged :class:`ACPFResult` rather than raising
        when the solve diverges — spec §7 makes voltage collapse a
        first-class physical result.

        Parameters
        ----------
        init : str, optional
            Initial-guess strategy passed to ``pandapower.runpp``.
            ``"auto"`` (default) delegates to pandapower's chooser;
            other useful values include ``"flat"``, ``"dc"``, and
            ``"results"``.

        Returns
        -------
        ACPFResult
            Populated result on convergence; empty per-bus dicts on
            non-convergence, ``converged=False`` in both cases if
            pandapower reports divergence.
        """
        pp = _require_pandapower()
        self._check_built()
        converged = False
        iterations = 0
        try:
            pp.runpp(self._net, init=init)
            converged = bool(getattr(self._net, "converged", True))
            iterations = int(getattr(self._net, "_ppc", {}).get("iterations", 0) or 0)
        except pp.LoadflowNotConverged:
            converged = False
        if not converged:
            return ACPFResult(
                converged=False,
                v_bus_pu={},
                v_bus_angle_deg={},
                p_bus_mw={},
                q_bus_mvar={},
                iterations=iterations,
                metadata={"backend": "pandapower", "reason": "LoadflowNotConverged"},
            )
        return self._extract_result(iterations=iterations)

    def continuation(self, direction: LoadDirection) -> PVCurve:
        """Continuation power flow — stub (work package ``acpf-continuation-pf``)."""
        del direction
        raise NotImplementedYetError(
            "PandapowerBackend.continuation",
            "acpf-continuation-pf",
        )

    # ------------------------------------------------------------------ helpers

    def _check_built(self) -> None:
        """Raise ``DataError`` if :meth:`build` has not been called."""
        if self._net is None:
            raise DataError(
                "PandapowerBackend.build(network, ac_case) must be called before "
                "inject_reactive() / solve() / continuation()"
            )

    def _resolve_bus(self, key: Any) -> int:
        """Translate a bus key (int index or str name) to a pp-net row index."""
        if isinstance(key, (int,)) and not isinstance(key, bool):
            if key in self._net.bus.index:
                return int(key)
            raise DataError(f"bus index {key!r} out of range in pandapower net")
        matches = self._net.bus[self._net.bus["name"] == str(key)]
        if len(matches) == 0:
            raise DataError(f"bus {key!r} not found in pandapower net")
        return int(matches.index[0])

    def _extract_result(self, *, iterations: int) -> ACPFResult:
        """Pull per-bus results out of the solved pandapower net."""
        bus = self._net.bus
        res = self._net.res_bus
        names = [
            str(n) if n is not None else str(idx)
            for idx, n in zip(bus.index, bus["name"], strict=True)
        ]
        return ACPFResult(
            converged=True,
            v_bus_pu={n: float(v) for n, v in zip(names, res["vm_pu"], strict=True)},
            v_bus_angle_deg={n: float(v) for n, v in zip(names, res["va_degree"], strict=True)},
            p_bus_mw={n: float(v) for n, v in zip(names, res["p_mw"], strict=True)},
            q_bus_mvar={n: float(v) for n, v in zip(names, res["q_mvar"], strict=True)},
            iterations=iterations,
            metadata={"backend": "pandapower"},
        )
