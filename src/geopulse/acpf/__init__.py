"""AC power-flow coupling layer for GIC → grid-impact studies.

Subsystem (1) of the AC-power-flow coupling programme laid out in
``scratchpad/geopulse-acpf-coupling-handoff.md``. Introduces the
:class:`ACPFBackend` abstract base class that mediates between
GeoPulse's DC GIC solve and any AC power-flow engine, plus one
concrete implementation, :class:`PandapowerBackend`, behind the
``[acpf]`` optional-dependency extra.

Importing this package without ``pandapower`` installed is safe — the
:class:`PandapowerBackend` constructor is the only place that raises,
and only when actually called. See :func:`_require_pandapower` in the
backend module for the exact error path.
"""

from __future__ import annotations

from geopulse.acpf.base import ACPFBackend, ACPFResult, LoadDirection, PVCurve
from geopulse.acpf.coupling import CoupledResult, TransformerContribution, solve_coupled

__all__ = [
    "ACPFBackend",
    "ACPFResult",
    "CoupledResult",
    "LoadDirection",
    "PVCurve",
    "TransformerContribution",
    "solve_coupled",
]
