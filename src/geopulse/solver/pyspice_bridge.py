"""PySpice bridge for nonlinear transient simulation — STUB (WP3).

Requires the optional ``[spice]`` extra.
"""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError
from geopulse.solver.base import Solver, SolverResult

__all__ = ["PySpiceSolver"]


class PySpiceSolver(Solver):
    """SPICE-backed nonlinear transient solver (WP3)."""

    def solve(
        self,
        network,
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
    ) -> SolverResult:
        """Translate to a SPICE netlist and run transient simulation (WP3)."""
        raise NotImplementedYetError("PySpiceSolver.solve", "WP3")
