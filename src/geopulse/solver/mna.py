"""Modified Nodal Analysis solver — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.exceptions import NotImplementedYetError
from geopulse.solver.base import Solver, SolverResult

__all__ = ["MNASolver"]


class MNASolver(Solver):
    """Modified Nodal Analysis solver (WP3) — supports reactive elements."""

    def solve(
        self,
        network,
        network_admittance: np.ndarray,
        earthing_impedance: np.ndarray,
        thevenin_voltages: np.ndarray,
    ) -> SolverResult:
        """Assemble & solve the MNA system (WP3)."""
        raise NotImplementedYetError("MNASolver.solve", "WP3")
