"""Circuit solvers: NAM (=LPm), MNA, and PySpice bridge."""

from __future__ import annotations

from geopulse.solver.base import Solver, SolverResult
from geopulse.solver.nam import NAMSolver, solve_nam

__all__ = ["NAMSolver", "Solver", "SolverResult", "solve_nam"]
