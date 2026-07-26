"""GeoPulse — Unified engine for geomagnetically induced currents.

A modular, extensible Python engine that computes GIC, induced voltages, and
total harmonic distortion across grounded infrastructure — submarine cables,
power grids, oil/gas pipelines, and railways — from a single core.

The top-level namespace intentionally exposes only the version string; heavy
submodules are imported lazily on first attribute access to keep ``import
geopulse`` fast (< 0.5 s cold start).

Environment variables
---------------------
``GEOPULSE_LOG_LEVEL``
    loguru level for the default stderr sink. Default ``INFO``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

from geopulse._version import __version__

__all__ = ["__version__", "logger"]

# --- Configure the default loguru sink ------------------------------------
# We remove loguru's built-in stderr handler so users see exactly one line
# per log record, formatted with green timestamps and cyan call-site info.
logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("GEOPULSE_LOG_LEVEL", "INFO"),
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)


# --- Lazy submodule imports ----------------------------------------------
# Heavy submodules (scipy.sparse in solver, h5py in io, matplotlib in viz)
# are only imported when the user first touches them. This keeps `python -c
# "import geopulse"` under half a second even on slow filesystems.
_LAZY_SUBMODULES = {
    "constants",
    "exceptions",
    "types",
    "sources",
    "earth",
    "efield",
    "network",
    "solver",
    "devices",
    "metrics",
    "uq",
    "io",
    "viz",
    "config",
    "cli",
}


def __getattr__(name: str) -> Any:
    """Lazy import of registered submodules."""
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"geopulse.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'geopulse' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from geopulse import (  # noqa: F401
        cli,
        config,
        constants,
        devices,
        earth,
        efield,
        exceptions,
        io,
        metrics,
        network,
        solver,
        sources,
        types,
        uq,
        viz,
    )
