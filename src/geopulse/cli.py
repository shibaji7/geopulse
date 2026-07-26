"""GeoPulse command-line interface.

Uses :mod:`argparse` (stdlib only). Deliberately no Click / Typer to keep the
core dependency list at six.

Subcommands
-----------
``run``
    Execute a pipeline described by a YAML config file.
``validate``
    Validate the current build against a published benchmark directory.
``info``
    Print version, install location, and detected optional extras.
``earth list``
    List the built-in 1-D Earth models.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from loguru import logger

from geopulse._version import __version__

__all__ = ["main", "build_parser"]


# --- Detect which optional extras are installed at import time ------------
_EXTRAS = {
    "data": ("xarray", "netCDF4", "pandas"),
    "viz": ("cartopy", "geopandas", "shapely"),
    "earth3d": ("SimPEG",),
    "spice": ("PySpice",),
    "service": ("fastapi", "uvicorn"),
    "dev": ("pytest", "ruff", "mypy"),
}


def _extras_installed() -> dict[str, bool]:
    """Return ``{extra_name: True/False}`` reflecting import availability."""
    status: dict[str, bool] = {}
    for name, modules in _EXTRAS.items():
        status[name] = all(importlib.util.find_spec(m) is not None for m in modules)
    return status


def _cmd_info(_args: argparse.Namespace) -> int:
    """``geopulse info`` — print version and optional-extras status."""
    print(f"GeoPulse v{__version__}")
    print(f"Install location: {Path(__file__).resolve().parent}")
    print(f"Python: {sys.version.split()[0]}")
    print("Installed optional extras:")
    for name, present in _extras_installed().items():
        mark = "yes" if present else "no "
        print(f"  [{mark}] {name}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """``geopulse run <config.yaml>`` — execute a pipeline."""
    from geopulse.config import load_config

    cfg = load_config(args.config)
    logger.info("Loaded config from {}", args.config)
    logger.info(
        "Pipeline: source={} earth={} network={} solver={}",
        cfg.source.type,
        cfg.earth.type,
        cfg.network.type,
        cfg.solver.type,
    )
    # Full pipeline execution ships with Phase 1+.
    logger.warning("Pipeline execution is not implemented in Phase 0 (WP1).")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """``geopulse validate <benchmark_dir>`` — validate against a benchmark."""
    p = Path(args.benchmark_dir)
    if not p.is_dir():
        logger.error("Benchmark directory not found: {}", p)
        return 1
    logger.info("Benchmark validation is not implemented in Phase 0 (WP2).")
    return 0


def _cmd_earth_list(_args: argparse.Namespace) -> int:
    """``geopulse earth list`` — list built-in 1-D Earth models."""
    try:
        from geopulse.earth.library import list_models

        for name in list_models():
            print(name)
    except Exception as exc:  # pragma: no cover - library ships in Phase 1
        logger.error("Cannot list models: {}", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser used by :func:`main`."""
    parser = argparse.ArgumentParser(
        prog="geopulse",
        description="GeoPulse — geomagnetically induced current engine.",
    )
    parser.add_argument("--version", action="version", version=f"geopulse {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run a pipeline from a YAML config.")
    p_run.add_argument("config", help="Path to the pipeline config YAML.")
    p_run.set_defaults(func=_cmd_run)

    p_val = subparsers.add_parser("validate", help="Validate against a benchmark.")
    p_val.add_argument("benchmark_dir", help="Path to a benchmark directory.")
    p_val.set_defaults(func=_cmd_validate)

    p_info = subparsers.add_parser("info", help="Show version and installed extras.")
    p_info.set_defaults(func=_cmd_info)

    p_earth = subparsers.add_parser("earth", help="Earth-model commands.")
    earth_sub = p_earth.add_subparsers(dest="earth_command", required=True)
    p_earth_list = earth_sub.add_parser("list", help="List built-in earth models.")
    p_earth_list.set_defaults(func=_cmd_earth_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector for testing. Defaults to :data:`sys.argv[1:]`.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
