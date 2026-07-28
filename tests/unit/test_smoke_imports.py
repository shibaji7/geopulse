"""Smoke test: every public submodule must import cleanly.

Two purposes:

1. **Catch import-time regressions** early — a stub with a typo would
   otherwise sit undetected until someone tried to call it.
2. **Contribute honest coverage** for stub modules. With ``raise
   NotImplementedYetError`` excluded via ``[tool.coverage.report]
   exclude_lines``, the only lines a stub file has left to cover are
   its ``import`` statements at the top — walking every submodule
   here exercises those.

Modules that are stubs but expose no importable name at package level
(``metrics/*``, ``viz/*``, some ``uq/*``) still get counted because
``pkgutil.walk_packages`` reaches them by module path.
"""

from __future__ import annotations

import importlib
import pkgutil

import geopulse


def _all_submodules() -> list[str]:
    names: list[str] = []
    for m in pkgutil.walk_packages(geopulse.__path__, prefix="geopulse."):
        names.append(m.name)
    return names


def test_every_submodule_imports() -> None:
    """Import every submodule under :mod:`geopulse`.

    Failure means either a genuine broken import (fix the module) or a
    new submodule was added that raises at import time (probably wrong
    — imports should be side-effect free; raises belong in function
    bodies).
    """
    failures: list[tuple[str, Exception]] = []
    for name in _all_submodules():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — capture-all is the point
            failures.append((name, exc))
    if failures:
        report = "\n".join(f"  {name}: {type(exc).__name__}: {exc}" for name, exc in failures)
        raise AssertionError(f"submodule import failures:\n{report}")


def test_walk_finds_expected_families() -> None:
    """Sanity-check the walker actually reaches every top-level family."""
    names = _all_submodules()
    expected_prefixes = {
        "geopulse.devices",
        "geopulse.earth",
        "geopulse.efield",
        "geopulse.io",
        "geopulse.metrics",
        "geopulse.network",
        "geopulse.solver",
        "geopulse.sources",
        "geopulse.uq",
        "geopulse.viz",
    }
    for pref in expected_prefixes:
        assert any(n.startswith(pref) for n in names), f"walker missed {pref!r}"
