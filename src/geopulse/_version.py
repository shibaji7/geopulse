"""Version string for GeoPulse.

Single source of truth. `pyproject.toml` reads `__version__` at build time
via `[tool.setuptools.dynamic]`; `docs/conf.py` imports it directly. Bump
here and here alone — everything else derives.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0a1"
