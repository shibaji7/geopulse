"""Custom exception hierarchy for GeoPulse.

All GeoPulse exceptions inherit from :class:`GeoPulseError` so users can
catch all of them with a single except clause.
"""

from __future__ import annotations

__all__ = [
    "GeoPulseError",
    "ConfigurationError",
    "DataError",
    "ShapeMismatchError",
    "ConvergenceError",
    "NotImplementedYetError",
]


class GeoPulseError(Exception):
    """Base exception for all GeoPulse errors."""


class ConfigurationError(GeoPulseError):
    """Raised when a configuration file is invalid or missing."""


class DataError(GeoPulseError):
    """Raised when input data is malformed, missing, or has wrong units."""


class ShapeMismatchError(DataError):
    """Raised when array shapes do not match the expected contract."""


class ConvergenceError(GeoPulseError):
    """Raised when an iterative solver fails to converge."""


class NotImplementedYetError(GeoPulseError):
    """Raised for features that are in the roadmap but not yet coded.

    Includes the work-package reference so users know when to expect it.

    Parameters
    ----------
    feature : str
        Short human-readable feature name (e.g. ``"TensorImpedance.apply"``).
    work_package : str, optional
        Roadmap work-package identifier (e.g. ``"WP2"``). Default: ``"TBD"``.
    """

    def __init__(self, feature: str, work_package: str = "TBD") -> None:
        super().__init__(
            f"{feature} is not yet implemented. See GeoPulse roadmap work package {work_package}."
        )
        self.feature = feature
        self.work_package = work_package
