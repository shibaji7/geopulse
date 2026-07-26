"""Earth conductivity models and the surface-impedance abstraction.

Every :class:`~geopulse.earth.base.EarthModel` produces an
:class:`~geopulse.earth.impedance.Impedance` object. The
:mod:`~geopulse.efield` module calls ``impedance.apply(Bx_f, By_f)`` and never
branches on model dimensionality — polymorphism is the whole point.
"""

from __future__ import annotations

from geopulse.earth.base import ConductivityLayer, EarthModel
from geopulse.earth.impedance import (
    Impedance,
    KernelImpedance,
    ScalarImpedance,
    TensorImpedance,
)
from geopulse.earth.structured_2d import CoastalCorrection2D

__all__ = [
    "CoastalCorrection2D",
    "ConductivityLayer",
    "EarthModel",
    "Impedance",
    "ScalarImpedance",
    "TensorImpedance",
    "KernelImpedance",
]
