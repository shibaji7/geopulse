"""Abstract base class for Earth conductivity models.

The design contract: every :class:`EarthModel` produces an
:class:`~geopulse.earth.impedance.Impedance` object. The
:class:`~geopulse.earth.impedance.Impedance` handles the actual E-field
computation via polymorphism. 1-D models produce
:class:`~geopulse.earth.impedance.ScalarImpedance`, 2-D produce
:class:`~geopulse.earth.impedance.TensorImpedance`, 3-D produce
:class:`~geopulse.earth.impedance.KernelImpedance`. The
:mod:`~geopulse.efield` layer calls ``impedance.apply(...)`` and never knows
which type it got.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from geopulse.earth.impedance import Impedance

__all__ = ["ConductivityLayer", "EarthModel"]


@dataclass(frozen=True)
class ConductivityLayer:
    """A single layer in a 1-D conductivity model.

    Attributes
    ----------
    thickness_m : float
        Layer thickness in meters. Use ``numpy.inf`` for the terminating
        half-space (bottom layer).
    conductivity_Sm : float
        Electrical conductivity in S/m.

    Notes
    -----
    Resistivity ``ρ = 1/σ`` is NOT stored — compute it when needed to avoid
    redundancy and potential inconsistency.

    Raises
    ------
    ValueError
        If ``conductivity_Sm`` is non-positive, or ``thickness_m`` is
        non-positive and not infinite.
    """

    thickness_m: float
    conductivity_Sm: float

    def __post_init__(self) -> None:
        """Validate physical positivity of both fields."""
        if self.conductivity_Sm <= 0:
            raise ValueError(f"Conductivity must be positive, got {self.conductivity_Sm} S/m")
        if self.thickness_m <= 0 and not np.isinf(self.thickness_m):
            raise ValueError(f"Thickness must be positive or inf, got {self.thickness_m} m")


class EarthModel(abc.ABC):
    """Abstract base class for Earth conductivity models.

    Subclasses represent different spatial complexity:

    * ``Layered1D`` — horizontally layered half-space (Wait recursion).
    * ``Structured2D`` — 2-D cross-section (finite-difference).
    * ``Unstructured3D`` — full volumetric (ModEM / SimPEG wrapper).

    All produce an :class:`~geopulse.earth.impedance.Impedance` object via
    :meth:`compute_impedance`.
    """

    @abc.abstractmethod
    def compute_impedance(self, freqs_Hz: np.ndarray) -> Impedance:
        """Compute the surface impedance at the given frequencies.

        Parameters
        ----------
        freqs_Hz : numpy.ndarray
            Frequency array in Hz. Shape: ``(n_freqs,)``. Must be
            non-negative. DC (0 Hz) is allowed but may return NaN because
            the plane-wave impedance is undefined at ω = 0.

        Returns
        -------
        Impedance
            An :class:`~geopulse.earth.impedance.Impedance` subclass matching
            this model's dimensionality.
        """

    @property
    @abc.abstractmethod
    def tier(self) -> int:
        """Dimensionality tier: 1 for 1-D, 2 for 2-D, 3 for 3-D."""

    @classmethod
    def from_library(cls, name: str) -> "EarthModel":
        """Load a named model from the built-in library.

        Parameters
        ----------
        name : str
            Model name. See :func:`geopulse.earth.library.list_models` for the
            registry.

        Returns
        -------
        EarthModel
            The requested model.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        from geopulse.earth.library import get_model

        return get_model(name)
