"""Abstract base class for nonlinear device models.

A :class:`DeviceModel` takes a GIC time series (from the solver) and computes
the device response: transformer magnetisation current, harmonic content,
hot-spot temperature proxy, or cathodic-protection shift.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["DeviceModel", "DeviceResponse"]


@dataclass(frozen=True)
class DeviceResponse:
    """Container for a device response to GIC injection.

    Attributes
    ----------
    time_s : numpy.ndarray
        Time array in seconds. Shape ``(n_times,)``.
    response_current_A : numpy.ndarray
        Device response current (e.g. magnetisation current). Shape
        ``(n_times,)``. Units: Amperes.
    thd : float
        Total harmonic distortion (0 to 1, or NaN if not applicable).
    harmonics : numpy.ndarray
        Harmonic amplitudes. Shape ``(n_harmonics,)``. Index 0 = fundamental.
    top_oil_C : numpy.ndarray or None
        Top-oil temperature time series in °C. Shape ``(n_times,)``. Set by
        :class:`~geopulse.devices.transformer.TransformerModel`; ``None`` for
        devices without a thermal model.
    hotspot_C : numpy.ndarray or None
        Winding hot-spot temperature time series in °C. Shape ``(n_times,)``.
        Set by :class:`~geopulse.devices.transformer.TransformerModel`.
    metadata : dict
        Additional device-specific outputs.
    """

    time_s: np.ndarray
    response_current_A: np.ndarray
    thd: float
    harmonics: np.ndarray
    top_oil_C: Optional[np.ndarray] = None
    hotspot_C: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class DeviceModel(abc.ABC):
    """Abstract base class for nonlinear device models.

    Subclasses:

    * ``TransformerModel`` — saturation curve → half-cycle → THD.
    * ``RectifierModel`` — bridge rectifier under DC bias.
    * ``CathodicProtectionModel`` — PSP shift under telluric E-field.
    """

    @abc.abstractmethod
    def inject_gic(
        self,
        time_s: np.ndarray,
        gic_A: np.ndarray,
        ac_voltage_V: float = 0.0,
        ac_frequency_Hz: float = 50.0,
    ) -> DeviceResponse:
        """Inject a GIC time series into the device and compute the response.

        Parameters
        ----------
        time_s : numpy.ndarray
            Time array in seconds. Shape ``(n_times,)``.
        gic_A : numpy.ndarray
            GIC current in Amperes. Shape ``(n_times,)``.
        ac_voltage_V : float, optional
            Nominal AC voltage (transformers only). Default: 0.0.
        ac_frequency_Hz : float, optional
            AC system frequency in Hz. Default: 50.0.

        Returns
        -------
        DeviceResponse
            Full response including harmonics and THD.
        """
