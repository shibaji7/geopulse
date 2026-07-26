"""Synthetic B-field waveforms for smoke tests and controlled experiments.

Supports three families:

* ``"gaussian_pulse"`` — ``B_x(t) = A · exp(-(t - t_0)² / (2σ²))`` (WP1 primary).
* ``"step"`` — Heaviside step, jumping from 0 to ``A`` at ``t = t_0``.
* ``"sinusoid"`` — single-frequency sinusoid at ``frequency_Hz``.

All waveforms are placed on the ``bx_T`` component only (north-south). ``by_T``
and ``bz_T`` are zero — the plane-wave MT relation then produces a pure ``Ey``
response, which makes analytic validation trivial.

Amplitudes are supplied in **nanoTesla** for scientific convenience but are
converted to Tesla at the loader boundary, per the SI-inside-the-core rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geopulse.constants import NT_TO_T
from geopulse.exceptions import DataError
from geopulse.sources.base import BFieldSource, BFieldTimeSeries

__all__ = ["SyntheticSource"]

_VALID_WAVEFORMS = frozenset({"gaussian_pulse", "step", "sinusoid"})


class SyntheticSource(BFieldSource):
    """Deterministic synthetic B-field waveforms.

    Parameters
    ----------
    waveform : {"gaussian_pulse", "step", "sinusoid"}
        Waveform family.
    amplitude_nT : float, optional
        Peak amplitude in nT. Default: 100.
    t0_s : float, optional
        Time of peak (``gaussian_pulse``) or edge (``step``), in seconds
        relative to ``start_s``. Default: half-way through the window.
    sigma_s : float, optional
        Gaussian half-width (standard deviation), seconds. Default: 300.
    frequency_Hz : float, optional
        Sinusoid frequency in Hz. Default: 0.01 (100 s period).
    phase_rad : float, optional
        Sinusoid phase in radians. Default: 0.
    latitude_deg, longitude_deg : float, optional
        Synthetic station coordinates. Default: (45, -75).

    Raises
    ------
    DataError
        If ``waveform`` is not one of the supported families.

    Examples
    --------
    >>> src = SyntheticSource("gaussian_pulse", amplitude_nT=500.0)
    >>> b = src.load(start_s=0, end_s=3600, dt_s=1.0)
    >>> b.bx_T.max() > 4.9e-7   # 500 nT ≈ 5e-7 T
    True
    """

    def __init__(
        self,
        waveform: str,
        amplitude_nT: float = 100.0,
        *,
        t0_s: float | None = None,
        sigma_s: float = 300.0,
        frequency_Hz: float = 0.01,
        phase_rad: float = 0.0,
        latitude_deg: float = 45.0,
        longitude_deg: float = -75.0,
        **extra: Any,
    ) -> None:
        if waveform not in _VALID_WAVEFORMS:
            raise DataError(f"Unknown waveform {waveform!r}. Valid: {sorted(_VALID_WAVEFORMS)}")
        self.waveform = waveform
        self.amplitude_nT = float(amplitude_nT)
        self.t0_s = t0_s
        self.sigma_s = float(sigma_s)
        self.frequency_Hz = float(frequency_Hz)
        self.phase_rad = float(phase_rad)
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)
        self.extra = extra

    def load(
        self,
        start_s: float,
        end_s: float,
        dt_s: float = 1.0,
        station_id: str = "SYN",
    ) -> BFieldTimeSeries:
        """Generate the configured waveform.

        Parameters
        ----------
        start_s : float
            Start time (seconds since epoch, or 0 for synthetic).
        end_s : float
            End time (seconds since epoch). Must satisfy ``end_s > start_s``.
        dt_s : float, optional
            Sampling interval, seconds. Default: 1.0.
        station_id : str, optional
            Station identifier stored on the output. Default: ``"SYN"``.

        Returns
        -------
        BFieldTimeSeries
            SI-unit B-field time series with the waveform on ``bx_T`` and
            zeros on ``by_T``, ``bz_T``.

        Raises
        ------
        DataError
            If the time window is invalid.
        """
        if end_s <= start_s:
            raise DataError(f"end_s ({end_s}) must exceed start_s ({start_s})")
        if dt_s <= 0:
            raise DataError(f"dt_s must be positive, got {dt_s}")

        # Absolute epoch times for downstream metadata; internal waveform uses
        # elapsed seconds since start_s so that t0_s is relative to the window.
        n = int(np.floor((end_s - start_s) / dt_s))
        time_s = start_s + np.arange(n, dtype=np.float64) * dt_s
        t_rel = time_s - start_s

        amp_T = self.amplitude_nT * NT_TO_T
        t0 = self.t0_s if self.t0_s is not None else 0.5 * (end_s - start_s)

        if self.waveform == "gaussian_pulse":
            # B_x(t) = A · exp(-(t - t0)² / (2σ²))  (canonical Gaussian pulse).
            bx = amp_T * np.exp(-((t_rel - t0) ** 2) / (2.0 * self.sigma_s**2))
        elif self.waveform == "step":
            # Heaviside step: 0 before t0, amp after. Value at the edge = amp/2
            # (the standard antisymmetric convention).
            bx = np.where(t_rel < t0, 0.0, amp_T)
            edge = np.argmin(np.abs(t_rel - t0))
            if np.isclose(t_rel[edge], t0):
                bx[edge] = 0.5 * amp_T
        elif self.waveform == "sinusoid":
            # B_x(t) = A · sin(2π f t + φ).
            bx = amp_T * np.sin(2.0 * np.pi * self.frequency_Hz * t_rel + self.phase_rad)
        else:  # pragma: no cover - guarded in __init__
            raise DataError(f"Unhandled waveform: {self.waveform}")

        zeros = np.zeros_like(bx)

        return BFieldTimeSeries(
            time_s=time_s,
            bx_T=bx.astype(np.float64),
            by_T=zeros,
            bz_T=zeros,
            station_id=station_id,
            latitude_deg=self.latitude_deg,
            longitude_deg=self.longitude_deg,
            sampling_rate_Hz=1.0 / dt_s,
            metadata={
                "source": "SyntheticSource",
                "waveform": self.waveform,
                "amplitude_nT": self.amplitude_nT,
                "t0_s": t0,
                "sigma_s": self.sigma_s,
                "frequency_Hz": self.frequency_Hz,
            },
        )
