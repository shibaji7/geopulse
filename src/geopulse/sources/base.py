"""Abstract base class for magnetic-field data sources.

Every source adapter (SuperMAG, INTERMAGNET, synthetic, MHD) implements this
interface. Downstream modules (efield, network) never know or care which
source produced the B-field data — they only see
:class:`BFieldTimeSeries`.

Design principle: load raw data → convert to SI (Tesla) at the boundary →
return a standardised container.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from geopulse.exceptions import ShapeMismatchError

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["BFieldSource", "BFieldTimeSeries"]


@dataclass(frozen=True)
class BFieldTimeSeries:
    """Standardised magnetic-field time-series container.

    All values are in SI units (Tesla for field, seconds for time). This is
    the ONLY object that crosses the source → efield boundary.

    Attributes
    ----------
    time_s : numpy.ndarray
        Time array in seconds since epoch. Shape: ``(n_times,)``. Must be
        monotonically increasing with (approximately) uniform spacing.
    bx_T : numpy.ndarray
        North-south magnetic-field component in Tesla. Shape: ``(n_times,)``.
        Geographic north positive.
    by_T : numpy.ndarray
        East-west magnetic-field component in Tesla. Shape: ``(n_times,)``.
        Geographic east positive.
    bz_T : numpy.ndarray
        Vertical component in Tesla. Downward positive (geophysics convention).
    station_id : str
        Station identifier (e.g. ``"OTT"`` for Ottawa).
    latitude_deg : float
        Geographic latitude, degrees north.
    longitude_deg : float
        Geographic longitude, degrees east.
    sampling_rate_Hz : float
        Sampling rate in Hz. Auto-computed from ``time_s`` if left at 0.
    metadata : dict
        Arbitrary metadata (source, processing history, ...).

    Notes
    -----
    Frozen dataclass — immutable after construction. To modify, build a new
    instance.
    """

    time_s: np.ndarray
    bx_T: np.ndarray
    by_T: np.ndarray
    bz_T: np.ndarray
    station_id: str
    latitude_deg: float
    longitude_deg: float
    sampling_rate_Hz: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and back-fill :attr:`sampling_rate_Hz` if needed."""
        n = len(self.time_s)
        for name, arr in (
            ("bx_T", self.bx_T),
            ("by_T", self.by_T),
            ("bz_T", self.bz_T),
        ):
            if len(arr) != n:
                raise ShapeMismatchError(
                    f"{name} has length {len(arr)}, expected {n} (matching time_s)."
                )
        if self.sampling_rate_Hz == 0.0 and n > 1:
            dt = float(np.median(np.diff(self.time_s)))
            object.__setattr__(self, "sampling_rate_Hz", 1.0 / dt)


class BFieldSource(abc.ABC):
    """Abstract base class for magnetic-field data sources.

    Subclasses must implement :meth:`load` which returns a
    :class:`BFieldTimeSeries`. All unit conversion (nT → T, local time → UTC
    epoch seconds) happens inside the subclass, never downstream.

    Examples
    --------
    >>> source = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=500)
    >>> b = source.load(start_s=0, end_s=3600, dt_s=1.0)   # doctest: +SKIP
    >>> b.bx_T.shape                                        # doctest: +SKIP
    (3600,)
    """

    @abc.abstractmethod
    def load(
        self,
        start_s: float,
        end_s: float,
        dt_s: float = 1.0,
        station_id: str = "SYN",
    ) -> BFieldTimeSeries:
        """Load or generate magnetic-field data for the given time window.

        Parameters
        ----------
        start_s : float
            Start time in seconds since epoch (or 0 for synthetic).
        end_s : float
            End time in seconds since epoch.
        dt_s : float, optional
            Sampling interval in seconds. Default: 1.0.
        station_id : str, optional
            Station identifier. Default: ``"SYN"``.

        Returns
        -------
        BFieldTimeSeries
            Validated, SI-unit magnetic-field data.
        """

    def to_frequency_domain(
        self,
        b_data: BFieldTimeSeries,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the FFT of the horizontal B-field components.

        Concrete helper — NOT abstract — because the FFT is identical
        regardless of source type.

        Parameters
        ----------
        b_data : BFieldTimeSeries
            Time-domain magnetic-field data.

        Returns
        -------
        freqs_Hz : numpy.ndarray
            Frequency array in Hz. Shape: ``(n_freqs,)``. Positive frequencies
            only, including DC and Nyquist.
        Bx_f : numpy.ndarray
            FFT of ``bx_T`` in Tesla. Shape: ``(n_freqs,)``, complex.
        By_f : numpy.ndarray
            FFT of ``by_T`` in Tesla. Shape: ``(n_freqs,)``, complex.

        Notes
        -----
        Uses :func:`scipy.fft.rfft` for real-valued input.
        """
        from scipy.fft import rfft, rfftfreq

        n = len(b_data.time_s)
        freqs_Hz = rfftfreq(n, d=1.0 / b_data.sampling_rate_Hz)
        Bx_f = rfft(b_data.bx_T)
        By_f = rfft(b_data.by_T)
        return freqs_Hz, Bx_f, By_f
