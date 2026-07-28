"""Harmonic-content extraction from a discrete current or voltage waveform.

This module handles the FFT-based side of the harmonics story: take a
time-domain waveform (e.g. a transformer excitation current under GIC
bias), return the amplitude of each integer harmonic of a chosen
fundamental frequency. The transformation is pure signal processing;
the physics — how a DC bias produces those harmonics through half-cycle
core saturation — is a *separate* problem addressed by the transformer
model in :mod:`geopulse.devices.transformer` and by the (deferred)
``half_cycle_harmonics`` predictor.

The extracted :class:`HarmonicSpectrum` is the canonical input to
:func:`geopulse.metrics.thd.compute_thd`.

Model-based half-cycle harmonic prediction from a DC bias alone (without
a waveform) is a v0.3 item — it needs the empirical Walling & Khan
(1991) or Girgis & Vedante (2012) transformer saturation curves, which
this repository does not yet ship.

References
----------
.. [1] Walling, R. A., Khan, A. N. (1991). *Characteristics of
   transformer exciting current during geomagnetic disturbances*. IEEE
   Trans. Power Delivery, 6(4), 1707-1714.
   https://doi.org/10.1109/61.97711
.. [2] Girgis, R., Vedante, K. (2012). *Effects of GIC on power
   transformers and power systems.* Proc. IEEE PES T&D Conference.
.. [3] IEEE Std 519-2014. *IEEE Recommended Practice and Requirements
   for Harmonic Control in Electric Power Systems.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, rfftfreq

from geopulse.exceptions import DataError, ShapeMismatchError

__all__ = ["HarmonicSpectrum", "extract_harmonics"]


@dataclass(frozen=True)
class HarmonicSpectrum:
    """Amplitude spectrum sampled at integer multiples of a fundamental.

    Attributes
    ----------
    fundamental_Hz : float
        The fundamental frequency the spectrum is expressed against
        (typically 50 or 60 Hz for power systems).
    orders : numpy.ndarray of int
        Integer harmonic orders, shape ``(n_orders,)``. Includes the
        fundamental as ``orders[0] = 1``. Higher orders in ascending
        integer sequence.
    amplitudes_A : numpy.ndarray of float
        RMS current amplitude at each order, in Amperes, shape
        ``(n_orders,)``. RMS is chosen (not peak) so that
        :math:`I_{total,\\,rms}^2 = \\sum_k a_k^2`.
    dc_A : float
        Zero-frequency (DC) component of the waveform in Amperes. Kept
        separate from ``amplitudes_A`` because the IEEE 519-2014 THD
        definition treats it separately (THD is defined against the
        fundamental, not the DC).
    """

    fundamental_Hz: float
    orders: np.ndarray
    amplitudes_A: np.ndarray
    dc_A: float


def extract_harmonics(
    time_s: np.ndarray,
    current_A: np.ndarray,
    fundamental_Hz: float,
    *,
    n_harmonics: int = 40,
) -> HarmonicSpectrum:
    """FFT extraction of harmonic amplitudes from a current waveform.

    Uses :func:`scipy.fft.rfft` and picks the frequency bin closest to
    each integer multiple of ``fundamental_Hz``. RMS-normalised so the
    sum of squared amplitudes equals the AC (mean-removed) RMS squared
    when the input is band-limited to those tones.

    Parameters
    ----------
    time_s : numpy.ndarray
        Uniformly sampled time base, seconds. Shape ``(n,)``.
    current_A : numpy.ndarray
        Current waveform in Amperes. Shape ``(n,)`` matching
        ``time_s``. Any dtype; internally cast to float64.
    fundamental_Hz : float
        Fundamental frequency of interest (e.g. 60.0). Must be strictly
        positive.
    n_harmonics : int, optional
        Number of harmonic orders to extract, including the fundamental.
        Default: 40 (fundamental + 39 higher orders).

    Returns
    -------
    HarmonicSpectrum
        RMS amplitudes at orders ``1, 2, ..., n_harmonics``, plus the DC
        component.

    Raises
    ------
    ShapeMismatchError
        If ``time_s`` and ``current_A`` shapes disagree, or aren't 1-D.
    DataError
        If ``fundamental_Hz`` is not strictly positive, if ``n_harmonics``
        is not a positive int, if the sample spacing is not uniform, or
        if the requested top harmonic exceeds Nyquist.

    Notes
    -----
    * The bin-picking method is deliberately simple. If your fundamental
      is not exactly a multiple of the frequency resolution
      ``1 / (n * dt)``, spectral leakage will bleed into neighbouring
      bins and the reported amplitudes will be biased low. For most GIC
      use cases (waveforms much longer than one cycle) this bias is
      < 1 %. Callers who need exact tone recovery for a non-integer
      number of cycles should window the waveform or interpolate the
      spectrum externally.
    * RMS convention: for a pure cosine of peak amplitude ``A``, the
      returned amplitude is ``A / sqrt(2)``.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.devices.harmonics import extract_harmonics
    >>> fs = 6000.0
    >>> t = np.arange(6000) / fs                          # 1 second
    >>> x = 100.0 * np.sin(2 * np.pi * 60.0 * t)          # 100 A peak, 60 Hz
    >>> s = extract_harmonics(t, x, fundamental_Hz=60.0, n_harmonics=5)
    >>> round(float(s.amplitudes_A[0]), 2)                # RMS = 100/sqrt(2)
    70.71
    >>> bool(np.all(s.amplitudes_A[1:] < 1e-6))
    True
    """
    t = np.asarray(time_s, dtype=np.float64)
    x = np.asarray(current_A, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1:
        raise ShapeMismatchError(
            f"extract_harmonics expects 1-D arrays; got time.shape={t.shape}, "
            f"current.shape={x.shape}"
        )
    if t.shape != x.shape:
        raise ShapeMismatchError(f"time_s shape {t.shape} does not match current_A shape {x.shape}")
    if not np.isfinite(fundamental_Hz) or fundamental_Hz <= 0.0:
        raise DataError(f"fundamental_Hz must be positive-finite, got {fundamental_Hz!r}")
    if not isinstance(n_harmonics, int) or n_harmonics <= 0:
        raise DataError(f"n_harmonics must be a positive int, got {n_harmonics!r}")

    n = t.size
    if n < 2:
        raise DataError("extract_harmonics needs at least 2 samples")

    dt = np.diff(t)
    dt0 = float(dt[0])
    if dt0 <= 0.0 or not np.allclose(dt, dt0, rtol=1e-6, atol=1e-12):
        raise DataError("extract_harmonics requires a uniformly-sampled time base")

    fs = 1.0 / dt0
    nyq = 0.5 * fs
    top_Hz = n_harmonics * fundamental_Hz
    if top_Hz > nyq:
        raise DataError(
            f"top requested harmonic {n_harmonics}*{fundamental_Hz} = {top_Hz} Hz "
            f"exceeds Nyquist {nyq} Hz — reduce n_harmonics or resample"
        )

    X = rfft(x)
    freqs = rfftfreq(n, d=dt0)
    peak_norm = 2.0 / n

    orders = np.arange(1, n_harmonics + 1, dtype=np.int64)
    amps = np.empty(n_harmonics, dtype=np.float64)
    for i, k in enumerate(orders):
        target = k * fundamental_Hz
        bin_idx = int(np.argmin(np.abs(freqs - target)))
        amps[i] = float(np.abs(X[bin_idx])) * peak_norm / np.sqrt(2.0)

    dc_A = float(np.real(X[0])) / n

    return HarmonicSpectrum(
        fundamental_Hz=float(fundamental_Hz),
        orders=orders,
        amplitudes_A=amps,
        dc_A=dc_A,
    )
