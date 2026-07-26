"""Plane-wave geoelectric field: ``E(ω) = Z(ω) · B(ω) / μ₀``.

Delegates to :meth:`Impedance.apply` so the caller does not know or care
whether ``Z`` is scalar (1-D), tensor (2-D), or kernel (3-D). The IFFT back
to the time domain is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np

from geopulse.earth.impedance import Impedance

__all__ = ["compute_efield_planewave"]


def compute_efield_planewave(
    freqs_Hz: np.ndarray,
    Bx_f: np.ndarray,
    By_f: np.ndarray,
    impedance: Impedance,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frequency-domain geoelectric field via the plane-wave relation.

    For a 1-D scalar impedance this reduces to::

        E_x(ω) =  Z(ω) / μ₀ · B_y(ω)
        E_y(ω) = -Z(ω) / μ₀ · B_x(ω)

    Parameters
    ----------
    freqs_Hz : numpy.ndarray
        Frequency array in Hz. Shape ``(n_freqs,)``. Accepted for interface
        symmetry; the impedance already carries its own frequency grid.
    Bx_f, By_f : numpy.ndarray
        FFT of the horizontal B-field components in Tesla. Shape
        ``(n_freqs,)``.
    impedance : Impedance
        Any :class:`~geopulse.earth.impedance.Impedance` subclass; the type
        is not inspected here.

    Returns
    -------
    Ex_f, Ey_f : numpy.ndarray
        FFT of the horizontal E-field components in V/m. Shape matches the
        impedance subclass semantics (``(n_freqs,)`` for scalar).

    Notes
    -----
    Standard magnetotelluric plane-wave relation; see Boteler (2014).
    """
    del freqs_Hz  # unused — impedance already knows its own frequency grid
    return impedance.apply(Bx_f, By_f)
