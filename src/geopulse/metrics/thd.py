"""Total Harmonic Distortion — IEEE 519-2014 definition.

Given a :class:`~geopulse.devices.harmonics.HarmonicSpectrum` (produced
by :func:`~geopulse.devices.harmonics.extract_harmonics` from a
transformer excitation waveform), the total harmonic distortion is

.. math::

    \\mathrm{THD} \\;=\\; \\frac{\\sqrt{\\sum_{k=2}^{N} a_k^2}}{a_1}

where :math:`a_k` is the RMS amplitude at order :math:`k`. The DC
component is deliberately excluded (IEEE 519-2014 §3.1); to get the
"distortion factor" definition that folds DC in as well, pass
``include_dc=True``.

References
----------
.. [1] IEEE Std 519-2014. *IEEE Recommended Practice and Requirements
   for Harmonic Control in Electric Power Systems.* Definition of THD.
"""

from __future__ import annotations

import numpy as np

from geopulse.devices.harmonics import HarmonicSpectrum
from geopulse.exceptions import DataError

__all__ = ["compute_thd"]


def compute_thd(spectrum: HarmonicSpectrum, *, include_dc: bool = False) -> float:
    """IEEE 519-2014 total harmonic distortion of a spectrum.

    Parameters
    ----------
    spectrum : HarmonicSpectrum
        Harmonic spectrum, typically produced by
        :func:`geopulse.devices.harmonics.extract_harmonics`. The
        fundamental must be present as ``spectrum.orders[0] == 1``.
    include_dc : bool, optional
        When ``True``, include the DC (zero-frequency) component in the
        numerator sum. This is the "distortion factor" convention; the
        default matches IEEE 519-2014 THD, which excludes DC. Default:
        ``False``.

    Returns
    -------
    float
        Dimensionless ratio in :math:`[0, \\infty)`. Multiply by 100 to
        report as a percentage.

    Raises
    ------
    DataError
        If ``spectrum.orders[0] != 1``, or if the fundamental amplitude
        is zero (THD is undefined against a zero fundamental).

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.devices.harmonics import HarmonicSpectrum
    >>> from geopulse.metrics.thd import compute_thd
    >>> s = HarmonicSpectrum(
    ...     fundamental_Hz=60.0,
    ...     orders=np.array([1, 2, 3]),
    ...     amplitudes_A=np.array([100.0, 3.0, 4.0]),  # 3% h2, 4% h3
    ...     dc_A=0.0,
    ... )
    >>> round(compute_thd(s) * 100, 2)                  # sqrt(3^2 + 4^2) / 100
    5.0
    """
    orders = np.asarray(spectrum.orders)
    amps = np.asarray(spectrum.amplitudes_A, dtype=np.float64)
    if orders.size == 0 or int(orders[0]) != 1:
        raise DataError("compute_thd: spectrum must include the fundamental as orders[0] = 1")
    a1 = float(amps[0])
    if a1 == 0.0:
        raise DataError("compute_thd: fundamental amplitude is zero — THD undefined")

    higher = amps[1:]
    if include_dc:
        higher = np.concatenate([[float(spectrum.dc_A)], higher])
    return float(np.sqrt(np.sum(higher * higher))) / a1
