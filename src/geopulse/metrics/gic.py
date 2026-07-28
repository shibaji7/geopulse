"""Descriptive statistics for GIC / branch-current / pipe-to-soil signals.

These are the plain summary numbers reports and regulatory filings ask
for: peak, RMS, percentiles of the signal magnitude, and total time
spent above chosen operational thresholds. No filtering, no smoothing,
no unit conversion — callers hand over a numpy array in whatever unit
is meaningful (Amperes for GIC, Volts for pipe-to-soil) and get back
the same-unit statistics.

Rationale
---------
The percentile choice is deliberately opinionated:

* ``p90`` matches the **NERC GIC-90** convention used in TPL-007-4 for
  transformer thermal screening.
* ``p50`` (median of ``|x|``) is the natural robust centre.
* ``p99`` gives the "extreme but not tail" number that engineers reach
  for when a single peak is suspicious.

Exceedance curves follow the empirical estimator of Pulkkinen et al.
(2012) — ``P(|X| >= x) = fraction of samples >= x``. No parametric fit
is imposed; users who want a Weibull, GPD, or Tsallis fit should take
the raw curve out and fit it themselves.

References
----------
.. [1] North American Electric Reliability Corporation. **TPL-007-4:
   Transmission System Planned Performance for Geomagnetic
   Disturbance Events.** In-force revision.
.. [2] Pulkkinen, A., Bernabeu, E., Eichner, J., Beggan, C.,
   Thomson, A. W. P. (2012). *Generation of 100-year geoelectric
   hazard maps for the mid-Atlantic United States*. Space Weather,
   10(4), S04003. https://doi.org/10.1029/2011SW000750
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from geopulse.exceptions import DataError, ShapeMismatchError

__all__ = ["GICStats", "exceedance_curve", "summary_stats"]

_DEFAULT_THRESHOLDS_A = (5.0, 10.0, 50.0, 100.0)


@dataclass(frozen=True)
class GICStats:
    """Descriptive statistics of a scalar time series.

    All fields are in the same unit as the input array (typically A for
    GIC, V for pipe-to-soil potential).

    Attributes
    ----------
    peak_abs : float
        ``max(|x|)`` — the single-sample worst-case magnitude.
    rms : float
        ``sqrt(mean(x**2))`` — root-mean-square amplitude, including
        sign (so a mean-zero storm signal's RMS is the AC magnitude).
    p50, p90, p99 : float
        Percentiles of ``|x|``. ``p90`` is the NERC GIC-90 convention;
        ``p50`` is the robust centre; ``p99`` is the extreme-but-not-tail
        anchor.
    n_samples : int
        Number of finite input samples used to compute the statistics.
        NaNs and infs are discarded before the reduction.
    duration_above : dict[float, float]
        Mapping ``{threshold: seconds_above}``. For each threshold in
        the caller-supplied list, the total wall-clock time (seconds)
        the signal magnitude ``|x|`` was strictly greater than the
        threshold. Uses ``dt_s`` * count.
    """

    peak_abs: float
    rms: float
    p50: float
    p90: float
    p99: float
    n_samples: int
    duration_above: dict[float, float] = field(default_factory=dict)


def summary_stats(
    x: np.ndarray,
    *,
    dt_s: float,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS_A,
) -> GICStats:
    """Compute descriptive statistics for a scalar signal.

    Parameters
    ----------
    x : numpy.ndarray
        Time series of the signal. Any real dtype; NaN / inf samples are
        dropped before the reduction (with the effective sample count
        reported on the returned :class:`GICStats`).
    dt_s : float
        Sampling interval in seconds. Must be strictly positive. Used
        only to convert threshold-exceedance sample counts into wall
        clock time.
    thresholds : Sequence[float], optional
        Absolute-value thresholds (same unit as ``x``). Each entry
        produces one entry in :attr:`GICStats.duration_above`. Duplicate
        values are allowed but collapse in the returned dict. Default:
        ``(5, 10, 50, 100)`` — sensible GIC screening steps in Amperes.

    Returns
    -------
    GICStats
        Peak, RMS, three percentiles, sample count, and per-threshold
        durations.

    Raises
    ------
    ShapeMismatchError
        If ``x`` is not one-dimensional.
    DataError
        If ``dt_s`` is not strictly positive, or if ``x`` contains no
        finite samples.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.metrics.gic import summary_stats
    >>> t = np.linspace(0, 60, 601)
    >>> x = 10.0 * np.sin(2 * np.pi * t / 10.0)      # 10 A sine, 10 s period
    >>> s = summary_stats(x, dt_s=0.1, thresholds=(5.0, 9.0))
    >>> round(s.peak_abs, 3)
    10.0
    >>> round(s.rms, 3)
    7.065
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ShapeMismatchError(f"summary_stats expects 1-D input, got shape {arr.shape}")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise DataError(f"dt_s must be positive-finite, got {dt_s!r}")

    finite = np.isfinite(arr)
    if not finite.any():
        raise DataError("summary_stats: input has no finite samples")
    clean = arr[finite].astype(np.float64, copy=False)
    abs_clean = np.abs(clean)

    peak_abs = float(abs_clean.max())
    rms = float(np.sqrt(np.mean(clean * clean)))
    p50, p90, p99 = (float(v) for v in np.percentile(abs_clean, [50.0, 90.0, 99.0]))

    duration: dict[float, float] = {}
    for thr in thresholds:
        t = float(thr)
        n_above = int(np.count_nonzero(abs_clean > t))
        duration[t] = n_above * float(dt_s)

    return GICStats(
        peak_abs=peak_abs,
        rms=rms,
        p50=p50,
        p90=p90,
        p99=p99,
        n_samples=int(clean.size),
        duration_above=duration,
    )


def exceedance_curve(
    x: np.ndarray,
    *,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical exceedance curve — ``P(|X| >= x)`` on a log/linear grid.

    Follows the estimator in Pulkkinen et al. (2012): sort ``|x|``,
    evaluate the survival function at ``n_points`` values spanning the
    non-zero range on a log grid (or linear when the range is narrow).

    Parameters
    ----------
    x : numpy.ndarray
        1-D signal. NaN / inf samples are dropped.
    n_points : int, optional
        Number of points on the returned curve. Default: 100.

    Returns
    -------
    levels : numpy.ndarray
        Threshold values (same unit as ``x``), sorted ascending. Shape
        ``(n_points,)``.
    prob : numpy.ndarray
        ``P(|X| >= levels[i])`` — empirical survival probability. Values
        in ``(0, 1]`` and monotonically non-increasing. Shape
        ``(n_points,)``.

    Raises
    ------
    ShapeMismatchError
        If ``x`` is not one-dimensional.
    DataError
        If ``x`` has no finite non-zero samples (a degenerate curve),
        or if ``n_points`` is not a positive integer.

    Notes
    -----
    * The smallest returned level is the smallest non-zero ``|x|`` in
      the signal, not zero — a zero level would sit at ``P = 1``
      trivially and add no information.
    * The curve is empirical; no parametric fit is imposed. Downstream
      Weibull / GPD fits should read this curve and fit externally.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.metrics.gic import exceedance_curve
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(scale=1.0, size=10_000)
    >>> levels, prob = exceedance_curve(x, n_points=50)
    >>> bool(prob[0] > prob[-1] and prob[-1] > 0.0)
    True
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ShapeMismatchError(f"exceedance_curve expects 1-D input, got shape {arr.shape}")
    if not isinstance(n_points, int) or n_points <= 0:
        raise DataError(f"n_points must be a positive int, got {n_points!r}")

    clean = arr[np.isfinite(arr)]
    abs_clean = np.abs(clean)
    nz = abs_clean[abs_clean > 0.0]
    if nz.size == 0:
        raise DataError("exceedance_curve: no finite non-zero samples")

    lo = float(nz.min())
    hi = float(nz.max())
    # Log-spaced when the dynamic range is wide; linear when the signal
    # is confined to a narrow band (avoids log-of-narrow-range warts).
    if hi / lo > 10.0:
        levels = np.geomspace(lo, hi, num=n_points)
    else:
        levels = np.linspace(lo, hi, num=n_points)

    total = float(abs_clean.size)
    prob = np.array(
        [float(np.count_nonzero(abs_clean >= level)) / total for level in levels],
        dtype=np.float64,
    )
    return levels, prob
