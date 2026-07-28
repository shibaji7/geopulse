"""Model-based half-cycle harmonic prediction — STUB (WP3.b).

*This module reserves the public API. The math is deferred until the
Walling & Khan (1991) or Girgis & Vedante (2012) empirical saturation
curves are transcribed into*  ``benchmarks/walling_khan_1991/`` *— see
the README there for the CSV format an implementer needs to provide.*

What this will do
-----------------
Predict the harmonic spectrum of a power transformer's excitation
current given **only a scalar DC bias** (i.e. the GIC magnitude flowing
through the neutral), without requiring a full time-domain waveform.
The physics is quasi-static core saturation: DC bias pushes the
operating point along the magnetisation curve, and one half of each AC
period drives the flux into the non-linear knee, producing a sharp
asymmetric excitation pulse rich in low-order harmonics (h2, h3, h4
dominant).

Empirical fits from field-measurement campaigns give the per-order
amplitude ratios :math:`a_k / a_1` as functions of the normalised DC
bias :math:`x = I_{dc} / I_{ex}` (DC current divided by the
transformer's rated exciting current). Walling & Khan (1991) Fig. 6
plots those curves for a typical 500 kV GSU. Different transformer
designs have different curves; the ``transformer_model`` argument
selects which fit to apply.

Contrast with :func:`~geopulse.devices.harmonics.extract_harmonics`
-------------------------------------------------------------------
* :func:`extract_harmonics` — pure signal processing. Give it a
  waveform, get back the FFT-derived spectrum. No physics assumed.
* :func:`half_cycle_harmonics` (this module) — pure model prediction
  from a DC scalar. No waveform touched. Useful when you have a NAM
  solve output (per-substation GIC in Amperes) and want an *estimated*
  harmonic spectrum without running a full nonlinear transformer
  simulation.

Both paths return :class:`~geopulse.devices.harmonics.HarmonicSpectrum`,
so downstream code (``compute_thd``, plotting, HDF5 export) treats
them identically.

Status
------
Raises :class:`~geopulse.exceptions.NotImplementedYetError` on every
call. Implementation is a **~1-day PR** once a contributor drops the
Walling & Khan curves as CSVs into ``benchmarks/walling_khan_1991/``
in the documented format, then swaps this body for a linear-interp
lookup. Test scaffolding for that PR is already in place at
``tests/unit/test_devices/test_half_cycle_harmonics.py`` — replace
the "raises" assertions with numerical checks against the paper's
Fig. 6 spot values.

References
----------
.. [1] Walling, R. A., Khan, A. N. (1991). *Characteristics of
   transformer exciting current during geomagnetic disturbances*. IEEE
   Trans. Power Delivery, 6(4), 1707-1714.
   https://doi.org/10.1109/61.97711
.. [2] Girgis, R., Vedante, K. (2012). *Effects of GIC on power
   transformers and power systems.* Proc. IEEE PES T&D Conference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from geopulse.exceptions import NotImplementedYetError

if TYPE_CHECKING:
    # HarmonicSpectrum lands in geopulse.devices.harmonics with PR #10; until
    # then mypy cannot resolve the symbol on this standalone stub branch.
    # Drop this ignore once #10 has merged.
    from geopulse.devices.harmonics import HarmonicSpectrum  # type: ignore[attr-defined]

__all__ = ["half_cycle_harmonics"]

_TransformerModelName = Literal["walling_khan_500kV_gsu", "girgis_vedante"]


def half_cycle_harmonics(
    dc_bias_A: float | np.ndarray,
    *,
    fundamental_Hz: float = 60.0,
    transformer_model: _TransformerModelName = "walling_khan_500kV_gsu",
    n_orders: int = 10,
    rated_exciting_current_A: float = 1.0,
) -> "HarmonicSpectrum":
    """Predict a transformer excitation-current harmonic spectrum from DC bias.

    .. warning::
       Not yet implemented. Raises :class:`NotImplementedYetError`. This
       function's signature is stable; the math is a v0.3 item awaiting
       transcription of the Walling & Khan (1991) or Girgis & Vedante
       (2012) empirical saturation curves into
       ``benchmarks/walling_khan_1991/``.

    Parameters
    ----------
    dc_bias_A : float or numpy.ndarray
        DC bias current through the transformer neutral, in Amperes.
        Scalar for a single spot prediction; array to vectorise across
        many substations. Sign is ignored (saturation is symmetric under
        polarity reversal).
    fundamental_Hz : float, optional
        Fundamental frequency the spectrum is expressed against.
        Default: ``60.0`` (North American power). Set to ``50.0`` for
        European / Asian grids.
    transformer_model : {"walling_khan_500kV_gsu", "girgis_vedante"}, optional
        Which set of empirical curves to apply. Default:
        ``"walling_khan_500kV_gsu"`` — representative of a large
        single-phase GSU. Additional models can be added by dropping a
        matching CSV into ``benchmarks/walling_khan_1991/`` and
        registering it here.
    n_orders : int, optional
        Number of harmonic orders returned, including the fundamental.
        Default: 10. Higher orders beyond ~10 are typically below noise
        for typical bias levels.
    rated_exciting_current_A : float, optional
        Rated no-load exciting current :math:`I_{ex}` of the
        transformer, in Amperes. Used to normalise the DC bias into the
        empirical curves' :math:`x = I_{dc} / I_{ex}` argument. Default:
        ``1.0`` (i.e. the caller has already normalised). Real values
        for GSUs are typically 0.1 - 1 % of rated primary current.

    Returns
    -------
    HarmonicSpectrum
        RMS amplitudes at orders ``1 .. n_orders``, with ``dc_A =
        dc_bias_A`` echoed back. The fundamental amplitude reflects the
        transformer's normal AC exciting current; higher orders reflect
        the half-cycle saturation contribution.

    Raises
    ------
    NotImplementedYetError
        Always — this is a WP3.b stub. When implemented, will also raise
        :class:`~geopulse.exceptions.DataError` on unknown
        ``transformer_model``, non-finite ``dc_bias_A``, or
        ``n_orders <= 0``.

    Notes
    -----
    * The empirical curves have limited validity outside the calibrated
      DC-bias range (roughly ``0 <= x <= 2 * I_ex``). Extrapolation
      beyond that saturates and will be capped by the implementation
      with a warning.
    * For work needing higher fidelity than the empirical fits provide,
      use :class:`~geopulse.devices.transformer.TransformerModel` to
      simulate a full time-domain excitation current, then pass it
      through :func:`~geopulse.devices.harmonics.extract_harmonics`.

    Examples
    --------
    >>> from geopulse.devices.half_cycle_harmonics import half_cycle_harmonics
    >>> half_cycle_harmonics(50.0)                       # doctest: +SKIP
    HarmonicSpectrum(fundamental_Hz=60.0, orders=array([1, 2, 3, ...]), ...)
    """
    raise NotImplementedYetError(
        "half_cycle_harmonics",
        "WP3.b — needs Walling & Khan (1991) or Girgis & Vedante (2012) "
        "empirical curves. See benchmarks/walling_khan_1991/README.md for "
        "the CSV format an implementer should provide.",
    )
