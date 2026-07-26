r"""2-D structured Earth models.

Two flavours live here:

* :class:`CoastalCorrection2D` — a semi-analytic **coast-effect** wrapper
  that interpolates between a land :class:`~geopulse.earth.layered_1d.Layered1D`
  and an ocean :class:`~geopulse.earth.layered_1d.Layered1D` model based on
  distance from the coast, producing a
  :class:`~geopulse.earth.impedance.TensorImpedance`. Fast, closed-form,
  captures the classic ocean/continent contrast that dominates coastal
  GIC risk. This is what operational GIC codes typically use in place of
  a true 2-D finite-difference solve.
* :class:`Structured2D` — placeholder for a proper finite-difference /
  finite-element 2-D MT solver (TE/TM modes on a discretised
  cross-section). Deferred; raises :class:`NotImplementedYetError`.

Coordinate convention (both classes)
------------------------------------
The coast is aligned with the ``y`` axis (east). Distance from the coast
``d_m`` is measured along ``x`` (north), **positive inland**, **negative
seaward**. Under this frame the impedance tensor is anti-diagonal:

.. math::

    \tilde{Z}(ω) = \begin{bmatrix} 0 & Z_\mathrm{TE}(ω) \\ -Z_\mathrm{TM}(ω) & 0 \end{bmatrix}

where TE (E-parallel-to-coast) and TM (E-perpendicular-to-coast) modes
decouple and can be computed independently.

For :class:`CoastalCorrection2D` the smooth interpolation used is::

    Z_TE(d, ω) = Z_land(ω) + [Z_ocean(ω) − Z_land(ω)] · f_TE(d, ω)
    Z_TM(d, ω) = Z_land(ω) + [Z_ocean(ω) − Z_land(ω)] · f_TM(d, ω)

with ``f_TE(d, ω) = ½ [1 − tanh(d / δ(ω))]``, ``f_TM`` similar but with
a shorter decay length (TM is more strongly perturbed near a
conductivity contrast per Weaver 1994 and Boteler-Pirjola 1998), and
``δ(ω) = √(2 / (ω μ₀ σ_eff))`` the skin depth at the geometric-mean
surface conductivity ``σ_eff = √(σ_land · σ_ocean)``. Limits check::

    d → +∞   ⇒   Z_TE, Z_TM → Z_land
    d → −∞   ⇒   Z_TE, Z_TM → Z_ocean
    d = 0    ⇒   Z_TE, Z_TM = (Z_land + Z_ocean) / 2

This is **not** a substitute for a full 2-D FD MT computation at
percent-level accuracy — the goal is a defensible, closed-form correction
that captures the right qualitative behaviour and asymptotic limits, and
that returns a proper :class:`TensorImpedance` so downstream E-field
computations exercise the tensor code path.

References
----------
.. [1] Weaver, J. T. (1994). Mathematical Methods for Geo-Electromagnetic
   Induction. Research Studies Press.
.. [2] Boteler, D. H., & Pirjola, R. J. (1998). The complex-image method
   for calculating the magnetic and electric fields at the surface of the
   Earth by the auroral electrojet. Geophysical Journal International,
   132(1), 31-40.
"""

from __future__ import annotations

import numpy as np

from geopulse.constants import MU_0
from geopulse.earth.base import EarthModel
from geopulse.earth.impedance import Impedance, TensorImpedance
from geopulse.earth.layered_1d import Layered1D
from geopulse.exceptions import DataError, NotImplementedYetError

__all__ = ["CoastalCorrection2D", "Structured2D"]


class CoastalCorrection2D(EarthModel):
    """Semi-analytic coast-effect 2-D Earth model.

    Interpolates between two 1-D reference models (land and ocean) via a
    tanh-shaped coupling function of distance from the coast, producing a
    :class:`~geopulse.earth.impedance.TensorImpedance` at each frequency.

    Parameters
    ----------
    land_model : Layered1D
        Layered 1-D model for the land side (well inland). Provides the
        far-field ``d → +∞`` asymptote.
    ocean_model : Layered1D
        Layered 1-D model for the ocean side (well offshore). Typically a
        thin high-conductivity seawater layer over the same deep
        conductivity structure. Provides the ``d → −∞`` asymptote.
    distance_from_coast_m : float
        Distance of the observation site from the coastline, in metres.
        Positive inland, negative seaward. ``0`` sits exactly at the coast.
    tm_decay_ratio : float, optional
        Ratio of TM to TE decay length ``L_TM / L_TE``. Smaller → TM
        transitions more sharply than TE. Default 0.7 matches the
        Boteler-Pirjola (1998) observation that TM is more perturbed than
        TE near the interface.

    Notes
    -----
    ``tier == 2``. Coordinate frame: coast along ``y`` (east); ``x``
    (north) points inland. The tensor is anti-diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.earth.base import ConductivityLayer
    >>> from geopulse.earth.layered_1d import Layered1D
    >>> land = Layered1D([ConductivityLayer(np.inf, 0.001)])
    >>> ocean = Layered1D([
    ...     ConductivityLayer(1_000.0, 3.3),
    ...     ConductivityLayer(np.inf, 0.001),
    ... ])
    >>> coast = CoastalCorrection2D(land, ocean, distance_from_coast_m=0.0)
    >>> imp = coast.compute_impedance(np.array([1e-3, 1e-2, 1e-1]))
    >>> imp.rank
    2
    """

    def __init__(
        self,
        land_model: Layered1D,
        ocean_model: Layered1D,
        distance_from_coast_m: float,
        tm_decay_ratio: float = 0.7,
    ) -> None:
        if not isinstance(land_model, Layered1D):
            raise DataError("land_model must be a Layered1D instance")
        if not isinstance(ocean_model, Layered1D):
            raise DataError("ocean_model must be a Layered1D instance")
        if tm_decay_ratio <= 0:
            raise DataError(f"tm_decay_ratio must be positive, got {tm_decay_ratio}")
        self.land_model = land_model
        self.ocean_model = ocean_model
        self.distance_from_coast_m = float(distance_from_coast_m)
        self.tm_decay_ratio = float(tm_decay_ratio)

    @property
    def tier(self) -> int:  # noqa: D401
        """Return ``2``."""
        return 2

    def compute_impedance(self, freqs_Hz: np.ndarray) -> Impedance:
        """Compute the surface impedance tensor per the coast-effect model.

        Parameters
        ----------
        freqs_Hz : numpy.ndarray
            Frequencies in Hz. Shape ``(n_freqs,)``. Must be non-negative;
            DC returns a zero tensor (no plane-wave response at ω = 0).

        Returns
        -------
        TensorImpedance
            Anti-diagonal 2×2 impedance tensor per frequency.
        """
        freqs_Hz = np.asarray(freqs_Hz, dtype=np.float64)
        if np.any(freqs_Hz < 0):
            raise DataError("All frequencies must be non-negative")

        Z_land = self.land_model.compute_impedance(freqs_Hz).Z_values
        Z_ocean = self.ocean_model.compute_impedance(freqs_Hz).Z_values

        omega = 2.0 * np.pi * freqs_Hz
        sigma_land = self.land_model.layers[0].conductivity_Sm
        sigma_ocean = self.ocean_model.layers[0].conductivity_Sm
        sigma_eff = float(np.sqrt(sigma_land * sigma_ocean))

        # Skin depth at each frequency using the geometric-mean surface σ;
        # DC (ω=0) → infinite skin depth → f = 0.5 (undefined coupling but
        # the DC E-field is anyway defined to be zero).
        with np.errstate(divide="ignore"):
            delta_m = np.where(
                omega > 0.0,
                np.sqrt(2.0 / (omega * MU_0 * sigma_eff)),
                np.inf,
            )

        d = self.distance_from_coast_m
        f_TE = 0.5 * (1.0 - np.tanh(d / delta_m))
        f_TM = 0.5 * (1.0 - np.tanh(d / (self.tm_decay_ratio * delta_m)))

        Z_TE = Z_land + (Z_ocean - Z_land) * f_TE
        Z_TM = Z_land + (Z_ocean - Z_land) * f_TM

        n = freqs_Hz.size
        Z_tensor = np.zeros((n, 2, 2), dtype=np.complex128)
        Z_tensor[:, 0, 1] = Z_TE
        Z_tensor[:, 1, 0] = -Z_TM

        return TensorImpedance(freqs_Hz, Z_tensor)


class Structured2D(EarthModel):
    """True 2-D finite-difference MT solver — STUB (deferred WP).

    Discretises a 2-D cross-section, solves TE and TM Maxwell modes on
    the grid with proper boundary conditions, and extracts the impedance
    tensor at surface locations. See :class:`CoastalCorrection2D` for the
    pragmatic closed-form alternative that is available now.
    """

    @property
    def tier(self) -> int:  # noqa: D401
        """Return ``2``."""
        return 2

    def compute_impedance(self, freqs_Hz: np.ndarray) -> Impedance:
        """Not yet implemented — use :class:`CoastalCorrection2D` instead."""
        raise NotImplementedYetError("Structured2D (full 2-D FD MT solver)", "WP-future")
