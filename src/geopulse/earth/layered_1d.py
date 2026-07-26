r"""1-D layered-Earth surface impedance via Wait (1954) recursion.

Given a horizontally layered half-space with ``N`` layers of conductivity
``σ_n`` and thickness ``h_n`` (the terminating layer ``N`` has
``h_N → ∞``), the surface impedance seen from above at angular frequency
``ω`` is computed by the bottom-up recursion::

    γ_n = sqrt(i·ω·μ₀·σ_n)
    η_n = i·ω·μ₀ / γ_n
    Z_N = η_N
    Z_n = η_n · (Z_{n+1} + η_n · tanh(γ_n · h_n))
                / (η_n + Z_{n+1} · tanh(γ_n · h_n))     for n = N-1 down to 1
    Z_surface = Z_1

For a **uniform** half-space with only the terminating layer, the recursion
degenerates to the analytic ``Z(ω) = sqrt(i·ω·μ₀/σ)``; that identity is the
critical acceptance test for Phase 1.

References
----------
.. [1] Wait, J. R. (1954). On the relation between telluric currents and the
   Earth's magnetic field. Geophysics, 19(2), 281-289.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geopulse.constants import MU_0
from geopulse.earth.base import ConductivityLayer, EarthModel
from geopulse.earth.impedance import ScalarImpedance
from geopulse.exceptions import DataError

__all__ = ["Layered1D"]


class Layered1D(EarthModel):
    """Horizontally layered Earth model.

    Parameters
    ----------
    layers : sequence of ConductivityLayer
        Top-down list of layers. The final entry must have
        ``thickness_m == numpy.inf`` (the terminating half-space).

    Raises
    ------
    DataError
        If ``layers`` is empty, or the last layer is not the infinite
        half-space.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.earth.base import ConductivityLayer
    >>> model = Layered1D([ConductivityLayer(np.inf, 0.01)])
    >>> imp = model.compute_impedance(np.array([1e-3, 1e-2, 1e-1]))
    >>> imp.Z_values.shape
    (3,)
    """

    def __init__(self, layers: Sequence[ConductivityLayer]) -> None:
        if len(layers) == 0:
            raise DataError("Layered1D requires at least one layer")
        if not np.isinf(layers[-1].thickness_m):
            raise DataError(
                "The last (bottom) layer must be the terminating half-space "
                "with thickness_m = numpy.inf"
            )
        self.layers = list(layers)

    @property
    def tier(self) -> int:  # noqa: D401
        """Return ``1``."""
        return 1

    def compute_impedance(self, freqs_Hz: np.ndarray) -> ScalarImpedance:
        """Compute the surface impedance via Wait recursion.

        Parameters
        ----------
        freqs_Hz : numpy.ndarray
            Frequencies in Hz. Shape ``(n_freqs,)``. Must be non-negative.
            ``ω = 0`` (DC) is handled specially: the surface impedance is
            defined to be zero because a static magnetic field induces no
            electric field for a finitely conducting Earth.

        Returns
        -------
        ScalarImpedance
            Surface impedance ``Z(ω)`` in Ohms. Shape ``(n_freqs,)``.

        Raises
        ------
        DataError
            If any frequency is negative.
        """
        freqs_Hz = np.asarray(freqs_Hz, dtype=np.float64)
        if np.any(freqs_Hz < 0):
            raise DataError("All frequencies must be non-negative")

        omega = 2.0 * np.pi * freqs_Hz
        Z = np.zeros_like(omega, dtype=np.complex128)

        nonzero = omega > 0.0
        if not np.any(nonzero):
            return ScalarImpedance(freqs_Hz, Z)

        w = omega[nonzero]
        # i·ω·μ₀ appears in both γ and η; compute once per frequency.
        i_w_mu0 = 1j * w * MU_0

        # Base case: bottom half-space impedance η_N = sqrt(iωμ₀/σ_N).
        sigma_N = self.layers[-1].conductivity_Sm
        gamma_N = np.sqrt(i_w_mu0 * sigma_N)
        eta_N = i_w_mu0 / gamma_N
        Z_n = eta_N

        # Recurse upward from layer N-1 to layer 1 (Wait, 1954, eq. above).
        for layer in reversed(self.layers[:-1]):
            sigma = layer.conductivity_Sm
            h = layer.thickness_m
            gamma = np.sqrt(i_w_mu0 * sigma)
            eta = i_w_mu0 / gamma
            tanh_gh = np.tanh(gamma * h)
            Z_n = eta * (Z_n + eta * tanh_gh) / (eta + Z_n * tanh_gh)

        Z[nonzero] = Z_n
        return ScalarImpedance(freqs_Hz, Z)
