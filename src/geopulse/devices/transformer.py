r"""Transformer thermal response — Mate et al. (2021) top-oil + hot-spot model.

Bilinear-discretised top-oil temperature dynamics + effective-GIC-driven
hot-spot rise, applied per-transformer. The relations follow Mate, Overbye,
Weiss & Trakas (2021) Sec. II-D, eqns 5–9:

.. math::

    \delta_u^t     &= \delta_r\, k(t)^2                     \quad (5)     \\
    \tau_e \frac{d\delta_e}{dt} + \delta_e &= \delta_u      \quad (6)     \\
    \zeta &= 2\tau_e / \Delta                                              \\
    \delta_e^t &= \frac{\delta_u^t + \delta_u^{t-1}}{1 + \zeta}
                  - \frac{1 - \zeta}{1 + \zeta}\, \delta_e^{t-1}
                                                             \quad (7,\ \mathrm{Tustin}) \\
    \rho_h^t &= \rho + \delta_e^t + \eta_e^t                 \quad (8)     \\
    \eta_e^t &= R_e\,|I_e^t|                                 \quad (9)

with

* ``ρ`` — ambient temperature (°C)
* ``δ_r`` — top-oil rise at rated load (°C)
* ``τ_e`` — top-oil thermal time constant (min)
* ``k(t)`` — fractional apparent-power loading (0..1+)
* ``δ_e^t`` — dynamic top-oil rise above ambient (°C)
* ``|I_e|`` — effective GIC magnitude through the winding (A)
* ``R_e`` — hot-spot rise coefficient (°C / A)

The Tustin discretisation of eqn 6 is unconditionally stable and gives
eqn 7 exactly.

References
----------
.. [1] Mate, A., Overbye, T. J., Weiss, K. R., & Trakas, D. N. (2021).
   Modeling and Simulation of Cascading Contingencies Considering
   Geomagnetic Disturbances. arXiv:2101.05042.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.exceptions import DataError

__all__ = ["TransformerModel", "ThermalParams"]


@dataclass(frozen=True)
class ThermalParams:
    """Transformer thermal parameters (Mate 2021 Table IV).

    Defaults are the "row 1" values from that table (single-phase GSU),
    which reproduce the paper's Fig 5 hot-spot curves.

    Attributes
    ----------
    ambient_C : float
        Ambient temperature ``ρ`` in °C. Default: 25.
    to_rated_rise_C : float
        Top-oil rise at rated load ``δ_r`` in °C. Default: 75.
    to_time_constant_min : float
        Top-oil thermal time constant ``τ_e`` in minutes. Default: 71.
    hs_coeff_C_per_A : float
        Hot-spot rise coefficient ``R_e`` in °C/A. Default: 0.63.
    hs_avg_limit_C : float
        8-hour hot-spot average limit in °C. Default: 240.
    hs_inst_limit_C : float
        1-hour hot-spot instantaneous limit in °C. Default: 280.
    """

    ambient_C: float = 25.0
    to_rated_rise_C: float = 75.0
    to_time_constant_min: float = 71.0
    hs_coeff_C_per_A: float = 0.63
    hs_avg_limit_C: float = 240.0
    hs_inst_limit_C: float = 280.0


class TransformerModel(DeviceModel):
    """Bilinear top-oil + hot-spot temperature model for a power transformer.

    Parameters
    ----------
    params : ThermalParams, optional
        Thermal parameters. Defaults to :class:`ThermalParams` factory
        defaults (Mate 2021 Table IV row 1).
    k_load : float or array_like, optional
        Fractional apparent-power loading ``k(t)`` (0..1+). Scalar means
        constant loading; array must match ``time_s`` length passed to
        :meth:`inject_gic`. Default: 0.63 — the value Mate 2021 uses to
        reproduce Fig 5's ~55 °C peak top-oil at 25 °C ambient.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.devices.transformer import TransformerModel
    >>> t_s = np.arange(0.0, 3600.0, 60.0)
    >>> gic_A = 20.0 * np.sin(2 * np.pi * t_s / 3600.0)
    >>> resp = TransformerModel().inject_gic(t_s, gic_A)
    >>> resp.hotspot_C.shape == t_s.shape
    True
    """

    def __init__(
        self,
        params: ThermalParams | None = None,
        k_load: float | np.ndarray = 0.63,
    ) -> None:
        self.params = params if params is not None else ThermalParams()
        self.k_load = k_load

    def inject_gic(
        self,
        time_s: np.ndarray,
        gic_A: np.ndarray,
        ac_voltage_V: float = 0.0,
        ac_frequency_Hz: float = 50.0,
    ) -> DeviceResponse:
        """Compute the thermal response for a GIC waveform.

        Parameters
        ----------
        time_s : numpy.ndarray
            Uniformly-sampled time array in seconds. Shape ``(n_times,)``.
        gic_A : numpy.ndarray
            GIC through the transformer winding in Amperes. Shape
            ``(n_times,)``. Only ``|gic_A|`` enters the hot-spot rise, so
            sign is irrelevant.
        ac_voltage_V, ac_frequency_Hz : float, optional
            Accepted for ABC symmetry — the pure thermal response does not
            depend on either.

        Returns
        -------
        DeviceResponse
            ``time_s``, ``response_current_A = |gic_A|``, ``top_oil_C``,
            ``hotspot_C``. ``thd`` is ``NaN`` (harmonic content lives in a
            separate device model); ``harmonics`` is empty.

        Raises
        ------
        DataError
            If ``time_s`` is not uniformly sampled or shapes don't match.

        Notes
        -----
        The time step ``Δ`` used for the Tustin update is the **median** of
        ``diff(time_s)`` converted to minutes to match the thermal
        time-constant units. Non-uniform time series are rejected.
        """
        del ac_voltage_V, ac_frequency_Hz  # unused; kept for ABC symmetry

        time_s = np.asarray(time_s, dtype=np.float64)
        gic_A = np.asarray(gic_A, dtype=np.float64)
        if time_s.ndim != 1 or time_s.shape != gic_A.shape:
            raise DataError(f"time_s {time_s.shape} and gic_A {gic_A.shape} shapes must match")
        n = time_s.size
        if n < 2:
            raise DataError("Need at least 2 samples to run the thermal solver")

        dt_s = np.diff(time_s)
        dt_med_s = float(np.median(dt_s))
        if not np.allclose(dt_s, dt_med_s, rtol=1e-3):
            raise DataError(
                "Thermal model requires uniformly-sampled time_s; found non-uniform spacing"
            )
        dt_min = dt_med_s / 60.0

        k = np.broadcast_to(np.asarray(self.k_load, dtype=np.float64), (n,)).copy()

        p = self.params
        # Steady-state top-oil rise (Mate eqn 5).
        delta_u_C = p.to_rated_rise_C * (k**2)

        # Tustin coefficients for eqn 7.
        zeta = 2.0 * p.to_time_constant_min / dt_min
        a = 1.0 / (1.0 + zeta)
        b = (1.0 - zeta) / (1.0 + zeta)

        # Dynamic top-oil rise (bilinear discretisation of eqn 6).
        delta_e_C = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            delta_e_C[i] = a * (delta_u_C[i] + delta_u_C[i - 1]) - b * delta_e_C[i - 1]

        # Hot-spot rise from GIC (eqn 9); absolute temperatures (eqn 8).
        eta_C = p.hs_coeff_C_per_A * np.abs(gic_A)
        top_oil_C = p.ambient_C + delta_e_C
        hotspot_C = p.ambient_C + delta_e_C + eta_C

        return DeviceResponse(
            time_s=time_s,
            response_current_A=np.abs(gic_A),
            thd=float("nan"),
            harmonics=np.zeros(0, dtype=np.float64),
            top_oil_C=top_oil_C,
            hotspot_C=hotspot_C,
            metadata={
                "model": "Mate2021_TopOilHotspot",
                "params": p,
                "peak_hotspot_C": float(np.max(hotspot_C)),
                "peak_top_oil_C": float(np.max(top_oil_C)),
                "avg_limit_exceeded": bool(np.max(hotspot_C) > p.hs_avg_limit_C),
                "inst_limit_exceeded": bool(np.max(hotspot_C) > p.hs_inst_limit_C),
            },
        )
