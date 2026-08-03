r"""Resistive-blocker mitigation-device model.

A **DC-current-blocking device** installed at a transformer neutral raises
the effective neutral-to-earth impedance so that most of the GIC that
*would* have flowed through the neutral instead redistributes to other
parts of the network. Two flavours in the literature:

* **DC-blocking capacitor** — ideally zero impedance for AC (line
  frequency and above), infinite for DC. Practically implemented with a
  series capacitor plus fault-mode-bypass thyristors; behaves as an
  open circuit for GIC purposes.
* **Neutral-grounding resistor (NGR)** — a plain resistor with a value
  chosen large enough to attenuate GIC by a factor of several, but small
  enough that fault-clearing behaviour is not compromised. Typical
  values 5–30 Ω.

For the current alpha of GeoPulse we model both as a purely **resistive**
element with a scalar :attr:`resistance_Ohm` attribute — an ideal
DC-blocking capacitor is the ``R → ∞`` limit of this model and can be
represented by a large finite resistance (10¹² Ω) in the same code path.

What this class **does**
------------------------
Given the current time series ``i(t)`` flowing through the blocker to
earth (i.e. the GIC at the blocker's substation neutral, as reported by
the solver), it computes:

* the voltage drop across the blocker ``v(t) = i(t) · R``,
* instantaneous dissipated power ``p(t) = i(t)² · R``,
* total dissipated energy ``E = ∫ p(t) dt``,
* peak power (for thermal rating checks).

These are surfaced in :attr:`DeviceResponse.metadata` under
``"blocker_*"`` keys. The device itself is linear and generates no
harmonics, so :attr:`DeviceResponse.thd` is ``0.0`` and
:attr:`DeviceResponse.harmonics` is a zero-length array.

What this class **does not** do
-------------------------------
* Modify the network topology. That belongs at the network layer —
  use :func:`geopulse.network.helpers.apply_resistive_blocker` to raise
  the earthing impedance at the substation before solving.
* Enforce a maximum voltage. The blocker will happily report
  megavolt-scale drops if you feed kilo-Ampere currents into a
  10¹² Ω resistor. Voltage-rating checks are the caller's
  responsibility (compare :attr:`DeviceResponse.metadata["blocker_peak_voltage_V"]`
  against your device datasheet).
* Model fault-mode bypass (thyristor firing at an over-voltage
  threshold). That's a v0.5+ item.

The intended workflow is a two-step composition:

1. Apply :func:`geopulse.network.helpers.apply_resistive_blocker` to
   modify the earthing-impedance matrix ``[Z^e]`` at the target
   substation.
2. Solve the network. Take the GIC at that substation neutral from
   :attr:`SolverResult.node_voltages_V` /
   :attr:`SolverResult.branch_currents_A` and hand it to
   :meth:`ResistiveBlocker.inject_gic` to get dissipation numbers.

Step 1 changes the physics; step 2 instruments the blocker for
reporting.

References
----------
.. [1] Bolduc, L., Granger, M., Paré, G., Saintonge, J., Brophy, L.
   (2005). Development of a DC current-blocking device for transformer
   neutrals. IEEE Trans. Power Delivery, 20(1), 163-168.
.. [2] Boteler, D. H. (2014). Methodology for simulation of
   geomagnetically induced currents in power systems. J. Space Weather
   Space Clim., 4, A21. (Redistribution physics.)
.. [3] IEEE Std 1613-2009. IEEE Standard Environmental and Testing
   Requirements for Communications Networking Devices Installed in
   Electric Power Substations. (Voltage-rating context.)
"""

from __future__ import annotations

import numpy as np

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.exceptions import DataError, ShapeMismatchError

__all__ = ["ResistiveBlocker"]


class ResistiveBlocker(DeviceModel):
    r"""Passive resistive DC-blocker at a transformer neutral.

    Parameters
    ----------
    resistance_Ohm : float
        Blocker resistance in Ohms. Must be strictly positive. Use a
        large finite value (e.g. :math:`10^{12}`) to approximate an
        ideal DC-blocking capacitor.

    Attributes
    ----------
    resistance_Ohm : float
        The value passed at construction, echoed back for reporting.

    Raises
    ------
    DataError
        If ``resistance_Ohm`` is not strictly positive-finite.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.devices.blocker import ResistiveBlocker
    >>> b = ResistiveBlocker(resistance_Ohm=10.0)
    >>> t = np.linspace(0, 60, 601)
    >>> i = 5.0 * np.ones_like(t)             # steady 5 A of GIC through the blocker
    >>> resp = b.inject_gic(t, i)
    >>> round(resp.metadata["blocker_peak_voltage_V"], 3)
    50.0
    >>> round(resp.metadata["blocker_dissipated_energy_J"], 1)
    15000.0
    """

    def __init__(self, resistance_Ohm: float) -> None:
        if not np.isfinite(resistance_Ohm) or resistance_Ohm <= 0.0:
            raise DataError(f"resistance_Ohm must be positive-finite, got {resistance_Ohm!r}")
        self.resistance_Ohm = float(resistance_Ohm)

    def inject_gic(
        self,
        time_s: np.ndarray,
        gic_A: np.ndarray,
        ac_voltage_V: float = 0.0,  # noqa: ARG002 — accepted for ABC parity, ignored
        ac_frequency_Hz: float = 50.0,  # noqa: ARG002 — accepted for ABC parity, ignored
    ) -> DeviceResponse:
        """Report the blocker's instrumentation for a given neutral current.

        The blocker is linear and passive: the response current
        equals the input GIC. What this method adds is the
        *instrumentation* — voltage, power, energy, peak — computed
        against the blocker resistance and surfaced in
        :attr:`DeviceResponse.metadata`.

        Parameters
        ----------
        time_s : numpy.ndarray
            Time array in seconds. Shape ``(n_times,)``. Must be
            strictly increasing and uniformly sampled.
        gic_A : numpy.ndarray
            GIC through the blocker in Amperes. Shape ``(n_times,)``,
            same length as ``time_s``. Sign convention is caller's —
            all reported metadata uses ``|i|`` where sign is irrelevant
            (energy, peak power).
        ac_voltage_V, ac_frequency_Hz : float, optional
            Accepted for API parity with the ABC. Ignored — this
            device is DC-only and linear.

        Returns
        -------
        DeviceResponse
            ``response_current_A = gic_A`` (passed through). ``thd = 0``
            and ``harmonics = array([])`` (linear device). ``metadata``
            keys:

            * ``blocker_resistance_Ohm`` — echoes the constructor value.
            * ``blocker_voltage_V`` — full time series of ``v(t) = i(t) · R``.
            * ``blocker_peak_voltage_V`` — ``max(|v(t)|)``.
            * ``blocker_peak_power_W`` — ``max(i(t)²) · R``.
            * ``blocker_dissipated_energy_J`` — trapezoidal integral of
              ``p(t) = i(t)² · R`` over ``time_s``.

        Raises
        ------
        ShapeMismatchError
            If ``time_s`` and ``gic_A`` shapes disagree or aren't 1-D.
        DataError
            If ``time_s`` is not strictly increasing, or shorter than 2
            samples (energy integration needs at least one interval).

        Notes
        -----
        Energy is integrated with the composite trapezoidal rule on the
        supplied ``time_s`` grid. For a uniformly sampled input this
        matches :func:`numpy.trapezoid` exactly; for a slightly
        non-uniform grid the result is still second-order accurate.
        """
        del ac_voltage_V, ac_frequency_Hz  # linear DC-only device
        t = np.asarray(time_s, dtype=np.float64)
        i = np.asarray(gic_A, dtype=np.float64)
        if t.ndim != 1 or i.ndim != 1:
            raise ShapeMismatchError(
                f"time_s and gic_A must be 1-D, got shapes {t.shape} and {i.shape}"
            )
        if t.shape != i.shape:
            raise ShapeMismatchError(
                f"time_s {t.shape} and gic_A {i.shape} must have the same length"
            )
        if t.size < 2:
            raise DataError("time_s must have at least 2 samples for energy integration")
        if not np.all(np.diff(t) > 0.0):
            raise DataError("time_s must be strictly increasing")

        r = self.resistance_Ohm
        v_t = i * r
        p_t = (i * i) * r
        energy_J = float(np.trapezoid(p_t, t))
        peak_v = float(np.max(np.abs(v_t)))
        peak_p = float(np.max(p_t))

        return DeviceResponse(
            time_s=t,
            response_current_A=i,
            thd=0.0,
            harmonics=np.zeros(0, dtype=np.float64),
            top_oil_C=None,
            hotspot_C=None,
            metadata={
                "blocker_resistance_Ohm": r,
                "blocker_voltage_V": v_t,
                "blocker_peak_voltage_V": peak_v,
                "blocker_peak_power_W": peak_p,
                "blocker_dissipated_energy_J": energy_J,
            },
        )
