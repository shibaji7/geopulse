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
from enum import Enum, auto
from functools import lru_cache
from importlib import resources
from typing import Literal

import numpy as np
import yaml  # type: ignore[import-untyped]

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.exceptions import DataError, NotImplementedYetError
from geopulse.uq.uncertain import Uncertain, propagate_uncertainty

__all__ = [
    "CoreType",
    "K_FACTOR_PLACEHOLDERS_MVAR_PER_A",
    "ThermalParams",
    "TransformerModel",
    "gic_to_reactive",
    "saturation_harmonics",
]


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


# ---------------------------------------------------------------------------
# ACPF coupling — subsystem (2) of the acpf-coupling handoff spec §6.2.
# GIC → transformer reactive-power absorption via a linear-in-|I_eff| model.
# See docs/acpf-coupling-handoff.md (spec §6.2) for the full derivation and
# calibration caveats.
# ---------------------------------------------------------------------------


class CoreType(Enum):
    """Transformer core construction determining GIC susceptibility.

    Core geometry is the strongest determinant of the reactive-power
    absorption a transformer draws per amp of DC neutral bias. Ordered
    from least- to most-susceptible (spec §6.2):

    * ``THREE_LIMB_CORE`` — three-legged three-phase core. High
      zero-sequence reluctance keeps GIC flux mostly out of the iron.
      Lowest susceptibility.
    * ``FIVE_LIMB_CORE`` — five-legged three-phase core. Outer return
      legs provide a low-reluctance path for the zero-sequence flux;
      moderately susceptible.
    * ``SHELL_FORM`` — three-phase shell-form construction. High
      susceptibility.
    * ``SINGLE_PHASE_BANK`` — three single-phase units. No inter-phase
      flux cancellation at all. Highest susceptibility of common designs.
    * ``AUTOTRANSFORMER`` — series + common winding sharing a neutral.
      The *effective* GIC current is a weighted sum of series and common
      winding currents rather than ``I_neutral / 3``. Because the
      weighting is genuinely subtle (spec §10 item 3), this enum value
      is **not** in the placeholder K-factor table:
      :func:`gic_to_reactive` requires ``k_override`` when called with
      ``AUTOTRANSFORMER``.
    """

    THREE_LIMB_CORE = auto()
    FIVE_LIMB_CORE = auto()
    SHELL_FORM = auto()
    SINGLE_PHASE_BANK = auto()
    AUTOTRANSFORMER = auto()


# WARNING (spec §6.2, §10 item 1):
#
# These are ORDER-OF-MAGNITUDE placeholders. They are ranked correctly
# (three-limb < five-limb < shell < single-phase bank) but their absolute
# values MUST be calibrated against the GIC literature (Kappenman;
# Overbye & Shetye; NERC TPL-007 committee data) before any published
# result. Use ``k_override`` on :func:`gic_to_reactive` to supply a
# calibrated value per transformer without touching this table.
#
# Units: MVAr/A. Multiplied by ``|I_eff|`` (A per phase) and ``V_pu``
# (per-unit operating voltage) to give absorbed reactive power in MVAr.
K_FACTOR_PLACEHOLDERS_MVAR_PER_A: dict[CoreType, float] = {
    CoreType.THREE_LIMB_CORE: 0.3,
    CoreType.FIVE_LIMB_CORE: 0.6,
    CoreType.SHELL_FORM: 0.8,
    CoreType.SINGLE_PHASE_BANK: 1.0,
    # AUTOTRANSFORMER intentionally absent — see CoreType docstring.
}


def _resolve_k(core_type: CoreType, k_override: Uncertain | float | None) -> Uncertain | float:
    """Return the K to use for this transformer, raising if none is available."""
    if k_override is not None:
        return k_override
    if core_type is CoreType.AUTOTRANSFORMER:
        raise DataError(
            "gic_to_reactive: CoreType.AUTOTRANSFORMER requires k_override — the "
            "effective-current weighting between series and common windings is a "
            "known-subtle calibration (spec §10 item 3) and no placeholder K is "
            "shipped for it. Supply k_override=Uncertain(...) with a calibrated "
            "value or a callable per-transformer estimate."
        )
    return K_FACTOR_PLACEHOLDERS_MVAR_PER_A[core_type]


def _scalar_gic_to_reactive(i_gic_eff: float, k: float, v_pu: float) -> float:
    """Deterministic kernel: ΔQ (MVAr) = K · |I_eff| · V_pu."""
    return float(k) * float(abs(i_gic_eff)) * float(v_pu)


def gic_to_reactive(
    i_gic_eff: Uncertain | float,
    core_type: CoreType,
    v_pu: float = 1.0,
    k_override: Uncertain | float | None = None,
) -> Uncertain:
    r"""Reactive-power absorption of one transformer under DC bias.

    Implements the linear GIC → ΔQ model (Kappenman; Overbye & Shetye;
    spec §6.2):

    .. math::

        \Delta Q \; [\mathrm{MVAr}] \; = \; K \cdot |I_{\mathrm{eff}}| \cdot V_{\mathrm{pu}}

    where ``|I_eff|`` is the effective per-phase DC current in A (for a
    three-phase transformer, ``I_neutral / 3``; autotransformers need
    the caller-supplied series+common winding weighting — see
    :class:`CoreType`), ``K`` is the reactive-absorption coefficient in
    MVAr/A depending on core construction, and ``V_pu`` scales for the
    operating voltage in per-unit.

    Parameters
    ----------
    i_gic_eff : Uncertain or float
        Effective per-phase DC current through the transformer, in
        amperes. Signed; magnitude is taken internally so the result is
        always non-negative. Uncertainty propagates through Monte Carlo
        automatically if this argument is an :class:`Uncertain`.
    core_type : CoreType
        Core construction of this transformer. Determines the placeholder
        ``K`` when ``k_override`` is not supplied.
    v_pu : float, optional
        Operating voltage in per-unit at the transformer's HV terminal.
        Default: ``1.0``.
    k_override : Uncertain or float, optional
        Override the placeholder ``K`` from the module-level table with
        a calibrated value (recommended for published results, required
        for ``CoreType.AUTOTRANSFORMER``). Default: ``None`` (use the
        placeholder).

    Returns
    -------
    Uncertain
        Reactive-power absorption in MVAr, always ≥ 0. If any input
        carries uncertainty, the result carries an ensemble of samples;
        otherwise a deterministic :class:`Uncertain`.

    Raises
    ------
    DataError
        If ``core_type is CoreType.AUTOTRANSFORMER`` and ``k_override``
        is not supplied.

    Notes
    -----
    * The linear model is standard in the GIC literature but breaks
      down at very deep saturation (rule of thumb: |I_eff| > ~30–50 A
      per phase, spec §10 item 4). Document this whenever reporting
      results near or above that regime.
    * The placeholder K-factors in
      :data:`K_FACTOR_PLACEHOLDERS_MVAR_PER_A` are order-of-magnitude
      only — see the module docstring warning.

    Examples
    --------
    Deterministic single-phase bank at 25 A per phase, 1.02 pu voltage:

    >>> import numpy as np
    >>> dq = gic_to_reactive(25.0, CoreType.SINGLE_PHASE_BANK, v_pu=1.02)
    >>> round(float(dq.nominal), 2)
    25.5

    Uncertainty in the DC current propagates:

    >>> from geopulse.uq.uncertain import Uncertain
    >>> i = Uncertain(nominal=25.0, distribution="gaussian", params={"std": 2.0})
    >>> dq_u = gic_to_reactive(i, CoreType.THREE_LIMB_CORE)
    >>> dq_u.n_samples > 0
    True
    """
    k = _resolve_k(core_type, k_override)
    # If neither input carries uncertainty, the deterministic path avoids
    # Monte-Carlo overhead entirely.
    inputs_are_deterministic = (
        not isinstance(i_gic_eff, Uncertain) or i_gic_eff.is_deterministic
    ) and (not isinstance(k, Uncertain) or k.is_deterministic)
    if inputs_are_deterministic:
        i_val = i_gic_eff.nominal if isinstance(i_gic_eff, Uncertain) else i_gic_eff
        k_val = k.nominal if isinstance(k, Uncertain) else k
        return Uncertain(nominal=_scalar_gic_to_reactive(i_val, k_val, v_pu))
    return propagate_uncertainty(
        _scalar_gic_to_reactive,
        i_gic_eff,
        k,
        v_pu=v_pu,
    )


# ---------------------------------------------------------------------------
# ACPF coupling — subsystem (3) of the handoff spec §6.3.
# GIC → harmonic-current injection at the transformer terminal.
# ---------------------------------------------------------------------------


# Map from CoreType members to the corresponding key in the YAML lookup
# table. AUTOTRANSFORMER is deliberately absent (spec §10 item 3) so a
# call with that core type falls back to the analytical hook.
_HARMONIC_TABLE_KEYS: dict[CoreType, str] = {
    CoreType.THREE_LIMB_CORE: "three_limb_core",
    CoreType.FIVE_LIMB_CORE: "five_limb_core",
    CoreType.SHELL_FORM: "shell_form",
    CoreType.SINGLE_PHASE_BANK: "single_phase_bank",
}


@lru_cache(maxsize=1)
def _load_harmonic_injection_table() -> dict[str, dict[str, list[float]]]:
    """Read the placeholder harmonic-injection table shipped with the package.

    Cached — the YAML is parsed at most once per process. Returns the
    raw dict so tests can inspect it directly.
    """
    with (
        resources.files("geopulse.devices.data")
        .joinpath("harmonic_injection.yaml")
        .open("r", encoding="utf-8") as f
    ):
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DataError(
            "harmonic_injection.yaml did not parse to a mapping — the shipped data file is corrupt"
        )
    return data


def _scalar_saturation_harmonics(
    i_gic_eff: float,
    core_type_key: str,
    max_order: int,
) -> np.ndarray:
    """Deterministic kernel: piecewise-linear lookup + fundamental = 100 %."""
    table = _load_harmonic_injection_table()
    row = table[core_type_key]
    breakpoints = np.asarray(row["breakpoints_A"], dtype=np.float64)
    out = np.zeros(max_order, dtype=np.float64)
    out[0] = 100.0  # fundamental is 100 % of itself by definition
    i_mag = abs(float(i_gic_eff))
    for order in range(2, max_order + 1):
        key = f"order_{order}_pct"
        if key not in row:
            # Table does not carry this order — the empirical model can
            # only claim what the calibration covers. Leave 0 rather than
            # extrapolating unknown physics.
            continue
        curve = np.asarray(row[key], dtype=np.float64)
        # np.interp clamps outside the table — a saturating extrapolation
        # is the physically-defensible behaviour at very high |I_eff|
        # where the linear-in-A curve fit no longer holds anyway.
        out[order - 1] = float(np.interp(i_mag, breakpoints, curve))
    return out


def saturation_harmonics(
    i_gic_eff: Uncertain | float,
    core_type: CoreType,
    model: Literal["empirical", "analytical"] = "empirical",
    max_order: int = 5,
) -> np.ndarray:
    r"""Harmonic-current injection driven by half-cycle transformer saturation.

    Returns an array of length ``max_order`` whose *i*-th entry is the
    RMS amplitude of harmonic order ``i + 1`` as a percentage of the
    fundamental. Element 0 is the fundamental (always 100.0 by
    convention); elements 1..max_order-1 are the 2nd..max_order-th
    harmonics.

    The empirical model looks up a piecewise-linear curve shipped with
    the package (``geopulse/devices/data/harmonic_injection.yaml``);
    replace or edit that file to re-calibrate without touching code.

    Parameters
    ----------
    i_gic_eff : Uncertain or float
        Effective per-phase DC current through the transformer, in
        amperes. Sign is ignored; only the magnitude enters the
        saturation depth. Uncertainty is propagated automatically if
        this is an :class:`Uncertain`.
    core_type : CoreType
        Core construction. ``CoreType.AUTOTRANSFORMER`` is not
        supported by the empirical table (spec §10 item 3); it routes
        into the analytical hook, which is currently a stub.
    model : {"empirical", "analytical"}, optional
        Which model to use. ``"empirical"`` (default) is the piecewise-
        linear table lookup. ``"analytical"`` is a hook for a future
        B-H saturation integration; it raises
        :class:`~geopulse.exceptions.NotImplementedYetError` today,
        per the existing repo convention for deferred features.
    max_order : int, optional
        Highest harmonic order to return. Must satisfy
        ``1 <= max_order <= 40``. Default: ``5``.

    Returns
    -------
    numpy.ndarray
        Shape ``(max_order,)``, dtype float64. Percent of fundamental
        for orders 1..max_order. Element 0 is exactly 100.0.

    Raises
    ------
    DataError
        If ``max_order`` is out of range, or the core type has no
        empirical table entry, or ``model`` is not one of the
        supported strings.
    NotImplementedYetError
        If ``model="analytical"`` — the analytical B-H integration is
        a future work package.

    Notes
    -----
    Values in the placeholder table are order-of-magnitude only. The
    qualitative structure is right — even harmonics dominate over odd
    at the same saturation depth (the asymmetric-saturation signature
    unique to DC-biased transformers), and susceptibility rises with
    core-type index in the same order as the reactive-absorption
    K-factors (spec §6.2). Absolute magnitudes must be calibrated
    against the GIC literature before any published result.

    When ``i_gic_eff`` is an :class:`Uncertain`, the returned array is
    the *nominal* result and callers who want uncertainty on the
    harmonics should use :func:`~geopulse.uq.propagate_uncertainty`
    directly.

    Examples
    --------
    Empirical lookup at 10 A on a shell-form transformer:

    >>> h = saturation_harmonics(10.0, CoreType.SHELL_FORM)
    >>> h.shape
    (5,)
    >>> float(h[0])
    100.0
    >>> bool(h[1] > h[2])   # even harmonics dominate odd
    True

    Higher current, deeper saturation:

    >>> h_deep = saturation_harmonics(30.0, CoreType.SHELL_FORM)
    >>> bool(h_deep[1] > h[1])   # 2nd-harmonic content grows
    True
    """
    if not 1 <= max_order <= 40:
        raise DataError(f"max_order must be in [1, 40], got {max_order}")
    if model not in ("empirical", "analytical"):
        raise DataError(f"model must be 'empirical' or 'analytical', got {model!r}")
    if model == "analytical":
        raise NotImplementedYetError(
            "saturation_harmonics(model='analytical')",
            "acpf-harmonics-analytical",
        )
    if core_type not in _HARMONIC_TABLE_KEYS:
        raise DataError(
            f"saturation_harmonics: no empirical table entry for {core_type.name}. "
            "AUTOTRANSFORMER effective-current weighting is a known-subtle "
            "calibration issue (spec §10 item 3) — supply a table entry via "
            "the analytical hook once it lands."
        )
    key = _HARMONIC_TABLE_KEYS[core_type]
    i_val = float(i_gic_eff.nominal if isinstance(i_gic_eff, Uncertain) else i_gic_eff)
    return _scalar_saturation_harmonics(i_val, key, max_order)
