"""Shi hypotheses — H2: how do resistive blockers change GIC?

Mechanism claim
---------------
Between the two hypothetical storms the operator changed the neutral-
side grounding at the target substation: raised a neutral-grounding
resistor, or installed a DC-blocking device. That local change reduces
GIC at the target but redistributes current onto the rest of the
network — a real trade-off documented in Boteler (2014) § 5.

We isolate that claim by holding the field fixed at the uniform
baseline (1 V/km east) and sweeping the earthing resistance at the
target substation from ``R0`` (as-delivered) to ``100·R0`` on a log
grid, plus a "full blocker" limiting case at ``10¹² Ω``. For every
scenario we record:

* ``|GIC|`` at the target substation (local protection metric).
* network-total ``Σ|GIC|`` (network-wide cost metric).

The plot puts both on twin y-axes, showing at a glance where local
protection is bought cheaply and where further hardening becomes
counter-productive.

Also demonstrates the two-step compose-mitigate-and-instrument
workflow: after the NGR and full-blocker solves we hand the target-
substation current to a :class:`geopulse.devices.blocker.ResistiveBlocker`
to report peak voltage / peak power / dissipated energy at the blocker
itself — the numbers asset-management engineers need for rating
checks and cumulative heating.

Draft quality — same convention as H1.

Run
---

.. code-block:: bash

    python examples/papers/shi_hypotheses/scripts/hypothesis2_mitigation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.devices.blocker import ResistiveBlocker
from geopulse.network.helpers import apply_resistive_blocker
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_NODE_ID = "dc_sub6"
E_LOCAL_VM = 1e-3  # 1 V/km east, uniform, across every scenario
BLOCKER_R_IDEAL_OHM = 1.0e12  # effective open (DC-blocking capacitor limit)
BLOCKER_R_NGR_OHM = 10.0  # realistic neutral-grounding resistor
# Log-spaced multipliers of R_gnd at the target: 1 → 100, 15 points.
R_MULTIPLIERS = np.logspace(0.0, 2.0, 15)

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE.parent / "figures"
_EPRI21 = _HERE.parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"


def _target_index(net: PowerGridNetwork, node_id: str) -> int:
    """Return the 0-based row index of ``node_id``."""
    for i, n in enumerate(net.get_nodes()):
        if n.node_id == node_id:
            return i
    raise ValueError(f"target node {node_id!r} not found")


def _solve(
    net: PowerGridNetwork,
    Y: np.ndarray,
    Z: np.ndarray,
    V_th: np.ndarray,
    target_idx: int,
) -> tuple[float, float]:
    """Solve one scenario; return (|GIC| at target, Σ|GIC|)."""
    r = NAMSolver().solve(net, Y, Z, V_th)
    earth_diag = np.diag(Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        gic = np.where(
            earth_diag > 0, r.node_voltages_V / np.where(earth_diag > 0, earth_diag, 1.0), 0.0
        )
    return float(abs(gic[target_idx])), float(np.sum(np.abs(gic)))


def _blocker_report(
    resistance_Ohm: float,
    gic_at_target_A: float,
    duration_s: float = 3600.0,
    n_samples: int = 601,
) -> dict:
    """Instrument the blocker device with a sustained-DC proxy."""
    b = ResistiveBlocker(resistance_Ohm=resistance_Ohm)
    t = np.linspace(0.0, duration_s, n_samples)
    i = np.full_like(t, gic_at_target_A)
    resp = b.inject_gic(t, i)
    return {
        "peak_V": resp.metadata["blocker_peak_voltage_V"],
        "peak_W": resp.metadata["blocker_peak_power_W"],
        "energy_J": resp.metadata["blocker_dissipated_energy_J"],
    }


def _plot_h2a(
    multipliers: np.ndarray,
    gic_target: np.ndarray,
    gic_total: np.ndarray,
    baseline_target: float,
    baseline_total: float,
    full_blocker_target: float,
    full_blocker_total: float,
    savepath: Path,
) -> None:
    """H2a: local |GIC| vs network Σ|GIC| under a target-only R_gnd sweep."""
    fig, ax_l = plt.subplots(figsize=(11, 6))
    ax_r = ax_l.twinx()

    red = "tab:red"
    blue = "tab:blue"

    ax_l.plot(
        multipliers,
        gic_target,
        "o-",
        color=red,
        lw=1.8,
        markersize=5,
        label="|GIC at target substation|",
    )
    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"Grounding resistance multiplier at target  ($R_{\rm gnd}/R_{\rm gnd,0}$)")
    ax_l.set_ylabel("|GIC at target substation|  (A)", color=red)
    ax_l.tick_params(axis="y", labelcolor=red)

    ax_r.plot(
        multipliers,
        gic_total,
        "s--",
        color=blue,
        lw=1.5,
        markersize=5,
        label=r"Network-total $\sum |GIC|$",
    )
    ax_r.set_ylabel(r"Network-total $\sum |GIC|$  (A)", color=blue)
    ax_r.tick_params(axis="y", labelcolor=blue)

    # Full-blocker limiting case as stars off to the right of the sweep.
    star_x = multipliers[-1] * 1.4
    ax_l.plot(
        star_x,
        full_blocker_target,
        "*",
        color=red,
        markersize=18,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    ax_r.plot(
        star_x,
        full_blocker_total,
        "*",
        color=blue,
        markersize=18,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    ax_l.annotate(
        "full blocker",
        xy=(star_x, full_blocker_target),
        xytext=(4, -14),
        textcoords="offset points",
        fontsize=9,
        color="0.3",
    )

    ax_l.annotate(
        "baseline",
        xy=(multipliers[0], baseline_target),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=9,
        color="0.3",
    )

    ax_l.axvline(multipliers[0], color="0.75", lw=0.6, linestyle=":")
    ax_l.grid(alpha=0.25, which="both")

    ax_l.set_title(
        r"H2a: local protection vs network cost — raising $R_{\rm gnd}$ at target substation"
    )

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(
        lines_l + lines_r, labels_l + labels_r, loc="center left", framealpha=0.95, fontsize=10
    )

    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the H2a sweep, print the table, save the figure."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    net = PowerGridNetwork.from_file(str(_EPRI21))
    Y = net.assemble_network_admittance()
    Z_base = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(ex_Vm=E_LOCAL_VM, ey_Vm=0.0)

    target_idx = _target_index(net, TARGET_NODE_ID)
    r0 = float(Z_base[target_idx, target_idx])

    print(f"Target: {TARGET_NODE_ID} (idx {target_idx})")
    print(f"Baseline R_gnd at target: {r0:.4f} Ω")
    print(f"Uniform driving field: {E_LOCAL_VM * 1e3:.1f} V/km east\n")

    # Baseline (multiplier = 1)
    base_t, base_all = _solve(net, Y, Z_base, V_th, target_idx)
    print(f"{'R_mult':>10s}  {'R_gnd [Ω]':>12s}  {'|GIC target| [A]':>18s}  {'Σ|GIC| [A]':>13s}")
    print("-" * 62)

    gic_target = np.zeros_like(R_MULTIPLIERS)
    gic_total = np.zeros_like(R_MULTIPLIERS)
    for k, m in enumerate(R_MULTIPLIERS):
        r_new = r0 * float(m)
        Z = apply_resistive_blocker(Z_base, [target_idx], r_new)
        g_t, g_all = _solve(net, Y, Z, V_th, target_idx)
        gic_target[k] = g_t
        gic_total[k] = g_all
        print(f"{m:10.3f}  {r_new:12.4f}  {g_t:18.2f}  {g_all:13.2f}")

    # Full-blocker limiting case
    Z_full = apply_resistive_blocker(Z_base, [target_idx], BLOCKER_R_IDEAL_OHM)
    full_t, full_all = _solve(net, Y, Z_full, V_th, target_idx)
    print(f"{'full':>10s}  {BLOCKER_R_IDEAL_OHM:12.2e}  {full_t:18.2f}  {full_all:13.2f}")

    fig_h2a = _FIG_DIR / "h2a_local_vs_global.png"
    _plot_h2a(
        R_MULTIPLIERS,
        gic_target,
        gic_total,
        base_t,
        base_all,
        full_t,
        full_all,
        fig_h2a,
    )
    print(f"wrote {fig_h2a}")

    # ---- blocker instrumentation on the NGR + full-open cases ----
    print()
    print("Blocker instrumentation (peak-plateau proxy: 1 h of sustained DC current)")
    print("-" * 78)
    for label, r_ohm in [
        (f"NGR 10 Ω @ {TARGET_NODE_ID}", BLOCKER_R_NGR_OHM),
        (f"ideal 1e12 Ω @ {TARGET_NODE_ID}", BLOCKER_R_IDEAL_OHM),
    ]:
        Z_scen = apply_resistive_blocker(Z_base, [target_idx], r_ohm)
        r = NAMSolver().solve(net, Y, Z_scen, V_th)
        earth_diag = np.diag(Z_scen)
        i_blocker = float(r.node_voltages_V[target_idx] / earth_diag[target_idx])
        try:
            rep = _blocker_report(r_ohm, i_blocker)
        except AttributeError as exc:
            print(f"  {label}: blocker device unavailable in this env ({exc})")
            continue
        rating = "OK" if rep["peak_V"] < 1e5 else "⚠ exceeds O(1e5 V) rating"
        print(f"  {label}")
        print(f"    through-blocker current : {i_blocker:.3e} A")
        print(f"    peak voltage            : {rep['peak_V']:.3e} V — {rating}")
        print(f"    peak power              : {rep['peak_W']:.3e} W")
        print(f"    dissipated energy (1 h) : {rep['energy_J']:.3e} J")
        print()


if __name__ == "__main__":
    main()
