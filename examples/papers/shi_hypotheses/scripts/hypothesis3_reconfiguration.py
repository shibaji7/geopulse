"""Shi hypotheses — H3: how do line outages and added ties change GIC?

Mechanism claim
---------------
Between the two hypothetical storms the network topology changed. A
transmission line was out of service (planned maintenance, forced
outage), or a new tie between two previously-disconnected corridors was
in service. Either shifts where the current divider concentrates GIC.

We hold the driving field fixed at the uniform 1 V/km east baseline and
sweep:

* every in-service transmission line (``dc_br*`` with ``length_m > 0``)
  is opened, one at a time.
* one added tie between the two geographically extreme substations
  (``dc_sub2`` NW-most and ``dc_sub8`` E-most) — a hypothetical long-
  distance reinforcement.

For each scenario we record ``|GIC|`` at ``dc_sub6`` (the paper target)
and rank the outages by how much they change the target current.

Interpretation
--------------
Lines whose removal *lowers* GIC at the target were the primary
carriers into the target substation; lines whose removal *raises* it
were competing paths that had been diverting current elsewhere. A new
tie can go either way depending on whether it opens a new path away
from the target or pulls current toward it.

Draft quality — same convention as H1/H2.

Run
---

.. code-block:: bash

    python examples/papers/shi_hypotheses/scripts/hypothesis3_reconfiguration.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.helpers import add_tie, open_line
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_NODE_ID = "dc_sub6"
E_LOCAL_VM = 1e-3
# Added-tie endpoints: the NW-most and E-most substations with valid
# coordinates. dc_sub2 sits at (34.31, -86.37), dc_sub8 at (34.20, -81.10).
TIE_FROM = "dc_sub2"
TIE_TO = "dc_sub8"
TIE_RESISTANCE_OHM = 1.0

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE.parent / "figures"
_EPRI21 = _HERE.parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"


@dataclass(frozen=True)
class Scenario:
    """One reconfiguration scenario: label, |GIC| at target, Σ|GIC|."""

    label: str
    target_gic_A: float
    total_gic_A: float


def _node_index(net: PowerGridNetwork, node_id: str) -> int:
    for i, n in enumerate(net.get_nodes()):
        if n.node_id == node_id:
            return i
    raise ValueError(f"node {node_id!r} not found")


def _solve(
    net: PowerGridNetwork,
    Y: np.ndarray,
    Z: np.ndarray,
    V_th: np.ndarray,
    target_idx: int,
) -> tuple[float, float]:
    r = NAMSolver().solve(net, Y, Z, V_th)
    earth = np.diag(Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        gic = np.where(earth > 0, r.node_voltages_V / np.where(earth > 0, earth, 1.0), 0.0)
    return float(abs(gic[target_idx])), float(np.sum(np.abs(gic)))


def _plot(scenarios: list[Scenario], baseline: Scenario, savepath: Path) -> None:
    """Ranked bar of Δ|GIC| at target across every reconfiguration scenario."""
    deltas = np.array([s.target_gic_A - baseline.target_gic_A for s in scenarios])
    order = np.argsort(deltas)
    labels = [scenarios[i].label for i in order]
    deltas = deltas[order]

    colors = ["tab:blue" if d < 0 else "tab:red" for d in deltas]

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.32 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # most negative at top
    ax.set_xlabel(r"$\Delta |GIC|$ at target substation  (A, relative to baseline)")
    ax.set_title(
        f"H3: single-line outage + added-tie impact on |GIC| at {TARGET_NODE_ID}\n"
        f"baseline |GIC| = {baseline.target_gic_A:.1f} A"
    )
    ax.grid(alpha=0.3, axis="x")

    for yy, d in zip(y, deltas, strict=True):
        ha = "left" if d >= 0 else "right"
        offset = 2 if d >= 0 else -2
        ax.text(d + offset, yy, f"{d:+.1f}", ha=ha, va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the H3 reconfiguration sweep."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    net = PowerGridNetwork.from_file(str(_EPRI21))
    Y_base = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(ex_Vm=E_LOCAL_VM, ey_Vm=0.0)

    target_idx = _node_index(net, TARGET_NODE_ID)
    baseline_gic, baseline_total = _solve(net, Y_base, Z, V_th, target_idx)
    baseline = Scenario("baseline (as-delivered)", baseline_gic, baseline_total)

    print(f"Target: {TARGET_NODE_ID} (idx {target_idx})")
    print(f"Uniform driving field: {E_LOCAL_VM * 1e3:.1f} V/km east")
    print(f"Baseline: |GIC| = {baseline_gic:.2f} A   Σ|GIC| = {baseline_total:.2f} A\n")

    nodes = list(net.get_nodes())
    node_idx = {n.node_id: i for i, n in enumerate(nodes)}
    branches = list(net.get_branches())

    scenarios: list[Scenario] = []

    # ---- single-line outages (transmission only) ----
    print(f"{'Outage':22s}  {'|GIC target| [A]':>18s}  {'Δ vs baseline':>14s}  {'Σ|GIC| [A]':>12s}")
    print("-" * 74)
    for br in branches:
        if br.length_m <= 0.0:
            continue  # transformer / low-side — skip
        g_line = 1.0 / br.resistance_Ohm
        i, j = node_idx[br.from_node], node_idx[br.to_node]
        Y_mod = open_line(Y_base, i, j, g_line)
        g_t, g_all = _solve(net, Y_mod, Z, V_th, target_idx)
        delta = g_t - baseline_gic
        label = f"open {br.branch_id} ({br.from_node}→{br.to_node})"
        print(f"{br.branch_id:22s}  {g_t:18.2f}  {delta:>+14.2f}  {g_all:12.2f}")
        scenarios.append(Scenario(label, g_t, g_all))

    # ---- added tie ----
    print()
    tie_from_idx = node_idx[TIE_FROM]
    tie_to_idx = node_idx[TIE_TO]
    Y_tie = add_tie(Y_base, tie_from_idx, tie_to_idx, TIE_RESISTANCE_OHM)
    g_t, g_all = _solve(net, Y_tie, Z, V_th, target_idx)
    tie_label = f"add tie {TIE_FROM}↔{TIE_TO} (R={TIE_RESISTANCE_OHM:g} Ω)"
    print(f"{tie_label:22s}  {g_t:18.2f}  {g_t - baseline_gic:>+14.2f}  {g_all:12.2f}")
    scenarios.append(Scenario(tie_label, g_t, g_all))

    figpath = _FIG_DIR / "h3_reconfiguration_bars.png"
    _plot(scenarios, baseline, figpath)
    print(f"\nwrote {figpath}")


if __name__ == "__main__":
    main()
