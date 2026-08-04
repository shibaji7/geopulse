"""Shi hypotheses — network-setup map (fig3 analogue).

Draws the Horton (2012) EPRI21 test network as the paper-frontispiece
map: substations grouped and labelled by AC-bus membership, transmission
lines coloured by voltage tier (500 kV vs 345 kV, derived from per-km
resistance), transformer / low-side branches drawn as thin greys, and
the baseline eastward E-field overlaid as a uniform quiver.

Matplotlib-only. No cartopy → no state boundaries / coastline; note this
in the paper caption. Voltage tiers are inferred from the branch
resistance-per-kilometre split visible in the MATPOWER file (Horton
2012 uses two clean R/km regimes for the two tiers).

Run
---

.. code-block:: bash

    python examples/papers/shi_hypotheses/scripts/plot_network_map_setup.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.powergrid import PowerGridNetwork

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE.parent / "figures"
_EPRI21 = _HERE.parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# Empirical split from Horton EPRI21: the 500 kV corridor lines sit at
# ≈ 0.0048 Ω/km; the 345 kV corridor at ≈ 0.0097 Ω/km. Threshold at
# the midpoint gives a clean two-tier classification.
R_PER_KM_500KV_MAX = 0.006


def _voltage_tier(resistance_Ohm: float, length_m: float) -> str:
    """Return '500', '345', or 'xfmr' for a branch."""
    if length_m <= 0.0:
        return "xfmr"
    r_per_km = resistance_Ohm / (length_m / 1000.0)
    return "500" if r_per_km < R_PER_KM_500KV_MAX else "345"


def _bus_groups_by_substation(nodes: list) -> dict[str, list[str]]:
    """Group dc_bus* ids by the dc_sub* they share coordinates with."""
    subs = {n.node_id: n for n in nodes if n.node_id.startswith("dc_sub")}
    groups: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if not n.node_id.startswith("dc_bus"):
            continue
        if not (np.isfinite(n.latitude_deg) and np.isfinite(n.longitude_deg)):
            continue
        # Match by coordinate equality (Horton pins each AC bus to its
        # parent substation's exact lat/lon).
        for sub_id, sub_node in subs.items():
            if (
                n.latitude_deg == sub_node.latitude_deg
                and n.longitude_deg == sub_node.longitude_deg
            ):
                groups[sub_id].append(n.node_id)
                break
    return groups


def _substation_short_label(sub_id: str, buses: list[str]) -> str:
    """Turn 'dc_sub6' + ['dc_bus6','dc_bus7','dc_bus8'] into 'S6: bus 6,7,8'."""
    idx = sub_id.replace("dc_sub", "")
    bus_nums = sorted(int(b.replace("dc_bus", "")) for b in buses)
    return f"S{idx}: bus " + ",".join(str(n) for n in bus_nums)


def _plot(net: PowerGridNetwork, savepath: Path) -> None:
    """Render the setup map to ``savepath``."""
    nodes = list(net.get_nodes())
    branches = list(net.get_branches())
    node_by_id = {n.node_id: n for n in nodes}
    sub_nodes = [
        n
        for n in nodes
        if n.node_id.startswith("dc_sub")
        and np.isfinite(n.latitude_deg)
        and np.isfinite(n.longitude_deg)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    tier_style = {
        "500": {"color": "crimson", "lw": 2.2, "ls": "-", "zorder": 2, "label": "500 kV line"},
        "345": {"color": "darkorange", "lw": 2.0, "ls": "-", "zorder": 2, "label": "345 kV line"},
        "xfmr": {"color": "0.55", "lw": 0.8, "ls": "-", "zorder": 1, "label": "GSU / low-side"},
    }
    tiers_seen: set[str] = set()
    for br in branches:
        a = node_by_id.get(br.from_node)
        b = node_by_id.get(br.to_node)
        if a is None or b is None:
            continue
        if not (np.isfinite(a.latitude_deg) and np.isfinite(b.latitude_deg)):
            continue
        tier = _voltage_tier(br.resistance_Ohm, br.length_m)
        style = tier_style[tier]
        kwargs = {k: v for k, v in style.items() if k != "label"}
        label = style["label"] if tier not in tiers_seen else None
        tiers_seen.add(tier)
        ax.plot(
            [a.longitude_deg, b.longitude_deg],
            [a.latitude_deg, b.latitude_deg],
            label=label,
            **kwargs,
        )

    groups = _bus_groups_by_substation(nodes)
    for n in sub_nodes:
        ax.scatter(
            n.longitude_deg,
            n.latitude_deg,
            s=170,
            c="crimson",
            edgecolor="black",
            linewidth=1.0,
            marker="o",
            zorder=5,
        )
        ax.plot(
            n.longitude_deg,
            n.latitude_deg,
            "+",
            color="black",
            markersize=9,
            markeredgewidth=1.4,
            zorder=6,
        )
        label = _substation_short_label(n.node_id, groups.get(n.node_id, []))
        ax.annotate(
            label,
            (n.longitude_deg, n.latitude_deg),
            xytext=(10, 6),
            textcoords="offset points",
            fontsize=9,
            zorder=7,
        )

    # E-field baseline quiver — uniform eastward (1 V/km).
    lons_all = np.array([n.longitude_deg for n in sub_nodes])
    lats_all = np.array([n.latitude_deg for n in sub_nodes])
    lon_pad = 0.8
    lat_pad = 0.8
    xlim = (float(lons_all.min()) - lon_pad, float(lons_all.max()) + lon_pad)
    ylim = (float(lats_all.min()) - lat_pad, float(lats_all.max()) + lat_pad)
    qx = np.arange(xlim[0] + 0.3, xlim[1] - 0.1, 0.6)
    qy = np.arange(ylim[0] + 0.3, ylim[1] - 0.1, 0.5)
    QX, QY = np.meshgrid(qx, qy)
    U = np.ones_like(QX)
    V = np.zeros_like(QX)
    ax.quiver(
        QX,
        QY,
        U,
        V,
        color="steelblue",
        alpha=0.55,
        scale=22,
        width=0.0025,
        zorder=0,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    ax.set_title("Horton (2012) EPRI21 test network — baseline setup for Shi hypothesis tests")

    handles, labels = ax.get_legend_handles_labels()
    # Preferred legend order: 500, 345, xfmr — plus a synthetic quiver hint.
    order = {"500 kV line": 0, "345 kV line": 1, "GSU / low-side": 2}
    paired = sorted(zip(handles, labels, strict=True), key=lambda p: order.get(p[1], 9))
    handles, labels = zip(*paired, strict=True)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        title=r"E-field vector: unit east (1 V/km)",
    )

    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the network setup map."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    net = PowerGridNetwork.from_file(str(_EPRI21))

    # Print tier counts for traceability
    lines_500 = sum(
        1 for b in net.get_branches() if _voltage_tier(b.resistance_Ohm, b.length_m) == "500"
    )
    lines_345 = sum(
        1 for b in net.get_branches() if _voltage_tier(b.resistance_Ohm, b.length_m) == "345"
    )
    xfmrs = sum(
        1 for b in net.get_branches() if _voltage_tier(b.resistance_Ohm, b.length_m) == "xfmr"
    )
    print(f"transmission tiers: 500 kV = {lines_500} lines, 345 kV = {lines_345} lines")
    print(f"transformer / low-side branches: {xfmrs}")

    figpath = _FIG_DIR / "fig3_network_map.png"
    _plot(net, figpath)
    print(f"wrote {figpath}")


if __name__ == "__main__":
    main()
