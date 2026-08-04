"""Shi hypotheses — H1: how does dE/dx impact GIC at the target substation?

Mechanism claim
---------------
The local geoelectric field at the observation-point substation stays
fixed between two hypothetical storms; only the *regional* structure
of the field elsewhere in the network changes. Because GIC is set by
the whole network's current divider — not by the local driving
voltage alone — a purely-spatial field variation can move the
observed GIC by an order of magnitude while the local field at the
target is held constant.

We test that quantitatively by sweeping the eastward gradient
``dE/dx`` across a physically-plausible range and re-solving on the
Horton (2012) EPRI21 test network. The gradient field is constructed
so that ``E(target) == E_local`` in every scenario — anchoring the
local physics — while the field elsewhere varies with position.

Method
------
1. Load Horton EPRI21 via :class:`PowerGridNetwork`.
2. Pick the target substation. Default: ``dc_sub6`` (largest single-
   node GIC under uniform 1 V/km east — a natural demonstration
   proxy for a real IK413-like observation point).
3. Anchor the baseline: uniform 1 V/km eastward. Reports GIC(target)
   and network-total |GIC|.
4. Sweep ``dE/dx`` across ``[5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]``
   V/km per km (physically plausible for storm-scale spatial
   structure over a few-hundred-km network).
5. For each ``dE/dx``: build ``Ex(x) = E_local − dE/dx · (x −
   x_target)`` such that ``Ex(x_target) = E_local``. Sample at each
   branch midpoint via
   :func:`geopulse.network.helpers.evaluate_field_at_branch_midpoints`
   and pass the resulting arrays to
   :meth:`PowerGridNetwork.compute_thevenin_voltages`.
6. Solve with :class:`NAMSolver`. Record GIC(target) and network-
   total |GIC|.
7. Save a draft-quality sensitivity plot and print the table.

Draft quality
-------------
Figures use plain matplotlib and land in
``examples/papers/shi_hypotheses/figures/``. Publication-quality
retrofit (e.g. ``geopulse.viz.presets.apply_preset("sw_2col")``) is
a two-line follow-up when the paper is being submitted.

Run
---

.. code-block:: bash

    python examples/papers/shi_hypotheses/scripts/hypothesis1_spatial_field.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from geopulse.geo import meridian_radius_m, prime_vertical_radius_m
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

# H1b — spatial-redistribution map is drawn at this single gradient
# value (chosen to match the reference GIC_HSR_Model figure).
H1B_DEDX_V_PER_KM_PER_KM = 5e-3

# Note: we compute per-branch midpoints manually in _sample_field_at_branches
# rather than calling geopulse.network.helpers.evaluate_field_at_branch_midpoints
# because that helper takes a naive mean over ALL node lats to build its
# projection origin — and Horton EPRI21 has two substations
# (dc_sub1, dc_sub7) with NaN coords in the MATPOWER file, which poisons
# the mean to NaN. A NaN-safe version of the helper is filed as a
# follow-up issue against PR #20.

# ---------------------------------------------------------------------------
# Configuration — edit these two lines to point at a different target /
# baseline field if the paper's argument changes.
# ---------------------------------------------------------------------------
TARGET_NODE_ID = "dc_sub6"
E_LOCAL_VM = 1e-3  # 1 V/km eastward at the target
# Signed sweep — dense linear grid across ±0.02 V/km per km so the
# reference-figure "±1 order-of-magnitude band" is visible as a smooth
# curve rather than a coarse trace. Excludes exactly zero (that's the
# uniform baseline, drawn as a horizontal reference line).
DEDX_SWEEP_V_PER_KM_PER_KM = tuple(float(v) for v in np.linspace(-0.02, 0.02, 31) if abs(v) > 1e-12)

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE.parent / "figures"
_EPRI21 = _HERE.parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"


def _target_index(net: PowerGridNetwork, node_id: str) -> int:
    """Return the 0-based row index of ``node_id`` in the network's nodes."""
    for i, n in enumerate(net.get_nodes()):
        if n.node_id == node_id:
            return i
    raise ValueError(f"target node {node_id!r} not found in network")


def _valid_mean_latlon(net: PowerGridNetwork) -> tuple[float, float]:
    """Mean lat/lon over nodes with valid coordinates.

    Filters out nodes whose lat or lon is NaN — Horton EPRI21 has two
    such substations in the MATPOWER file, and a naive ``np.mean``
    would poison the projection origin.
    """
    nodes = list(net.get_nodes())
    valid = [n for n in nodes if np.isfinite(n.latitude_deg) and np.isfinite(n.longitude_deg)]
    lat0 = float(np.mean([n.latitude_deg for n in valid]))
    lon0 = float(np.mean([n.longitude_deg for n in valid]))
    return lat0, lon0


def _target_local_xy_km(net: PowerGridNetwork, target_idx: int) -> tuple[float, float]:
    """Local (east km, north km) of the target in the paper's projection."""
    lat0, lon0 = _valid_mean_latlon(net)
    m_per_deg_lat = meridian_radius_m(lat0) * np.pi / 180.0
    m_per_deg_lon = prime_vertical_radius_m(lat0) * float(np.cos(np.radians(lat0))) * np.pi / 180.0
    tgt = list(net.get_nodes())[target_idx]
    x_km = (tgt.longitude_deg - lon0) * m_per_deg_lon / 1000.0
    y_km = (tgt.latitude_deg - lat0) * m_per_deg_lat / 1000.0
    return float(x_km), float(y_km)


def _sample_field_at_branches(
    net: PowerGridNetwork,
    field_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-branch (Ex, Ey) via NaN-safe local equirectangular projection.

    Branches touching a NaN-coord node get (0, 0) — those are zero-
    length (co-located with parent AC bus), so no induced voltage is
    the right physics.
    """
    lat0, lon0 = _valid_mean_latlon(net)
    m_per_deg_lat = meridian_radius_m(lat0) * np.pi / 180.0
    m_per_deg_lon = prime_vertical_radius_m(lat0) * float(np.cos(np.radians(lat0))) * np.pi / 180.0

    nodes_by_id = {n.node_id: n for n in net.get_nodes()}
    branches = list(net.get_branches())
    ex = np.zeros(len(branches), dtype=np.float64)
    ey = np.zeros(len(branches), dtype=np.float64)
    for k, br in enumerate(branches):
        a, b = nodes_by_id[br.from_node], nodes_by_id[br.to_node]
        lat_mid = 0.5 * (a.latitude_deg + b.latitude_deg)
        lon_mid = 0.5 * (a.longitude_deg + b.longitude_deg)
        if not (np.isfinite(lat_mid) and np.isfinite(lon_mid)):
            continue  # zero-length degenerate branch → leave (0, 0)
        x_km = (lon_mid - lon0) * m_per_deg_lon / 1000.0
        y_km = (lat_mid - lat0) * m_per_deg_lat / 1000.0
        val = field_fn(float(x_km), float(y_km))
        ex[k], ey[k] = float(val[0]), float(val[1])
    return ex, ey


def _gic_at(node_idx: int, node_voltages_V: np.ndarray, earth_Z_diag: np.ndarray) -> float:
    """GIC at a node = V / R_ground; 0 when the node is ungrounded."""
    z = float(earth_Z_diag[node_idx])
    if z <= 0.0 or not np.isfinite(z):
        return 0.0
    return float(node_voltages_V[node_idx] / z)


def _network_total_abs_gic(node_voltages_V: np.ndarray, earth_Z_diag: np.ndarray) -> float:
    """Sum of |GIC| across every grounded node."""
    with np.errstate(divide="ignore", invalid="ignore"):
        gic = np.where(
            earth_Z_diag > 0, node_voltages_V / np.where(earth_Z_diag > 0, earth_Z_diag, 1.0), 0.0
        )
    return float(np.sum(np.abs(gic)))


def _solve_uniform_baseline(
    net: PowerGridNetwork,
    Y: np.ndarray,
    Z: np.ndarray,
    target_idx: int,
) -> tuple[float, float]:
    """Baseline: uniform 1 V/km east. Returns (target_GIC, network_total)."""
    V_th = net.compute_thevenin_voltages(ex_Vm=E_LOCAL_VM, ey_Vm=0.0)
    r = NAMSolver().solve(net, Y, Z, V_th)
    earth_diag = np.diag(Z)
    return _gic_at(target_idx, r.node_voltages_V, earth_diag), _network_total_abs_gic(
        r.node_voltages_V, earth_diag
    )


def _solve_with_gradient(
    net: PowerGridNetwork,
    Y: np.ndarray,
    Z: np.ndarray,
    target_idx: int,
    dEx_dx: float,
) -> tuple[float, float]:
    """Solve with Ex(x) = E_local - dEx_dx * (x - x_target), Ey = 0.

    Field at target is invariant across all dEx_dx values in the sweep.
    Only the spatial variation elsewhere changes.
    """
    x_target_km, _ = _target_local_xy_km(net, target_idx)

    # Unit book-keeping:
    #   E_LOCAL_VM       is V/m
    #   dEx_dx           is V/km per km  (paper convention, matches
    #                    the units used in the mechanism narrative)
    #   x_km             is km
    #
    # Ex_V_per_km = (E_LOCAL_VM * 1e3) - dEx_dx * (x_km - x_target_km)
    # Ex_V_per_m  = Ex_V_per_km * 1e-3
    e_local_v_per_km = E_LOCAL_VM * 1e3

    def field_fn(x_km: float, y_km: float) -> tuple[float, float]:
        del y_km  # H1 tests eastward-only spatial variation
        ex_v_per_km = e_local_v_per_km - dEx_dx * (x_km - x_target_km)
        return (ex_v_per_km * 1e-3, 0.0)

    ex, ey = _sample_field_at_branches(net, field_fn)
    V_th = net.compute_thevenin_voltages(ex_Vm=ex, ey_Vm=ey)
    r = NAMSolver().solve(net, Y, Z, V_th)
    earth_diag = np.diag(Z)
    return _gic_at(target_idx, r.node_voltages_V, earth_diag), _network_total_abs_gic(
        r.node_voltages_V, earth_diag
    )


def _per_node_abs_gic(
    net: PowerGridNetwork,
    Y: np.ndarray,
    Z: np.ndarray,
    V_th: np.ndarray,
) -> dict[str, float]:
    """Solve once, return {node_id: |GIC| A} for every grounded node."""
    r = NAMSolver().solve(net, Y, Z, V_th)
    earth_diag = np.diag(Z)
    out: dict[str, float] = {}
    for i, n in enumerate(net.get_nodes()):
        z = float(earth_diag[i])
        if z > 0.0 and np.isfinite(z):
            out[n.node_id] = float(abs(r.node_voltages_V[i] / z))
        else:
            out[n.node_id] = 0.0
    return out


def _plot_h1b(
    net: PowerGridNetwork,
    gic_uniform: dict[str, float],
    gic_gradient: dict[str, float],
    target_id: str,
    dEx_dx: float,
    savepath: Path,
) -> None:
    """H1b: two-panel per-substation GIC map (uniform vs gradient field)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

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

    vmax = max(max(gic_uniform.values()), max(gic_gradient.values()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("plasma")

    # Compute a shared, padded extent across both panels so all
    # substations and the target ring fit; equal-aspect is then set
    # with adjustable="box" so limits are honoured.
    lons_all = np.array([n.longitude_deg for n in sub_nodes])
    lats_all = np.array([n.latitude_deg for n in sub_nodes])
    lon_pad = 0.6
    lat_pad = 0.5
    xlim = (float(lons_all.min()) - lon_pad, float(lons_all.max()) + lon_pad)
    ylim = (float(lats_all.min()) - lat_pad, float(lats_all.max()) + lat_pad)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    panel_titles = (
        f"(a) uniform field  $E_x = {E_LOCAL_VM * 1e3:.1f}$ V/km",
        f"(b) gradient  $dE_x/dx = {dEx_dx:.3f}$   "
        f"[$E_x$ at target still $= {E_LOCAL_VM * 1e3:.1f}$ V/km]",
    )
    for ax, gic_dict, ttl in zip(axes, (gic_uniform, gic_gradient), panel_titles, strict=True):
        for br in branches:
            a = node_by_id.get(br.from_node)
            b = node_by_id.get(br.to_node)
            if a is None or b is None:
                continue
            if not (np.isfinite(a.latitude_deg) and np.isfinite(b.latitude_deg)):
                continue
            ax.plot(
                [a.longitude_deg, b.longitude_deg],
                [a.latitude_deg, b.latitude_deg],
                color="0.5",
                lw=0.8,
                zorder=1,
            )
        for n in sub_nodes:
            val = gic_dict.get(n.node_id, 0.0)
            ax.scatter(
                n.longitude_deg,
                n.latitude_deg,
                s=340,
                c=[cmap(norm(val))],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                f"{val:.0f} A",
                (n.longitude_deg, n.latitude_deg),
                xytext=(9, 4),
                textcoords="offset points",
                fontsize=9,
                zorder=4,
            )
        target = node_by_id.get(target_id)
        if target is not None:
            # Ring the target in blue. Radius in data units (degrees).
            r_deg = 0.22
            ring = Circle(
                (target.longitude_deg, target.latitude_deg),
                r_deg,
                fill=False,
                edgecolor="deepskyblue",
                linewidth=2.8,
                zorder=5,
            )
            ax.add_patch(ring)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")

    fig.suptitle(
        "H1b: how a fixed-at-target field but tilted-elsewhere redistributes GIC",
        fontsize=12,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.5,
        pad=0.12,
        aspect=40,
    )
    cbar.set_label("|GIC to ground| per substation  (A)")

    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity(
    dEx_dx_values: np.ndarray,
    gic_target: np.ndarray,
    baseline_target: float,
    savepath: Path,
) -> None:
    """H1a: single-panel signed GIC vs dEx/dx at the target substation."""
    fig, ax = plt.subplots(figsize=(9, 6))

    band_lo = min(baseline_target / 10.0, baseline_target * 10.0)
    band_hi = max(baseline_target / 10.0, baseline_target * 10.0)
    ax.axhspan(band_lo, band_hi, color="tab:blue", alpha=0.12, label="±1 order of magnitude band")

    ax.axhline(
        baseline_target,
        color="black",
        linestyle=":",
        lw=1.2,
        label=f"uniform-field GIC = {baseline_target:.1f} A",
    )
    ax.axvline(-0.02, color="0.5", linestyle="--", lw=0.8)
    ax.axvline(0.02, color="0.5", linestyle="--", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.5)

    ax.plot(
        dEx_dx_values,
        gic_target,
        "o-",
        color="crimson",
        lw=1.6,
        markersize=5,
        label=f"gradient-field GIC at {TARGET_NODE_ID}",
    )

    ax.set_xlim(-0.022, 0.022)
    ax.set_xlabel(r"$dE_x/dx$  (V/km per km, local east)")
    ax.set_ylabel(r"GIC at target (A)  [local $E_x$ fixed at 1.0 V/km]")
    ax.set_title(
        "H1a: spatial-field sensitivity of GIC at target substation\n"
        "(local field at target is IDENTICAL across all runs)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the full H1 sweep, print the table, save the figure."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    net = PowerGridNetwork.from_file(str(_EPRI21))
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    target_idx = _target_index(net, TARGET_NODE_ID)

    baseline_gic, baseline_total = _solve_uniform_baseline(net, Y, Z, target_idx)
    print(f"Target: {TARGET_NODE_ID} (idx {target_idx})")
    print(f"Local field at target: {E_LOCAL_VM * 1e3:.1f} V/km east (invariant across sweep)")
    print(
        f"\nUniform baseline (dE/dx = 0):  GIC(target) = {baseline_gic:+.2f} A  "
        f"|  network total = {baseline_total:.2f} A"
    )
    print()
    print(
        f"{'dEx/dx [V/km/km]':>16s}  {'GIC(target) [A]':>17s}  {'ratio to baseline':>18s}  "
        f"{'network total [A]':>18s}"
    )
    print("-" * 78)

    dEx_dx_arr = np.array(DEDX_SWEEP_V_PER_KM_PER_KM, dtype=np.float64)
    gic_target = np.zeros_like(dEx_dx_arr)
    gic_total = np.zeros_like(dEx_dx_arr)
    for i, dEx_dx in enumerate(dEx_dx_arr):
        g_t, g_all = _solve_with_gradient(net, Y, Z, target_idx, dEx_dx)
        gic_target[i] = g_t
        gic_total[i] = g_all
        ratio = g_t / baseline_gic if baseline_gic else float("nan")
        print(f"{dEx_dx:16.4e}  {g_t:17.2f}  {ratio:18.3f}  {g_all:18.2f}")

    figpath = _FIG_DIR / "h1a_sensitivity_curve.png"
    _plot_sensitivity(dEx_dx_arr, gic_target, baseline_gic, figpath)
    print(f"\nwrote {figpath}")

    # ---- H1b: per-substation redistribution map (uniform vs one gradient) ----
    V_uniform = net.compute_thevenin_voltages(ex_Vm=E_LOCAL_VM, ey_Vm=0.0)
    gic_uniform = _per_node_abs_gic(net, Y, Z, V_uniform)

    x_target_km, _ = _target_local_xy_km(net, target_idx)
    e_local_v_per_km = E_LOCAL_VM * 1e3

    def _grad_field(x_km: float, y_km: float) -> tuple[float, float]:
        del y_km
        ex_v_per_km = e_local_v_per_km - H1B_DEDX_V_PER_KM_PER_KM * (x_km - x_target_km)
        return (ex_v_per_km * 1e-3, 0.0)

    ex_arr, ey_arr = _sample_field_at_branches(net, _grad_field)
    V_grad = net.compute_thevenin_voltages(ex_Vm=ex_arr, ey_Vm=ey_arr)
    gic_gradient = _per_node_abs_gic(net, Y, Z, V_grad)

    figpath_b = _FIG_DIR / "h1b_redistribution_map.png"
    _plot_h1b(net, gic_uniform, gic_gradient, TARGET_NODE_ID, H1B_DEDX_V_PER_KM_PER_KM, figpath_b)
    print(f"wrote {figpath_b}")


if __name__ == "__main__":
    main()
