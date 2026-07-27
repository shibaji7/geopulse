"""
GeoPulse Case Study #22: Gannon Storm — Multi-Infrastructure Hindcast
=======================================================================

The Gannon storm (10-13 May 2024) is the first extreme event in the
modern dense-observation era. Published data exists or is emerging for
power grids, submarine cables, pipelines, and railways. Ranks third
since 1868 on the aa* index, after March 1989 and September 1941.

Infrastructure: Full engine — all four types, all Earth tiers.
Key references: Clilverd et al. (2025); Wilkerson et al. (2026);
    Chakraborty et al. (2026, SWW poster)
Status: PARTIAL — power-grid AND pipeline slices run cleanly under a
    common Gannon-scale E-field. Cable + railway slices are blocked
    because `network/cable` and `network/railway` are stubs.

This example demonstrates:
  - A single driving E-field feeds two infrastructure ABCs simultaneously.
  - Multi-infrastructure hindcasting from one engine — the eventual
    v2.0 capstone in miniature.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.geo import prime_vertical_radius_m
from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# Common driving field: Gannon-scale peak inferred ~3 V/km east on the
# NA mid-latitude corridor (representative; region-specific hindcasts
# vary from 2 to 8 V/km depending on latitude and ground σ).
E_east_Vm = 3e-3

# ---------------------------------------------------------------------------
# 1. GRID slice — Horton EPRI21 PROXY.
# ---------------------------------------------------------------------------
grid = PowerGridNetwork.from_file(str(_BENCH))
Y_g = grid.assemble_network_admittance()
Z_g = grid.assemble_earthing_impedance()
V_th_g = grid.compute_thevenin_voltages(ex_Vm=E_east_Vm, ey_Vm=0.0)
r_g = NAMSolver().solve(grid, Y_g, Z_g, V_th_g)
diag_zg = np.diag(Z_g)
gic_grid_A = np.where(diag_zg > 0, r_g.node_voltages_V / np.where(diag_zg > 0, diag_zg, 1.0), 0.0)
print(f"[GRID] peak substation |GIC| = {np.max(np.abs(gic_grid_A)):.2f} A")

# ---------------------------------------------------------------------------
# 2. PIPELINE slice — 300 km at 45 N.
# ---------------------------------------------------------------------------
lat = 45.0
L_m = 300_000.0
N_m = prime_vertical_radius_m(lat)
dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
pipe = PipelineNetwork(
    PipelineParameters(
        length_m=L_m,
        series_impedance_Ohm_per_m=3e-4,
        shunt_admittance_S_per_m=1e-6,
        start_lat_deg=lat,
        start_lon_deg=-75.0,
        end_lat_deg=lat,
        end_lon_deg=-75.0 + dlon,
        n_segments=80,
    )
)
Y_p = pipe.assemble_network_admittance()
Z_p = pipe.assemble_earthing_impedance()
V_th_p = pipe.compute_thevenin_voltages(ex_Vm=E_east_Vm, ey_Vm=0.0)
r_p = NAMSolver().solve(pipe, Y_p, Z_p, V_th_p)
psp_V = r_p.node_voltages_V
print(f"[PIPE] peak |V pipe-to-soil| = {np.max(np.abs(psp_V)):.2f} V")

# --- CHAIN STOPS HERE (partial): network/cable + network/railway are stubs. ---
# TODO: cable_net = CableNetwork.from_file("benchmarks/scubas/tat8.yaml")
# TODO: rail_net  = RailwayNetwork.from_file("benchmarks/patterson2023/glasgow_edinburgh.yaml")
# TODO: run both under the same E_east_Vm — same driving field, four
#       infrastructures, one engine. That is the v2.0 capstone.

# ---------------------------------------------------------------------------
# 3. Plot: grid GIC bar + pipeline PSP profile side-by-side.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
subs = [nid for i, nid in enumerate(r_g.node_ids) if diag_zg[i] > 0]
axes[0].bar(subs, gic_grid_A[diag_zg > 0])
axes[0].set_ylabel("Substation GIC (A)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_title("Case 22a: Gannon 2024 — GRID (Horton proxy)")

x_km = pipe.node_positions_m / 1000.0
axes[1].plot(x_km, psp_V)
axes[1].set_xlabel("Along-pipe distance (km)")
axes[1].set_ylabel("V pipe-to-soil (V)")
axes[1].set_title("Case 22b: Gannon 2024 — PIPELINE (300 km, 45 N)")
axes[1].grid(alpha=0.3)
fig.suptitle(f"Common driving field: E = {E_east_Vm * 1e3:.1f} V/km east", y=1.02)
fig.tight_layout()
out = _OUT / "case_22_gannon_multi_infra.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
