"""
GeoPulse Case Study #12: NZ Gas Pipeline — 2015-2022
======================================================

Ingham & Rodger (2018) and Ingham et al. (2022) studied CP monitoring
data from North Island gas pipelines. PSP variations were driven by the
EAST-WEST geoelectric field component rather than the north-south —
contrary to the simple auroral-electrojet expectation. Attributed to
complex NZ conductivity structure.

Infrastructure: Pipeline in complex mid-latitude geology.
Key references: Ingham & Rodger (2018) Space Weather; Ingham et al.
    (2022) Space Weather
Status: PARTIAL — DSTL runs; demonstrating the E-W vs N-S sensitivity is
    trivial once we orient the pipe. The CP monitoring correlation needs
    `devices/cp_unit` (stub).

This example demonstrates:
  - Peak PSP for a NZ-orientated pipeline under N-only vs E-only fields.
  - Directional PSP dependence explaining Ingham 2018's east-west
    correlation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.solver.lpm import LPMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# 1. NZ North Island Maui pipeline is largely N-S. Model a 300 km N-S run.
lat_start, lat_end = -37.5, -40.2  # Taranaki → Wellington
L_m = 300_000.0
from geopulse.geo import meridian_radius_m  # noqa: E402

M = meridian_radius_m(0.5 * (lat_start + lat_end))
dlat = float(np.degrees(-L_m / M))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=3e-4,
    shunt_admittance_S_per_m=2e-6,
    start_lat_deg=lat_start,
    start_lon_deg=174.0,
    end_lat_deg=lat_start + dlat,
    end_lon_deg=174.0,  # pure N-S run
    n_segments=80,
)
net = PipelineNetwork(params)

# 2. Compare a pure-N vs pure-E field of the same magnitude.
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
E_mag = 1e-3  # 1 V/km

V_th_east = net.compute_thevenin_voltages(ex_Vm=E_mag, ey_Vm=0.0)
V_th_north = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=E_mag)
r_east = LPMSolver().solve(net, Y, Z, V_th_east)
r_north = LPMSolver().solve(net, Y, Z, V_th_north)

peak_east = float(np.max(np.abs(r_east.node_voltages_V)))
peak_north = float(np.max(np.abs(r_north.node_voltages_V)))
print(f"E-only field (perpendicular to pipe):  peak |V| = {peak_east:.2f} V")
print(f"N-only field (parallel to pipe):        peak |V| = {peak_north:.2f} V")
print("Ingham (2018) reported E-W dominance for the Maui pipeline; for a")
print("pure N-S pipe geometry, the LARGER response should come from the")
print("pipe-PARALLEL (N-only here) component. The Ingham puzzle is that")
print("their E-W dominance implies an unexpected geometry / geology.")

# --- CHAIN STOPS HERE: devices/cp_unit is a stub. ---
# TODO: correlate branch-current time series with CP monitoring records
#       via CathodicProtectionModel — reproduce the E-W correlation
#       observed at multiple pipeline sites (Ingham 2018 Fig 5-7).

# 3. Plot both PSP profiles.
fig, ax = plt.subplots(figsize=(9, 4))
x_km = net.node_positions_m / 1000.0
ax.plot(x_km, np.abs(r_north.node_voltages_V), "b-", label="E north-only (∥ pipe)")
ax.plot(x_km, np.abs(r_east.node_voltages_V), "r-", label="E east-only (⊥ pipe)")
ax.set_xlabel("Along-pipe distance (km)")
ax.set_ylabel("|V pipe-to-soil| (V)")
ax.set_title("Case 12: NZ N-S pipeline — E-direction sensitivity")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_12_nz_gas_pipeline.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
