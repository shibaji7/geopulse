"""
GeoPulse Case Study #13: Czech Oil Pipelines — Halloween Storm 2003
====================================================================

Hejda & Bochníček (2005) reported pipe-to-soil voltage variations in
Czech oil pipelines during the October 2003 Halloween storm,
demonstrating that European mid-latitude pipelines are exposed to
significant telluric interference during major storms.

Infrastructure: Pipeline
Key references: Hejda & Bochníček (2005) Annals of Geophysics
Status: COMPLETE — DSTL solve of a Bohemian-region pipeline segment
        under a scaled Halloween-storm E-field.

This example demonstrates:
  - PSP magnitudes on a mid-latitude European pipeline under storm-strength
    telluric excitation (~2-3 V/km peak).
  - How a moderately conductive Central-European crust (~0.005 S/m) yields
    smaller PSP than the Scandinavian shield for the same B-field.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.geo import prime_vertical_radius_m
from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.solver.lpm import LPMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Pipeline: ~250 km oil pipe across Bohemia at 50° N (Družba pipeline
#    corridor).
# ---------------------------------------------------------------------------
L_m = 250_000.0
lat = 50.0
N_m = prime_vertical_radius_m(lat)
dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=3e-4,
    shunt_admittance_S_per_m=2e-6,  # slightly better coating than case 10
    start_lat_deg=lat,
    start_lon_deg=13.0,
    end_lat_deg=lat,
    end_lon_deg=13.0 + dlon,
    n_segments=80,
)
net = PipelineNetwork(params)

# ---------------------------------------------------------------------------
# 2. Sweep several E-field strengths that bracket the Halloween-storm
#    range: 1 V/km (typical strong storm) → 3 V/km (Halloween peak inferred
#    from Hejda & Bochníček).
# ---------------------------------------------------------------------------
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
E_kVpkm_list = [1.0, 2.0, 3.0]

fig, ax = plt.subplots(figsize=(8, 4))
for E_Vkm in E_kVpkm_list:
    V_th = net.compute_thevenin_voltages(ex_Vm=E_Vkm * 1e-3, ey_Vm=0.0)
    r = LPMSolver().solve(net, Y, Z, V_th)
    peak = float(np.max(np.abs(r.node_voltages_V)))
    print(f"E = {E_Vkm} V/km east  →  peak |V p-to-soil| = {peak:.1f} V")
    ax.plot(net.node_positions_m / 1000.0, np.abs(r.node_voltages_V), label=f"E = {E_Vkm} V/km")

ax.set_xlabel("Along-pipe distance (km)")
ax.set_ylabel("|V pipe-to-soil| (V)")
ax.set_title("Case 13: Czech oil pipeline — Halloween 2003 E-field sweep")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_13_czech_pipeline.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
