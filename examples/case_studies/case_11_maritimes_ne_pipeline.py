"""
GeoPulse Case Study #11: Maritimes & Northeast Pipeline (Rix & Boteler 2001)
==============================================================================

Rix & Boteler (2001) incorporated telluric current considerations into
the cathodic protection design for this offshore-onshore gas pipeline.
Demonstrates DSTL used proactively in CP design rather than post-event
analysis.

Infrastructure: Pipeline (Nova Scotia).
Key references: Rix & Boteler (2001) CORROSION 2001; Rix & Boteler
    (2001) Ocean Resources
Status: PARTIAL — DSTL solve of the pipeline runs; the CP-design
    integration needs `devices/cp_unit` (stub).

This example demonstrates:
  - Peak PSP for a 400 km Maritimes-scale pipeline on typical Atlantic-
    Canada shield-margin conductivity.
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

# 1. 400 km pipeline at 45 N (Maritimes Canada).
L_m = 400_000.0
lat = 45.0
N_m = prime_vertical_radius_m(lat)
dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=3e-4,
    shunt_admittance_S_per_m=1.5e-6,
    start_lat_deg=lat,
    start_lon_deg=-66.0,  # ~Saint John, NB
    end_lat_deg=lat,
    end_lon_deg=-66.0 + dlon,
    n_segments=80,
)
net = PipelineNetwork(params)

# 2. Sweep the E-field direction (eastward + northward components) to
#    reflect the range of storm-time incidence a Maritimes pipeline sees.
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
angles_deg = np.arange(0.0, 181.0, 30.0)
peaks = []
for theta in angles_deg:
    ex = 1e-3 * np.cos(np.radians(theta))
    ey = 1e-3 * np.sin(np.radians(theta))
    V_th = net.compute_thevenin_voltages(ex_Vm=ex, ey_Vm=ey)
    r = LPMSolver().solve(net, Y, Z, V_th)
    peaks.append(float(np.max(np.abs(r.node_voltages_V))))
    print(f"θ = {theta:5.0f}° (E from north)  peak |V p-to-soil| = {peaks[-1]:.2f} V")

# --- CHAIN STOPS HERE: devices/cp_unit is a stub. ---
# TODO: feed the branch-current series into CathodicProtectionModel and
#       size rectifier stations for the resulting PSP swings.

# 3. Plot.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(angles_deg, peaks, "o-")
ax.set_xlabel("E-field direction (° from north)")
ax.set_ylabel("Peak |V pipe-to-soil| (V)")
ax.set_title("Case 11: Maritimes NE pipe — PSP vs incidence angle at |E|=1 V/km")
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_11_maritimes_ne_pipeline.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
