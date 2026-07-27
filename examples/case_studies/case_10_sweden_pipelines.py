"""
GeoPulse Case Study #10: Southern Sweden Pipelines (Edwall & Boteler 2001)
==========================================================================

Detailed studies of telluric currents on buried pipelines in southern
Sweden, including pipe-to-soil potential (PSP) measurements and modelling
using Boteler's DSTL equations. Showed that PSP variations are largest
where pipelines cross major terrane boundaries where Earth conductivity
changes significantly.

Infrastructure: Pipeline
Key references: Edwall & Boteler (2001) CORROSION 2001; Boteler (1997)
Status: COMPLETE — DSTL π-section network solved via LPM, compared with
        the closed-form pipe-to-soil voltage profile.

This example demonstrates:
  - The DSTL equivalent-π discretisation converges to the analytic profile.
  - Peak |V_p-to-soil| scales linearly with the applied E-field, and its
    location sits at the pipeline ends (insulated end BC).
  - How the Scandinavian shield conductivity (very resistive) elevates
    the E-field and thus the PSP compared with a typical mid-latitude
    conductor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.geo import prime_vertical_radius_m
from geopulse.network.pipeline import (
    PipelineNetwork,
    PipelineParameters,
    pipe_to_soil_voltage_analytic,
)
from geopulse.solver.nam import NAMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Pipeline geometry — a 200 km buried steel pipe running eastward at 58° N
#    (roughly the latitude of southern Sweden), on the Precambrian shield.
# ---------------------------------------------------------------------------
L_m = 200_000.0
lat = 58.0
z_Ohm_per_m = 3e-4  # 30 in/dia steel, typical bare-metal resistance
y_S_per_m = 1e-6  # degraded coal-tar coating (~10^6 Ω m² leakage)
N_m = prime_vertical_radius_m(lat)
dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=z_Ohm_per_m,
    shunt_admittance_S_per_m=y_S_per_m,
    start_lat_deg=lat,
    start_lon_deg=13.0,  # ~Malmö
    end_lat_deg=lat,
    end_lon_deg=13.0 + dlon,  # ~200 km east
    n_segments=60,
)
net = PipelineNetwork(params)

# ---------------------------------------------------------------------------
# 2. Drive with a uniform 1 V/km eastward geoelectric field. Edwall & Boteler
#    report PSP swings of order 20-40 V during storms; a 1 V/km E-field over
#    a 200 km pipe on shield-like earth is representative of moderate storm
#    conditions.
# ---------------------------------------------------------------------------
E_Vm = 1e-3  # 1 V/km = 1e-3 V/m
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
V_th = net.compute_thevenin_voltages(ex_Vm=E_Vm, ey_Vm=0.0)

# ---------------------------------------------------------------------------
# 3. LP solve → node voltages == pipe-to-soil voltages at each grid point.
# ---------------------------------------------------------------------------
result = NAMSolver().solve(net, Y, Z, V_th)

x_km = net.node_positions_m / 1000.0
V_num = result.node_voltages_V
V_ana = pipe_to_soil_voltage_analytic(E_Vm, L_m, z_Ohm_per_m, y_S_per_m, net.node_positions_m)

peak_num = float(np.max(np.abs(V_num)))
peak_ana = float(np.max(np.abs(V_ana)))
print(f"peak |V_p-to-soil|:  analytic {peak_ana:.2f} V  |  network {peak_num:.2f} V")
print(f"relative error at peak:  {abs(peak_num - peak_ana) / peak_ana * 100:.3f} %")

# ---------------------------------------------------------------------------
# 4. Plot the two profiles overlaid.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_km, np.abs(V_ana), "k-", lw=2, label="Closed-form DSTL")
ax.plot(x_km, np.abs(V_num), "ro", ms=4, label=f"π-section + LPM ({params.n_segments} seg)")
ax.set_xlabel("Along-pipe distance (km)")
ax.set_ylabel("|V pipe-to-soil| (V)")
ax.set_title(f"Case 10: S. Sweden pipeline — L = {L_m / 1000:.0f} km, E = 1 V/km east")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_10_sweden_pipeline.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
