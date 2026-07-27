"""Pipeline DSTL — pipe-to-soil voltage profile.

Discretises a 100 km buried steel pipeline into 40 equivalent-π segments,
solves the LP GIC network under a uniform 1 V/km eastward geoelectric
field, and overlays the closed-form DSTL solution.

Reference: Boteler (1997) Fig 3 shape — antisymmetric about the midpoint,
peak |V| at the ends equal to (E/γ)·tanh(γL/2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.pipeline import (
    PipelineNetwork,
    PipelineParameters,
    pipe_to_soil_voltage_analytic,
)
from geopulse.solver.nam import NAMSolver

# 100 km pipeline, typical steel + degraded coating.
L_m = 100_000.0
z, y = 3e-4, 1e-6  # Ω/m, S/m
dlon = L_m / (111_000.0 * np.cos(np.radians(45.0)))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=z,
    shunt_admittance_S_per_m=y,
    start_lat_deg=45.0,
    start_lon_deg=-75.0,
    end_lat_deg=45.0,
    end_lon_deg=-75.0 + dlon,
    n_segments=40,
)
net = PipelineNetwork(params)
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)  # 1 V/km east
result = NAMSolver().solve(net, Y, Z, V_th)

x_km = net.node_positions_m / 1000.0
V_analytic = pipe_to_soil_voltage_analytic(1e-3, L_m, z, y, net.node_positions_m)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_km, np.abs(V_analytic), "k-", lw=2, label="Closed-form DSTL")
ax.plot(x_km, np.abs(result.node_voltages_V), "ro", ms=4, label="Equivalent-π + LPM (40 seg)")
ax.set_xlabel("Along-pipe distance (km)")
ax.set_ylabel("|V_pipe-to-soil| (V)")
ax.set_title(f"DSTL pipeline: L = {L_m / 1000:.0f} km, E = 1 V/km east")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

out = Path(__file__).parent / "output" / "pipeline_dstl.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print(f"wrote {out}")
print(
    f"peak |V|:  analytic {np.max(np.abs(V_analytic)):.2f} V, "
    f"network {np.max(np.abs(result.node_voltages_V)):.2f} V"
)
