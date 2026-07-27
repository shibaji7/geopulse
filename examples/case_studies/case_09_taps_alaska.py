"""
GeoPulse Case Study #09: Trans-Alaska Pipeline System (TAPS)
==============================================================

The 1300 km pipeline runs from 62° to 69° geomagnetic latitude beneath
the auroral electrojet. Campbell (1978, 1980) first documented telluric
currents causing pipe-to-soil potential fluctuations that obscure
cathodic protection measurements.

Infrastructure: Pipeline under the auroral electrojet.
Key references: Campbell (1978) Pure Appl. Geophys.; Campbell (1980)
    Geophys. J. R. Astr. Soc.; Degerstedt et al. (1995)
Status: PARTIAL — DSTL solve of a 1300 km auroral pipeline runs; the
    cathodic-protection unit model (`devices/cp_unit`) is a stub, so
    the CP-shift calculation is deferred.

This example demonstrates:
  - Long-pipeline DSTL asymptotics on TAPS-scale geometry.
  - Peak PSP scales linearly with the applied E-field.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.geo import meridian_radius_m
from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.solver.nam import NAMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# 1. TAPS: ~1287 km, runs roughly N-S from Prudhoe Bay (70.3 N) to Valdez.
lat_start, lat_end = 70.3, 61.0
L_m = 1_287_000.0
# Endpoint offset via meridional radius at mid-lat.
mid_lat = 0.5 * (lat_start + lat_end)
M_m = meridian_radius_m(mid_lat)
dlat = float(np.degrees(-L_m / M_m))  # negative = southward
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=3e-4,
    shunt_admittance_S_per_m=1e-6,
    start_lat_deg=lat_start,
    start_lon_deg=-148.5,
    end_lat_deg=lat_start + dlat,
    end_lon_deg=-146.4,  # slight east drift
    n_segments=100,
)
net = PipelineNetwork(params)

# 2. Auroral-latitude storm E-field: 2 V/km north (aligned with pipe).
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=2e-3)
r = NAMSolver().solve(net, Y, Z, V_th)
peak_V = float(np.max(np.abs(r.node_voltages_V)))
print(f"L = {L_m / 1000:.0f} km, E = 2 V/km north  →  peak |V p-to-soil| = {peak_V:.1f} V")

# --- CHAIN STOPS HERE: devices/cp_unit is a stub. ---
# TODO: cp_response = geopulse.devices.cp_unit.CathodicProtectionModel(...)
#           .inject_gic(time_s, r.branch_currents_A, ...)  # WP3
# TODO: compare rectifier-station output shifts to Degerstedt et al. (1995)
#       "telluric-nulled" measurement traces.

# 3. Plot.
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(net.node_positions_m / 1000.0, np.abs(r.node_voltages_V))
ax.set_xlabel("Along-pipe distance from Prudhoe Bay (km)")
ax.set_ylabel("|V pipe-to-soil| (V)")
ax.set_title(f"Case 09: TAPS — L = {L_m / 1000:.0f} km, E = 2 V/km north")
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_09_taps_alaska.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
