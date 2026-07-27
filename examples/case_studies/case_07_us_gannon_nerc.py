"""
GeoPulse Case Study #07: US Power Grid — Gannon Storm & NERC Benchmark
=======================================================================

Wilkerson et al. (2026) analysed GIC observations across the US during
the Gannon storm. NERC benchmark defines a peak surface electric field
of 8 V/km at 60° N for the reference GMD event (TPL-007-4). Peak GIC
magnitude scales linearly with geomagnetic latitude and ground
conductivity.

Infrastructure: Power grid + regulatory benchmark.
Key references: Wilkerson et al. (2026) Space Weather; NERC TPL-007-4
Status: PARTIAL — LPM runs under the NERC 8 V/km design field. Real US
    grid topology is not distributed; USGS physiographic-region models
    are not in the library yet (would be a natural WP2 addition).

This example demonstrates:
  - LPM under the NERC 8 V/km design E-field.
  - Effect of ground conductivity on peak substation |GIC| (sweep).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. NERC reference E-field.
E_NERC_Vm = 8e-3  # 8 V/km, per TPL-007-4

# 2. Sweep conductivity of an assumed 1-D half-space earth to show the
#    Wilkerson (2026) scaling with ground conductivity.
sigma_Sm_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
peak_gic_A = []
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)

for sigma in sigma_Sm_list:
    # The 8 V/km design field is applied directly; the earth model here is
    # just documentation — for a plane-wave uniform E-field, no impedance
    # step is needed. Print |Z(10 mHz)| for context.
    earth = Layered1D([ConductivityLayer(np.inf, sigma)])
    Zref = float(np.abs(earth.compute_impedance(np.array([1e-2])).Z_values[0]))
    V_th = net.compute_thevenin_voltages(ex_Vm=E_NERC_Vm, ey_Vm=0.0)
    r = NAMSolver().solve(net, Y, Z, V_th)
    ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
    peak_gic_A.append(float(np.max(np.abs(ng))))
    print(f"σ = {sigma:.0e} S/m  |Z@10mHz| = {Zref:.2e} Ω  peak |GIC| = {peak_gic_A[-1]:.1f} A")

# --- CHAIN STOPS HERE: US grid topology + USGS regions not distributed. ---
# TODO: expand earth/library with the USGS physiographic-region models
#       (Fernberg 2012 EPRI report).
# TODO: replace Horton grid with a bulk US 500 kV network file.

# 3. Plot GIC vs σ.
fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(sigma_Sm_list, peak_gic_A, "o-")
ax.set_xlabel("Half-space σ (S/m)")
ax.set_ylabel("Peak substation |GIC| under NERC 8 V/km (A)")
ax.set_title("Case 07: NERC benchmark GIC vs ground conductivity")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = _OUT / "case_07_us_gannon_nerc.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
