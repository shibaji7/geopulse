"""
GeoPulse Case Study #02: Swedish Malmö Blackout — Halloween 30 Oct 2003
========================================================================

During the Halloween superstorm, GIC knocked out part of the Swedish
high-voltage transmission system. Malmö lost power for 20-50 minutes,
affecting ~50 000 customers. The largest GIC ever measured at that time
— nearly 300 A — was detected in Sweden.

Infrastructure: Power grid — with strong coastal-effect exposure (Baltic).
Key references: Pulkkinen et al. (2005) Space Weather; Wik et al. (2008)
Status: PARTIAL — CoastalCorrection2D + Layered1D drive the LPM solve on
    the Horton grid PROXY. The Swedish 400/220 kV topology itself is not
    distributed here.

This example demonstrates:
  - CoastalCorrection2D (Scandinavian shield vs. Baltic seawater).
  - How the TE-mode E-field is amplified near the coast — the dominant
    driver of the reported 300 A Sweden GIC record.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.earth.structured_2d import CoastalCorrection2D
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.lpm import LPMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Land: Fennoscandian shield (~1e-4 S/m). Ocean: 200 m Baltic + shield.
land = Layered1D([ConductivityLayer(np.inf, 1e-4)])
ocean = Layered1D(
    [
        ConductivityLayer(200.0, 3.3),  # shallow Baltic
        ConductivityLayer(np.inf, 1e-4),
    ]
)

# 2. Sweep site distance from the coast — Malmö sits ~5 km inland.
freqs = np.logspace(-3, -1, 30)
distances_km = np.array([-50, -10, 0, 5, 10, 50, 200])
Z_TE_amp = []
for d_km in distances_km:
    coast = CoastalCorrection2D(land, ocean, distance_from_coast_m=d_km * 1000.0)
    imp = coast.compute_impedance(freqs)
    # |Z_TE| at a representative frequency (~10 mHz, 100 s period).
    idx = int(np.argmin(np.abs(freqs - 1e-2)))
    Z_TE_amp.append(float(np.abs(imp.Z_tensor[idx, 0, 1])))
print("Distance-from-coast → |Z_TE| at f=10 mHz:")
for d, z in zip(distances_km, Z_TE_amp, strict=True):
    print(f"  d = {d:>4} km   |Z_TE| = {z:.4e}  Ω")

# 3. Solve LPM on the proxy grid using the coast-adjacent impedance to derive
#    a peak-storm E-field. Halloween 2003 in Sweden inferred ~5 V/km peak.
E_east_Vm = 5e-3
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
V_th = net.compute_thevenin_voltages(ex_Vm=E_east_Vm, ey_Vm=0.0)
r = LPMSolver().solve(net, Y, Z, V_th)
ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
print(f"\npeak substation |GIC| (proxy grid, 5 V/km east) = {np.max(np.abs(ng)):.1f} A")

# --- CHAIN STOPS HERE: Swedish 400 kV grid topology not distributed. ---
# TODO: swap `_BENCH` for a Svenska Kraftnät network file.
# TODO: drive with a real Uppsala magnetometer trace via INTERMAGNETSource.

# 4. Plot.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(distances_km, Z_TE_amp, "o-")
axes[0].set_xlabel("Distance from coast (km, + inland)")
axes[0].set_ylabel("|Z_TE| at 10 mHz (Ω)")
axes[0].set_title("Case 02: Baltic coast-effect on |Z_TE|")
axes[0].grid(alpha=0.3)
subs = [nid for i, nid in enumerate(r.node_ids) if diag_z[i] > 0]
axes[1].bar(subs, ng[diag_z > 0])
axes[1].set_ylabel("Substation GIC (A)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_title("Proxy grid LPM at 5 V/km east")
fig.tight_layout()
out = _OUT / "case_02_malmo_2003.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
