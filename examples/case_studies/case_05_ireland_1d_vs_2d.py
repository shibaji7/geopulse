"""
GeoPulse Case Study #05: Irish Power Grid — 1-D vs 2-D Earth Comparison
=========================================================================

Blake et al. (2016, 2018) built a detailed model of the Irish
400/220/110 kV transmission network and simulated GIC for the March
1989 and October 2003 storms, showing that the choice of Earth model
significantly affects results. Peak modelled GIC 23-26 A.

Infrastructure: Power grid.
Key references: Blake et al. (2016, 2018) Space Weather
Status: PARTIAL — 1-D vs 2-D (CoastalCorrection2D) impedance comparison
    runs cleanly; Horton grid used as TOPOLOGY PROXY since the Irish
    network file is not distributed.

This example demonstrates:
  - Difference in |Z| between homogeneous 1-D and coastal 2-D earth.
  - Ripple through to per-substation GIC on the proxy grid.
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

# 1. Two earth models: homogeneous 1-D vs coastal 2-D (Ireland is an island).
land = Layered1D([ConductivityLayer(np.inf, 5e-4)])
ocean = Layered1D(
    [
        ConductivityLayer(200.0, 3.3),
        ConductivityLayer(np.inf, 5e-4),
    ]
)
d_from_coast_m = 20_000.0  # ~inland Ireland

freqs = np.logspace(-3, -1, 25)
imp_1d = land.compute_impedance(freqs)
imp_2d = CoastalCorrection2D(land, ocean, d_from_coast_m).compute_impedance(freqs)
Z1 = np.abs(imp_1d.Z_values)
Z2_TE = np.abs(imp_2d.Z_tensor[:, 0, 1])

# 2. Solve LPM on proxy grid for both. Storm-representative 3 V/km E east.
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
V_th = net.compute_thevenin_voltages(ex_Vm=3e-3, ey_Vm=0.0)
r = LPMSolver().solve(net, Y, Z, V_th)
ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
print(f"peak substation |GIC| at 3 V/km east (proxy grid) = {np.max(np.abs(ng)):.2f} A")
print(f"|Z(1e-2 Hz)|  1-D = {Z1[np.argmin(abs(freqs - 1e-2))]:.3e} Ω")
print(f"|Z_TE(1e-2 Hz)|  2-D = {Z2_TE[np.argmin(abs(freqs - 1e-2))]:.3e} Ω")

# --- CHAIN STOPS HERE: Irish network topology not distributed. ---
# TODO: replace `_BENCH` with an Irish 400/220/110 kV network file.

# 3. Plot.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].loglog(freqs, Z1, "b-", label="1-D land")
axes[0].loglog(freqs, Z2_TE, "r-", label="2-D coastal TE (20 km inland)")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("|Z| (Ω)")
axes[0].set_title("Case 05: Ireland — 1-D vs 2-D impedance")
axes[0].legend()
axes[0].grid(True, alpha=0.3, which="both")
subs = [nid for i, nid in enumerate(r.node_ids) if diag_z[i] > 0]
axes[1].bar(subs, ng[diag_z > 0])
axes[1].set_ylabel("Substation GIC (A)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_title("Proxy-grid LPM at E = 3 V/km east")
fig.tight_layout()
out = _OUT / "case_05_ireland_1d_vs_2d.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
