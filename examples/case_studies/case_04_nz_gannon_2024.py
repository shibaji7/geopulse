"""
GeoPulse Case Study #04: New Zealand Gannon Storm — May 2024
==============================================================

During the May 2024 Gannon storm (Dst_min ~ -412 nT), Transpower NZ
implemented a real-time GIC mitigation strategy (TP2022NZ) via network
reconfiguration. GIC measurements at Halfway Bush (Dunedin) showed
correlations between DC current, even-order harmonic generation, and
increased reactive power consumption in three-phase three-limb
transformers.

Infrastructure: Power grid + transformer thermal chain.
Key references: Clilverd et al. (2025) Space Weather; Mac Manus et al.
    (2025) Space Weather
Status: PARTIAL — full E → GIC → thermal chain runs; harmonics /
    even-order THD (`devices/harmonics`, `metrics/thd`) are stubs, so
    the Clilverd (2025) harmonic reproduction cannot be attempted yet.

This example demonstrates:
  - Gannon-scale (2500 nT peak) B-field over Otago-region 1-D earth.
  - Peak-day per-substation GIC snapshot on the Horton grid PROXY.
  - Transformer thermal envelope over the storm main phase.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft

from geopulse.devices.transformer import TransformerModel
from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver
from geopulse.sources.synthetic import SyntheticSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Otago-region earth (mid-latitude, moderately conductive):
#    single half-space at 0.003 S/m as a placeholder.
earth = Layered1D([ConductivityLayer(np.inf, 0.003)])

# 2. Gannon-scale B-field: 2500 nT, σ = 25 min.
src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=2500.0, sigma_s=1500.0)
b = src.load(0.0, 5.0 * 3600.0, 15.0)  # 15-s sampling, 5 h

# 3. E-field.
freqs, Bx_f, By_f = src.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))
print(f"peak |Ey| = {np.max(np.abs(Ey_t)) * 1e3:.2f} mV/m")

# 4. Per-timestep LPM on Horton proxy grid.
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
branch_ids = [b_.branch_id for b_ in net.get_branches()]
target_idx = branch_ids.index("dc_xf10_hi")

gic_A = np.empty_like(b.time_s)
solver = NAMSolver()
for i, e in enumerate(Ey_t):
    V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=e)
    r = solver.solve(net, Y, Z, V_th)
    gic_A[i] = r.branch_currents_A[target_idx]
print(f"peak |I| through target winding = {np.max(np.abs(gic_A)):.1f} A")

# 5. Peak-time per-substation GIC snapshot.
V_th_peak = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=float(np.max(np.abs(Ey_t))))
r_peak = solver.solve(net, Y, Z, V_th_peak)
ng_peak = np.where(diag_z > 0, r_peak.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)

# 6. Transformer thermal.
resp = TransformerModel(k_load=0.65).inject_gic(b.time_s, gic_A)
print(f"peak top-oil = {resp.metadata['peak_top_oil_C']:.1f} °C")
print(f"peak hot-spot = {resp.metadata['peak_hotspot_C']:.1f} °C")

# --- CHAIN STOPS HERE: even-order harmonics / THD are stubs. ---
# TODO: harmonics = geopulse.devices.harmonics.extract_harmonics(t, gic_A, ...)
# TODO: thd_series = geopulse.metrics.thd.compute_thd(harmonics, fundamental_Hz=50)
# TODO: compare to Clilverd et al. (2025) Fig 3-5 harmonic time series.
# TODO: quantify Mac Manus et al. (2025) 16 % reduction from mitigation by
#       toggling branch statuses via a case-configuration helper (WP2).

# 7. Plot.
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].plot(b.time_s / 3600.0, Ey_t * 1e3)
axes[0, 0].set_ylabel("Ey (mV/m)")
axes[0, 0].set_title("Case 04: NZ Gannon — main-phase E-field")
axes[0, 1].plot(b.time_s / 3600.0, np.abs(gic_A))
axes[0, 1].set_ylabel("|I| through winding (A)")
axes[0, 1].set_title("Target-transformer GIC")
axes[1, 0].plot(b.time_s / 3600.0, resp.top_oil_C, label="Top oil")
axes[1, 0].plot(b.time_s / 3600.0, resp.hotspot_C, label="Hot spot")
axes[1, 0].set_ylabel("Temperature (°C)")
axes[1, 0].set_xlabel("time (hours)")
axes[1, 0].legend()
subs = [nid for i, nid in enumerate(r_peak.node_ids) if diag_z[i] > 0]
axes[1, 1].bar(subs, ng_peak[diag_z > 0])
axes[1, 1].set_ylabel("Substation GIC (A)")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].set_title("Peak per-substation GIC")
fig.tight_layout()
out = _OUT / "case_04_nz_gannon_2024.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
