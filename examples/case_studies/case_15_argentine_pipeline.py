"""
GeoPulse Case Study #15: Argentine Buried Pipeline — Corrosion Study
======================================================================

Osella et al. (1998) studied geomagnetic-storm-induced currents in
Argentine buried pipelines as a cause of corrosion, demonstrating
southern-hemisphere mid-latitude vulnerability.

Infrastructure: Pipeline in southern hemisphere.
Key references: Osella et al. (1998) J. Applied Geophysics
Status: PARTIAL — DSTL solve runs; corrosion-rate quantification would
    need `devices/cp_unit` and `metrics/exceedance` (both stubs).

This example demonstrates:
  - PSP time-history for a 500 km southern-cone pipeline under a synthetic
    storm E-field envelope.
  - Cumulative PSP-swing histogram — the input to future corrosion-rate
    integration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.geo import prime_vertical_radius_m
from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.solver.lpm import LPMSolver
from geopulse.sources.synthetic import SyntheticSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# 1. Southern-cone pipeline: 500 km, ~-35 S.
L_m = 500_000.0
lat = -35.0
N_m = prime_vertical_radius_m(lat)
dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
params = PipelineParameters(
    length_m=L_m,
    series_impedance_Ohm_per_m=3e-4,
    shunt_admittance_S_per_m=1e-6,
    start_lat_deg=lat,
    start_lon_deg=-63.0,  # ~Neuquén
    end_lat_deg=lat,
    end_lon_deg=-63.0 + dlon,
    n_segments=100,
)
net = PipelineNetwork(params)

# 2. Storm envelope → E(t).
earth = Layered1D([ConductivityLayer(np.inf, 5e-3)])
src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=800.0, sigma_s=1500.0)
b = src.load(0.0, 5.0 * 3600.0, 20.0)
freqs, Bx_f, By_f = src.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))

# 3. Per-time-step DSTL solve → downsampled midpoint PSP time series.
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
solver = LPMSolver()
# Track voltage at the pipe midpoint node.
mid_node_idx = params.n_segments // 2
V_mid_t = np.empty_like(b.time_s)
V_end_t = np.empty_like(b.time_s)
for i, e in enumerate(Ey_t):
    V_th = net.compute_thevenin_voltages(ex_Vm=e, ey_Vm=0.0)
    r = solver.solve(net, Y, Z, V_th)
    V_mid_t[i] = r.node_voltages_V[mid_node_idx]
    V_end_t[i] = r.node_voltages_V[-1]

print(f"peak |V| at midpoint = {np.max(np.abs(V_mid_t)):.2f} V")
print(f"peak |V| at end      = {np.max(np.abs(V_end_t)):.2f} V")

# --- CHAIN STOPS HERE: devices/cp_unit + metrics/exceedance are stubs. ---
# TODO: integrate |dV/dt|·Δt over the storm to estimate coating charge
#       transfer → corrosion rate via cp_unit.
# TODO: fit return-period statistics via metrics/exceedance for
#       1-in-100-year PSP swings.

# 4. Plot.
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(b.time_s / 3600.0, Ey_t * 1e3)
axes[0].set_ylabel("Ex (mV/m)")
axes[0].set_title("Case 15: Argentine pipeline — PSP time-history")
axes[1].plot(b.time_s / 3600.0, V_mid_t, label="midpoint")
axes[1].plot(b.time_s / 3600.0, V_end_t, label="end")
axes[1].set_ylabel("V pipe-to-soil (V)")
axes[1].set_xlabel("time (hours)")
axes[1].legend()
fig.tight_layout()
out = _OUT / "case_15_argentine_pipeline.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
