"""
GeoPulse Case Study #06: Japanese Power Grid — Low-Latitude GIC
=================================================================

Watari et al. (2009, 2015) measured GIC in a 500 kV transformer at
Shin-Fukushima. Despite Japan's low geomagnetic latitude, GIC of up to
30 A was calculated during the Halloween storm. Telegraph lines between
Tokyo and regional cities were affected by an 1909 storm.

Infrastructure: Power grid at low geomagnetic latitude.
Key references: Watari et al. (2009) Earth Planets Space; Watari et al.
    (2015)
Status: PARTIAL — Halloween-scale B-field over a Japan-representative
    resistivity produces the expected sub-100 A GIC on the Horton PROXY.
    The real Tokyo Electric 500 kV grid is not distributed.

This example demonstrates:
  - GIC is NOT negligible at low geomagnetic latitudes.
  - Reproduces the Watari (2015) order-of-magnitude on the proxy grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.lpm import LPMSolver
from geopulse.sources.synthetic import SyntheticSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Japan sits on a moderately conductive volcanic-arc crust (~0.005 S/m).
earth = Layered1D([ConductivityLayer(np.inf, 0.005)])

# 2. Halloween 2003 at low latitude: B-field magnitudes are smaller
#    (~500 nT peak vs auroral thousands).
src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=500.0, sigma_s=1200.0)
b = src.load(0.0, 4.0 * 3600.0, 15.0)

# 3. E-field.
freqs, Bx_f, By_f = src.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))
peak_Ey_Vm = float(np.max(np.abs(Ey_t)))
print(f"peak |Ey| = {peak_Ey_Vm * 1e3:.2f} mV/m  (low-lat, 500 nT B, 0.005 S/m)")

# 4. LPM at peak.
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=peak_Ey_Vm)
r = LPMSolver().solve(net, Y, Z, V_th)
ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
print(f"peak substation |GIC| (proxy) = {np.max(np.abs(ng)):.2f} A")
print("(Watari 2015 reported ~30 A at Shin-Fukushima)")

# --- CHAIN STOPS HERE: real Japanese 500 kV grid not distributed. ---
# TODO: replace with a Tokyo Electric grid file.

# 5. Plot.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(b.time_s / 3600.0, Ey_t * 1e3)
axes[0].set_ylabel("Ey (mV/m)")
axes[0].set_xlabel("time (hours)")
axes[0].set_title("Case 06: Japan (low-lat) — E-field over Halloween envelope")
subs = [nid for i, nid in enumerate(r.node_ids) if diag_z[i] > 0]
axes[1].bar(subs, ng[diag_z > 0])
axes[1].set_ylabel("Substation GIC (A)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_title("Proxy-grid LPM at peak E")
fig.tight_layout()
out = _OUT / "case_06_japan_low_latitude.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
