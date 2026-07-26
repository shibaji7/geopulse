"""
GeoPulse Case Study #08: Brazilian Low-Latitude GIC — Itumbiara 500 kV
========================================================================

Barbosa et al. (2015) analysed long-term GIC monitoring at Itumbiara.
Measured GIC reached ~30 A during Halloween 2003. The GIC frequency
distribution follows a q-exponential Tsallis distribution — a
return-period analysis natural for `metrics/exceedance`.

Infrastructure: Power grid at equatorial latitude.
Key references: Barbosa et al. (2015) J. Space Weather Space Clim.
Status: PARTIAL — LPM runs on the Horton PROXY driven by an equatorial-
    scale B-field envelope; `metrics/exceedance` (return-period stats)
    is a stub, so the Tsallis analysis is deferred.

This example demonstrates:
  - Order-of-magnitude Itumbiara-scale GIC on the proxy grid.
  - A histogram of peak-GIC per synthetic sub-storm (~return period).
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
rng = np.random.default_rng(42)

# 1. Brazilian shield-like earth: moderately conductive (~0.005 S/m).
earth = Layered1D([ConductivityLayer(np.inf, 0.005)])

# 2. Simulate 200 sub-storm-scale Gaussian pulses with amplitudes drawn from a
#    heavy-tailed distribution — approximates the Tsallis regime.
n_events = 200
amps_nT = rng.lognormal(mean=np.log(80), sigma=1.0, size=n_events).clip(20, 2000)
peaks_A = []
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
solver = LPMSolver()

for amp in amps_nT:
    src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=float(amp), sigma_s=600.0)
    b = src.load(0.0, 1800.0, 5.0)
    freqs, Bx_f, By_f = src.to_frequency_domain(b)
    imp = earth.compute_impedance(freqs)
    Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
    peak_E = float(np.max(np.abs(irfft(Ey_f, n=len(b.time_s)))))
    V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=peak_E)
    r = solver.solve(net, Y, Z, V_th)
    ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
    peaks_A.append(float(np.max(np.abs(ng))))

peaks_A = np.asarray(peaks_A)
print(f"peak-GIC over {n_events} synthetic events:")
print(f"  median = {np.median(peaks_A):.2f} A   max = {np.max(peaks_A):.2f} A")
print(f"  99th %ile = {np.percentile(peaks_A, 99):.2f} A")

# --- CHAIN STOPS HERE: metrics/exceedance is a stub. ---
# TODO: fit q-exponential Tsallis distribution to `peaks_A` via
#       geopulse.metrics.exceedance.return_period(...) (WP4).

# 3. Plot: input amplitude distribution + output peak-GIC distribution.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(amps_nT, bins=30, color="steelblue")
axes[0].set_xlabel("Input B-field amplitude (nT)")
axes[0].set_ylabel("count")
axes[0].set_title("Case 08: input storm-strength distribution")
axes[1].hist(peaks_A, bins=30, color="tomato")
axes[1].set_xlabel("Peak substation |GIC| (A)")
axes[1].set_ylabel("count")
axes[1].set_title("Peak-GIC distribution (proxy grid)")
axes[1].set_yscale("log")
fig.tight_layout()
out = _OUT / "case_08_brazil_itumbiara.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
