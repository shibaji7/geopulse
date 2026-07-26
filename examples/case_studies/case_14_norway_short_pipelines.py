"""
GeoPulse Case Study #14: Norwegian Short Gas Pipelines (Hesjevik & Birketveit 2001)
====================================================================================

Hesjevik & Birketveit (2001) studied telluric currents on SHORT gas
pipelines in Norway, finding that even short pipelines are affected —
challenging the assumption that only long pipelines are vulnerable.

Infrastructure: Pipeline
Key references: Hesjevik & Birketveit (2001); Trichtchenko et al. (2001)
Status: COMPLETE — DSTL run over a sweep of pipeline lengths, showing
        the peak PSP saturating as L exceeds ~2/γ.

This example demonstrates:
  - Even short pipelines (L << 1/γ) develop measurable PSP.
  - Peak PSP saturates at (E/γ) as L → ∞: the "short-pipeline" regime is
    everything below ~2 × (1/γ).
  - Where the transition sits for a typical Norwegian pipeline
    (steel, moderate coating, mid-latitude shield-like earth).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.geo import prime_vertical_radius_m
from geopulse.network.pipeline import PipelineNetwork, PipelineParameters
from geopulse.solver.lpm import LPMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DSTL parameters representative of a coated Norwegian gas pipeline.
#    γ = sqrt(z·y) ≈ 1.7e-5 /m so the characteristic length 1/γ ≈ 59 km.
# ---------------------------------------------------------------------------
z_Ohm_per_m = 3e-4
y_S_per_m = 1e-6
gamma = float(np.sqrt(z_Ohm_per_m * y_S_per_m))
one_over_gamma_km = 1e-3 / gamma
print(f"characteristic length 1/γ  ≈  {one_over_gamma_km:.1f} km")

# ---------------------------------------------------------------------------
# 2. Sweep pipeline length from 5 km ("short") to 400 km ("long").
# ---------------------------------------------------------------------------
E_Vm = 1e-3  # 1 V/km east — moderate storm
lat = 60.0  # Norway
N_m = prime_vertical_radius_m(lat)

length_km_list = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
peak_V_list = []
for L_km in length_km_list:
    L_m = L_km * 1000.0
    dlon = float(np.degrees(L_m / (N_m * np.cos(np.radians(lat)))))
    params = PipelineParameters(
        length_m=L_m,
        series_impedance_Ohm_per_m=z_Ohm_per_m,
        shunt_admittance_S_per_m=y_S_per_m,
        start_lat_deg=lat,
        start_lon_deg=10.0,
        end_lat_deg=lat,
        end_lon_deg=10.0 + dlon,
        n_segments=max(20, int(L_km / 2)),
    )
    net = PipelineNetwork(params)
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(ex_Vm=E_Vm, ey_Vm=0.0)
    r = LPMSolver().solve(net, Y, Z, V_th)
    peak = float(np.max(np.abs(r.node_voltages_V)))
    peak_V_list.append(peak)
    print(f"L = {L_km:>4} km  →  peak |V p-to-soil| = {peak:6.2f} V")

# ---------------------------------------------------------------------------
# 3. Asymptote:  peak |V| →  (E/γ) tanh(γL/2)  →  E/γ  as γL/2 >> 1.
# ---------------------------------------------------------------------------
asymptote_V = E_Vm / gamma  # ≈ 59 V for 1 V/km east
print(f"\nasymptote (γL/2 → ∞):  |V|_peak → E/γ = {asymptote_V:.2f} V")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(length_km_list, peak_V_list, "bo-", label="DSTL π-section + LPM")
ax.axhline(asymptote_V, color="red", ls="--", label=f"asymptote  E/γ = {asymptote_V:.1f} V")
ax.axvline(one_over_gamma_km, color="grey", ls=":", label=f"1/γ = {one_over_gamma_km:.0f} km")
ax.set_xlabel("Pipeline length L (km)")
ax.set_ylabel("Peak |V pipe-to-soil| (V)")
ax.set_title("Case 14: short-pipeline saturation — Hesjevik & Birketveit 2001")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_14_norway_short_pipelines.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
