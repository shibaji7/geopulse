"""
GeoPulse Case Study #03: South African Transformer Damage — Oct/Nov 2003
==========================================================================

The Halloween storm damaged 15 power transformers at ESKOM mid-latitude
substations through internal heating from GIC-driven half-cycle
saturation. One 665 MW generator transformer failed weeks after the
storm. This event proved that mid-latitude countries with resistive
lithospheres are also at significant risk.

Infrastructure: Power grid + transformer thermal chain.
Key references: Gaunt & Coetzee (2007); Koen & Gaunt (2003); Bernhardi
    et al. (2008)
Status: PARTIAL — Kaapvaal-region resistive earth + Horton grid PROXY.
    Full transformer thermal cascade is exercised. The Kaapvaal
    conductivity model here is a placeholder single-half-space; a
    published MT model would replace it.

This example demonstrates:
  - How a highly resistive craton (~5e-4 S/m) amplifies E for the same B.
  - Sustained GIC drives transformer hot-spot into the danger zone.
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
from geopulse.solver.lpm import LPMSolver
from geopulse.sources.synthetic import SyntheticSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Kaapvaal-craton-like resistive earth: 5e-4 S/m half-space.
#    (Placeholder; a published Kaapvaal MT model would replace it.)
earth = Layered1D([ConductivityLayer(np.inf, 5e-4)])

# 2. Multi-hour Halloween-like storm B-field envelope: 1500 nT, σ = 30 min.
src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=1500.0, sigma_s=1800.0)
b = src.load(start_s=0.0, end_s=6.0 * 3600.0, dt_s=10.0)  # 10-s sampling, 6 h

# 3. E-field.
freqs, Bx_f, By_f = src.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))
print(f"peak |Ey| over resistive Kaapvaal = {np.max(np.abs(Ey_t)) * 1e3:.2f} mV/m")

# 4. LPM at each time step — feeds a time series into the thermal model.
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
branch_ids = [br.branch_id for br in net.get_branches()]
target_idx = branch_ids.index("dc_xf10_hi")

gic_A = np.empty_like(b.time_s)
solver = LPMSolver()
for i, e in enumerate(Ey_t):
    V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=e)
    r = solver.solve(net, Y, Z, V_th)
    gic_A[i] = r.branch_currents_A[target_idx]
print(f"peak |I| through target winding = {np.max(np.abs(gic_A)):.1f} A")

# 5. Feed into transformer thermal.
resp = TransformerModel(k_load=0.70).inject_gic(b.time_s, gic_A)
print(f"peak top-oil = {resp.metadata['peak_top_oil_C']:.1f} °C")
print(f"peak hot-spot = {resp.metadata['peak_hotspot_C']:.1f} °C")
print(f"8-h avg limit exceeded? {resp.metadata['avg_limit_exceeded']}")

# --- CHAIN STOPS HERE: Kaapvaal MT model + real ESKOM grid missing. ---
# TODO: replace the single-half-space with a published Kaapvaal-region
#       MT-derived Layered1D.
# TODO: replace Horton grid with the ESKOM 400/765 kV topology.
# TODO: drive with a real Hermanus (HER) magnetometer trace via
#       INTERMAGNETSource.

# 6. Plot: E-field, GIC, temperatures.
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(b.time_s / 3600.0, Ey_t * 1e3)
axes[0].set_ylabel("Ey (mV/m)")
axes[0].set_title("Case 03: South Africa 2003 — resistive earth amplifies GIC")
axes[1].plot(b.time_s / 3600.0, np.abs(gic_A))
axes[1].set_ylabel("|I| through winding (A)")
axes[2].plot(b.time_s / 3600.0, resp.top_oil_C, label="Top oil")
axes[2].plot(b.time_s / 3600.0, resp.hotspot_C, label="Hot spot")
axes[2].axhline(240, color="k", ls="--", alpha=0.5, label="8-h limit 240 °C")
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_xlabel("time (hours)")
axes[2].legend(loc="upper right")
fig.tight_layout()
out = _OUT / "case_03_south_africa_2003.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
