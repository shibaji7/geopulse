"""
GeoPulse Case Study #01: Hydro-Québec Blackout — 13 March 1989
================================================================

The foundational GIC event. A severe geomagnetic storm caused a complete
blackout of the Hydro-Québec system at 07:45 UT, leaving millions
without power for up to nine hours. GIC of 220 A was measured at the
Radisson substation on the Radisson–Sandy Pond HVDC link, causing loss
of the link.

Infrastructure: Power grid
Key references: Bolduc (2002) J. Atmos. Sol.-Terr. Phys.; Czech et al.
    (1992); Béland & Small (2004)
Status: PARTIAL — GeoPulse ships the Québec 7-layer earth model, the LPM
    solver, and the transformer thermal model. The Hydro-Québec grid
    topology itself is not distributed here (proprietary), so this
    script uses the Horton EPRI21 grid as a topology PROXY driven by a
    Québec-region earth model + a scaled 1989-like B-field pulse.

This example demonstrates:
  - Wait-recursion impedance for the Québec crust (Precambrian shield).
  - Peak Ey response to a ~2000 nT / 15-min B-field disturbance.
  - LPM per-substation GIC snapshot at the storm peak (proxy grid).
  - Transformer hot-spot response to a target-branch current.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft

from geopulse.devices.transformer import TransformerModel
from geopulse.earth.library import get_model
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver
from geopulse.sources.synthetic import SyntheticSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Québec-region earth (Precambrian shield, top-crust ~2.5e-4 S/m).
earth = get_model("quebec_7layer")

# 2. Synthetic March-1989-scale B-field: 2000 nT peak Gaussian, 15-min half-width.
src = SyntheticSource(waveform="gaussian_pulse", amplitude_nT=2000.0, sigma_s=900.0)
b = src.load(start_s=0.0, end_s=7200.0, dt_s=1.0)

# 3. E-field.
freqs, Bx_f, By_f = src.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ex_t = irfft(Ex_f, n=len(b.time_s))
Ey_t = irfft(Ey_f, n=len(b.time_s))
peak_Ey_Vm = float(np.max(np.abs(Ey_t)))
print(f"peak |Ey| over Québec-shield 1-D earth = {peak_Ey_Vm * 1e3:.2f} mV/m")

# 4. LPM on the Horton EPRI21 grid — TOPOLOGY PROXY (Hydro-Québec not shipped).
net = PowerGridNetwork.from_file(str(_BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=peak_Ey_Vm)
r = NAMSolver().solve(net, Y, Z, V_th)
ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
peak_gic_A = float(np.max(np.abs(ng)))
peak_node = r.node_ids[int(np.argmax(np.abs(ng)))]
print(f"peak substation |GIC| (proxy grid) = {peak_gic_A:.1f} A  at  {peak_node}")

# 5. Transformer hot-spot for the highest-GIC winding.
branch_ids = r.branch_ids
target = "dc_xf10_hi"  # T10 GSU at Sub 8 (matches examples/05)
idx = branch_ids.index(target)
gic_A_target = np.full_like(b.time_s, r.branch_currents_A[idx])
# Drive over the same 2-hour window
resp = TransformerModel(k_load=0.6).inject_gic(b.time_s, gic_A_target)
print(f"transformer top-oil peak = {resp.metadata['peak_top_oil_C']:.1f} °C")
print(f"transformer hot-spot peak = {resp.metadata['peak_hotspot_C']:.1f} °C")

# --- CHAIN STOPS HERE: real Hydro-Québec topology not distributed. ---
# TODO: replace `_BENCH` with a Québec-region network file once available.
# TODO: drive with a real SuperMAG/OTT magnetometer trace via
#       SuperMAGSource.load(...) rather than the synthetic Gaussian.

# 6. Plot: B, E, and the modelled per-substation GIC bar chart.
fig, axes = plt.subplots(3, 1, figsize=(9, 8))
axes[0].plot(b.time_s / 60.0, b.bx_T * 1e9)
axes[0].set_ylabel("Bx (nT)")
axes[0].set_title("Case 01: Hydro-Québec 1989 (proxy topology, Québec 7-layer earth)")
axes[1].plot(b.time_s / 60.0, Ey_t * 1e3)
axes[1].set_ylabel("Ey (mV/m)")
axes[1].set_xlabel("time (min)")
grounded = diag_z > 0
subs = [r.node_ids[i] for i in range(len(r.node_ids)) if grounded[i]]
axes[2].bar(subs, ng[grounded])
axes[2].set_ylabel("Substation GIC (A)")
axes[2].tick_params(axis="x", rotation=45)
fig.tight_layout()
out = _OUT / "case_01_hydro_quebec_1989.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
