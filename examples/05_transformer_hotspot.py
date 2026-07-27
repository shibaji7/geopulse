"""Transformer hot-spot temperature under a triangular geomagnetic event.

Reproduces the shape of Fig 5 in Mate et al. 2021 (arXiv:2101.05042):

    1. Load the Horton EPRI21 grid.
    2. Drive it with a triangular E-field ramp (0 → 3.2 V/km → 0 V/km east
       over 7 hours; the paper's Fig 4).
    3. At each time step, solve the LP GIC problem, extract |I| through a
       chosen transformer branch.
    4. Feed |I|(t) into TransformerModel to get top-oil + hot-spot curves.
    5. Save a 3-panel figure: E(t), |I|(t), and top-oil / hot-spot / limit.

Writes ``examples/output/transformer_hotspot.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.devices.transformer import ThermalParams, TransformerModel
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

BENCH = Path(__file__).parent.parent / "benchmarks" / "horton2012" / "epri21.m"

# 1. Load the network once — Y and Z are static.
net = PowerGridNetwork.from_file(str(BENCH))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()

# Pick a transformer branch to sample. Mate 2021 Fig 5 uses T10/T11 (GSUs at
# Sub 8, bus 12↔bus 13/14). In our epri21.m parse those are branches named
# dc_xf10_hi and dc_xf11_hi. Both carry the same current; use the first.
TARGET_BRANCH_ID = "dc_xf10_hi"

# 2. Triangular E-field ramp (Mate 2021 Fig 4).
t_hours = np.arange(0.0, 7.0 + 1.0 / 60.0, 1.0 / 60.0)  # 1-min sampling
t_s = t_hours * 3600.0
E_break_h = np.array([1.0, 4.0, 7.0])
E_break_Vkm = np.array([0.0, 3.2, 0.0])
E_Vkm = np.interp(t_hours, E_break_h, E_break_Vkm, left=0.0, right=0.0)
E_Vm = E_Vkm * 1e-3  # SI

# 3. At each time step, solve for the branch current through TARGET_BRANCH_ID.
branch_ids = [b.branch_id for b in net.get_branches()]
target_idx = branch_ids.index(TARGET_BRANCH_ID)

gic_A = np.empty_like(t_s)
solver = NAMSolver()
for i, e in enumerate(E_Vm):
    V_th = net.compute_thevenin_voltages(ex_Vm=e, ey_Vm=0.0)
    result = solver.solve(net, Y, Z, V_th)
    gic_A[i] = result.branch_currents_A[target_idx]

# 4. Thermal solve. Constant k_load = 0.63 per Mate 2021.
transformer = TransformerModel(params=ThermalParams(), k_load=0.63)
resp = transformer.inject_gic(t_s, gic_A)

# 5. Plot.
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
axes[0].plot(t_hours, E_Vkm, color="tab:blue")
axes[0].set_ylabel("E east (V/km)")
axes[0].set_title(f"Horton EPRI21 · {TARGET_BRANCH_ID} · triangular ramp")

axes[1].plot(t_hours, np.abs(gic_A), color="tab:orange")
axes[1].set_ylabel("|I| through winding (A)")

axes[2].plot(t_hours, resp.top_oil_C, label="Top-oil", color="tab:green")
axes[2].plot(t_hours, resp.hotspot_C, label="Hot-spot", color="tab:red")
axes[2].axhline(
    transformer.params.hs_inst_limit_C,
    color="k",
    ls="--",
    alpha=0.5,
    label="1-h inst limit (280 °C)",
)
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_xlabel("Time (hours)")
axes[2].legend(loc="upper right")

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()

out = Path(__file__).parent / "output" / "transformer_hotspot.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print(f"wrote {out}")
print(
    f"peak |I| = {np.max(np.abs(gic_A)):.2f} A, "
    f"peak top-oil = {resp.metadata['peak_top_oil_C']:.1f} °C, "
    f"peak hot-spot = {resp.metadata['peak_hotspot_C']:.1f} °C"
)
