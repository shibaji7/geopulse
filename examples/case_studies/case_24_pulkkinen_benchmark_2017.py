"""
GeoPulse Case Study #24: Pulkkinen et al. (2017) GIC Benchmark Suite
======================================================================

An expanded benchmark test suite for GIC modelling codes, providing
additional test cases beyond Horton (2012).

Infrastructure: Power grid (synthetic benchmark).
Key references: Pulkkinen et al. (2017)
Status: PARTIAL — LPM machinery runs on synthetic variants of the
    Horton grid under different uniform E-field directions and
    magnitudes. Actual Pulkkinen 2017 test-case topologies + expected
    GIC values are not transcribed yet.

This example demonstrates:
  - The LPM solver's response to a rotation sweep of a uniform E-field.
  - Sanity checks: at 0° the result matches Horton Table VII northward;
    at 90° it matches eastward.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.lpm import LPMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012"

# 1. Load network + Horton golden values for sanity anchors.
net = PowerGridNetwork.from_file(str(_BENCH / "epri21.m"))
Y = net.assemble_network_admittance()
Z = net.assemble_earthing_impedance()
diag_z = np.diag(Z)
paper: dict[str, tuple[float, float]] = {}
with (_BENCH / "expected_gic.csv").open() as fh:
    for row in csv.reader(fh):
        if not row or row[0].startswith("#") or row[0] == "node_id":
            continue
        paper[row[0]] = (float(row[1]), float(row[2]))

# 2. Sweep the field direction from 0° (north) → 180° in 15° steps.
theta_deg = np.arange(0.0, 181.0, 15.0)
E_mag_Vm = 1e-3
solver = LPMSolver()
peak_per_theta: dict[str, list[float]] = {n: [] for n in paper}

for theta in theta_deg:
    ex = E_mag_Vm * np.sin(np.radians(theta))
    ey = E_mag_Vm * np.cos(np.radians(theta))
    V_th = net.compute_thevenin_voltages(ex_Vm=ex, ey_Vm=ey)
    r = solver.solve(net, Y, Z, V_th)
    ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
    for i, nid in enumerate(r.node_ids):
        if nid in paper:
            peak_per_theta[nid].append(float(ng[i]))

# 3. Sanity-anchor against Horton values at 0° (pure north) and 90° (pure east).
print("\nSanity — GeoPulse should match Horton (2012) Table VII at 0° and 90°:")
for nid, (n_ref, e_ref) in paper.items():
    if n_ref == 0.0 and e_ref == 0.0:
        continue
    got_n = peak_per_theta[nid][0]
    got_e = peak_per_theta[nid][6]  # θ = 90°
    print(
        f"  {nid}  N  ref={n_ref:>7.2f}  got={got_n:>7.2f}   E ref={e_ref:>7.2f}  got={got_e:>7.2f}"
    )

# --- CHAIN STOPS HERE: Pulkkinen (2017) test-case topologies + golden values not transcribed. ---
# TODO: add `benchmarks/pulkkinen2017/*.m` and expected_gic.csv, then
#       assert per-node agreement within the paper's tolerance.

# 4. Plot GIC-vs-angle for every substation.
fig, ax = plt.subplots(figsize=(9, 5))
for nid, series in peak_per_theta.items():
    if all(v == 0.0 for v in series):
        continue
    ax.plot(theta_deg, series, "-o", ms=3, label=nid)
ax.axvline(0, color="k", ls=":", alpha=0.3)
ax.axvline(90, color="k", ls=":", alpha=0.3)
ax.set_xlabel("E-field angle θ (° from north; 0=N, 90=E)")
ax.set_ylabel("Substation GIC (A)  at E = 1 V/km")
ax.set_title("Case 24: GIC vs. field-direction sweep (Horton EPRI21)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
out = _OUT / "case_24_pulkkinen_benchmark.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
