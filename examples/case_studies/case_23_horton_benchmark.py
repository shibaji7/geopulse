"""
GeoPulse Case Study #23: Horton et al. (2012) EPRI21 Benchmark
================================================================

A synthetic test grid specifically designed for benchmarking GIC
calculations. Published network topology, earthing resistances, and
expected GIC values for uniform 1 V/km electric fields (northward and
eastward). This is not a real event but a controlled benchmark — the
formal Phase 2 acceptance target for GeoPulse.

Infrastructure: Power grid (21-bus, 8-substation EHV test system)
Key references: Horton et al. (2012) IEEE Trans. Power Delivery 27(4)
Status: COMPLETE — reproduces Table VII within 1 % on every substation.

This example demonstrates:
  - MATPOWER-GMD .m file ingestion → ConductorNetwork.
  - LPM solve driven by a uniform 1 V/km E-field in each direction.
  - Sub-percent agreement with Horton (2012) Table VII.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_BENCH = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012"

# ---------------------------------------------------------------------------
# 1. Load the network + the paper's Table VII golden values.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Solve for both uniform-field directions.
# ---------------------------------------------------------------------------
def _solve(ex_Vm: float, ey_Vm: float) -> dict[str, float]:
    V_th = net.compute_thevenin_voltages(ex_Vm=ex_Vm, ey_Vm=ey_Vm)
    r = NAMSolver().solve(net, Y, Z, V_th)
    ng = np.where(diag_z > 0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
    return dict(zip(r.node_ids, ng, strict=True))


north = _solve(0.0, 1e-3)
east = _solve(1e-3, 0.0)

# ---------------------------------------------------------------------------
# 3. Compare and print. Track worst-case relative error.
# ---------------------------------------------------------------------------
print(f"\n{'node':<8} {'field':<6} {'paper (A)':>10} {'mine (A)':>10} {'err %':>7}")
worst = 0.0
for label, table, col_idx in [("N", north, 0), ("E", east, 1)]:
    for nid, (n_ref, e_ref) in paper.items():
        ref = (n_ref, e_ref)[col_idx]
        got = table[nid]
        if ref == 0.0:
            continue
        err = 100.0 * (got - ref) / ref
        worst = max(worst, abs(err))
        print(f"{nid:<8} {label:<6} {ref:>10.2f} {got:>10.2f} {err:>+7.3f}")

print(f"\nworst-case relative error: {worst:.3f} %  (Phase 2 target: <1 %)")

# ---------------------------------------------------------------------------
# 4. Bar-chart comparison, per direction.
# ---------------------------------------------------------------------------
subs = [k for k in paper if paper[k] != (0.0, 0.0)]
x = np.arange(len(subs))
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, table, idx, title in [
    (axes[0], north, 0, "Northward 1 V/km"),
    (axes[1], east, 1, "Eastward 1 V/km"),
]:
    ref = [paper[s][idx] for s in subs]
    mine = [table[s] for s in subs]
    ax.bar(x - 0.2, ref, 0.4, label="Horton (2012) Table VII")
    ax.bar(x + 0.2, mine, 0.4, label="GeoPulse LPM")
    ax.set_ylabel("GIC (A)")
    ax.set_title(f"Case 23: EPRI21 — {title}")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
axes[1].set_xticks(x, subs, rotation=45)
fig.tight_layout()
out = _OUT / "case_23_horton_benchmark.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
