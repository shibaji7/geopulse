"""Horton et al. (2012) LPM benchmark.

Loads the EPRI21 test network from ``benchmarks/horton2012/epri21.m``,
drives it with uniform 1 V/km northward and eastward geoelectric fields,
and verifies the per-substation ground GIC matches **Table VII** of

    R. Horton, D. Boteler, T. J. Overbye, R. Pirjola, R. C. Dugan,
    "A Test Case for the Calculation of Geomagnetically Induced Currents,"
    IEEE Trans. Power Delivery, 27(4), 2368-2373 (2012).

Reference values are stored in ``benchmarks/horton2012/expected_gic.csv``.
Tolerance is ``rtol = 1e-2`` (1 %). With the WGS84 projection now enabled
per-segment (matching Horton Appendix eqns A3/A7), the residual is
dominated by Table VII's own 2-decimal rounding — the largest error sits
on the smallest-magnitude entry (Sub 4 Northward ≈ 20 A, where ±0.005 A
rounding = 0.03 % noise).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.lpm import LPMSolver

_BENCH_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "horton2012"
EPRI21 = _BENCH_DIR / "epri21.m"
EXPECTED_CSV = _BENCH_DIR / "expected_gic.csv"

BENCHMARK_RTOL = 1e-2  # 1 % — matches Section 22 Phase 2 acceptance criterion


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not EPRI21.is_file() or not EXPECTED_CSV.is_file(),
        reason="Horton (2012) benchmark data not present in benchmarks/horton2012/",
    ),
]


def _load_expected() -> dict[str, tuple[float, float]]:
    """Parse ``expected_gic.csv`` → ``{node_id: (gic_north_A, gic_east_A)}``."""
    out: dict[str, tuple[float, float]] = {}
    with EXPECTED_CSV.open() as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "node_id":
                continue
            out[row[0]] = (float(row[1]), float(row[2]))
    return out


def _solve_uniform(ex_Vm: float, ey_Vm: float):
    net = PowerGridNetwork.from_file(str(EPRI21))
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(ex_Vm=ex_Vm, ey_Vm=ey_Vm)
    result = LPMSolver().solve(net, Y, Z, V_th)
    diag_z = np.diag(Z)
    grounded = diag_z > 0.0
    node_gic_A = np.zeros_like(result.node_voltages_V)
    node_gic_A[grounded] = result.node_voltages_V[grounded] / diag_z[grounded]
    return dict(zip(result.node_ids, node_gic_A, strict=True))


@pytest.mark.parametrize(
    "field_label, ex_Vm, ey_Vm, column_index",
    [
        ("northward_1_Vkm", 0.0, 1e-3, 0),
        ("eastward_1_Vkm", 1e-3, 0.0, 1),
    ],
)
def test_matches_horton_2012_table_vii(field_label, ex_Vm, ey_Vm, column_index):
    expected = _load_expected()
    node_gic = _solve_uniform(ex_Vm, ey_Vm)

    errors: dict[str, tuple[float, float]] = {}
    for node_id, (n_A, e_A) in expected.items():
        ref = (n_A, e_A)[column_index]
        got = node_gic.get(node_id)
        if got is None:
            pytest.fail(f"Node {node_id} not present in solver output")
        # Reference-zero nodes (Sub 1, Sub 7) must produce zero to floating-point noise.
        if ref == 0.0:
            assert abs(got) < 1e-6, f"{node_id}: expected 0 A, got {got:.6g} A"
        else:
            rel = abs(got - ref) / abs(ref)
            errors[node_id] = (ref, got)
            assert rel <= BENCHMARK_RTOL, (
                f"{field_label} {node_id}: expected {ref:.2f} A, got {got:.2f} A "
                f"(rel err {rel:.4f} > {BENCHMARK_RTOL})"
            )


def test_zero_field_gives_zero_gic():
    node_gic = _solve_uniform(0.0, 0.0)
    for node_id, gic in node_gic.items():
        assert abs(gic) < 1e-9, f"{node_id}: {gic}"
