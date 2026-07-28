"""
GeoPulse Case Study #07: US Power Grid — Gannon Storm & NERC Benchmark
=======================================================================

Wilkerson et al. (2026) analysed GIC observations across the US during
the Gannon storm (10-13 May 2024). The NERC benchmark defines a peak
surface electric field of 8 V/km at 60° N for the reference GMD event
(TPL-007-4). Peak GIC magnitude scales linearly with geomagnetic
latitude and ground conductivity.

Infrastructure: Power grid + regulatory benchmark.
Key references: Wilkerson et al. (2026) Space Weather; NERC TPL-007-4;
    USGS Fredericksburg (FRD) 2024-05-10/11 magnetogram.
Status: COMPLETE — real Gannon-storm IAGA-2002 ingestion drives an
    end-to-end B(t) → E(t) → NAM solve on the Horton grid (US-proxy),
    with a companion NERC 8 V/km sensitivity sweep over σ.

This example demonstrates:
  - IAGA-2002 ingestion via `INTERMAGNETSource` on real observatory data.
  - Plane-wave `B(f) → E(f)` step through a 1-D layered Earth.
  - Per-timestep NAM solve producing a GIC(t) trace at every substation.
  - Sensitivity of the NERC 8 V/km design GIC to half-space conductivity.

Real US topology is not distributed; the Horton EPRI21 grid is used as a
mid-Atlantic-plausible bulk-transmission proxy. Replace `_BENCH` with a
real US 500 kV network file when one becomes available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.network.powergrid import PowerGridNetwork
from geopulse.solver.nam import NAMSolver
from geopulse.sources.base import BFieldTimeSeries
from geopulse.sources.intermagnet import INTERMAGNETSource

_OUT = Path(__file__).parent.parent / "output"
_OUT.mkdir(exist_ok=True)
_ROOT = Path(__file__).parent.parent.parent / "benchmarks"
_GRID_FILE = _ROOT / "horton2012" / "epri21.m"
_FRD_FILE = _ROOT / "gannon2024" / "frd_20240510_1min.min"

# ---------------------------------------------------------------------------
# 1. Real Gannon-day B(t) from FRD (Fredericksburg, VA — mid-Atlantic US).
# ---------------------------------------------------------------------------
src = INTERMAGNETSource(str(_FRD_FILE))
b = src.load()  # entire 48-h window at 1-min cadence
n_samples = int(b.bx_T.size)
dt_s = 1.0 / b.sampling_rate_Hz
print(
    f"[SRC] {b.station_id} lat={b.latitude_deg:.3f} lon={b.longitude_deg:.3f}  "
    f"n={n_samples} @ dt={dt_s:.0f} s"
)

# Remove the quiet-time mean so we only feed storm-relevant variations
# through the plane-wave / DSTL chain. NaN-safe: if bundled data ever
# introduces a gap, drop it.
if not np.isfinite(b.bx_T).all() or not np.isfinite(b.by_T).all():
    raise RuntimeError("bundled FRD file has NaNs — fix the block-mean step")
bx_var_T = b.bx_T - float(np.mean(b.bx_T))
by_var_T = b.by_T - float(np.mean(b.by_T))

# ---------------------------------------------------------------------------
# 2. Plane-wave E(t) through a 1-D Piedmont-like Earth (σ = 5e-3 S/m).
# ---------------------------------------------------------------------------
earth = Layered1D([ConductivityLayer(np.inf, 5e-3)])
# Use to_frequency_domain via BFieldSource; the variation-only signal has
# the same time base as `b`, so wrap it in a fresh timeseries.
b_var = BFieldTimeSeries(
    time_s=b.time_s,
    bx_T=bx_var_T,
    by_T=by_var_T,
    bz_T=b.bz_T - float(np.mean(b.bz_T)),
    station_id=b.station_id,
    latitude_deg=b.latitude_deg,
    longitude_deg=b.longitude_deg,
    sampling_rate_Hz=b.sampling_rate_Hz,
    metadata=b.metadata,
)
freqs, Bx_f, By_f = src.to_frequency_domain(b_var)
imp = earth.compute_impedance(freqs)
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ex_t_Vm = irfft(Ex_f, n=n_samples)
Ey_t_Vm = irfft(Ey_f, n=n_samples)

peak_E_mVm = float(max(np.max(np.abs(Ex_t_Vm)), np.max(np.abs(Ey_t_Vm)))) * 1e3
print(f"[E]   peak |E| under σ=5e-3 S/m half-space = {peak_E_mVm:.2f} mV/m")

# ---------------------------------------------------------------------------
# 3. Per-timestep NAM solve on the Horton grid (US mid-Atlantic proxy).
# ---------------------------------------------------------------------------
grid = PowerGridNetwork.from_file(str(_GRID_FILE))
Y = grid.assemble_network_admittance()
Z = grid.assemble_earthing_impedance()
diag_z = np.diag(Z)
solver = NAMSolver()

n_nodes = int(diag_z.size)
gic_A = np.zeros((n_samples, n_nodes), dtype=np.float64)
for i in range(n_samples):
    V_th = grid.compute_thevenin_voltages(ex_Vm=float(Ex_t_Vm[i]), ey_Vm=float(Ey_t_Vm[i]))
    r = solver.solve(grid, Y, Z, V_th)
    gic_A[i, :] = np.where(diag_z > 0.0, r.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)

peak_abs = np.max(np.abs(gic_A), axis=0)  # per-substation peak |GIC| A
imax = int(peak_abs.argmax())
tmax = int(np.abs(gic_A[:, imax]).argmax())
node_id = r.node_ids[imax]
tmax_utc = datetime.fromtimestamp(float(b.time_s[tmax]), tz=timezone.utc)
print(
    f"[GIC] peak substation |GIC| = {peak_abs[imax]:.1f} A at node {node_id!r}, "
    f"t = {tmax_utc.strftime('%Y-%m-%d %H:%M UTC')}"
)

# ---------------------------------------------------------------------------
# 4. Ground-conductivity sensitivity of the FRD-hindcast peak substation
#    GIC. σ modulates the plane-wave impedance, and hence E, and hence GIC.
#    Also prints the NERC 8 V/km reference solve as the design-envelope
#    upper anchor (single value — a uniform E is σ-invariant downstream).
# ---------------------------------------------------------------------------
sigma_Sm_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
peak_gic_sweep_A: list[float] = []
for sigma in sigma_Sm_list:
    e_half = Layered1D([ConductivityLayer(np.inf, sigma)])
    imp_s = e_half.compute_impedance(freqs)
    Ex_f_s, Ey_f_s = compute_efield_planewave(freqs, Bx_f, By_f, imp_s)
    Ex_t_s = irfft(Ex_f_s, n=n_samples)
    Ey_t_s = irfft(Ey_f_s, n=n_samples)
    peak_gic_i = 0.0
    for i in range(n_samples):
        V_th_i = grid.compute_thevenin_voltages(ex_Vm=float(Ex_t_s[i]), ey_Vm=float(Ey_t_s[i]))
        r_i = solver.solve(grid, Y, Z, V_th_i)
        gic_i = np.where(diag_z > 0, r_i.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
        m = float(np.max(np.abs(gic_i)))
        if m > peak_gic_i:
            peak_gic_i = m
    peak_gic_sweep_A.append(peak_gic_i)
    Zref = float(np.abs(e_half.compute_impedance(np.array([1e-2])).Z_values[0]))
    print(f"[SWEEP] σ={sigma:.0e} S/m  |Z@10 mHz|={Zref:.2e} Ω  peak|GIC|={peak_gic_i:.1f} A")

E_NERC_Vm = 8e-3  # 8 V/km, per TPL-007-4 — σ-invariant downstream
V_th_nerc = grid.compute_thevenin_voltages(ex_Vm=E_NERC_Vm, ey_Vm=0.0)
rn = solver.solve(grid, Y, Z, V_th_nerc)
gic_nerc = np.where(diag_z > 0, rn.node_voltages_V / np.where(diag_z > 0, diag_z, 1.0), 0.0)
peak_gic_NERC_A = float(np.max(np.abs(gic_nerc)))
print(f"[NERC] design 8 V/km peak |GIC| = {peak_gic_NERC_A:.1f} A (σ-invariant reference)")

# ---------------------------------------------------------------------------
# 5. Plot: real hindcast on top, NERC sensitivity on the bottom.
# ---------------------------------------------------------------------------
t_h = (b.time_s - b.time_s[0]) / 3600.0

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.25)

ax_b = fig.add_subplot(gs[0, :])
ax_b.plot(t_h, bx_var_T * 1e9, label="ΔBx (north)", lw=0.9)
ax_b.plot(t_h, by_var_T * 1e9, label="ΔBy (east)", lw=0.9, alpha=0.85)
ax_b.set_ylabel("ΔB (nT)")
ax_b.set_title("FRD magnetogram (10-11 May 2024) — 1-min cadence, quiet mean removed")
ax_b.legend(loc="upper left", framealpha=0.9)
ax_b.grid(alpha=0.3)

ax_e = fig.add_subplot(gs[1, :], sharex=ax_b)
ax_e.plot(t_h, Ex_t_Vm * 1e3, label="Ex (north)", lw=0.9)
ax_e.plot(t_h, Ey_t_Vm * 1e3, label="Ey (east)", lw=0.9, alpha=0.85)
ax_e.set_ylabel("E (mV/m)")
ax_e.set_title("Plane-wave E-field via 1-D Piedmont-like Earth (σ = 5 mS/m)")
ax_e.legend(loc="upper left", framealpha=0.9)
ax_e.grid(alpha=0.3)

ax_g = fig.add_subplot(gs[2, 0], sharex=ax_b)
ax_g.plot(t_h, gic_A[:, imax], color="tab:red", lw=1.0)
ax_g.set_xlabel("Time (h from 2024-05-10 00:00 UTC)")
ax_g.set_ylabel("GIC (A)")
ax_g.set_title(f"Peak substation ({node_id}) GIC — Horton grid (US proxy)")
ax_g.grid(alpha=0.3)

ax_n = fig.add_subplot(gs[2, 1])
ax_n.loglog(sigma_Sm_list, peak_gic_sweep_A, "o-", color="tab:blue", label="FRD hindcast")
ax_n.axhline(peak_gic_NERC_A, color="tab:orange", ls="--", label="NERC 8 V/km design")
ax_n.set_xlabel("Half-space σ (S/m)")
ax_n.set_ylabel("Peak |GIC| (A)")
ax_n.set_title("σ-sensitivity of FRD-hindcast peak GIC")
ax_n.legend(fontsize=8, loc="lower left")
ax_n.grid(True, which="both", alpha=0.3)

out = _OUT / "case_07_us_gannon_nerc.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
