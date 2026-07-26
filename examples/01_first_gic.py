"""First GIC — end-to-end smoke chain in <30 lines.

Synthetic Gaussian B-field → Québec 1-D Earth → plane-wave E-field.
Saves a two-panel plot to ``examples/output/first_gic.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.fft import irfft

from geopulse.earth.library import get_model
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.sources.synthetic import SyntheticSource

source = SyntheticSource("gaussian_pulse", amplitude_nT=500.0, sigma_s=300.0)
b = source.load(start_s=0.0, end_s=3600.0, dt_s=1.0)

earth = get_model("quebec_7layer")
freqs, Bx_f, By_f = source.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)

Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
axes[0].plot(b.time_s / 60.0, b.bx_T * 1e9)
axes[0].set_ylabel("Bx (nT)")
axes[1].plot(b.time_s / 60.0, Ey_t * 1e3)
axes[1].set_ylabel("Ey (mV/m)")
axes[1].set_xlabel("Time (minutes)")
fig.suptitle("GeoPulse — Gaussian B → Québec 1-D → plane-wave E")
fig.tight_layout()

out = Path(__file__).parent / "output" / "first_gic.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print(f"wrote {out}")
