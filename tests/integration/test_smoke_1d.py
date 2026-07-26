"""End-to-end smoke test: synthetic B → 1-D Earth → plane-wave E.

Verifies that the full Phase-1 chain produces a physically reasonable E-field
in the mV/m range for a 500 nT storm-like pulse over a Québec-like layered
Earth (order-of-magnitude check against Boteler 2014 Table 1).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import irfft

from geopulse.earth.library import get_model
from geopulse.efield.planewave import compute_efield_planewave
from geopulse.sources.synthetic import SyntheticSource


@pytest.mark.integration
def test_end_to_end_gaussian_over_quebec():
    source = SyntheticSource("gaussian_pulse", amplitude_nT=500.0, sigma_s=300.0)
    b = source.load(start_s=0.0, end_s=3600.0, dt_s=1.0)

    earth = get_model("quebec_7layer")
    freqs, Bx_f, By_f = source.to_frequency_domain(b)
    impedance = earth.compute_impedance(freqs)

    Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, impedance)
    Ex_t = irfft(Ex_f, n=len(b.time_s))
    Ey_t = irfft(Ey_f, n=len(b.time_s))

    # By is zero, so Ex = (Z/μ₀) · By must be identically zero.
    np.testing.assert_allclose(Ex_t, 0.0, atol=1e-30)

    # Ey is driven by Bx; peak magnitude should sit in the mV/m ballpark.
    peak_Ey_Vm = float(np.max(np.abs(Ey_t)))
    peak_Ey_mVm = peak_Ey_Vm * 1e3
    assert 1e-4 < peak_Ey_mVm < 1e2, f"Ey peak {peak_Ey_mVm:.3e} mV/m out of range"

    # Output is finite everywhere.
    assert np.all(np.isfinite(Ey_t))
