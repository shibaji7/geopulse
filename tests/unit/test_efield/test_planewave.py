"""Tests for :func:`geopulse.efield.planewave.compute_efield_planewave`."""

from __future__ import annotations

import numpy as np

from geopulse.constants import MU_0
from geopulse.earth.impedance import ScalarImpedance
from geopulse.efield.planewave import compute_efield_planewave


def test_planewave_matches_hand_calculation():
    freqs = np.array([1e-3, 1e-2, 1e-1])
    Z = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    imp = ScalarImpedance(freqs, Z)

    Bx = np.array([0.5 + 0j, 1.0, 2.0])
    By = np.array([1.0 + 0j, 0.5, 0.25])

    Ex, Ey = compute_efield_planewave(freqs, Bx, By, imp)

    np.testing.assert_allclose(Ex, Z * By / MU_0)
    np.testing.assert_allclose(Ey, -Z * Bx / MU_0)


def test_planewave_delegates_to_apply():
    """The function must be a thin polymorphic wrapper over Impedance.apply."""

    class _Recording(ScalarImpedance):
        def apply(self, Bx_f, By_f):
            self._called = (Bx_f, By_f)
            return super().apply(Bx_f, By_f)

    freqs = np.array([1e-2])
    imp = _Recording(freqs, np.array([1 + 1j]))
    Bx = np.array([1.0 + 0j])
    By = np.array([0.5 + 0j])
    compute_efield_planewave(freqs, Bx, By, imp)
    assert np.array_equal(imp._called[0], Bx)
    assert np.array_equal(imp._called[1], By)
