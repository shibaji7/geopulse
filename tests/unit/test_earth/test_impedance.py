"""Tests for :mod:`geopulse.earth.impedance`."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from geopulse.constants import MU_0
from geopulse.earth.impedance import ScalarImpedance, TensorImpedance
from geopulse.exceptions import ShapeMismatchError


def test_scalar_impedance_apply_matches_formula():
    """E_x = (Z/μ₀) B_y, E_y = -(Z/μ₀) B_x — exact algebra check."""
    freqs = np.array([1e-3, 1e-2, 1e-1])
    Z = np.array([1 + 1j, 3 + 3j, 10 + 10j])
    imp = ScalarImpedance(freqs, Z)

    Bx = np.array([1.0 + 0j, 2.0, 3.0])
    By = np.array([4.0 + 0j, 5.0, 6.0])
    Ex, Ey = imp.apply(Bx, By)

    np.testing.assert_allclose(Ex, (Z / MU_0) * By)
    np.testing.assert_allclose(Ey, -(Z / MU_0) * Bx)


def test_scalar_impedance_shape_validation():
    freqs = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ShapeMismatchError):
        ScalarImpedance(freqs, np.array([1 + 1j, 2 + 2j]))


def test_scalar_impedance_apply_shape_mismatch_raises():
    freqs = np.array([1.0, 2.0])
    imp = ScalarImpedance(freqs, np.array([1 + 1j, 2 + 2j]))
    with pytest.raises(ShapeMismatchError):
        imp.apply(np.array([1.0]), np.array([1.0]))


def test_scalar_impedance_hdf5_roundtrip(tmp_hdf5_path):
    freqs = np.linspace(1e-4, 1.0, 32)
    Z = np.sqrt(1j * 2 * np.pi * freqs * MU_0 / 0.01)
    imp = ScalarImpedance(freqs, Z)

    with h5py.File(tmp_hdf5_path, "w") as fh:
        g = fh.create_group("imp")
        imp.to_hdf5(g)

    with h5py.File(tmp_hdf5_path, "r") as fh:
        back = ScalarImpedance.from_hdf5(fh["imp"])

    np.testing.assert_array_equal(back.freqs_Hz, imp.freqs_Hz)
    np.testing.assert_array_equal(back.Z_values, imp.Z_values)


# --- TensorImpedance tests -----------------------------------------------


def _random_tensor(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 2, 2)) + 1j * rng.standard_normal((n, 2, 2))).astype(
        np.complex128
    )


def test_tensor_impedance_shape_validation():
    freqs = np.array([1e-3, 1e-2])
    with pytest.raises(ShapeMismatchError):
        TensorImpedance(freqs, np.zeros((3, 2, 2), dtype=complex))


def test_tensor_impedance_apply_matches_formula():
    freqs = np.array([1e-3, 1e-2, 1e-1])
    Z = _random_tensor(3, seed=42)
    imp = TensorImpedance(freqs, Z)
    Bx = np.array([1 + 0.5j, 2 + 0j, -1.5 + 1j])
    By = np.array([0.5 + 1j, -1.0, 2 + 2j])
    Ex, Ey = imp.apply(Bx, By)
    expected_Ex = (Z[:, 0, 0] * Bx + Z[:, 0, 1] * By) / MU_0
    expected_Ey = (Z[:, 1, 0] * Bx + Z[:, 1, 1] * By) / MU_0
    np.testing.assert_allclose(Ex, expected_Ex)
    np.testing.assert_allclose(Ey, expected_Ey)


def test_tensor_impedance_reduces_to_scalar_for_antidiag():
    """
    A tensor with Z[0,1]=Z, Z[1,0]=-Z, zeros on the diagonal must give
    the same E-field as ScalarImpedance(Z).
    """
    freqs = np.array([1e-3, 1e-2, 1e-1])
    Z_scalar = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    Z_tensor = np.zeros((3, 2, 2), dtype=complex)
    Z_tensor[:, 0, 1] = Z_scalar
    Z_tensor[:, 1, 0] = -Z_scalar
    imp_t = TensorImpedance(freqs, Z_tensor)
    imp_s = ScalarImpedance(freqs, Z_scalar)
    Bx = np.array([1.0 + 0j, 0.5, -0.2])
    By = np.array([0.1 + 0j, -0.4, 0.7])
    Et_x, Et_y = imp_t.apply(Bx, By)
    Es_x, Es_y = imp_s.apply(Bx, By)
    np.testing.assert_allclose(Et_x, Es_x)
    np.testing.assert_allclose(Et_y, Es_y)


def test_tensor_impedance_apply_shape_mismatch_raises():
    freqs = np.array([1e-3, 1e-2])
    imp = TensorImpedance(freqs, _random_tensor(2))
    with pytest.raises(ShapeMismatchError):
        imp.apply(np.array([1 + 0j]), np.array([1 + 0j]))


def test_tensor_impedance_hdf5_roundtrip(tmp_hdf5_path):
    freqs = np.linspace(1e-4, 1.0, 16)
    Z = _random_tensor(16, seed=7)
    imp = TensorImpedance(freqs, Z)
    with h5py.File(tmp_hdf5_path, "w") as fh:
        imp.to_hdf5(fh.create_group("t"))
    with h5py.File(tmp_hdf5_path, "r") as fh:
        back = TensorImpedance.from_hdf5(fh["t"])
    np.testing.assert_array_equal(back.freqs_Hz, imp.freqs_Hz)
    np.testing.assert_array_equal(back.Z_tensor, imp.Z_tensor)
    with h5py.File(tmp_hdf5_path, "r") as fh:
        assert fh["t"].attrs["impedance_type"] == "tensor_2d"
        assert int(fh["t"].attrs["schema_version"]) == 1
