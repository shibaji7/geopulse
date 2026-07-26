"""Tests for :mod:`geopulse.types` aliases and :class:`Uncertain`."""

from __future__ import annotations

import numpy as np

from geopulse.types import ComplexArray, FloatArray, LatLon, Uncertain


def test_type_aliases_importable():
    # Type aliases are objects at runtime; verify they exist and are usable.
    x: FloatArray = np.zeros(3, dtype=np.float64)
    z: ComplexArray = np.zeros(3, dtype=np.complex128)
    p: LatLon = (45.0, -75.0)
    assert x.dtype == np.float64
    assert z.dtype == np.complex128
    assert p == (45.0, -75.0)


def test_uncertain_deterministic_defaults():
    u = Uncertain(nominal=3.14)
    assert u.is_deterministic
    assert u.n_samples == 0
    assert u.mean == 3.14
    assert u.std == 0.0


def test_uncertain_sampled_mean_std_scalar():
    u = Uncertain(
        nominal=1.0,
        distribution="gaussian",
        params={"std": 0.5},
    )
    samples = u.generate_samples(1000, rng=np.random.default_rng(0))
    u_pop = Uncertain(nominal=1.0, samples=samples, distribution="ensemble")
    # Mean ≈ 1.0 within 3σ / √1000
    assert abs(u_pop.mean - 1.0) < 0.1
    assert 0.3 < float(u_pop.std) < 0.7


def test_uncertain_array_std_shape():
    u = Uncertain(nominal=np.zeros(4))
    assert np.all(u.std == 0)
    assert u.std.shape == (4,)
