"""Tests for :mod:`geopulse.uq.uncertain`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.uq.uncertain import Uncertain, propagate_uncertainty


def test_generate_samples_deterministic_returns_replicas():
    u = Uncertain(nominal=7.0)
    samples = u.generate_samples(5)
    assert samples == [7.0] * 5


def test_generate_samples_gaussian_reproducible():
    u = Uncertain(nominal=0.0, distribution="gaussian", params={"std": 1.0})
    s1 = u.generate_samples(500, rng=np.random.default_rng(42))
    s2 = u.generate_samples(500, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(s1, s2)


def test_generate_samples_uniform_requires_bounds():
    u = Uncertain(nominal=0.0, distribution="uniform")
    with pytest.raises(ValueError, match="uniform"):
        u.generate_samples(3, rng=np.random.default_rng(0))


def test_generate_samples_unknown_raises():
    u = Uncertain(nominal=0.0, distribution="cauchy")
    with pytest.raises(ValueError, match="Unknown distribution"):
        u.generate_samples(3)


def test_propagate_uncertainty_scales_gaussian():
    u = Uncertain(nominal=1.0, distribution="gaussian", params={"std": 0.1})
    out = propagate_uncertainty(lambda x: 2.0 * x, u, n_samples=500, seed=1)
    assert out.n_samples == 500
    assert abs(out.mean - 2.0) < 0.05
    assert 0.15 < float(out.std) < 0.25


def test_propagate_uncertainty_pass_through_deterministic():
    out = propagate_uncertainty(lambda x, y: x + y, 3.0, 4.0, n_samples=10)
    assert out.mean == 7.0
    assert float(out.std) == 0.0
