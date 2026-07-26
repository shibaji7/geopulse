"""Tests for :mod:`geopulse.earth.library`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.earth.layered_1d import Layered1D
from geopulse.earth.library import BUILT_IN_MODELS, get_model, list_models


def test_list_models_returns_registered_names():
    names = list_models()
    assert "uniform_100" in names
    assert "quebec_7layer" in names
    assert names == sorted(names)  # stable, alphabetical


def test_get_model_returns_layered_1d():
    m = get_model("uniform_100")
    assert isinstance(m, Layered1D)
    # Uniform 100 Ω·m → σ = 0.01 S/m; single infinite half-space.
    assert len(m.layers) == 1
    assert np.isinf(m.layers[0].thickness_m)
    assert m.layers[0].conductivity_Sm == pytest.approx(0.01)


def test_get_model_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="No such earth model"):
        get_model("model_that_definitely_does_not_exist")


def test_builtin_models_are_all_layered_1d_instantiable():
    """Every entry in the registry must construct a valid Layered1D."""
    for name in BUILT_IN_MODELS:
        m = get_model(name)
        assert isinstance(m, Layered1D)
        # And compute_impedance must run without error.
        imp = m.compute_impedance(np.array([1e-3, 1e-2, 1e-1]))
        assert imp.Z_values.shape == (3,)
