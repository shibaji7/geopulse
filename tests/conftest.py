"""Shared pytest fixtures for GeoPulse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geopulse.earth.base import ConductivityLayer
from geopulse.earth.layered_1d import Layered1D
from geopulse.sources.base import BFieldTimeSeries


@pytest.fixture
def sample_bfield() -> BFieldTimeSeries:
    """A deterministic 1-hour Gaussian pulse B-field at 1 Hz sampling.

    Kept independent of :class:`SyntheticSource` (which is a Phase-1 stub) so
    the fixture works in Phase 0.
    """
    dt_s = 1.0
    n = 3600
    t = np.arange(n) * dt_s
    amplitude_T = 500e-9  # 500 nT in SI
    t0 = 1800.0
    sigma = 300.0
    bx = amplitude_T * np.exp(-((t - t0) ** 2) / (2 * sigma**2))
    return BFieldTimeSeries(
        time_s=t,
        bx_T=bx,
        by_T=np.zeros_like(t),
        bz_T=np.zeros_like(t),
        station_id="TEST",
        latitude_deg=45.0,
        longitude_deg=-75.0,
    )


@pytest.fixture
def uniform_halfspace() -> Layered1D:
    """Uniform 100 Ω·m half-space (σ = 0.01 S/m)."""
    return Layered1D(layers=[ConductivityLayer(thickness_m=np.inf, conductivity_Sm=0.01)])


@pytest.fixture
def quebec_model() -> Layered1D:
    """Placeholder Québec-like layered model.

    In Phase 0 the built-in library is a stub, so we assemble a small stack
    here directly rather than call :func:`get_model`.
    """
    layers = [
        ConductivityLayer(thickness_m=15_000, conductivity_Sm=2.5e-4),
        ConductivityLayer(thickness_m=10_000, conductivity_Sm=5.0e-4),
        ConductivityLayer(thickness_m=125_000, conductivity_Sm=1.0e-3),
        ConductivityLayer(thickness_m=200_000, conductivity_Sm=3.33e-3),
        ConductivityLayer(thickness_m=np.inf, conductivity_Sm=1.0),
    ]
    return Layered1D(layers=layers)


@pytest.fixture
def tmp_hdf5_path(tmp_path: Path) -> Path:
    """Path to a temporary ``.h5`` file inside pytest's tmp_path."""
    return tmp_path / "geopulse_test.h5"
