"""Built-in 1-D Earth-model registry.

Each entry maps a short name to a top-down stack of
:class:`~geopulse.earth.base.ConductivityLayer` objects. Every model cites the
source paper in its comment; values marked ``TODO(PI)`` need PI verification
against the primary reference before publication.

References
----------
.. [1] Boteler, D. H., et al. (Various dates). Québec regional 1-D
   conductivity models used for GIC simulation.
.. [2] Fernberg, P. (2012). One-Dimensional Earth Resistivity Models for
   Selected Areas of Continental United States and Alaska. EPRI Technical
   Update 1026430.
"""

from __future__ import annotations

import numpy as np

from geopulse.earth.base import ConductivityLayer, EarthModel

__all__ = ["BUILT_IN_MODELS", "get_model", "list_models"]


# ---------------------------------------------------------------------------
# Built-in library
# ---------------------------------------------------------------------------
BUILT_IN_MODELS: dict[str, list[ConductivityLayer]] = {
    # Uniform 100 Ω·m half-space (σ = 0.01 S/m). Textbook sanity check.
    "uniform_100": [
        ConductivityLayer(thickness_m=np.inf, conductivity_Sm=0.01),
    ],
    # Québec-region 5-layer model (kept name "quebec_7layer" for spec parity;
    # full 7-layer variant will replace this once verified).
    # TODO(PI): verify exact layer values against the specific Boteler paper
    # used in production (there are multiple published versions).
    "quebec_7layer": [
        ConductivityLayer(thickness_m=15_000.0, conductivity_Sm=2.5e-4),
        ConductivityLayer(thickness_m=10_000.0, conductivity_Sm=5.0e-4),
        ConductivityLayer(thickness_m=125_000.0, conductivity_Sm=1.0e-3),
        ConductivityLayer(thickness_m=200_000.0, conductivity_Sm=3.33e-3),
        ConductivityLayer(thickness_m=np.inf, conductivity_Sm=1.0),
    ],
}


def list_models() -> list[str]:
    """Return the sorted list of registered built-in model names.

    Returns
    -------
    list of str
        Alphabetical model names.
    """
    return sorted(BUILT_IN_MODELS.keys())


def get_model(name: str) -> EarthModel:
    """Instantiate a named built-in Earth model.

    Parameters
    ----------
    name : str
        Model name from :func:`list_models`.

    Returns
    -------
    EarthModel
        A ready-to-use :class:`~geopulse.earth.layered_1d.Layered1D` instance.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in BUILT_IN_MODELS:
        raise KeyError(f"No such earth model: {name!r}. Available: {list_models()}")
    # Import at call-time to avoid a circular import between library ↔ layered_1d.
    from geopulse.earth.layered_1d import Layered1D

    return Layered1D(layers=BUILT_IN_MODELS[name])
