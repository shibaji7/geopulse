"""3-D unstructured Earth model (ModEM / SimPEG wrapper) — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.earth.base import EarthModel
from geopulse.earth.impedance import Impedance
from geopulse.exceptions import NotImplementedYetError

__all__ = ["Unstructured3D"]


class Unstructured3D(EarthModel):
    """3-D volumetric Earth model — WP3."""

    @property
    def tier(self) -> int:  # noqa: D401
        """Return ``3``."""
        return 3

    def compute_impedance(self, freqs_Hz: np.ndarray) -> Impedance:
        """Compute a :class:`KernelImpedance` (WP3)."""
        raise NotImplementedYetError("Unstructured3D.compute_impedance", "WP3")
