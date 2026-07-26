"""Bridge-rectifier model under DC bias — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.exceptions import NotImplementedYetError

__all__ = ["RectifierModel"]


class RectifierModel(DeviceModel):
    """Full-wave bridge rectifier under superimposed GIC bias (WP3)."""

    def inject_gic(
        self,
        time_s: np.ndarray,
        gic_A: np.ndarray,
        ac_voltage_V: float = 0.0,
        ac_frequency_Hz: float = 50.0,
    ) -> DeviceResponse:
        raise NotImplementedYetError("RectifierModel.inject_gic", "WP3")
