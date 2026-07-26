"""Cathodic-protection unit response to telluric E-field — STUB (WP3)."""

from __future__ import annotations

import numpy as np

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.exceptions import NotImplementedYetError

__all__ = ["CathodicProtectionModel"]


class CathodicProtectionModel(DeviceModel):
    """Cathodic-protection unit: PSP shift under telluric E-field (WP3)."""

    def inject_gic(
        self,
        time_s: np.ndarray,
        gic_A: np.ndarray,
        ac_voltage_V: float = 0.0,
        ac_frequency_Hz: float = 50.0,
    ) -> DeviceResponse:
        raise NotImplementedYetError("CathodicProtectionModel.inject_gic", "WP3")
