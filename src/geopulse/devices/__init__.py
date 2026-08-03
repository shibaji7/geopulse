"""Nonlinear device models: transformers, rectifiers, cathodic protection.

Also includes :class:`ResistiveBlocker` — a *passive* mitigation-device
model that reports blocker voltage / power / dissipated-energy given a
neutral GIC time series. Blockers are network-topology interventions,
not GIC-response devices; the topology change lives at the network
layer (see :func:`geopulse.network.helpers.apply_resistive_blocker`).
"""

from __future__ import annotations

from geopulse.devices.base import DeviceModel, DeviceResponse
from geopulse.devices.blocker import ResistiveBlocker

__all__ = ["DeviceModel", "DeviceResponse", "ResistiveBlocker"]
