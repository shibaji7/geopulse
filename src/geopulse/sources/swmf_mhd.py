"""SWMF / Gamera MHD output adapter — STUB (WP4)."""

from __future__ import annotations

from geopulse.exceptions import NotImplementedYetError
from geopulse.sources.base import BFieldSource, BFieldTimeSeries

__all__ = ["SWMFSource"]


class SWMFSource(BFieldSource):
    """Reader for SWMF / Gamera MHD ground-magnetic output at a station."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(
        self,
        start_s: float,
        end_s: float,
        dt_s: float = 1.0,
        station_id: str = "SYN",
    ) -> BFieldTimeSeries:
        """Read MHD B-field time series at the requested station (WP4)."""
        raise NotImplementedYetError("SWMFSource.load", "WP4")
