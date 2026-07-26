"""B-field data adapters — SuperMAG, INTERMAGNET, MHD, synthetic.

Downstream modules never know which adapter produced the data; they only
consume :class:`~geopulse.sources.base.BFieldTimeSeries`.
"""

from __future__ import annotations

from geopulse.sources.base import BFieldSource, BFieldTimeSeries

__all__ = ["BFieldSource", "BFieldTimeSeries"]
