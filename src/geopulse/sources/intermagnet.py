r"""INTERMAGNET IAGA-2002 loader.

Wraps :func:`geopulse.io.iaga2002.read_iaga2002` and standardises the
result into a :class:`BFieldTimeSeries` — nT → T conversion, HDZF frame
converted to geographic X (north) / Y (east), and cropped to the
requested time window.

Frame conventions
-----------------
* **XYZF** reporting is the easy case: ``X`` = geographic north,
  ``Y`` = geographic east, ``Z`` = downward, all in nT. Direct mapping to
  ``bx_T``, ``by_T``, ``bz_T``.
* **HDZF** reporting: ``H`` = horizontal magnitude (nT), ``D`` =
  declination (**minutes of arc** in the IAGA convention — most
  observatories still emit this way; some emit degrees). The declination
  is the angle from geographic north to the horizontal-field direction,
  measured positive east. So::

      X = H · cos(D_rad),   Y = H · sin(D_rad).

  Pass ``declination_unit="degrees"`` if the observatory reports D in
  degrees (D max magnitude around 20 is a giveaway).

Missing samples (IAGA 99999 / 88888 sentinels) surface as ``NaN`` on the
returned series; downstream code must filter or interpolate.
"""

from __future__ import annotations

import numpy as np

from geopulse.constants import DEG_TO_RAD, NT_TO_T
from geopulse.exceptions import DataError
from geopulse.io.iaga2002 import read_iaga2002
from geopulse.sources.base import BFieldSource, BFieldTimeSeries

__all__ = ["INTERMAGNETSource"]


class INTERMAGNETSource(BFieldSource):
    """Loader for INTERMAGNET IAGA-2002 magnetometer data.

    Parameters
    ----------
    file_path : str
        Path to the ``.min`` / ``.sec`` / ``.iaga`` file.
    declination_unit : {"minutes", "degrees"}, optional
        Interpretation of the ``D`` column for HDZF files. Default:
        ``"minutes"`` (IAGA-2002 native). Ignored for XYZF files.

    Examples
    --------
    >>> src = INTERMAGNETSource("OTT_20240510.min")     # doctest: +SKIP
    >>> b = src.load(start_s=0, end_s=86400)             # doctest: +SKIP
    >>> b.bx_T.shape                                     # doctest: +SKIP
    (1440,)
    """

    def __init__(
        self,
        file_path: str,
        declination_unit: str = "minutes",
    ) -> None:
        if declination_unit not in {"minutes", "degrees"}:
            raise DataError(
                f"declination_unit must be 'minutes' or 'degrees', got {declination_unit!r}"
            )
        self.file_path = file_path
        self.declination_unit = declination_unit

    def load(
        self,
        start_s: float = -np.inf,
        end_s: float = np.inf,
        dt_s: float = 1.0,
        station_id: str = "",
    ) -> BFieldTimeSeries:
        """Load and standardise the file.

        Parameters
        ----------
        start_s, end_s : float, optional
            Time window in UNIX epoch seconds (UTC). Defaults return the
            full file. Samples outside the window are dropped.
        dt_s : float, optional
            Ignored — the file's native sampling rate is used. Present for
            ABC symmetry.
        station_id : str, optional
            Override for the station identifier. If empty, the file's
            ``IAGA CODE`` is used.

        Returns
        -------
        BFieldTimeSeries
            Bx/By/Bz in Tesla, time in UTC epoch seconds.

        Raises
        ------
        DataError
            On unrecognised reporting orientation or empty windowed slice.
        """
        del dt_s  # unused — file's native sampling wins
        f = read_iaga2002(self.file_path)

        mask = (f.time_utc >= start_s) & (f.time_utc <= end_s)
        if not np.any(mask):
            raise DataError(f"No IAGA-2002 samples in [{start_s}, {end_s}] for {self.file_path}")

        t = f.time_utc[mask]
        c1 = f.col1_nT[mask]
        c2 = f.col2_nT[mask]
        c3 = f.col3_nT[mask]

        # Convert to geographic X (north) / Y (east) / Z (down) in nT.
        if f.reporting == "XYZF":
            bx_nT, by_nT, bz_nT = c1, c2, c3
        elif f.reporting in {"HDZF", "DHZF"}:
            # Column order for DHZF puts D first; swap so H = c1.
            if f.reporting == "DHZF":
                c1, c2 = c2, c1
            H = c1
            D = c2
            if self.declination_unit == "minutes":
                D_rad = (D / 60.0) * DEG_TO_RAD
            else:
                D_rad = D * DEG_TO_RAD
            bx_nT = H * np.cos(D_rad)
            by_nT = H * np.sin(D_rad)
            bz_nT = c3
        elif f.reporting == "HEZF":
            # H, E (east) — H is horizontal magnitude, E is east component;
            # solve X = sqrt(H^2 - E^2) with sign from declination convention.
            H, E = c1, c2
            X_sq = H * H - E * E
            bx_nT = np.sqrt(np.clip(X_sq, 0.0, None))
            by_nT = E
            bz_nT = c3
        else:  # pragma: no cover - guarded in read_iaga2002
            raise DataError(f"Unhandled reporting: {f.reporting}")

        # nT → T at the boundary.
        bx_T = bx_nT * NT_TO_T
        by_T = by_nT * NT_TO_T
        bz_T = bz_nT * NT_TO_T

        # Sampling rate from the median dt (robust to occasional gaps).
        dt_s_median = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        sampling_rate_Hz = 1.0 / dt_s_median if dt_s_median > 0.0 else 0.0

        return BFieldTimeSeries(
            time_s=t.astype(np.float64),
            bx_T=bx_T.astype(np.float64),
            by_T=by_T.astype(np.float64),
            bz_T=bz_T.astype(np.float64),
            station_id=station_id or f.station_code,
            latitude_deg=f.latitude_deg,
            longitude_deg=f.longitude_deg,
            sampling_rate_Hz=sampling_rate_Hz,
            metadata={
                "source": "INTERMAGNETSource",
                "file_path": str(self.file_path),
                "reporting": f.reporting,
                "declination_unit": self.declination_unit,
                "iaga_header": dict(f.header),
                "n_missing": int(np.sum(np.isnan(bx_T))),
            },
        )
