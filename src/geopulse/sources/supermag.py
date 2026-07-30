"""SuperMAG CSV / ASCII loader.

SuperMAG (https://supermag.jhuapl.edu) provides ground-magnetometer data
from a global network of observatories in a consistent frame (baseline-
subtracted N/E/Z, geographic components in nT).

Their common CSV export has:

* ``#``-prefixed comment lines at the top (metadata),
* one header row naming columns: at minimum ``Date_UTC, IAGA, N, E, Z``
  (may also include ``GEOLAT, GEOLON, MAGLAT, MAGLON, MLT, SZA, ...``),
* one row per (station, timestamp).

If the file contains multiple stations, a specific one can be selected by
passing ``iaga=`` or ``station_id=`` at load time; otherwise the first
station encountered is used.

References
----------
* Gjerloev, J. W. (2012). The SuperMAG data processing technique.
  Journal of Geophysical Research: Space Physics, 117(A9).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geopulse.constants import NT_TO_T
from geopulse.exceptions import DataError
from geopulse.sources.base import BFieldSource, BFieldTimeSeries

__all__ = ["SuperMAGSource"]

# Column-name aliases we tolerate across SuperMAG export flavours.
_TIME_KEYS = ("Date_UTC", "date_utc", "DATE_UTC", "Date", "timestamp")
_IAGA_KEYS = ("IAGA", "iaga", "Station", "station")
_N_KEYS = ("N", "n", "N_nez", "dbn_nez")
_E_KEYS = ("E", "e", "E_nez", "dbe_nez")
_Z_KEYS = ("Z", "z", "Z_nez", "dbz_nez")
_LAT_KEYS = ("GEOLAT", "geolat", "Latitude", "lat")
_LON_KEYS = ("GEOLON", "geolon", "Longitude", "lon")


def _first_key(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row:
            return k
    return None


def _parse_supermag_time(value: str) -> float:
    """Parse SuperMAG UTC timestamps into epoch seconds.

    Handles both ``'2024-05-10T00:00:00'`` (ISO) and
    ``'2024-05-10 00:00:00'`` (space-separated) forms.
    """
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    # Fallback: fromisoformat handles fractional seconds.
    try:
        dt = datetime.fromisoformat(value.replace("T", " "))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError as exc:
        raise DataError(f"Cannot parse SuperMAG timestamp {value!r}") from exc


class SuperMAGSource(BFieldSource):
    """Loader for SuperMAG station CSV data.

    Parameters
    ----------
    file_path : str
        Path to the SuperMAG-format CSV file.
    iaga : str, optional
        IAGA station code to select when the file contains multiple
        stations. Case-insensitive. If ``None``, uses the first station
        encountered.

    Examples
    --------
    >>> src = SuperMAGSource("supermag_may2024.csv", iaga="OTT")   # doctest: +SKIP
    >>> b = src.load(start_s=0, end_s=86400)                        # doctest: +SKIP
    >>> b.station_id                                                # doctest: +SKIP
    'OTT'
    """

    def __init__(self, file_path: str, iaga: str | None = None) -> None:
        self.file_path = file_path
        self.iaga = iaga.upper() if iaga else None

    def load(  # noqa: C901 — SuperMAG CSV format varies enough that consolidating the ingest here beats splitting.
        self,
        start_s: float = -np.inf,
        end_s: float = np.inf,
        dt_s: float = 1.0,
        station_id: str = "",
    ) -> BFieldTimeSeries:
        """Load a SuperMAG CSV file into a standardised :class:`BFieldTimeSeries`.

        Parameters
        ----------
        start_s, end_s : float, optional
            Time window in UNIX epoch seconds (UTC). Defaults are open.
        dt_s : float, optional
            Ignored — the file's native sampling rate is used.
        station_id : str, optional
            Override for the returned ``station_id`` field. If empty, the
            IAGA code found in the file is used.

        Returns
        -------
        BFieldTimeSeries
            N/E/Z channels mapped directly to Bx (north) / By (east) /
            Bz (down), converted from nT to Tesla.

        Raises
        ------
        DataError
            If the file is missing required columns or contains no rows for
            the selected station in the requested window.
        """
        del dt_s
        p = Path(self.file_path)
        if not p.is_file():
            raise DataError(f"SuperMAG file not found: {p}")

        times: list[float] = []
        n_list: list[float] = []
        e_list: list[float] = []
        z_list: list[float] = []
        chosen_iaga: str | None = None
        lat = float("nan")
        lon = float("nan")

        with p.open("r", encoding="utf-8", errors="replace") as fh:
            # Skip comment lines at the top.
            reader_lines = (ln for ln in fh if not ln.lstrip().startswith("#") and ln.strip())
            reader = csv.DictReader(reader_lines)
            fieldnames = reader.fieldnames or []

            k_time = _first_key({k: None for k in fieldnames}, _TIME_KEYS)
            k_iaga = _first_key({k: None for k in fieldnames}, _IAGA_KEYS)
            k_n = _first_key({k: None for k in fieldnames}, _N_KEYS)
            k_e = _first_key({k: None for k in fieldnames}, _E_KEYS)
            k_z = _first_key({k: None for k in fieldnames}, _Z_KEYS)
            k_lat = _first_key({k: None for k in fieldnames}, _LAT_KEYS)
            k_lon = _first_key({k: None for k in fieldnames}, _LON_KEYS)

            if not all((k_time, k_iaga, k_n, k_e, k_z)):
                raise DataError(
                    f"SuperMAG file {p} missing required columns; "
                    f"have {fieldnames}, need Date_UTC/IAGA/N/E/Z (or aliases)"
                )

            for row in reader:
                iaga = row[k_iaga].strip().upper()
                if self.iaga is not None and iaga != self.iaga:
                    continue
                if chosen_iaga is None:
                    chosen_iaga = iaga
                    if k_lat is not None and row.get(k_lat):
                        lat = float(row[k_lat])
                    if k_lon is not None and row.get(k_lon):
                        lon = float(row[k_lon])
                elif iaga != chosen_iaga:
                    # File has more stations after our first pick — ignore
                    # them silently; user should pass iaga= to disambiguate.
                    continue
                t = _parse_supermag_time(row[k_time])
                if t < start_s or t > end_s:
                    continue
                times.append(t)
                n_list.append(float(row[k_n]))
                e_list.append(float(row[k_e]))
                z_list.append(float(row[k_z]))

        if not times:
            raise DataError(
                f"SuperMAG file {p} yielded no rows for iaga={self.iaga!r} "
                f"in window [{start_s}, {end_s}]"
            )

        t_arr = np.asarray(times, dtype=np.float64)
        bx_T = np.asarray(n_list, dtype=np.float64) * NT_TO_T
        by_T = np.asarray(e_list, dtype=np.float64) * NT_TO_T
        bz_T = np.asarray(z_list, dtype=np.float64) * NT_TO_T

        dt_median_s = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 0.0
        sampling_rate_Hz = 1.0 / dt_median_s if dt_median_s > 0.0 else 0.0

        return BFieldTimeSeries(
            time_s=t_arr,
            bx_T=bx_T,
            by_T=by_T,
            bz_T=bz_T,
            station_id=station_id or chosen_iaga or "?",
            latitude_deg=lat,
            longitude_deg=lon,
            sampling_rate_Hz=sampling_rate_Hz,
            metadata={
                "source": "SuperMAGSource",
                "file_path": str(self.file_path),
                "iaga_requested": self.iaga,
            },
        )
