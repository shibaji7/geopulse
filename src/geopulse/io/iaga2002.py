"""IAGA-2002 magnetometer format parser.

IAGA-2002 is the standard geomagnetic observatory format used by
INTERMAGNET and many observatories. Files have:

* a fixed-width 12-line header (each line 69 chars, terminated with ``|``),
* optional ``#``-prefixed comment lines,
* one column-header line (also ``|``-terminated),
* data rows: ``YYYY-MM-DD HH:MM:SS.sss  DOY  <col1>  <col2>  <col3>  <col4>``.

The header includes the IAGA station code, geodetic lat/lon, and a
``Reporting`` field naming the four data columns — typically ``XYZF``
(geographic X-north, Y-east, Z-down, total F, all in nT) or ``HDZF``
(horizontal, declination, vertical, total).

This module reads the file and returns raw arrays; the SI conversion and
frame-standardisation (D → X, Y) happen in
:class:`geopulse.sources.intermagnet.INTERMAGNETSource`.

References
----------
* IAGA-2002 format specification, INTERMAGNET Technical Reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geopulse.exceptions import DataError

__all__ = ["Iaga2002File", "read_iaga2002"]

# Sentinel values IAGA-2002 uses for missing data.
_MISSING_SENTINELS = {99999.0, 88888.0}


@dataclass(frozen=True)
class Iaga2002File:
    """Parsed contents of a single IAGA-2002 file.

    Attributes
    ----------
    station_code : str
        3-letter IAGA observatory code (e.g. ``"OTT"``).
    latitude_deg : float
        Geodetic latitude, degrees north.
    longitude_deg : float
        Geodetic longitude, degrees east (0-360 in the file, normalised to
        [-180, 180] here).
    reporting : str
        4-character orientation string from the header (e.g. ``"XYZF"``).
    time_utc : numpy.ndarray
        UNIX epoch seconds (UTC). Shape ``(n_samples,)``.
    col1_nT, col2_nT, col3_nT, col4_nT : numpy.ndarray
        The four data columns in nT (D columns remain in minutes of arc;
        the source adapter is responsible for angle interpretation).
        Sentinel values (``99999``, ``88888``) are replaced with ``NaN``.
    header : dict
        Full raw header key/value pairs, keys lowercased and stripped.
    """

    station_code: str
    latitude_deg: float
    longitude_deg: float
    reporting: str
    time_utc: np.ndarray
    col1_nT: np.ndarray
    col2_nT: np.ndarray
    col3_nT: np.ndarray
    col4_nT: np.ndarray
    header: dict


def _clean(v: float) -> float:
    """Replace IAGA sentinel values with NaN; pass floats through."""
    return float("nan") if v in _MISSING_SENTINELS else v


def _parse_time(date_str: str, time_str: str) -> float:
    """Parse ``'YYYY-MM-DD'`` + ``'HH:MM:SS.sss'`` into UTC epoch seconds."""
    ts = f"{date_str}T{time_str}"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise DataError(f"Cannot parse IAGA timestamp {ts!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_iaga2002(path: str | Path) -> Iaga2002File:
    """Read an IAGA-2002 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.iaga`` / ``.min`` / ``.sec`` file.

    Returns
    -------
    Iaga2002File
        Parsed file contents. Component columns are in nT (with sentinels
        NaN'd); the ``reporting`` field tells you what the four columns
        actually represent (XYZF vs HDZF vs ...).

    Raises
    ------
    DataError
        If the file is missing a required header field, has an unrecognised
        reporting orientation, or fails to parse.

    Examples
    --------
    >>> from geopulse.io.iaga2002 import read_iaga2002
    >>> f = read_iaga2002("some_file.min")  # doctest: +SKIP
    >>> f.station_code                       # doctest: +SKIP
    'OTT'
    """
    p = Path(path)
    if not p.is_file():
        raise DataError(f"IAGA-2002 file not found: {p}")

    header: dict[str, str] = {}
    data_header_line: str | None = None

    with p.open("r", encoding="utf-8", errors="replace") as fh:
        # --- Header block: pipe-terminated key/value pairs ----------------
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if not line.endswith("|"):
                raise DataError(f"IAGA-2002 header terminated unexpectedly in {p}: {line!r}")
            body = line[:-1].rstrip()
            if body.lstrip().startswith("#"):
                continue
            if body.lstrip().startswith("DATE"):
                data_header_line = body
                break
            key = body[:24].strip().lower()
            value = body[24:].strip()
            header[key] = value

    if data_header_line is None:
        raise DataError(f"IAGA-2002 file {p} has no data-column header row")

    try:
        station_code = header["iaga code"]
        lat = float(header["geodetic latitude"])
        lon = float(header["geodetic longitude"])
        reporting = header.get("reporting", "").strip().upper()
    except KeyError as exc:
        raise DataError(f"IAGA-2002 header missing field: {exc}") from exc

    if len(reporting) != 4 or reporting not in {"XYZF", "HDZF", "HEZF", "DHZF"}:
        raise DataError(f"Unrecognised IAGA reporting orientation {reporting!r} in {p}")

    if lon > 180.0:
        lon -= 360.0

    times: list[float] = []
    c1: list[float] = []
    c2: list[float] = []
    c3: list[float] = []
    c4: list[float] = []

    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            body = raw.rstrip().rstrip("|").rstrip()
            if body.lstrip().startswith("DATE"):
                break
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 6:
                continue
            t = _parse_time(tokens[0], tokens[1])
            times.append(t)
            c1.append(_clean(float(tokens[3])))
            c2.append(_clean(float(tokens[4])))
            c3.append(_clean(float(tokens[5])))
            c4.append(_clean(float(tokens[6])) if len(tokens) > 6 else float("nan"))

    if not times:
        raise DataError(f"IAGA-2002 file {p} has no data rows")

    return Iaga2002File(
        station_code=station_code,
        latitude_deg=lat,
        longitude_deg=lon,
        reporting=reporting,
        time_utc=np.asarray(times, dtype=np.float64),
        col1_nT=np.asarray(c1, dtype=np.float64),
        col2_nT=np.asarray(c2, dtype=np.float64),
        col3_nT=np.asarray(c3, dtype=np.float64),
        col4_nT=np.asarray(c4, dtype=np.float64),
        header=header,
    )
