"""Schema-versioned HDF5 reader/writer.

Every file written by GeoPulse carries a ``schema_version`` attribute at the
root. The reader migrates older schemas forward automatically so that files
written by previous versions remain readable. Writers always emit the current
schema.

Layout (schema v1)::

    /
      attrs:
        schema_version   int
        geopulse_version str
        created_utc      str  (ISO-8601)
        description      str
      source/            (optional)
        attrs: {station_id, latitude_deg, longitude_deg, sampling_rate_Hz}
        time_s, bx_T, by_T, bz_T   datasets
      earth/             (optional)
        attrs: {model_name, tier}
        impedance/       group with impedance-specific layout
      results/           (optional)
        efield/          Ex_Vm, Ey_Vm
        gic/             node_ids, currents_A
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from geopulse._version import __version__ as _GEOPULSE_VERSION
from geopulse.exceptions import DataError

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "read_results",
    "write_results",
]

CURRENT_SCHEMA_VERSION: int = 1
"""Integer schema version — bump on any breaking change to the layout."""


def _write_group(group: h5py.Group, payload: dict[str, Any]) -> None:
    """Recursively write a nested dict into an HDF5 group."""
    for key, value in payload.items():
        if isinstance(value, dict):
            sub = group.create_group(key)
            _write_group(sub, value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype.kind in {"U", "S", "O"}:
                # Store variable-length UTF-8 strings.
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(key, data=np.asarray(value, dtype=object), dtype=dt)
            else:
                group.create_dataset(key, data=arr)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            group.attrs[key] = value
        elif isinstance(value, str):
            group.attrs[key] = value
        else:
            raise DataError(f"Unsupported value type for HDF5: {type(value)} at {key}")


def _read_group(group: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group into a nested dict."""
    out: dict[str, Any] = {k: group.attrs[k] for k in group.attrs.keys()}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item)
        else:
            out[key] = item[()]
    return out


def _migrate(data: dict[str, Any], from_version: int) -> dict[str, Any]:
    """Migrate a payload dict forward to :data:`CURRENT_SCHEMA_VERSION`.

    Currently a no-op — there is only one schema version. Future breaking
    changes should add a branch here (v1 → v2 → ...).
    """
    if from_version == CURRENT_SCHEMA_VERSION:
        return data
    # Future migration ladder goes here.
    return data


def write_results(path: str | Path, description: str = "", **groups: Any) -> None:
    """Write results to HDF5 with the current schema version.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path. Any existing file is overwritten.
    description : str, optional
        Free-text description recorded at the root.
    **groups
        Named groups to write. Each value should be a nested dict of arrays,
        scalars, or sub-dicts.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False).name
    >>> write_results(tmp, description="demo",
    ...               source={"time_s": np.arange(10.0)})
    >>> data = read_results(tmp)
    >>> data["source"]["time_s"].shape
    (10,)
    >>> os.unlink(tmp)
    """
    path = Path(path)
    with h5py.File(path, "w") as fh:
        fh.attrs["schema_version"] = CURRENT_SCHEMA_VERSION
        fh.attrs["geopulse_version"] = _GEOPULSE_VERSION
        fh.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        fh.attrs["description"] = description
        for name, payload in groups.items():
            grp = fh.create_group(name)
            if isinstance(payload, dict):
                _write_group(grp, payload)
            else:
                raise DataError(f"Top-level group {name!r} must be a dict; got {type(payload)}")


def read_results(path: str | Path) -> dict[str, Any]:
    """Read an HDF5 file, migrating older schemas forward.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to read.

    Returns
    -------
    dict
        Nested dict of the file contents, including root attributes under
        keys ``schema_version``, ``geopulse_version``, ``created_utc``, and
        ``description``.

    Raises
    ------
    DataError
        If the file is missing the ``schema_version`` attribute.
    """
    path = Path(path)
    with h5py.File(path, "r") as fh:
        if "schema_version" not in fh.attrs:
            raise DataError(f"{path} is missing 'schema_version' — not a GeoPulse file")
        payload = _read_group(fh)
    version = int(payload.get("schema_version", CURRENT_SCHEMA_VERSION))
    return _migrate(payload, version)
