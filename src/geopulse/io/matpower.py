"""MATPOWER-GMD (``.m``) parser for power-grid GIC test cases.

Extracts only the GMD/DC-network-relevant sections used by
:class:`~geopulse.network.powergrid.PowerGridNetwork`:

* ``mpc.gmd_bus``   — DC ground nodes and grounding conductance ``g_gnd`` [S]
* ``mpc.gmd_branch`` — DC-equivalent branches (lines, transformer windings)
  with per-branch resistance ``br_r`` [Ω] and length ``len_km`` [km]
* ``mpc.bus_gmd``    — (lat, lon) per AC bus
* ``mpc.bus`` / ``mpc.branch`` — AC-side metadata retained for downstream
  configuration helpers (baseKV, ratings, statuses)

This is a **narrow** parser — enough to load ``epri21.m`` (Horton et al.
2012) and equivalent PowerModelsGMD.jl test files. It is NOT a general
MATPOWER reader; that is out of scope for GeoPulse core.

References
----------
.. [1] Horton, R., Boteler, D., Overbye, T. J., Pirjola, R., & Dugan, R. C.
   (2012). A test case for the calculation of geomagnetically induced
   currents. IEEE Transactions on Power Delivery, 27(4), 2368-2373.
.. [2] LANL-ANSI PowerModelsGMD.jl — reference implementation of the
   MATPOWER-GMD format extensions.

Notes
-----
Row-position vs. AC bus number: the ``f_bus`` / ``t_bus`` fields inside
``mpc.gmd_branch`` reference **1-based row positions** within
``mpc.gmd_bus``, NOT AC bus numbers. Callers must respect this convention.
The ``parent_index`` column on each ``gmd_bus`` row instead names the AC
bus (substation) that DC node belongs to; use it for geographic lookup via
``mpc.bus_gmd``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from geopulse.exceptions import DataError

__all__ = ["MatpowerGMDCase", "read_matpower_gmd"]

# Regex: match `mpc.<name> = { ... };` OR `mpc.<name> = [ ... ];`.
# The block delimiters differ between MATPOWER text formats; both are
# tolerated here.
_BLOCK_RE_TEMPLATE = r"mpc\.{name}\s*=\s*[\{{\[](.*?)[\}}\]]\s*;"


def _extract_block(text: str, name: str) -> list[list[str]]:
    """Pull one ``mpc.<name>`` block as a list of tokenised rows.

    Parameters
    ----------
    text : str
        Full file contents.
    name : str
        Block name (e.g. ``"gmd_bus"``).

    Returns
    -------
    list of list of str
        One inner list per non-empty, non-comment source line.
    """
    pattern = _BLOCK_RE_TEMPLATE.format(name=re.escape(name))
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    rows: list[list[str]] = []
    for raw in body.strip().splitlines():
        # Drop inline `% ...` comments, then tokenise on whitespace while
        # preserving single-quoted strings as one token (used for names).
        cleaned = raw.split("%")[0].strip()
        if not cleaned:
            continue
        tokens = re.findall(r"'[^']*'|\S+", cleaned)
        rows.append(tokens)
    return rows


def _num(tok: str) -> float:
    """Cast a MATPOWER token to ``float``; raise :class:`DataError` on failure."""
    try:
        return float(tok)
    except ValueError as exc:
        raise DataError(f"Cannot parse numeric token {tok!r}") from exc


@dataclass(frozen=True)
class MatpowerGMDCase:
    """Structured contents of a MATPOWER-GMD ``.m`` file.

    Attributes
    ----------
    source_path : pathlib.Path
        Path the case was loaded from.
    n_dc_nodes : int
        Number of DC (grounding) nodes.
    dc_nodes : list of dict
        One entry per DC node, ordered by 1-based row position within
        ``mpc.gmd_bus``. Fields:

        * ``row``           1-based row index (DC node id used by branches)
        * ``parent_ac_bus`` AC bus number this DC node grounds
        * ``status``        1 = in service, 0 = out
        * ``g_gnd_S``       grounding conductance in Siemens
        * ``name``          human name (e.g. ``"dc_sub1"``)
    dc_branches : list of dict
        DC branches. Fields:

        * ``from_row``, ``to_row``   1-based DC node row indices
        * ``parent_ac_branch``       1-based AC branch index (or 0 for none)
        * ``status``                 1 in service, 0 out
        * ``resistance_Ohm``         DC resistance
        * ``length_km``              geographic length (km); 0 for xfmr windings
        * ``name``                   human name
    bus_latlon : dict of int -> (float, float)
        ``ac_bus_id -> (latitude_deg, longitude_deg)``.
    ac_bus_kv : dict of int -> float
        ``ac_bus_id -> baseKV``.
    ac_branches : list of dict
        AC-side branches, retained for case-configuration helpers (each has
        ``f``, ``t``, ``rateA``, ``ratio``, ``status``).
    """

    source_path: Path
    n_dc_nodes: int
    dc_nodes: list[dict] = field(default_factory=list)
    dc_branches: list[dict] = field(default_factory=list)
    bus_latlon: dict[int, tuple[float, float]] = field(default_factory=dict)
    ac_bus_kv: dict[int, float] = field(default_factory=dict)
    ac_branches: list[dict] = field(default_factory=list)


def read_matpower_gmd(path: str | Path) -> MatpowerGMDCase:
    """Parse a MATPOWER-GMD ``.m`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.m`` file.

    Returns
    -------
    MatpowerGMDCase
        Structured case data.

    Raises
    ------
    DataError
        If the file is missing required blocks (``gmd_bus``, ``gmd_branch``,
        ``bus``, ``bus_gmd``) or has malformed rows.
    """
    p = Path(path)
    if not p.is_file():
        raise DataError(f"MATPOWER-GMD file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")

    gmd_bus_rows = _extract_block(text, "gmd_bus")
    gmd_branch_rows = _extract_block(text, "gmd_branch")
    bus_rows = _extract_block(text, "bus")
    bus_gmd_rows = _extract_block(text, "bus_gmd")
    branch_rows = _extract_block(text, "branch")

    for name, rows in [
        ("gmd_bus", gmd_bus_rows),
        ("gmd_branch", gmd_branch_rows),
        ("bus", bus_rows),
        ("bus_gmd", bus_gmd_rows),
    ]:
        if not rows:
            raise DataError(f"MATPOWER-GMD file {p} is missing block mpc.{name}")

    # --- AC bus metadata (id -> kV) ---
    ac_bus_kv: dict[int, float] = {}
    ac_bus_ids: list[int] = []
    for r in bus_rows:
        bid = int(_num(r[0]))
        ac_bus_ids.append(bid)
        ac_bus_kv[bid] = _num(r[9])

    # --- AC branch metadata (retained for case helpers) ---
    ac_branches: list[dict] = []
    for r in branch_rows:
        ac_branches.append(
            {
                "f": int(_num(r[0])),
                "t": int(_num(r[1])),
                "rateA": _num(r[5]),
                "ratio": _num(r[8]),
                "status": int(_num(r[10])),
            }
        )

    # --- DC nodes: 1-based row position IS the id used by branches ---
    dc_nodes: list[dict] = []
    for row_i, r in enumerate(gmd_bus_rows, start=1):
        parent_idx = int(_num(r[0]))
        status = int(_num(r[1]))
        g_gnd = _num(r[2])
        name = r[3].strip("'") if len(r) > 3 else f"dc_node_{row_i}"
        dc_nodes.append(
            {
                "row": row_i,
                "parent_ac_bus": parent_idx,
                "status": status,
                # Downstream expects "in service" nodes only to contribute
                # grounding; carry raw g_gnd but zero it if status == 0.
                "g_gnd_S": g_gnd if status == 1 else 0.0,
                "name": name,
            }
        )

    # --- DC branches ---
    dc_branches: list[dict] = []
    for r in gmd_branch_rows:
        f_row = int(_num(r[0]))
        t_row = int(_num(r[1]))
        # parent_type is r[2] (unused here); parent_index at r[3] is 1-based
        # index into mpc.branch (0 when the DC element has no AC parent).
        parent_ac = int(_num(r[3]))
        br_status = int(_num(r[4]))
        br_r = _num(r[5])
        len_km = _num(r[7]) if len(r) > 7 else 0.0
        name = r[8].strip("'") if len(r) > 8 else f"dc_br_{len(dc_branches) + 1}"
        dc_branches.append(
            {
                "from_row": f_row,
                "to_row": t_row,
                "parent_ac_branch": parent_ac,
                "status": br_status,
                "resistance_Ohm": br_r,
                "length_km": len_km,
                "name": name,
            }
        )

    # --- AC bus lat/lon (rows in the same order as mpc.bus) ---
    bus_latlon: dict[int, tuple[float, float]] = {}
    if len(bus_gmd_rows) != len(ac_bus_ids):
        raise DataError(
            f"bus_gmd row count ({len(bus_gmd_rows)}) does not match bus count "
            f"({len(ac_bus_ids)}) in {p}"
        )
    for bus_id, r in zip(ac_bus_ids, bus_gmd_rows, strict=True):
        bus_latlon[bus_id] = (_num(r[0]), _num(r[1]))

    return MatpowerGMDCase(
        source_path=p,
        n_dc_nodes=len(dc_nodes),
        dc_nodes=dc_nodes,
        dc_branches=dc_branches,
        bus_latlon=bus_latlon,
        ac_bus_kv=ac_bus_kv,
        ac_branches=ac_branches,
    )
