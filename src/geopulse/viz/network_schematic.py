"""Publication-style DC-equivalent circuit schematics for GIC modelling.

Companion to :mod:`geopulse.viz.network_map` (which draws the same
network on a geographic canvas). Where the map answers *where* nodes
sit, this module answers *what the circuit actually looks like* — the
transformer windings, the neutral point, the grounding resistor, and
the coupling into transmission-line induced-voltage sources that a
GIC modeller has to reason about.

Flavours
--------
Three levels of ambition, all sharing the same schemdraw backend:

1. :func:`plot_transformer_dc_equivalent` — one transformer winding
   configuration rendered as its DC-equivalent circuit. Reproduces
   Mate, Barnes, Bent & Cotilla-Sanchez (2021) Fig 1 for
   ``gywe-delta``, ``gywe-gywe``, and ``gywe-gywe-auto``. Fixed
   layout, data-free — a reference figure for the methods section
   of any GIC-modelling paper.
2. :func:`plot_substation_dc_equivalent` — one substation from a real
   ``PowerGridNetwork`` rendered as its DC-equivalent circuit, with
   real transformer resistances, real ``R_gnd``, outgoing transmission-
   line stubs annotated with induced voltage ``V = E·L``, and live
   ``I_GIC`` at the neutral if a solve result is supplied.
3. :func:`plot_network_dc_equivalent` — the whole network as a
   composed schematic: each substation as a compact block, connected
   by transmission-line elements labelled with real ``Ω`` values.

Backend
-------
Uses `schemdraw <https://schemdraw.readthedocs.io/>`_ for IEEE
electrical symbols (transformer windings, resistors, sources, ground
symbols) — publication-quality out of the box, matplotlib-based, so
:mod:`geopulse.viz.presets` still applies for column-width sizing
and font choices.

``schemdraw`` lives under the ``[viz]`` optional dependency group.
If it is not installed a :class:`DataError` with an install hint is
raised on first call — the rest of ``geopulse.viz`` continues to work.

References
----------
.. [1] Mate, A., Barnes, A. K., Bent, R., & Cotilla-Sanchez, E. (2021).
   Analyzing and Mitigating the Impacts of GMD and EMP Events on the
   Electrical Grid with PowerModelsGMD.jl. arXiv:2101.05042v2. Fig 1.
.. [2] Boteler, D. H. (2014). Methodology for simulation of
   geomagnetically induced currents in power systems. J. Space
   Weather Space Clim., 4, A21.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geopulse.exceptions import DataError

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = [
    "TransformerConfig",
    "plot_network_dc_equivalent",
    "plot_substation_dc_equivalent",
    "plot_transformer_dc_equivalent",
]

TransformerConfig = Literal["gywe-delta", "gywe-gywe", "gywe-gywe-auto"]


def _require_schemdraw() -> tuple[Any, Any]:
    """Import and return the ``schemdraw`` module, or raise with a hint."""
    try:
        import schemdraw
        import schemdraw.elements as elm
    except ImportError as exc:
        raise DataError(
            "schemdraw is required for circuit schematics; install with "
            "`pip install geopulse[viz]` (or `pip install schemdraw`)."
        ) from exc
    return schemdraw, elm


def plot_transformer_dc_equivalent(
    config: TransformerConfig,
    *,
    savepath: str | Path | None = None,
    dpi: int = 150,
    show_current_arrows: bool = True,
    label_grounding_admittance: bool = True,
) -> "Figure":
    r"""Render the DC-equivalent circuit of one transformer winding config.

    Reproduces Fig 1 of Mate, Barnes, Bent & Cotilla-Sanchez (2021) for
    the three winding types that GIC-capable network models routinely
    represent:

    * ``"gywe-delta"`` — grounded-wye / delta (typical GSU or
      load-side step-down). Only the wye winding conducts GIC to
      ground; the delta blocks it.
    * ``"gywe-gywe"`` — grounded-wye / grounded-wye (typical
      transmission-transmission bank). Both windings conduct.
    * ``"gywe-gywe-auto"`` — grounded-wye autotransformer with a
      series and a common winding sharing a neutral. Series and
      common branches conduct in parallel through the neutral.

    All three end at the same shared grounding admittance
    :math:`a_i = 1 / (3 R_{gi})` — the factor of three comes from
    the three-phase-in-parallel reduction to a single-phase equivalent
    per Boteler (2014).

    Parameters
    ----------
    config : {"gywe-delta", "gywe-gywe", "gywe-gywe-auto"}
        Which winding configuration to render.
    savepath : str or pathlib.Path, optional
        If given, save the figure to this path (PNG by default; use a
        ``.svg`` / ``.pdf`` suffix for vector output). Default: don't
        save.
    dpi : int, optional
        Raster DPI when ``savepath`` ends in ``.png``. Default: 150.
    show_current_arrows : bool, optional
        Overlay :math:`I_h^t`, :math:`I_l^t`, :math:`I_s^t`,
        :math:`I_c^t` current-direction arrows on the windings.
        Default: ``True``.
    label_grounding_admittance : bool, optional
        Label the grounding element as ``aᵢ = 1/(3 R_gi)`` rather
        than a plain resistor. Default: ``True``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the schematic.

    Raises
    ------
    DataError
        If ``config`` is not one of the three recognised strings, or
        if ``schemdraw`` is not installed.

    Examples
    --------
    Draft-quality reproduction of Mate et al. (2021) Fig 1a:

    >>> from geopulse.viz.network_schematic import plot_transformer_dc_equivalent  # doctest: +SKIP
    >>> fig = plot_transformer_dc_equivalent("gywe-delta", savepath="fig1a.png")   # doctest: +SKIP

    All three panels side-by-side (see the ``examples/`` script for
    the composed 3-panel version).

    >>> for c in ("gywe-delta", "gywe-gywe", "gywe-gywe-auto"):                    # doctest: +SKIP
    ...     plot_transformer_dc_equivalent(c, savepath=f"fig_{c}.png")             # doctest: +SKIP
    """
    if config not in ("gywe-delta", "gywe-gywe", "gywe-gywe-auto"):
        raise DataError(
            f"config must be one of gywe-delta / gywe-gywe / gywe-gywe-auto, got {config!r}"
        )
    schemdraw, elm = _require_schemdraw()

    ground_label = "aᵢ = 1/(3 R_gᵢ)" if label_grounding_admittance else "R_gᵢ"

    if config == "gywe-delta":
        fig = _draw_gywe_delta(schemdraw, elm, ground_label, show_current_arrows)
    elif config == "gywe-gywe":
        fig = _draw_gywe_gywe(schemdraw, elm, ground_label, show_current_arrows)
    else:  # gywe-gywe-auto
        fig = _draw_gywe_gywe_auto(schemdraw, elm, ground_label, show_current_arrows)

    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")
    return fig


def _draw_gywe_delta(schemdraw: Any, elm: Any, ground_label: str, arrows: bool) -> "Figure":
    """Grounded-wye / delta: only the wye winding conducts to ground."""
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.4, fontsize=12, lw=1.4)
    d += elm.Dot().label(r"$V_l^t$", loc="top", ofst=(-0.05, 0.15))
    d += elm.Line().right().length(1.6)
    d += (wye := elm.Resistor().down().label(r"$a_c$", loc="right", ofst=(0.1, 0)))
    if arrows:
        d += elm.CurrentLabel(reverse=True, ofst=0.4).at(wye).label(r"$I_h^t$")
    d += elm.Line().down().length(0.5)
    d += elm.Resistor().down().label(ground_label, loc="right", ofst=(0.15, 0))
    d += elm.Ground()
    fig = d.draw().fig
    return fig


def _draw_gywe_gywe(schemdraw: Any, elm: Any, ground_label: str, arrows: bool) -> "Figure":
    """Grounded-wye / grounded-wye: both windings conduct in parallel to ground."""
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.4, fontsize=12, lw=1.4)
    # Top bus with two terminals
    d += (top_h := elm.Dot().label(r"$V_h^t$", loc="top", ofst=(-0.05, 0.15)))
    d += elm.Line().right().length(3.0)
    d += (top_l := elm.Dot().label(r"$V_l^t$", loc="top", ofst=(0.05, 0.15)))
    # Left winding a_h down from V_h terminal
    d.move_from(top_h.center, dx=0, dy=0)
    d += (r_hv := elm.Resistor().down().label(r"$a_h$", loc="left", ofst=(-0.15, 0)))
    if arrows:
        d += elm.CurrentLabel(reverse=True, ofst=0.4).at(r_hv).label(r"$I_h^t$")
    left_bot = d.here
    # Right winding a_l down from V_l terminal
    d.move_from(top_l.center, dx=0, dy=0)
    d += (r_lv := elm.Resistor().down().label(r"$a_l$", loc="right", ofst=(0.15, 0)))
    if arrows:
        d += elm.CurrentLabel(ofst=0.4).at(r_lv).label(r"$I_l^t$")
    right_bot = d.here
    # Neutral bus joining left_bot and right_bot
    d += elm.Line().at(left_bot).to(right_bot)
    # Mid-point of the neutral bus down to ground
    mid_x = 0.5 * (left_bot[0] + right_bot[0])
    d += elm.Line().at((mid_x, left_bot[1])).down().length(0.5)
    d += elm.Resistor().down().label(ground_label, loc="right", ofst=(0.15, 0))
    d += elm.Ground()
    fig = d.draw().fig
    return fig


def _draw_gywe_gywe_auto(schemdraw: Any, elm: Any, ground_label: str, arrows: bool) -> "Figure":
    """Autotransformer: series + common windings share a neutral."""
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.4, fontsize=12, lw=1.4)
    d += elm.Dot().label(r"$V_h^t$", loc="top", ofst=(-0.05, 0.15))
    d += (r_s := elm.Resistor().down().label(r"$a_s$", loc="right", ofst=(0.15, 0)))
    if arrows:
        d += elm.CurrentLabel(reverse=True, ofst=0.4).at(r_s).label(r"$I_s^t$")
    d += elm.Dot().label(r"$V_l^t$", loc="right", ofst=(0.2, 0))
    d += (r_c := elm.Resistor().down().label(r"$a_c$", loc="right", ofst=(0.15, 0)))
    if arrows:
        d += elm.CurrentLabel(reverse=True, ofst=0.4).at(r_c).label(r"$I_c^t$")
    d += elm.Line().down().length(0.5)
    d += elm.Resistor().down().label(ground_label, loc="right", ofst=(0.15, 0))
    d += elm.Ground()
    fig = d.draw().fig
    return fig


# ---------------------------------------------------------------------------
# Flavour (2): per-substation DC-equivalent from a parsed network
# ---------------------------------------------------------------------------


def plot_substation_dc_equivalent(
    network: Any,
    sub_id: str,
    *,
    node_voltages_V: Any = None,
    field_source_ex_Vm: float | None = None,
    savepath: str | Path | None = None,
    dpi: int = 150,
) -> "Figure":
    r"""Render one real substation's DC-equivalent circuit.

    Walks the parsed ``PowerGridNetwork`` topology to find the target
    substation's high-side AC bus (via zero-length transformer branches),
    every transformer branch touching it, and every transmission line
    incident on the high-side bus. Draws:

    * HV bus at top (thick coloured line, labelled with its ``dc_bus*`` id).
    * One vertical stub per outgoing transmission line, each rendered as
      a series resistor labelled with its real ``R_line`` and a
      Thévenin voltage source labelled with the induced ``V = E·L`` if
      the caller passes ``field_source_ex_Vm``.
    * Parallel transformer windings from the HV bus down to the neutral
      bus, each labelled with its real winding resistance from the
      MATPOWER file.
    * Neutral bus at bottom (coloured, labelled with the ``dc_sub*`` id).
    * Grounding resistor labelled with the real ``R_gnd`` in Ω.
    * Optional ``I_GIC`` current-arrow annotation on the grounding leg
      when ``node_voltages_V`` is provided (from a solver run).

    Parameters
    ----------
    network : PowerGridNetwork
        Parsed network. Only ``get_nodes`` / ``get_branches`` /
        ``assemble_earthing_impedance`` are used.
    sub_id : str
        Node id of the target substation (e.g. ``"dc_sub6"``).
    node_voltages_V : numpy.ndarray, optional
        Per-node DC voltages from a solver run. If given, the
        substation's live GIC is annotated on the grounding leg as
        ``I_GIC = V_sub / R_gnd``.
    field_source_ex_Vm : float, optional
        Uniform eastward driving field, V/m. If given, transmission-
        line stubs are labelled with the induced Thévenin voltage
        approximated as ``E_x * L_km * 1000`` (single-component
        proxy; the real solver uses the full 2-D integral).
    savepath : str or pathlib.Path, optional
        If given, save the figure to this path.
    dpi : int, optional
        Raster DPI for ``.png`` output. Default: 150.

    Returns
    -------
    matplotlib.figure.Figure
        The schematic.

    Raises
    ------
    DataError
        If ``sub_id`` is not a node in the network, or ``schemdraw``
        is not installed.

    Examples
    --------
    >>> net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")  # doctest: +SKIP
    >>> fig = plot_substation_dc_equivalent(                                # doctest: +SKIP
    ...     net, "dc_sub6", field_source_ex_Vm=1e-3,                        # doctest: +SKIP
    ...     savepath="sub6_schematic.png",                                  # doctest: +SKIP
    ... )                                                                    # doctest: +SKIP
    """
    schemdraw, elm = _require_schemdraw()

    nodes = list(network.get_nodes())
    node_by_id = {n.node_id: n for n in nodes}
    if sub_id not in node_by_id:
        raise DataError(f"substation {sub_id!r} not found in network")

    branches = list(network.get_branches())
    # Transformers touching this substation = zero-length branches at sub_id.
    xf_branches = [b for b in branches if b.length_m == 0.0 and sub_id in (b.from_node, b.to_node)]
    if not xf_branches:
        raise DataError(
            f"{sub_id!r} has no transformer branches — nothing to schematise. "
            "Isolated substations (Sub 1, Sw.Sta 7 in Horton EPRI21) are "
            "solver-transparent and not renderable as circuits."
        )

    hv_bus_ids: list[str] = []
    for b in xf_branches:
        other = b.to_node if b.from_node == sub_id else b.from_node
        if other not in hv_bus_ids:
            hv_bus_ids.append(other)
    hv_bus = hv_bus_ids[0]  # Canonical HV bus (usually unique per sub in Horton).

    # Outgoing transmission lines from the HV bus.
    line_branches = [b for b in branches if b.length_m > 0.0 and hv_bus in (b.from_node, b.to_node)]

    # R_gnd from earthing impedance diagonal.
    Z = network.assemble_earthing_impedance()
    sub_idx = next(i for i, n in enumerate(nodes) if n.node_id == sub_id)
    r_gnd = float(Z[sub_idx, sub_idx])

    # Live GIC if solver voltages supplied.
    gic_A: float | None = None
    if node_voltages_V is not None and r_gnd > 0.0:
        gic_A = float(abs(node_voltages_V[sub_idx] / r_gnd))

    return _draw_substation(
        schemdraw,
        elm,
        sub_id=sub_id,
        hv_bus=hv_bus,
        xf_branches=xf_branches,
        line_branches=line_branches,
        r_gnd=r_gnd,
        gic_A=gic_A,
        field_ex_Vm=field_source_ex_Vm,
        savepath=savepath,
        dpi=dpi,
    )


def _draw_substation(
    schemdraw: Any,
    elm: Any,
    *,
    sub_id: str,
    hv_bus: str,
    xf_branches: list,
    line_branches: list,
    r_gnd: float,
    gic_A: float | None,
    field_ex_Vm: float | None,
    savepath: str | Path | None,
    dpi: int,
) -> "Figure":
    """Render the substation schematic from resolved topology."""
    n_lines = len(line_branches)
    n_xfmrs = len(xf_branches)
    # Wider bus_span so vertical stubs at top have room for their side labels.
    bus_span = max(n_lines, n_xfmrs, 2) * 2.8

    d = schemdraw.Drawing(show=False)
    d.config(unit=2.0, fontsize=11, lw=1.4)

    # Top: outgoing transmission line stubs (R + Vsrc + terminal label).
    # Alternate R and V_e labels left/right per stub to avoid crowding.
    line_x_positions = _evenly_spaced(n_lines, bus_span)
    top_y = 5.5
    for k, (x, br) in enumerate(zip(line_x_positions, line_branches, strict=True)):
        other_bus = br.to_node if br.from_node == hv_bus else br.from_node
        r_side = "left" if k % 2 == 0 else "right"
        v_side = "right" if k % 2 == 0 else "left"
        d += elm.Dot().at((x, top_y + 3.4)).label(f"to {other_bus}", loc="top", fontsize=9)
        d += elm.Line().at((x, top_y + 3.4)).to((x, top_y + 2.8))
        d += (
            elm.Resistor()
            .at((x, top_y + 2.8))
            .to((x, top_y + 1.6))
            .label(
                f"{br.resistance_Ohm:.2f} Ω",
                loc=r_side,
                fontsize=9,
                ofst=(-0.15 if r_side == "left" else 0.15, 0),
            )
        )
        if field_ex_Vm is not None:
            v_ind = field_ex_Vm * br.length_m  # V/m * m = V (single-component proxy)
            d += (
                elm.SourceV(reverse=True)
                .at((x, top_y + 1.6))
                .to((x, top_y + 0.4))
                .label(
                    f"$V_e$ = {v_ind:+.0f} V",
                    loc=v_side,
                    fontsize=9,
                    ofst=(-0.15 if v_side == "left" else 0.15, 0),
                )
            )
        else:
            d += elm.Line().at((x, top_y + 1.6)).to((x, top_y + 0.4))
        d += elm.Line().at((x, top_y + 0.4)).to((x, top_y))

    # HV bus (horizontal thick coloured line).
    d += elm.Line().at((-0.5, top_y)).to((bus_span + 0.5, top_y)).linewidth(3).color("crimson")
    d += (
        elm.Label()
        .at((bus_span / 2, top_y + 0.25))
        .label(f"{hv_bus} (HV)", fontsize=10, color="crimson")
    )

    # Middle: parallel transformer windings from HV bus down to neutral bus.
    xf_x_positions = _evenly_spaced(n_xfmrs, bus_span)
    neutral_y = 2.0
    for x, br in zip(xf_x_positions, xf_branches, strict=True):
        d += elm.Line().at((x, top_y)).to((x, top_y - 0.4))
        # Transformer element — a Resistor is the "electrical-equivalent" per
        # Boteler (2014) for DC analysis (the reactance vanishes at DC).
        # We label with the winding branch id so the reader can trace back
        # to the MATPOWER row.
        xf_label = br.branch_id.replace("dc_xf", "T").split("_")[0]
        d += (
            elm.Resistor()
            .at((x, top_y - 0.4))
            .to((x, neutral_y + 0.4))
            .label(f"{xf_label}\n{br.resistance_Ohm:.3f} Ω", loc="right", fontsize=9, ofst=(0.1, 0))
        )
        d += elm.Line().at((x, neutral_y + 0.4)).to((x, neutral_y))

    # Neutral bus.
    d += (
        elm.Line()
        .at((-0.5, neutral_y))
        .to((bus_span + 0.5, neutral_y))
        .linewidth(3)
        .color("darkorange")
    )
    d += (
        elm.Label()
        .at((bus_span / 2, neutral_y - 0.35))
        .label(f"{sub_id} (neutral)", fontsize=10, color="darkorange")
    )

    # Grounding leg: R_gnd + Ground.
    gnd_x = bus_span / 2
    d += elm.Line().at((gnd_x, neutral_y)).to((gnd_x, neutral_y - 0.9))
    d += (
        elm.Resistor()
        .at((gnd_x, neutral_y - 0.9))
        .to((gnd_x, neutral_y - 2.1))
        .label(f"$R_{{gnd}}$ = {r_gnd:.2f} Ω", loc="right", fontsize=10, ofst=(0.1, 0))
    )
    if gic_A is not None:
        d += (
            elm.CurrentLabel(ofst=0.4)
            .at((gnd_x, neutral_y - 1.5))
            .label(f"$I_{{GIC}}$ = {gic_A:.0f} A")
        )
    d += elm.Ground().at((gnd_x, neutral_y - 2.1))

    fig = d.draw().fig
    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")
    return fig


def _evenly_spaced(n: int, span: float) -> list[float]:
    """N evenly-spaced x-positions along [0, span]."""
    if n <= 0:
        return []
    if n == 1:
        return [span / 2]
    step = span / (n + 1)
    return [step * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Flavour (3): whole-network DC-equivalent
# ---------------------------------------------------------------------------


def plot_network_dc_equivalent(
    network: Any,
    *,
    node_voltages_V: Any = None,
    savepath: str | Path | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (14, 8),
) -> "Figure":
    r"""Compact whole-network DC-equivalent circuit.

    Every substation with valid coordinates is drawn as a compact
    circuit block (a resistor-to-ground icon representing its ``R_gnd``,
    with the ``dc_sub*`` id above and live GIC below if
    ``node_voltages_V`` is provided). Substations are laid out
    according to their geographic coordinates (scaled to the figure
    canvas), and every transmission line drawn as a two-terminal
    element labelled with its real resistance in Ω. Isolated substations
    (no transformer branches) are skipped — see the docstring of
    :func:`plot_substation_dc_equivalent` for the physics.

    The rendering is a graph of circuit elements, **not** a geographic
    map — coordinates are used only to seed a topologically-clean
    layout. Companion to :func:`geopulse.viz.network_map.plot_network_map`
    which does the geographic view.

    Parameters
    ----------
    network : PowerGridNetwork
        Parsed network with substation and branch data.
    node_voltages_V : numpy.ndarray, optional
        Per-node DC voltages from a solver run. Live GIC values are
        annotated under each substation icon.
    savepath : str or pathlib.Path, optional
        If given, save the figure to this path.
    dpi : int, optional
        Raster DPI for ``.png`` output. Default: 150.
    figsize : tuple[float, float], optional
        Matplotlib figure size in inches. Default ``(14, 8)`` fits the
        Horton corridor at 2-column width.

    Returns
    -------
    matplotlib.figure.Figure
        The whole-network schematic.

    Raises
    ------
    DataError
        If ``schemdraw`` is not installed, or the network has no
        substation with valid coordinates.
    """
    schemdraw, elm = _require_schemdraw()
    nodes = list(network.get_nodes())
    branches = list(network.get_branches())
    Z = network.assemble_earthing_impedance()

    sub_nodes = _finite_coord_subs(nodes)
    live_sub_ids = _live_sub_ids(sub_nodes, branches)
    bus_to_sub = _bus_to_sub_map(nodes, sub_nodes)
    layout = _sub_layout(sub_nodes)

    d = schemdraw.Drawing(show=False, figsize=figsize)
    d.config(fontsize=9, lw=1.2)
    sub_positions = _draw_all_sub_icons(
        d,
        elm,
        sub_nodes,
        live_sub_ids,
        layout,
        Z,
        node_voltages_V,
    )
    _draw_all_inter_sub_lines(d, elm, branches, bus_to_sub, sub_positions)

    fig = d.draw().fig
    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")
    return fig


def _finite_coord_subs(nodes: list) -> list[tuple[int, Any]]:
    """Substations that have finite lat/lon, paired with their row index."""
    import numpy as np

    subs = [
        (i, n)
        for i, n in enumerate(nodes)
        if n.node_id.startswith("dc_sub")
        and np.isfinite(n.latitude_deg)
        and np.isfinite(n.longitude_deg)
    ]
    if not subs:
        raise DataError("network has no substation with finite coordinates")
    return subs


def _live_sub_ids(sub_nodes: list, branches: list) -> set[str]:
    """Substations with at least one transformer branch (renderable as a circuit)."""
    live = {
        n.node_id
        for _, n in sub_nodes
        if any(b.length_m == 0.0 and n.node_id in (b.from_node, b.to_node) for b in branches)
    }
    if not live:
        raise DataError("network has no substation with transformer branches")
    return live


def _bus_to_sub_map(nodes: list, sub_nodes: list) -> dict[str, str]:
    """dc_bus* → dc_sub* by coordinate equality (Horton convention)."""
    import numpy as np

    out: dict[str, str] = {}
    for _, sub in sub_nodes:
        for m in nodes:
            if (
                m.node_id.startswith("dc_bus")
                and np.isfinite(m.latitude_deg)
                and m.latitude_deg == sub.latitude_deg
                and m.longitude_deg == sub.longitude_deg
            ):
                out[m.node_id] = sub.node_id
    return out


def _sub_layout(sub_nodes: list) -> dict[str, tuple[float, float]]:
    """Scale geographic coordinates to a (12, 6) canvas rectangle."""
    import numpy as np

    lons = np.array([n.longitude_deg for _, n in sub_nodes])
    lats = np.array([n.latitude_deg for _, n in sub_nodes])
    lon_span = lons.max() - lons.min()
    lat_span = lats.max() - lats.min()
    x_scale = 12.0 / (lon_span if lon_span > 0 else 1.0)
    y_scale = 6.0 / (lat_span if lat_span > 0 else 1.0)
    return {
        n.node_id: (
            (n.longitude_deg - lons.min()) * x_scale,
            (n.latitude_deg - lats.min()) * y_scale,
        )
        for _, n in sub_nodes
    }


def _draw_all_sub_icons(
    d: Any,
    elm: Any,
    sub_nodes: list,
    live_sub_ids: set[str],
    layout: dict[str, tuple[float, float]],
    Z: Any,
    node_voltages_V: Any,
) -> dict[str, tuple[float, float]]:
    """Draw one icon per live substation; return {sub_id: (x, y)}."""
    positions: dict[str, tuple[float, float]] = {}
    for i, n in sub_nodes:
        if n.node_id not in live_sub_ids:
            continue
        x, y = layout[n.node_id]
        positions[n.node_id] = (x, y)
        r_gnd = float(Z[i, i])
        gic_A: float | None = None
        if node_voltages_V is not None and r_gnd > 0.0:
            gic_A = float(abs(node_voltages_V[i] / r_gnd))
        _draw_sub_icon(elm, d, x, y, n.node_id, r_gnd, gic_A)
    return positions


def _draw_all_inter_sub_lines(
    d: Any,
    elm: Any,
    branches: list,
    bus_to_sub: dict[str, str],
    sub_positions: dict[str, tuple[float, float]],
) -> None:
    """Draw one grey wire + Ω label per unique inter-substation line pair."""
    drawn: set[tuple[str, str]] = set()
    for b in branches:
        if b.length_m == 0.0:
            continue
        pair = _resolve_inter_sub_pair(b, bus_to_sub, sub_positions)
        if pair is None or pair in drawn:
            continue
        drawn.add(pair)
        (x1, y1) = sub_positions[pair[0]]
        (x2, y2) = sub_positions[pair[1]]
        d += elm.Line().at((x1, y1 + 0.6)).to((x2, y2 + 0.6)).color("gray").linewidth(0.9)
        mid_x = 0.5 * (x1 + x2)
        mid_y = 0.5 * (y1 + y2) + 0.75
        d += (
            elm.Label()
            .at((mid_x, mid_y))
            .label(f"{b.resistance_Ohm:.1f} Ω", fontsize=7, color="dimgray")
        )


def _resolve_inter_sub_pair(
    b: Any,
    bus_to_sub: dict[str, str],
    sub_positions: dict[str, tuple[float, float]],
) -> tuple[str, str] | None:
    """Return (sub_a, sub_b) for an inter-sub branch, or None to skip."""
    a_sub = bus_to_sub.get(b.from_node)
    b_sub = bus_to_sub.get(b.to_node)
    if a_sub is None or b_sub is None or a_sub == b_sub:
        return None
    if a_sub not in sub_positions or b_sub not in sub_positions:
        return None
    return (a_sub, b_sub) if a_sub < b_sub else (b_sub, a_sub)


def _draw_sub_icon(
    elm: Any,
    d: Any,
    x: float,
    y: float,
    sub_id: str,
    r_gnd: float,
    gic_A: float | None,
) -> None:
    """Compact per-substation icon: label + R + ground, all under (x, y)."""
    d += elm.Label().at((x, y + 1.4)).label(sub_id.replace("dc_sub", "S"), fontsize=10)
    d += elm.Line().at((x, y + 0.6)).to((x, y + 0.2)).linewidth(2.4).color("crimson")
    d += (
        elm.Resistor()
        .at((x, y + 0.2))
        .to((x, y - 0.6))
        .label(f"{r_gnd:.2f}Ω", loc="right", fontsize=7, ofst=(0.05, 0))
    )
    d += elm.Ground().at((x, y - 0.6))
    if gic_A is not None:
        # Place GIC well below the ground symbol so it never collides.
        d += elm.Label().at((x, y - 1.4)).label(f"{gic_A:.0f} A", fontsize=8, color="crimson")
