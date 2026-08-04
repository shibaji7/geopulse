# Shi hypotheses — spatial-field, mitigation, and reconfiguration effects on GIC

**Paper-companion directory.** Scripts and figures here reproduce the
three hypothesis tests underpinning an in-preparation paper on why two
similarly-active geomagnetic storms produced order-of-magnitude
different GIC observations at the same substation.

## Question

A single substation shows similar magnetic-field variability during
two geomagnetic storms yet reports GIC magnitudes different by roughly
an order of magnitude. Three mechanisms could each, alone or in
combination, produce that difference:

- **H1 (spatial-field)** — the *local* geoelectric field at the target
  substation is comparable between the two storms, but the *regional*
  field structure differs. Because GIC is set by the whole network's
  current divider, a purely-spatial change in the field elsewhere in
  the grid can change GIC at the observation point by an order of
  magnitude while its local driving field stays fixed.
- **H2 (mitigation)** — the operator changed the neutral-side
  grounding: added a DC-blocking device, raised a neutral-grounding
  resistor, or otherwise increased the effective earthing impedance
  at the target substation (or across many substations) between the
  two storms.
- **H3 (reconfiguration)** — the network topology changed: a
  transmission line was out of service, or a new tie was in service,
  redistributing where the GIC concentrates.

Each hypothesis is a testable, isolated claim about the physics. The
scripts here quantify how large an effect each mechanism can produce
on the Horton (2012) 21-bus EPRI test network, which is representative
of a mid-latitude EHV corridor and is shipped with the repo under
`benchmarks/horton2012/epri21.m`.

## Files

```
examples/papers/shi_hypotheses/
├── README.md                                  ← this file
├── scripts/
│   ├── hypothesis1_spatial_field.py           ← dE/dx sweep at target
│   ├── hypothesis2_mitigation.py              ← blocker at all / one substation
│   ├── hypothesis3_reconfiguration.py         ← line outage + added tie
│   └── plot_network_map_setup.py              ← baseline network + GIC map
├── notebooks/
│   └── main.ipynb                             ← narrative + inline figure calls
└── figures/                                    ← generated outputs (PNG)
```

Every script is standalone. From the repo root:

```bash
python examples/papers/shi_hypotheses/scripts/hypothesis1_spatial_field.py
python examples/papers/shi_hypotheses/scripts/hypothesis2_mitigation.py
python examples/papers/shi_hypotheses/scripts/hypothesis3_reconfiguration.py
python examples/papers/shi_hypotheses/scripts/plot_network_map_setup.py
```

The notebook imports and calls each script's headline function, so
`Restart Kernel & Run All` reproduces every figure inline.

## What this study *does not* do

- **No real storm data.** The point of the three hypothesis tests is
  to quantify the *mechanism*: how much GIC change can each cause
  independently, given synthetic uniform or gradient fields? Real
  spatially-resolved storm E-fields would come from a SECS driver
  (Weygand et al. 2011), which is a v0.5 GeoPulse item and outside the
  scope of this reproduction.
- **No publication-quality figures.** Everything here is draft-quality
  matplotlib. When the paper is being submitted, `geopulse.viz.presets`
  provides journal-specific presets (`jgr_2col`, `sw_2col`, …) that
  will be dropped in via a two-line change to each script's save call.
- **No transformer thermal / harmonics.** Deferred by design; those
  are separate downstream questions once the three mechanism-level
  hypotheses are settled.

## Design choices

- **Target substation**: `dc_sub6` (the substation with the largest
  single-node GIC under a uniform 1 V/km eastward field in Horton
  EPRI21 — a good demonstration proxy for a real IK413-like
  observation point).
- **Baseline field**: 1 V/km eastward, uniform. All three hypothesis
  scripts sanity-check against this baseline before sweeping their
  variable.
- **Blocker resistance for H2**: 10¹² Ω (effective open-circuit DC
  blocker) via `geopulse.network.helpers.apply_resistive_blocker`. The
  same helper is used with a more moderate 10 Ω resistance in a
  secondary run to show a partial-mitigation intermediate.
- **Reconfiguration for H3**: single-line outages walked through
  every high-impact line + one added tie between distant substations.

## Reproducibility

- Every script prints its headline numbers to stdout for traceability.
- All figures are re-generated from scratch on each run — no cached
  intermediates.
- Uses only what ships with a plain `pip install geopulse` (no
  optional extras beyond `matplotlib`, which is already core).

## Archive plan

This directory is intended to be Zenodo-archived alongside the paper
at a future release; do not treat the scripts as stable API. When the
archive lands, the DOI will be added to the paper and this README.
