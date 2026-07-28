# Walling & Khan (1991) — transformer half-cycle saturation curves

This directory is a **placeholder**. It documents the CSV format a
contributor needs to drop in to unblock
[`geopulse.devices.half_cycle_harmonics`](../../src/geopulse/devices/half_cycle_harmonics.py),
which is currently a stub raising `NotImplementedYetError`.

## What's needed

Empirical harmonic-ratio curves from:

> Walling, R. A., & Khan, A. N. (1991). *Characteristics of transformer
> exciting current during geomagnetic disturbances*. IEEE Trans. Power
> Delivery, 6(4), 1707–1714. https://doi.org/10.1109/61.97711

Specifically Figure 6 (and equivalents in Girgis & Vedante 2012) plot
the ratio `a_k / a_1` (order-`k` amplitude over fundamental) as a
function of the normalised DC bias `x = I_dc / I_ex`, where `I_ex` is
the transformer's rated no-load exciting current. Each transformer
family (500 kV GSU, single-phase auto, three-phase core-type, etc.)
has its own curve set.

## File layout (per transformer family)

```
walling_khan_1991/
├── README.md                                  ← this file
├── walling_khan_500kV_gsu.csv                 ← Walling & Khan 1991 Fig. 6
├── girgis_vedante_three_phase_core.csv        ← Girgis & Vedante 2012
└── <your_paper>_<transformer_family>.csv
```

Each CSV:

```
# Provenance: Walling & Khan (1991) Fig. 6, digitised by <you> on <date>.
# Transformer family: 500 kV single-phase GSU (representative).
# Rows: measured points on the empirical curves.
x_normalised, a2_over_a1, a3_over_a1, a4_over_a1, a5_over_a1, a6_over_a1, a7_over_a1, a8_over_a1, a9_over_a1, a10_over_a1
0.0,   0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
0.2,   0.15,  0.08,  0.05,  0.03,  0.02,  0.01,  0.01,  0.00,  0.00
0.5,   0.40,  0.25,  0.15,  0.10,  0.07,  0.05,  0.03,  0.02,  0.01
1.0,   0.65,  0.40,  0.25,  0.15,  0.10,  0.07,  0.05,  0.03,  0.02
2.0,   0.80,  0.50,  0.30,  0.20,  0.13,  0.08,  0.06,  0.04,  0.03
```

*(the numbers above are illustrative shape only — do NOT commit these
placeholder values; transcribe the real curves from the paper.)*

## Implementation task

After the CSV is in place, unblock the stub by:

1. Reading the CSV into a table (`numpy.genfromtxt` with the header
   skipped).
2. For each requested order `k`, linear-interpolate `a_k / a_1` at the
   caller's `x = dc_bias_A / rated_exciting_current_A`.
3. Choose an absolute scale for `a_1` (either return the ratios
   directly, or take a caller-supplied `fundamental_amplitude_A` — TBD
   during implementation).
4. Populate and return a `HarmonicSpectrum` in
   [`geopulse.devices.harmonics`](../../src/geopulse/devices/harmonics.py).
5. Add a filename → key mapping so `transformer_model="walling_khan_500kV_gsu"`
   resolves to `walling_khan_500kV_gsu.csv`.
6. Replace the "raises NotImplementedYetError" assertions in
   [`tests/unit/test_devices/test_half_cycle_harmonics.py`](../../tests/unit/test_devices/test_half_cycle_harmonics.py)
   with numerical checks against 2–3 spot values transcribed from the
   paper (e.g. "at `x = 0.5`, `a2 / a1` should be `0.40 ± 5 %`").

Expected size: a few hours of transcription + a few hours of coding,
call it a one-day contribution.

## Licensing / redistribution

The Walling & Khan and Girgis & Vedante papers are IEEE-copyrighted.
Numerical values digitised from the figures (i.e. small tables of
`(x, ratio)` pairs) are *facts* and not themselves copyrightable in
most jurisdictions, so a CSV of transcribed points is safe to
distribute under the same Apache-2.0 licence as GeoPulse. **Do not**
redistribute PDF pages, figure images, or long verbatim excerpts.

If in doubt, add a `provenance:` header line in the CSV pointing to
the DOI, and mark rows with an inline comment noting they are
transcribed measurements.
