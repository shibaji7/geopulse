# GeoPulse Case Study Examples

Examples demonstrating GeoPulse's capabilities against 24 documented
GIC events from the literature. See `private/geopulse_case_studies.md`
for the full source references and paper citations for each case.

Every script is **self-contained** — runnable with plain
`python examples/case_studies/case_NN_*.py` from the repo root; no
external data files or credentials required. Real magnetometer data
is replaced with synthetic proxies where needed, with clear TODO
comments pointing at the loader that would replace them.

## Status legend

- **✅ COMPLETE** — all required GeoPulse modules are implemented; the
  chain runs end to end and can be compared quantitatively against
  published values.
- **🟡 PARTIAL** — the core physics chain runs, but at least one module
  needed for the full published analysis is a stub. The script marks
  where the chain stops with `# --- CHAIN STOPS HERE ---` and shows
  what would come next as `# TODO:` comments.
- **❌ NOT PRESENT** — the case is blocked by a stub module for the
  entire chain (currently: submarine cables + railways). No script
  ships in this directory for these cases; they will land when the
  relevant network subclass is implemented.

## Case index

| # | Script | Event | Status | What it shows |
|---|---|---|---|---|
| 1 | `case_01_hydro_quebec_1989.py` | Hydro-Québec Blackout | 🟡 PARTIAL | Québec 7-layer earth + LPM on proxy grid + xfmr thermal |
| 2 | `case_02_malmo_halloween_2003.py` | Malmö / Halloween storm | 🟡 PARTIAL | Baltic coast-effect on \|Z\|; peak-storm LPM snapshot |
| 3 | `case_03_south_africa_transformer_2003.py` | ESKOM xfmr damage | 🟡 PARTIAL | Resistive Kaapvaal earth amplifies GIC → hot-spot |
| 4 | `case_04_nz_gannon_2024.py` | NZ Gannon storm | 🟡 PARTIAL | Full E→GIC→thermal chain; harmonics stubbed |
| 5 | `case_05_ireland_1d_vs_2d.py` | Irish grid model | 🟡 PARTIAL | 1-D vs 2-D CoastalCorrection2D impedance comparison |
| 6 | `case_06_japan_low_latitude.py` | Japan low-lat GIC | 🟡 PARTIAL | Watari-scale GIC on proxy grid |
| 7 | `case_07_us_gannon_nerc.py` | US Gannon / NERC benchmark | ✅ COMPLETE | FRD IAGA-2002 → 1-D Earth → NAM hindcast + σ-sweep + NERC 8 V/km anchor |
| 8 | `case_08_brazil_itumbiara.py` | Brazil Itumbiara | 🟡 PARTIAL | Peak-GIC distribution over 200 synthetic events |
| 9 | `case_09_taps_alaska.py` | Trans-Alaska pipeline | 🟡 PARTIAL | 1300 km DSTL under auroral E; cp_unit stubbed |
| 10 | `case_10_sweden_pipelines.py` | Sweden pipelines | ✅ COMPLETE | DSTL π-section vs closed-form profile |
| 11 | `case_11_maritimes_ne_pipeline.py` | Maritimes NE pipeline | 🟡 PARTIAL | PSP vs E-incidence angle; cp_unit stubbed |
| 12 | `case_12_nz_gas_pipeline.py` | NZ gas pipeline | 🟡 PARTIAL | E-direction sensitivity per Ingham 2018 |
| 13 | `case_13_czech_oil_pipelines.py` | Czech oil pipelines | ✅ COMPLETE | Halloween E-field sweep on Bohemian pipeline |
| 14 | `case_14_norway_short_pipelines.py` | Norway short pipelines | ✅ COMPLETE | Peak-PSP saturation vs pipe length |
| 15 | `case_15_argentine_pipeline.py` | Argentine pipeline | 🟡 PARTIAL | PSP time-history; cp_unit + exceedance stubbed |
| 16 | (not shipped) | Sweden rail 1982 | ❌ | `network/railway` is a stub |
| 17 | (not shipped) | UK rail Patterson | ❌ | `network/railway` is a stub |
| 18 | (not shipped) | Russian railways | ❌ | `network/railway` is a stub |
| 19 | (not shipped) | Boteler 2021 rail model | ❌ | `network/railway` is a stub |
| 20 | (not shipped) | TAT-8 cable 1989 | ❌ | `network/cable` is a stub |
| 21 | (not shipped) | SCUBAS Gannon 2024 | ❌ | `network/cable` is a stub |
| 22 | `case_22_gannon_multi_infra.py` | Gannon multi-infra | 🟡 PARTIAL | Grid + pipeline slice under one E-field |
| 23 | `case_23_horton_benchmark.py` | Horton (2012) benchmark | ✅ COMPLETE | Table VII match within 1% every substation |
| 24 | `case_24_pulkkinen_benchmark_2017.py` | Pulkkinen 2017 benchmark | 🟡 PARTIAL | Field-direction sweep on Horton network |

**Coverage: 18 scripts ship — 4 COMPLETE, 14 PARTIAL, 6 blocked on
`network/cable` or `network/railway` stubs.**

## Running

Run one:

```bash
python examples/case_studies/case_23_horton_benchmark.py
```

Run all in sequence (from repo root):

```bash
for f in examples/case_studies/case_*.py; do
    echo "=== $(basename "$f") ==="
    python "$f" || echo "FAILED: $f"
done
```

Each script writes a PNG to `examples/output/` and prints headline
numbers to stdout.

## What to do with the ❌ NOT PRESENT cases

Cases 16-21 require the `network/railway` (WP4) and `network/cable`
(WP2 — SCUBAS refactor target) stubs to be filled in. When those
land, the six missing scripts should be added following the same
template as the pipeline scripts:

- 16, 17, 18, 19 mirror `case_10_sweden_pipelines.py` but for
  `RailwayNetwork` with track-circuit-appropriate parameters.
- 20 mirrors `case_10` for `CableNetwork` with TAT-8 geometry, and can
  compare against Boteler (2024) reported voltages.
- 21 is a direct regression test of the SCUBAS refactor; ideally
  should reproduce PI's own SCUBAS output for the Gannon 2024 storm to
  numerical parity.

See `private/geopulse_case_studies.md` § "Recommended Implementation
Order" for the phased priority ordering.
