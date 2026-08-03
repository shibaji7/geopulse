# GeoPulse roadmap

*Last updated: 2026-07-30.*

This is the single source of truth for **what is done, what is
partial, what is planned, and what is not touched yet** across the
GeoPulse codebase, plus a per-release tentative timeline with
pre-release (`aN` / `bN` / `rcN`) tag conventions.

Legend used throughout:

| Status | Meaning |
|--------|---------|
| ✅ **Done** | Implemented, tested, docs shipped, and matches its target paper / spec within its stated tolerance. |
| 🟡 **Partial** | Core physics chain works; at least one dependency is a stub or a placeholder default. The chain runs end-to-end but a downstream number needs the deferred piece to be defensible in a paper. |
| 🔵 **Planned** | Not started. Scope + citations agreed. Waiting for a slot on the release train. |
| ⬛ **Not touched** | Named in the architecture but no design work done. |

---

## Release train (tentative timeline)

Semantic versioning ([SemVer 2.0](https://semver.org/)). Pre-release
tags follow the Python packaging convention:

* **`aN` (alpha)** — API not stable; expect breaking changes; testing
  package plumbing (build, publish, CI, docs). Currently we're here.
* **`bN` (beta)** — API mostly stable; integration testing, bug
  hunting, real-storm shakedowns. No new features after `b1`.
* **`rcN` (release candidate)** — feature-freeze, critical-fixes only.
* **`.0`** — stable release. Docs, examples, and CHANGELOG all match.

See [`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-conventions) for
the decision rules on when to bump MAJOR / MINOR / PATCH / pre-release
suffix.

| Milestone | Scope highlight | Pre-release phases | Target |
|-----------|-----------------|--------------------|--------|
| **v0.1.0** (current alpha) | Base engine: sources, 1-D earth, plane-wave E, NAM/LPm solver, power-grid + pipeline networks, transformer thermal, IAGA/HDF5 I/O, Horton benchmark <1 % | `0.1.0aN` alphas (live) → `0.1.0b1` → `0.1.0` | end Q3 2026 |
| **v0.2.0** | Merge the open PRs. Adds: viz stacked-panel plotter, GIC statistics + exceedance, FFT harmonics + IEEE 519 THD, MNA (ideal V-sources), RailwayNetwork DSTL, issue templates, Gannon-day FRD case study, coverage cleanup, **network.helpers + devices.blocker + viz.network_map + viz.presets** (PR #20) | `0.2.0aN` alphas → `0.2.0bN` betas → `0.2.0rc1` → `0.2.0` | Q4 2026 |
| **v0.3.0** | CableNetwork (SCUBAS refactor) unlocks cases #20 (TAT-8) + #21 (SCUBAS Gannon). Model-based `half_cycle_harmonics` with Walling & Khan curves. USGS physiographic 1-D Earth library. First journal-scale viz presets (JGR SpP / Space Weather). | `0.3.0a1..N` → `0.3.0b1..2` → `0.3.0rc1` → `0.3.0` | H1 2027 |
| **v0.5.0** | Reactive-element MNA (companion models + trapezoidal integration). PySpice bridge for nonlinear transient. Structured 2-D FD MT Earth solver (replaces `CoastalCorrection2D` pragmatic stand-in). **SECS** driver (Weygand et al. 2011) producing 2-D `B(x,y,t)` from sparse magnetometer arrays, paired with `efield.nonuniform` for spatially-varying `E(x,y,t)`. Rectifier + cathodic-protection device models. | `0.5.0a1..N` → beta → rc → release | H2 2027 |
| **v0.9.0** | 3-D Earth via ModEM / SimPEG wrapper. Uncertainty propagation (Monte Carlo + Sobol). Sparse solver for continent-scale grids. FastAPI service layer for hosted GIC hindcasts. | `0.9.0a` → `b` → `rc` → release | 2027–2028 |
| **v1.0.0** | Stable public API. JOSS paper submitted. Zenodo DOI hooked up. Every implemented module ≥ 90 % coverage. Every case study either ✅ or explicitly retired. | `1.0.0rc1..N` → `1.0.0` | 2028 |

Version bumps between minors are driven by *scope*, not calendar. If a
target slips, everything on that milestone slips with it — no partial
releases.

---

## Currently live on `main`

- **PyPI**: [`geopulse`](https://pypi.org/project/geopulse/) — see
  the PyPI page for the current alpha tag (single source of truth
  lives in `src/geopulse/_version.py`). Base engine works
  end-to-end; 117 tests pass; benchmarks reproduce Horton (2012)
  Table VII within 0.774 %.
- **Docs**: <https://geopulse.readthedocs.io/> — Sphinx autosummary
  over the public API + hand-written getting-started.
- **CI**: GitHub Actions, pytest matrix over Py 3.10–3.13, coverage
  gated at 75 % (rising to 80 % after PR #14).

## Recently merged

- ✅ **#8** (issue templates), **#14** (coverage cleanup + `fail_under=80`), **#16** (guardrails: interrogate + pylint dup + gitlint + ruff D/N/C901/ANN)

## In flight (open PRs)

| PR | Branch | What lands | Merge risk |
|----|--------|------------|-----------|
| [#6](https://github.com/shibaji7/geopulse/pull/6) | `feature/case-07-gannon-ingestion` | Real Gannon FRD IAGA-2002 case study | Clean |
| [#7](https://github.com/shibaji7/geopulse/pull/7) | `feature/11-viz-timeseries` | `plot_timeseries` + `TimeSeriesPanel` | Conflict w/ #17 on `viz/__init__.py`; trivial |
| [#9](https://github.com/shibaji7/geopulse/pull/9) | `feature/10-metrics-gic` | `summary_stats` + `exceedance_curve` | `pyproject.toml` conflict w/ #10 |
| [#10](https://github.com/shibaji7/geopulse/pull/10) | `feature/12-devices-harmonics-thd` | FFT `extract_harmonics` + IEEE 519 THD | Conflict w/ #17 on `devices/__init__.py`; trivial |
| [#11](https://github.com/shibaji7/geopulse/pull/11) | `feature/half-cycle-harmonics-stub` | Model-based `half_cycle_harmonics` stub (blocked on W&K curves) | Clean |
| [#12](https://github.com/shibaji7/geopulse/pull/12) | `feature/09-network-railway` | `RailwayNetwork` DSTL + bonded grounds | Conflict w/ #17 on `network/__init__.py`; trivial |
| [#13](https://github.com/shibaji7/geopulse/pull/13) | `feature/13-solver-mna` | `MNASolver` with ideal V-sources | Trivial |
| **#17** (this PR) | `feature/core-additions` | `network.helpers` + `devices.blocker` + `viz.network_map` + `viz.presets` (with Level-2a) | Small `__init__.py` conflicts w/ #7, #10, #12; trivial |

Merge order recommendation: **#6 → #11 → #13 → #7 → #10 → #12 → #9 → #17**. Land #17 last so it rebases once on top of every module-family addition, rather than forcing all seven other PRs to rebase.

---

## Roadmap by work-package

### Core physics

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `sources.synthetic` | ✅ Done | 0.1.0 | Gaussian pulse, step, sinusoid |
| `sources.intermagnet` (IAGA-2002) | ✅ Done | 0.1.0 | XYZF / HDZF / HEZF / DHZF |
| `sources.supermag` (CSV) | ✅ Done | 0.1.0 | Sentinel handling, frame convert |
| `sources.secs` (Spherical Elementary Current Systems) | 🔵 Planned | 0.5.0 | Weygand et al. (2011, *JGR Space Phys.*, `10.1029/2010JA016177`) — practical SECS implementation for the North-American / Greenland magnetometer array; the standard reference for GIC-focused SECS work. Produces **2-D spatial B-field time series** `B(x, y, t)` from a sparse ground array by fitting divergence-free + curl-free elementary current sheets. Feeds `efield.nonuniform`; unlocks realistic non-plane-wave storm fields for continent-scale hindcasts |
| `sources.swmf_mhd` | ⬛ Not touched | v0.5+ | Coupling to global MHD outputs (an alternative 2-D driver — SECS is data-driven, SWMF is physics-driven) |
| `earth.layered_1d` (Wait 1954 recursion) | ✅ Done | 0.1.0 | Analytic half-space match at `rtol=1e-12` |
| `earth.library` (Québec, Kaapvaal, …) | 🟡 Partial | 0.3.0 | Quebec verified; Kaapvaal is a placeholder; USGS physiographic regions planned |
| `earth.structured_2d` (CoastalCorrection2D) | 🟡 Partial | 0.5.0 | Boteler-Pirjola tanh interp shipped; full FD MT solver in 0.5.0 |
| `earth.unstructured_3d` | ⬛ Not touched | v0.9 | ModEM / SimPEG wrapper |
| `efield.planewave` | ✅ Done | 0.1.0 | `E(f) = Z(f) · B(f) / μ₀` in frequency domain |
| `efield.convolution` | 🔵 Planned | 0.3.0 | Time-domain causal convolution |
| `efield.coastal` | 🔵 Planned | 0.5.0 | Standalone wrapper (2-D correction already usable) |
| `efield.nonuniform` | 🔵 Planned | 0.5.0 | Non-plane-wave source-field E. Pairs with `sources.secs`: takes a 2-D `B(x,y,t)` array and computes the spatially-varying `E(x,y,t)` via a lateral-derivative extension of the plane-wave impedance (or a full 2-D MT kernel when `earth.structured_2d` matures). Deferred until SECS lands so we have a real input to test against |

### Network models

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `network.powergrid` (MATPOWER-GMD loader) | ✅ Done | 0.1.0 | Horton EPRI21 reproduces Table VII < 1 % |
| `network.pipeline` (DSTL π-section) | ✅ Done | 0.1.0 | Matches Boteler 1997 analytic to 0.056 % |
| `network.railway` (DSTL + bonded grounds) | 🟡 Partial (PR #12) | 0.2.0 | Cites Boteler 2021, Patterson 2023/2024; unblocks cases #16–#19 |
| `network.cable` (SCUBAS refactor) | ⬛ Not touched | 0.3.0 | Unlocks cases #20 (TAT-8), #21 (SCUBAS Gannon) |
| `network.graph` / `network.thevenin` | ⬛ Not touched | 0.5.0 | Shared topology / Thévenin helpers |
| `network.helpers` (blocker / line-outage / add-tie mutations + per-branch field sampling) | ✅ Done (PR #17) | 0.2.0 | Post-hoc matrix mutations + spatial-field evaluator; used by hypothesis case studies |

### Circuit solvers

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `solver.nam` (Nodal Admittance Matrix / LPm) | ✅ Done | 0.1.0 | Pirjola et al. 2022, symmetric-PD `([Yⁿ]+[Yᵉ])V = Jᵉ` |
| `solver.mna` (Modified Nodal Analysis, ideal V-sources) | 🟡 Partial (PR #13) | 0.2.0 | DC + ideal V-sources; reactive elements deferred |
| `solver.mna` (reactive + companion models) | 🔵 Planned | 0.5.0 | L, C via trapezoidal companion models |
| `solver.pyspice_bridge` (nonlinear transient) | 🔵 Planned | 0.5.0 | Delegates to PySpice; needs `[spice]` extra |
| `solver.sparse` (continent-scale) | ⬛ Not touched | 0.9.0 | Cholesky/CG on sparse Y+Ze |

### Devices

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `devices.transformer` (Mate 2021 thermal) | ✅ Done | 0.1.0 | Top-oil + hot-spot, bilinear/Tustin |
| `devices.harmonics.extract_harmonics` (FFT-from-waveform) | 🟡 Partial (PR #10) | 0.2.0 | FFT + IEEE 519 THD; needs W&K curves for the model-based sibling |
| `devices.half_cycle_harmonics` (model-based, W&K curves) | 🟡 Stub (PR #11) | 0.3.0 | API reserved; blocked on curve transcription |
| `devices.rectifier` (full-wave under DC bias) | 🔵 Planned | 0.5.0 | Diode drop + harmonic content |
| `devices.cp_unit` (cathodic protection response) | 🔵 Planned | 0.5.0 | Rectifier-station shifts from PSP swings; unblocks cases #9, #11, #15 |
| `devices.blocker` (resistive DC-blocker instrumentation) | ✅ Done (PR #17) | 0.2.0 | Reports V / P / dissipated-energy at a blocker neutral; pairs with `network.helpers.apply_resistive_blocker` for the topology change |

### Metrics

Every metric is unit-transparent (Amperes in → Amperes out).

| Function | Status | Version | Notes |
|----------|--------|---------|-------|
| `metrics.gic.summary_stats` (peak / RMS / p50 / p90 / p99 + threshold durations) | 🟡 Partial (PR #9) | 0.2.0 | NERC GIC-90 alignment |
| `metrics.gic.exceedance_curve` (Pulkkinen 2012 style) | 🟡 Partial (PR #9) | 0.2.0 | Log/linear-adaptive grid |
| `metrics.thd.compute_thd` (IEEE 519-2014) | 🟡 Partial (PR #10) | 0.2.0 | DC excluded by default, opt-in flag |
| `metrics.hotspot` (transformer hot-spot proxies) | 🔵 Planned | 0.3.0 | Wrapping `devices.transformer` outputs |
| `metrics.exceedance` (multi-signal / spatial) | 🔵 Planned | 0.5.0 | Per-substation exceedance maps |

### Visualisation

Deliberately opinionated. Every viz helper accepts an `axes=` handle
so callers can compose into a larger figure and returns a `Figure`
for post-tweaks.

| Function | Status | Version | Notes |
|----------|--------|---------|-------|
| `viz.timeseries.plot_timeseries` (stacked B / E / V / I) | 🟡 Partial (PR #7) | 0.2.0 | Elapsed-numeric OR UTC-datetime x-axis; auto legend; ax injection |
| `viz.spectra.plot_spectrum` (log-log Bode-style) | 🔵 Planned | 0.3.0 | For impedance, harmonics, exceedance |
| `viz.network_map.plot_network_map` (substation scatter + line segments) | ✅ Done (PR #17) | 0.2.0 | Matplotlib-only (no cartopy dep); ax injection; log-scale colour; dict-or-array `node_values` |
| `viz.presets` (journal / poster / presentation registry + `apply_preset` + `save_figure`) | ✅ Done (PR #17) | 0.2.0 | 11 named presets; Level-1 warning + opt-in Level-2a mechanical fixes; see table below |

#### Journal / venue figure presets (`viz.presets`)

11-preset registry shipped in PR #17. Two-call API: `apply_preset(name)` before you build the figure sets rcParams globally, then `save_figure(fig, path, preset, ...)` resizes to the venue's exact column width and writes every required format. Backward-compat kwargs let opt-in Level-2a mechanical fixes (rotate ticks, pack legend, scale-up-tiny-text, tight-layout) run on save.

| Preset name(s) | Column width (mm) | Font stack | DPI | Formats | Notes |
|---------------|-------------------|-----------|-----|---------|-------|
| `jgr_1col` / `jgr_2col` | 89 / 183 | serif, 8 pt | 300 | PDF + PNG | AGU house style |
| `sw_1col` / `sw_2col` | 89 / 183 | serif, 8 pt | 300 | PDF + PNG | Space Weather (AGU) |
| `nature_1col` / `nature_2col` | 89 / 183 | sans-serif, 7 pt | 300 | PDF + PNG + EPS | Strict per Nature guide |
| `ieee_1col` / `ieee_2col` | 88 / 181 | serif, 8 pt | 300 | PDF + PNG | PES / TPD conventions |
| `agu_poster` | 254 (10 in), 4:3 | sans-serif, 20 pt | 150 | PNG | Poster panels |
| `presentation` | 254 (10 in), 16:9 | sans-serif, 20 pt | 150 | PNG | Slide-deck content |
| `preprint` | 170 (golden) | serif, 10 pt | 200 | PDF | arXiv / ESSOAr drafts |

Adding a new venue is a one-line dict entry in `PRESETS`. Level-2b heuristics (auto-promote 1col → 2col, auto-color-palette, semantic label placement) are explicitly out of scope — see the "not modelled" list in the module docstring for why.

### I/O

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `io.hdf5` (schema-versioned reader/writer) | ✅ Done | 0.1.0 | Handles `Impedance`, `BFieldTimeSeries`, `SolverResult` |
| `io.iaga2002` (INTERMAGNET) | ✅ Done | 0.1.0 | All four reporting orientations |
| `io.matpower` (MATPOWER-GMD `.m`) | ✅ Done | 0.1.0 | Handles the "row-position vs AC-bus" quirk |
| `io.geojson` (network map export) | 🔵 Planned | 0.3.0 | For `viz.network_map` + external GIS |

### UQ / Config / CLI

| Module | Status | Version | Notes |
|--------|--------|---------|-------|
| `uq.uncertain` (generic `Uncertain[T]`) | ✅ Done | 0.1.0 | Frozen dataclass, HDF5 roundtrip |
| `uq.montecarlo` (MC helpers over `Uncertain[T]`) | 🔵 Planned | 0.5.0 | Vectorised propagation |
| `uq.sensitivity` (Sobol / Morris) | 🔵 Planned | 0.9.0 | For hazard-map studies |
| `uq.distributions` (canonical priors) | 🔵 Planned | 0.5.0 | Log-normal, Weibull, GPD |
| `config` (YAML → dataclass) | ✅ Done | 0.1.0 | Used by CLI |
| `cli` (`geopulse run|validate|info|earth list`) | ✅ Done | 0.1.0 | argparse-based |

---

## Case-study coverage

Full source in [`examples/case_studies/`](examples/case_studies/README.md).

| # | Event / benchmark | Status | Blocked on |
|---|-------------------|--------|-----------|
| 01 | Hydro-Québec 1989 | 🟡 Partial | Verify Québec-7 earth vs Boteler paper |
| 02 | Malmö / Halloween 2003 | 🟡 Partial | Real SMAG magnetogram + `network.railway` links |
| 03 | ESKOM xfmr damage 2003 | 🟡 Partial | Kaapvaal MT model verification |
| 04 | NZ Gannon 2024 | 🟡 Partial | `half_cycle_harmonics` (PR #11 unblock) |
| 05 | Ireland 1-D vs 2-D | 🟡 Partial | Real MT points |
| 06 | Japan low-lat GIC | 🟡 Partial | Real magnetogram + JAPGRID topology |
| 07 | US Gannon / NERC | ✅ Done (PR #6) | — |
| 08 | Brazil Itumbiara | 🟡 Partial | Real MHD-driven storm ensemble |
| 09 | TAPS Alaska pipeline | 🟡 Partial | `devices.cp_unit` |
| 10 | Sweden pipelines | ✅ Done | — |
| 11 | Maritimes NE pipeline | 🟡 Partial | `devices.cp_unit` |
| 12 | NZ gas pipeline | 🟡 Partial | Ingham-2018 magnetogram |
| 13 | Czech oil pipelines | ✅ Done | — |
| 14 | Norway short pipelines | ✅ Done | — |
| 15 | Argentine pipeline | 🟡 Partial | `devices.cp_unit` + `metrics.exceedance` |
| 16 | Sweden rail 1982 | ❌ Not shipped | `network.railway` (PR #12 unblock) + Alm/Lejdström traces |
| 17 | UK rail Patterson 2023 | ❌ Not shipped | `network.railway` |
| 18 | Russian railways Eroshenko 2010 | ❌ Not shipped | `network.railway` |
| 19 | Boteler 2021 rail model | ❌ Not shipped | `network.railway` |
| 20 | TAT-8 cable 1989 | ❌ Not shipped | `network.cable` (v0.3) |
| 21 | SCUBAS Gannon 2024 | ❌ Not shipped | `network.cable` |
| 22 | Gannon multi-infrastructure | 🟡 Partial | `network.cable` + `network.railway` |
| 23 | Horton (2012) benchmark | ✅ Done | — (0.77 % worst-case) |
| 24 | Pulkkinen (2017) benchmark | 🟡 Partial | Pulkkinen test-case topologies |

**Coverage as of today**: 4 ✅ · 14 🟡 · 6 ❌.
**After v0.2.0** (with PR #12 railway landed): 5 ✅ · 14 🟡 · 5 ❌
(cases 16–19 flip from ❌ to 🟡 pending magnetograms).
**After v0.3.0** (with `network.cable`): 5 ✅ · 16 🟡 · 3 ❌.

---

## Meta / release infrastructure

| Task | Status |
|------|--------|
| PyPI Trusted Publishing (OIDC) | ✅ Done |
| Read the Docs build | ✅ Done |
| Codecov integration | ✅ Done (89 %) |
| Pre-commit gate (ruff / mypy / whitespace + pre-push wheel + docs smoke) | ✅ Done |
| GitHub issue templates | 🟡 Partial (PR #8) |
| CONTRIBUTING.md branch strategy + versioning conventions | 🟡 In this branch |
| CODEOWNERS | 🔵 Planned (v0.2) |
| Governance / RFC template | 🔵 Planned (v0.3) |
| Zenodo DOI | 🔵 Planned (bundled with JOSS submission) |
| JOSS paper | 🔵 Planned (v1.0) |

---

## How to contribute

1. Pick an item marked 🔵 **Planned** or 🟡 **Partial** from the
   tables above.
2. Open a **Feature request** issue (link to this file's row) — see
   [`.github/ISSUE_TEMPLATE/feature_request.yml`](.github/ISSUE_TEMPLATE/feature_request.yml)
   (lands with PR #8).
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) — especially the
   [versioning conventions](CONTRIBUTING.md#versioning-conventions)
   for how your change affects the next release tag.
4. Cut a `feature/<work-package>-<short-slug>` branch, follow the
   commit-message conventions, run `pre-commit run --all-files` before
   pushing.
5. PR into `main`. Reference this file's table row so the roadmap can
   be updated in the same PR.

Have a paper whose GIC numbers should be reproduced? Use the
[Benchmark or citation request](.github/ISSUE_TEMPLATE/benchmark_or_citation.yml)
template instead — that gets a dedicated triage lane.
