# Changelog

All notable changes to GeoPulse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Minimum numpy raised from `>=1.24` to `>=2.0`** (`pyproject.toml`,
  `environment.yml`, `environment-minimal.yml`). Fixes issue #21:
  `ResistiveBlocker.inject_gic` calls `numpy.trapezoid`, which was
  introduced in numpy 2.0. The previous floor advertised compatibility
  with numpy 1.24–1.26 but the device model crashed at runtime on
  those versions. Bumping the floor removes the false advertisement and
  matches the "no back-compat shims when we can just change the code"
  project preference.

### Fixed
- `network.helpers.evaluate_field_at_branch_midpoints` no longer
  produces `NaN` for every branch when *any* node in the network has
  `NaN` coordinates (issue #23). The local-projection origin now
  averages over only the finite-coord nodes, and branches touching a
  `NaN`-coord endpoint sample as `(0, 0)` — the right physics for a
  zero-length degenerate branch. Three regression tests cover the
  poisoned-mean, degenerate-branch, and all-NaN-network cases.

## [0.2.0a1] - 2026-07-30

Opens the v0.2 alpha train. First feature drop is a coordinated set
of four reusable library modules (PR #20 / `feature/core-additions`)
that unblock the upcoming `paper/shi-hypotheses` paper reproduction
and every future case study that touches network mutations, mitigation
devices, network maps, or venue-specific figure output.

### Added

- `geopulse.network.helpers` — post-hoc network-topology mutation
  helpers: `apply_resistive_blocker`, `open_line`, `add_tie`, plus a
  spatial-field sampler `evaluate_field_at_branch_midpoints` that
  evaluates a caller-supplied `f(x_km, y_km) → (Ex, Ey)` at each
  branch midpoint and returns the per-branch arrays that
  `PowerGridNetwork.compute_thevenin_voltages` accepts. All matrix
  helpers copy their input (scenario sweeps correct-by-construction).
- `geopulse.devices.blocker.ResistiveBlocker` — passive DeviceModel
  that instruments a resistive DC-blocker at a transformer neutral.
  Reports blocker voltage / peak-voltage / peak-power / dissipated-
  energy given a neutral GIC time series. Pairs with
  `network.helpers.apply_resistive_blocker` for the two-step
  compose-mitigate-and-instrument workflow.
- `geopulse.viz.network_map.plot_network_map` — matplotlib-only
  substation-scatter + line-segment network plot. No cartopy
  dependency (equirectangular is visually indistinguishable at ≤
  few hundred km extents). Handles `dict[node_id, value]` or
  per-node array, log-scale colour, axes injection for compose-into-
  larger-figure, savepath.
- `geopulse.viz.presets` — venue-figure preset registry (11 named
  presets: JGR / SW / Nature / IEEE 1col & 2col, AGU poster,
  presentation, preprint) with two-function API:
  `apply_preset(name)` sets matplotlib rcParams before you build
  the figure; `save_figure(fig, path, preset, ...)` resizes to the
  exact column width and writes every venue-required format.
  Level-1 readability warning always on; opt-in Level-2a mechanical
  fixes (rotate ticks on overlap, pack legend on overflow, scale-up
  tiny text, tight-layout) via new kwargs on `save_figure`.

### Tests

Full suite grows from 117 → 204 (+87 net across four new module test
files). All guardrails green (ruff D/N/C901/ANN, mypy, interrogate
≥ 85 %, pylint duplicate-code, check-no-version-drift).

### Docs

`docs/api.rst` gains a new Visualisation section and per-module rows
under Networks and Devices. `ROADMAP.md` flips four rows to Done and
refreshes the venue-preset sub-section to match the shipped registry.

## [0.1.0a3] - 2026-07-27

Renames the solver from "LPM" to "NAM" (=LPm) per collaborator feedback
(D. Boteler). The Nodal Admittance Matrix method is the long-established
label in the power industry (Boteler 2014); the "Lehtinen-Pirjola
modified" (LPm) formulation of Pirjola, Boteler, Tuck & Marsal (2022)
is mathematically identical. This release aligns GeoPulse's public
naming with both — one class, two labels, one code path.

### Changed (**breaking — no compat shim; PyPI users must update imports**)
- `geopulse.solver.lpm` → `geopulse.solver.nam`.
- `LPMSolver` → `NAMSolver(method_label="NAM" | "LPm")`. Both labels
  execute identical numerics; the choice only affects the string
  written into `SolverResult.metadata["solver_method"]` for report
  attribution.
- `solve_lpm` → `solve_nam`.
- `tests/integration/test_lpm_benchmark.py` → `test_nam_benchmark.py`.
- Entry point `[project.entry-points."geopulse.solvers"]` key renamed
  from `lpm` to `nam`.
- README architecture diagram updated: `LPM` → `NAM (=LPm)`, driving
  equation shown as `([Yⁿ] + [Yᵉ])V = Jᵉ`.

### Added
- `SolverResult.metadata: dict` field. Populated by `NAMSolver` with
  `{"solver_method": "NAM" | "LPm"}` so downstream reports can attribute
  the solve unambiguously.
- Full `Pirjola et al. (2022)` citation in the solver module docstring,
  identifying LPm and NAM as identical methods.

### Note on the equivalence
The current implementation always solved `([Yⁿ] + [Yᵉ])V = Jᵉ` — i.e.
the NAM/LPm formulation — never the original `(1 + Yⁿ·Zᵉ)⁻¹` form. The
0.1.0a0–a2 docstring called that "mathematically equivalent to" the
original LP method, which was correct but understated. This release
makes the label match what the math has always done.

## [0.1.0a2] - 2026-07-26

Metadata-only release. Publishes updated package authorship to PyPI.

### Changed
- `pyproject.toml` `authors` list now includes Xueling Shi and
  Michael Hartinger alongside Shibaji Chakraborty and David Boteler.
  The prose citation string in README.md and the docs landing page
  had already been updated to the same list in `7c84841`; this bump
  is what lands the change on PyPI package metadata (which is
  per-version immutable).

## [0.1.0a1] - 2026-07-26

Infrastructure release — no user-facing API changes. First release
published to real PyPI via OIDC Trusted Publishing.

### Added
- **Docs site** at https://geopulse.readthedocs.io/ — 107-page Sphinx
  build (landing + getting-started walkthrough + 54 auto-generated API
  pages). `.readthedocs.yaml` + full `docs/conf.py` with autosummary,
  myst-parser, intersphinx to numpy/scipy/h5py/matplotlib.
- **OIDC Trusted Publishing workflow**: `.github/workflows/release.yml`
  builds sdist+wheel on every `v*` tag push and publishes to PyPI
  without long-lived API tokens. Manual `workflow_dispatch` publishes
  to TestPyPI for rehearsal.
- **Pre-commit gate**: 8-hook pipeline (ruff lint + format, whitespace,
  EOF, YAML/TOML, large-file guard, mypy) + a `pre-commit` stage cleanup
  hook and two `pre-push` smoke checks (wheel build, Sphinx docs build).
- `[docs]` optional dependency group so RTD installs only what the build
  needs; `build` and `twine` added to `[dev]`.

### Changed
- CONTRIBUTING.md now documents GitHub flow (`main` + `feature/*`) as
  the current branch strategy, with gitflow described as the planned
  model once contributor count justifies it. PR checklist points to
  `pre-commit run --all-files` instead of enumerating ruff commands.
- WGS84 curvature-radius helpers now `float(...)`-cast their returns to
  satisfy strict mypy.
- `Layered1D.compute_impedance` return type narrowed from `Impedance`
  to `ScalarImpedance` (covariant return; unblocks `.Z_values` access
  in `CoastalCorrection2D`).

### Fixed
- 9 mypy errors flagged by CI (and newer numpy type stubs): missing
  `float()` casts in `geo.py`, wide return type in `layered_1d.py`,
  unbounded-TypeVar `# type: ignore` annotations in `uq/uncertain.py`,
  `np.ndarray` cast in `pipeline.py`, `import-untyped` on yaml.
- 12 empty `__init__.py` / `.gitkeep` files were missing terminal
  newlines (fixed by the new `end-of-file-fixer` pre-commit hook).
- `ruff-pre-commit` pin bumped from `v0.4.0` to `v0.16.0` to match the
  local dev env; the old pin didn't recognise `UP045` in `ruff.toml`.

## [0.1.0a0] - 2026-07-25

First alpha release. Base engine is pip-installable, imports cleanly in a
fresh virtualenv, ships with 100 passing tests, and reproduces two
published benchmarks (Horton 2012 Table VII and Boteler 1997 DSTL) at
sub-1 % relative error.

### Added
- **Sources**: `SyntheticSource` (Gaussian, step, sinusoid), plus
  `INTERMAGNETSource` (IAGA-2002) and `SuperMAGSource` (CSV) with
  XYZF/HDZF frame conversion and sentinel handling.
- **Earth**: `Layered1D` (Wait 1954 recursion, matches uniform half-space
  analytic at `rtol=1e-12`), model registry (`get_model("quebec_7layer")`
  etc.), `CoastalCorrection2D` (Boteler-Pirjola-style TE/TM interpolation
  → `TensorImpedance`).
- **Impedance**: full `Impedance` ABC + `ScalarImpedance` (with HDF5
  roundtrip) + `TensorImpedance` (2×2 tensor per frequency, HDF5, apply,
  reduces to ScalarImpedance in the anti-diagonal limit).
- **Network**: `PowerGridNetwork` with MATPOWER-GMD (`epri21.m`) loader
  handling the "row-position vs AC-bus" quirk; `PipelineNetwork` with
  DSTL equivalent-π discretisation.
- **Solver**: `NAMSolver` with delta-winding-safe active-subspace solve;
  `Solver.solve()` ABC signature accepts the `ConductorNetwork`.
- **Devices**: `TransformerModel` port of Mate 2021 top-oil + hot-spot
  (bilinear/Tustin discretisation, `ThermalParams`). `DeviceResponse`
  extended with `top_oil_C` / `hotspot_C` fields.
- **Geo**: WGS84 projection helpers (`meridian_radius_m`,
  `prime_vertical_radius_m`, `latlon_to_local_xy_wgs84_m`) — networks now
  use per-segment WGS84 line integrals matching Horton (2012) Appendix.
- **UQ**: `Uncertain[T]` generic + `propagate_uncertainty` MC.
- **I/O**: schema-versioned HDF5 reader/writer, MATPOWER-GMD parser,
  IAGA-2002 parser.
- **CLI**: `geopulse run|validate|info|earth list`.
- **Benchmarks**: Horton 2012 EPRI21 test case + `expected_gic.csv` from
  paper Table VII; passing test at `rtol=1e-2`.
- **Examples**: `01_first_gic.py` (Gaussian pulse smoke test),
  `04_pipeline_dstl.py` (DSTL vs analytic profile),
  `05_transformer_hotspot.py` (Mate 2021 hot-spot reproduction).
- **Packaging**: `py.typed` marker; wheel + sdist build clean via
  `python -m build`; installs in a fresh venv.
- **Docs**: Sphinx scaffold with logo + favicon.

### Deferred (stubs raise `NotImplementedYetError` with WP reference)
- `CableNetwork` (WP2), `RailwayNetwork` (WP4).
- `MNASolver` (WP3), `PySpiceSolver` (WP3).
- `RectifierModel`, `CathodicProtectionModel`, harmonics (WP3).
- `KernelImpedance` + `Unstructured3D` (WP3).
- `Structured2D` full FD MT solver (WP-future — `CoastalCorrection2D`
  is the pragmatic stand-in).
- `metrics/*` summary statistics (WP2/3/4).
- `viz/*` plotting helpers (WP1/2).

## [0.0.1] - 2026-07-24

### Added
- Initial scaffold.
- Repository layout: `src/geopulse/`, `tests/`, `examples/`, `benchmarks/`, `docs/`.
- Core constants (`MU_0`, `EPSILON_0`, `R_EARTH_M`, unit conversions).
- Type aliases (`FloatArray`, `ComplexArray`, `LatLon`).
- Exception hierarchy (`GeoPulseError`, `ConfigurationError`, `DataError`,
  `ShapeMismatchError`, `ConvergenceError`, `NotImplementedYetError`).
- Abstract base classes for `BFieldSource`, `EarthModel`, `ConductorNetwork`,
  `Solver`, `DeviceModel`.
- `Impedance` ABC with `ScalarImpedance` fully implemented; `TensorImpedance`
  and `KernelImpedance` stubs.
- `Uncertain[T]` generic wrapper and `propagate_uncertainty()` Monte Carlo helper.
- Schema-versioned HDF5 I/O.
- YAML config loader (dataclass-based, no pydantic).
- argparse-based CLI: `run`, `validate`, `info`, `earth list`.
- Tooling: `pyproject.toml`, `ruff.toml`, `mypy.ini`, `.pre-commit-config.yaml`.
- Conda environments (`environment.yml`, `environment-minimal.yml`).
- Apache-2.0 license, contributing guide, README.
