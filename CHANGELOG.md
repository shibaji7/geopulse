# Changelog

All notable changes to GeoPulse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Solver**: `LPMSolver` with delta-winding-safe active-subspace solve;
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
