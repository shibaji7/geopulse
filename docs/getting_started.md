# Getting Started

This page walks through **installing GeoPulse and running the smoke-test
chain end-to-end** — synthetic magnetic-field pulse → 1-D layered Earth →
plane-wave geoelectric field. About five minutes.

## Installation

GeoPulse ships as a normal Python package. Pick the path that matches
what you're doing:

**1. Core only** — six dependencies, no plotting extras, no dev tools.

```bash
pip install geopulse
```

**2. With optional data tools** (xarray, netCDF4, pandas):

```bash
pip install geopulse[data]
```

**3. Full development environment** (recommended if you're going to
contribute or run the full test suite):

```bash
git clone https://github.com/shibaji7/geopulse.git
cd geopulse
conda env create -f environment.yml
conda activate geopulse-dev
pre-commit install
```

**4. Pre-built wheel from the latest release**:

```bash
pip install https://github.com/shibaji7/geopulse/releases/download/v0.1.0a0/geopulse-0.1.0a0-py3-none-any.whl
```

Available optional extras: `[data]`, `[viz]`, `[earth3d]`, `[spice]`,
`[service]`, `[dev]`, `[all]`.

## Verify the install

```bash
geopulse info
```

Should print the version, install location, and which optional extras
are present.

## First GIC — the smoke-test chain

The complete pipeline in ~15 lines of user code
([`examples/01_first_gic.py`](https://github.com/shibaji7/geopulse/blob/main/examples/01_first_gic.py)):

```python
from geopulse.sources.synthetic import SyntheticSource
from geopulse.earth.library import get_model
from geopulse.efield.planewave import compute_efield_planewave
from scipy.fft import irfft
import matplotlib.pyplot as plt

# 1. Gaussian B-field pulse: 500 nT peak, 1-hour window at 1 Hz sampling.
source = SyntheticSource("gaussian_pulse", amplitude_nT=500.0, sigma_s=300.0)
b = source.load(start_s=0.0, end_s=3600.0, dt_s=1.0)

# 2. Load a 1-D Earth model and compute the surface impedance.
earth = get_model("quebec_7layer")
freqs, Bx_f, By_f = source.to_frequency_domain(b)
imp = earth.compute_impedance(freqs)

# 3. Plane-wave geoelectric field.
Ex_f, Ey_f = compute_efield_planewave(freqs, Bx_f, By_f, imp)
Ey_t = irfft(Ey_f, n=len(b.time_s))

# 4. Plot.
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(b.time_s / 60.0, b.bx_T * 1e9);  ax[0].set_ylabel("Bx (nT)")
ax[1].plot(b.time_s / 60.0, Ey_t * 1e3);    ax[1].set_ylabel("Ey (mV/m)")
plt.savefig("first_gic.png")
```

The `Ey` peak sits in the mV/m range for a 500 nT storm-like pulse over
a Québec-like layered Earth — consistent with Boteler (2014) Table 1.

## The three architectural spines

Once the smoke test runs, three abstractions do most of the work in the
rest of the library. Skim these in the API reference:

**{class}`~geopulse.earth.impedance.Impedance`** — the polymorphism spine.
Every Earth model returns an `Impedance` object; every E-field module
calls `impedance.apply(Bx_f, By_f)`. The 1-D case
({class}`~geopulse.earth.impedance.ScalarImpedance`) uses a complex
scalar per frequency; the 2-D case
({class}`~geopulse.earth.impedance.TensorImpedance`) uses a 2×2
tensor; a planned 3-D case uses a spatial kernel. Downstream code
never branches on tier.

**{class}`~geopulse.network.base.ConductorNetwork`** — the
infrastructure-agnostic spine. Cables, power grids, pipelines, and
railways all implement the same ABC. The
{class}`~geopulse.solver.lpm.LPMSolver` sees only nodes, branches,
admittances, and Thévenin voltages — it doesn't care what kind of
network it's solving.

**{class}`~geopulse.uq.uncertain.Uncertain`** — the uncertainty spine.
Wrap any value in `Uncertain(nominal=..., distribution="gaussian",
params={"std": ...})` and pass it through the pipeline; the
{func}`~geopulse.uq.uncertain.propagate_uncertainty` helper runs
Monte-Carlo propagation on demand. Deterministic values carry zero
overhead.

## Where to look next

- {doc}`api` — the full auto-generated API reference (~50 pages of
  docstrings).
- [`examples/`](https://github.com/shibaji7/geopulse/tree/main/examples)
  in the repo — runnable scripts for the smoke test, pipeline DSTL, and
  transformer hot-spot.
- [`benchmarks/horton2012/`](https://github.com/shibaji7/geopulse/tree/main/benchmarks/horton2012)
  — the LANL PowerModelsGMD.jl EPRI21 test case + `expected_gic.csv`
  transcribed from Horton (2012) Table VII.
- {doc}`contributing` — pre-commit setup, branch strategy, PR checklist.
