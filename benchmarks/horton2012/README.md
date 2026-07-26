# Horton et al. (2012) EPRI 21-bus GIC benchmark

`epri21.m` is the LANL PowerModelsGMD.jl test case
(`test/data/matpower/epri21.m`) transcribing the Horton et al. (2012)
21-bus, 8-substation EHV test system (345/500 kV, centered on TN/GA/AL).
It contains real grounding conductances, DC-line resistances, and
substation lat/lon.

## Citation

Horton, R., Boteler, D., Overbye, T. J., Pirjola, R., & Dugan, R. C. (2012).
*A test case for the calculation of geomagnetically induced currents*.
IEEE Transactions on Power Delivery, 27(4), 2368-2373.

## Source

The `.m` file is used verbatim from
[lanl-ansi/PowerModelsGMD.jl](https://github.com/lanl-ansi/PowerModelsGMD.jl)
(BSD-3-Clause). See that repository for its full license.

## Benchmark data

`expected_gic.csv` transcribes **Table VII** of Horton et al. (2012) — total
GIC into each substation's ground grid, in Amperes, for uniform 1 V/km
northward and 1 V/km eastward geoelectric fields.

`tests/integration/test_lpm_benchmark.py` compares GeoPulse's LPM solve
against these values at **`rtol = 1e-2`** (1 %). The residual comes from
the spherical-vs-WGS84 projection; the paper uses WGS84 (Appendix eqns
A3/A7), GeoPulse currently uses a spherical Earth of mean radius 6371 km.

## Parent-index convention (important)

Inside `mpc.gmd_bus`, the `parent_index` column means:

* AC bus number for `dc_bus*` nodes
* **substation number** (1–8) for `dc_sub*` nodes

`PowerGridNetwork` handles this by inferring `dc_sub*` node coordinates
from a connected `dc_bus*` neighbour via a zero-length transformer branch.
