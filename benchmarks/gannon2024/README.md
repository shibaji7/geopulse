# Gannon storm (10–11 May 2024) — Fredericksburg 1-min IAGA-2002

This directory bundles a compact, IAGA-2002 formatted magnetogram covering
the Gannon storm, suitable for smoke-testing the full GeoPulse ingestion
chain end-to-end without a network fetch.

## Contents

| File | Description |
| ---- | ----------- |
| `frd_20240510_1min.min` | XYZF, 1-minute cadence, 48 hours (2 880 samples) |

## Station metadata

| Field | Value |
| ----- | ----- |
| Station | Fredericksburg (FRD), Virginia, USA |
| Operator | USGS Geomagnetism Program |
| Geodetic latitude | 38.205° N |
| Geodetic longitude | 282.627° E (−77.373° W) |
| Elevation | 69 m |
| Sampling | 60 s (block-mean of a 1-second source) |
| Reporting | XYZF (geographic, nT) |
| Data type | definitive |

## Provenance

The upstream data are USGS/INTERMAGNET **provisional 1-second** definitives
for FRD spanning 2024-05-10 00:00 UTC → 2024-05-11 24:00 UTC.

The 1-second source used to build this file was a per-second CSV export
held under
`SubmarineCableSuperStromAnalysis/data/2024/SMAG/frd20240510psec.sec.csv`
in a companion project directory on the PI's workstation. That CSV was
block-mean downsampled by a factor of 60 into the 1-minute IAGA-2002
file distributed here, chosen because:

- 1-minute cadence is sufficient to resolve the sub-hour Gannon
  variations that drive GIC (peak d*B*/d*t* of order 10² nT s⁻¹ contains
  most of its power well below 0.01 Hz).
- The full 1-second file (~9 MB) is too large to bundle inside the
  wheel; the 1-minute file (~200 KiB) is not.

For higher-cadence work, fetch the original 1-second file directly from
the USGS Geomagnetism Program (https://geomag.usgs.gov/) or INTERMAGNET
(https://intermagnet.github.io/data_download.html).

## Storm signature (as bundled)

| Quantity | Value |
| -------- | ----- |
| Peak Δ*B*ₓ (min-mean domain) | ≈ 1 155 nT |
| Peak Δ*B*ᵧ (min-mean domain) | ≈ 528 nT |
| 1-second-source peak \|d*B*/d*t*\| | 112 nT s⁻¹ at 2024-05-11 17:45 UTC |
| Storm peak window | UT evening 2024-05-11 (Gannon main phase) |

## Attribution when re-used

If you use this file in a downstream paper, cite the underlying USGS
Fredericksburg observatory data (definitive) alongside GeoPulse.
