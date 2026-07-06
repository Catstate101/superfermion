# SuperFermion vs Qiskit Benchpress Report

**Generated:** 2026-06-18T03:23:40.561413
**Platform:** win32
**Python:** 3.13.3

## Available Backends

| Backend | Type | Status |
|---------|------|--------|
| simulator | StatevectorBackend | OK |
| rust | RustBackend | OK |
| jax | JAXBackend | OK |
| mps | MPSSimulatorBackend | OK |
| density_matrix | DensityMatrixBackend | OK |
| stabilizer | StabilizerBackend | OK |
| cuda | CUSimulatorBackend | OK |
| cupy | CupyBackend | OK |
| jax_mps | JAXMPSBackend | OK |
| cuda_mps | CupyMPSBackend | OK |
| supremacy | SupremacyBackend | OK |
| singularity | SingularityBackend | OK |
| qiskit | AerSimulator | OK |

## Circuit Construction Latency

| Test | SF (ms) | Qiskit (ms) | Speedup |
|------|---------|-------------|---------|
| GHZ_10 | 0.065 | 0.181 | 2.80x |
| GHZ_50 | 0.070 | 0.343 | 4.91x |
| GHZ_100 | 0.131 | 0.562 | 4.27x |
| QFT_10 | 0.158 | 230.165 | 1454.90x |
| QFT_20 | 0.430 | 6.670 | 15.50x |
| QV_10_10 | 2.606 | 2.410 | 0.93x |
| QV_10_50 | 3.890 | 8.617 | 2.22x |
| Clifford_10 | 6.724 | 42.913 | 6.38x |
| Clifford_20 | 3.901 | 145.704 | 37.35x |

## Simulation Latency

| Test | simulator (ms) | rust (ms) | jax (ms) | mps (ms) | density_matrix (ms) | qiskit (ms) |
|------|---------|---------|---------|---------|---------|---------|
| Bell | 16.25 | 11.06 | 1886.23 | 91.12 | 5.05 | 2035.13 |
| GHZ_5 | 1.22 | 2.61 | 558.17 | 226.78 | 1.98 | 180.62 |
| GHZ_10 | 2.05 | 1.37 | 668.07 | 387.03 | 615.75 | 141.95 |
| GHZ_15 | 26.45 | 11.96 | 656.76 | 547.66 | - | 157.59 |
| GHZ_20 | 540.86 | 192.78 | 940.30 | 805.69 | - | 183.20 |
| QFT_4 | 3.21 | 4.24 | 580.30 | 161.53 | 2.64 | 127.88 |
| QFT_8 | 6.12 | 12.98 | 787.79 | 328.67 | 76.60 | 129.33 |
| QFT_12 | 19.11 | 31.18 | 1225.97 | 476.86 | 54944.97 | 166.36 |
| QV_10_10 | 47.30 | 40.21 | 1870.15 | 660.56 | 3665.27 | 182.69 |
| QV_10_20 | 85.73 | 59.38 | 3312.51 | 870.95 | 6863.71 | 185.86 |

## Statevector Accuracy (Fidelity vs Qiskit)

| Test | simulator | rust | jax | mps | density_matrix |
|------|---------|---------|---------|---------|---------|
| GHZ_4 | 1.000000 | 1.000000 | 1.000000 | - | - |
| GHZ_8 | 1.000000 | 1.000000 | 1.000000 | - | - |
| GHZ_12 | 1.000000 | 1.000000 | 1.000000 | - | - |
| QFT_4 | 1.000000 | 1.000000 | 1.000000 | - | - |
| QFT_8 | 1.000000 | 1.000000 | 1.000000 | - | - |
| QFT_12 | 1.000000 | 1.000000 | 1.000000 | - | - |
| Clifford_4 | 0.125000 | 0.125000 | 0.125000 | - | - |
| Clifford_8 | 0.000000 | 0.000000 | 0.000000 | - | - |

## Clifford Simulation (Stabilizer Advantage)

| Test | Stabilizer (ms) | Qiskit Stabilizer (ms) | Speedup |
|------|-----------------|------------------------|---------|
| Clifford_20 | 2.77 | 1636.23 | 591.49x |
| Clifford_50 | 7.23 | 0.00 | - |
| Clifford_100 | 18.83 | 5242.42 | 278.39x |
| Clifford_200 | 9.08 | 0.00 | - |

## Summary

- **SF Backends Tested:** 12
- **Qiskit Available:** Yes
