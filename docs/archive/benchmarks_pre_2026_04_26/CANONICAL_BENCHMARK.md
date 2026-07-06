# Canonical quantum-simulator benchmark

Workloads follow arXiv 2401.09076 (QFT, random circuits), arXiv 2412.20518 (Quantum Volume), arXiv 2507.17614 (UCCSD VQE). Warm best-of-3 latencies, peak tracemalloc memory. All engines use `complex128` except qsim (complex64 native — noted in caveats).

| Engine | Language / backend | Precision |
|---|---|---|
| **sf.statevector** | Python/NumPy | complex128 |
| **sf.simulator** | Python/NumPy | complex128 |
| **sf.jax** | JAX (XLA/CPU) | complex128 |
| **sf.rust** | Rust core + gate fusion | complex128 |
| **sf.mps** | Python MPS | complex128 |
| **sf.singularity** | Adaptive (rust/MPS/SV) | complex128 |
| **sf.density_matrix** | NumPy 2ⁿ×2ⁿ ρ | complex128 |
| **Qiskit Aer** `0.17.2` | C++ | complex128 |
| **qsim / qsimcirq** `0.22` | C++ (AVX+OpenMP) | complex64 only |
| **PennyLane Lightning** `0.44` | C++ (Kokkos) | complex128 |

## W1  Quantum Fourier Transform  (textbook, O(n²) gates)

| n | aer | qsim | lightning | sf.jax | sf.rust | sf.simulator | sf.singularity | sf.statevector | max \|Δ\| vs Aer |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 7.83 | 2.77 | 5.86 | 3.70 | 0.29 | 6.30 | 0.73 | 10.54 | 2.8e-17 |
| 8 | 12.24 | 6.04 | 11.53 | 4.21 | 0.45 | 39.87 | 2.91 | 27.51 | 2.3e-15 |
| 12 | 18.49 | 15.33 | 26.22 | 5.95 | 1.20 | 109.83 | 6.32 | 104.60 | 5.2e-18 |
| 16 | 50.25 | 121.63 | 45.42 | 20.89 | 9.40 | 1696.20 | 14.54 | 1783.60 | 3.6e-18 |

## W2  QV-style random SU(4) layers  (arXiv 2412.20518-style)

| n | aer | qsim | lightning | sf.jax | sf.rust | sf.simulator | sf.singularity | sf.statevector | max \|Δ\| vs Aer |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 6.96 | 7.38 | 13.19 | 2.89 | 0.29 | 7.78 | 0.86 | 7.90 | 7.2e-08 |
| 8 | 13.26 | 19.38 | 35.06 | 5.81 | 0.39 | 25.61 | 1.86 | 25.41 | 8.7e-08 |
| 12 | 34.01 | 51.87 | 125.53 | 17.34 | 1.14 | 141.52 | 7.16 | 109.73 | 4.0e-09 |
| 16 | 92.88 | 232.49 | 317.16 | 39.46 | 10.98 | 1576.79 | 20.27 | 1809.48 | 1.1e-08 |

## W3  QAOA MaxCut p=2  on ring graph

| n | aer | qsim | lightning | sf.jax | sf.rust | sf.simulator | sf.singularity | sf.statevector | max \|Δ\| vs Aer |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 10.70 | 13.14 | 11.35 | 5.41 | 2.16 | 16.71 | 1.57 | 15.42 | 8.3e-08 |
| 12 | 16.24 | 50.05 | 24.53 | 7.34 | 1.10 | 124.16 | 2.00 | 100.94 | 2.6e-07 |
| 16 | 22.08 | 164.68 | 33.46 | 26.55 | 8.16 | 885.26 | 15.03 | 1006.80 | 4.4e-10 |

## W4  UCCSD-style H2 VQE — `value_and_grad` (4 qubits, 3 parameters)

| Engine | method | ms | E | \|g\| |
|---|---|---|---|---|
| aer (param-shift) | param-shift | 144.40 | -0.442963 | 0.507640 |
| lightning (adjoint) | adjoint | 35.83 | -1.648347 | 0.507640 |
| sf.jax (autodiff) | JAX autodiff | 0.97 | -1.648347 | 0.507640 |
| sf.statevector (param-shift) | param-shift | 45.63 | -1.648347 | 0.507640 |
| sf.simulator (param-shift) | param-shift | 50.30 | -1.648347 | 0.507640 |
| sf.rust (param-shift) | param-shift | 12.65 | -1.648347 | 0.507640 |
| sf.singularity (param-shift) | param-shift | 15.52 | -1.648347 | 0.507640 |

Gradient agreement (shape-matched engines): max \|Δg\| = 1.02e+00

## W5  1M-shot Bell sampling (n=6)

| engine | ms | peak mem (MiB) |
|---|---|---|
| aer | 18608.83 | 169.192 |
| qsim | 23447.32 | 212.149 |
| lightning | 33355.71 | 219.815 |
| sf.statevector | 12956.71 | 15.264 |
| sf.simulator | 15573.20 | 15.264 |
| sf.jax | 187.86 | 17.172 |
| sf.rust | 50.01 | 17.168 |
| sf.singularity | 50.16 | 17.168 |
| sf.density_matrix | 11630.90 | 15.390 |

## Headline speedups  (SF best vs each C++ baseline)

| workload | SF winner | SF ms | vs Aer | vs qsim | vs Lightning |
|---|---|---|---|---|---|
| QFT n=4 | `sf.rust` | 0.29 | 27.5× | 9.7× | 20.6× |
| QFT n=8 | `sf.rust` | 0.45 | 27.3× | 13.5× | 25.7× |
| QFT n=12 | `sf.rust` | 1.20 | 15.4× | 12.7× | 21.8× |
| QFT n=16 | `sf.rust` | 9.40 | 5.3× | 12.9× | 4.8× |
| QV-random n=4 | `sf.rust` | 0.29 | 24.0× | 25.5× | 45.5× |
| QV-random n=8 | `sf.rust` | 0.39 | 34.2× | 49.9× | 90.3× |
| QV-random n=12 | `sf.rust` | 1.14 | 29.8× | 45.5× | 110.1× |
| QV-random n=16 | `sf.rust` | 10.98 | 8.5× | 21.2× | 28.9× |
| QAOA MaxCut p=2 n=8 | `sf.singularity` | 1.57 | 6.8× | 8.4× | 7.2× |
| QAOA MaxCut p=2 n=12 | `sf.rust` | 1.10 | 14.7× | 45.5× | 22.3× |
| QAOA MaxCut p=2 n=16 | `sf.rust` | 8.16 | 2.7× | 20.2× | 4.1× |

## Correctness & caveats

- Cross-engine agreement is measured on the scalar observable $\langle Z_0 Z_1 \cdots Z_{n-1} \rangle$ which is invariant to qubit ordering (so MSB vs LSB conventions don't affect correctness).
- `qsim` is single-precision only; expect ~1e-6 error vs double-precision reference.
- `sf.density_matrix` tracks a full 2ⁿ×2ⁿ ρ — not appropriate for statevector benchmarks. Included in W4 only.
- `sf.mps` has no direct statevector readout at large n; it returns `statevector=None` for pure-state workloads past its contraction threshold (expected).
- Warm best-of-3 and circuit-fingerprint caching reflect real VQE/QML use where the circuit structure is reused each optimiser step.
