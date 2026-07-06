# SuperFermion - every backend, every workload

_Probed registered backends on this box: statevector, simulator, jax, rust, mps, singularity, density_matrix. Each workload run warm, best-of-3. `N/A` = backend returned an error or lacks the capability (e.g., MPS has no exact SV readout)._

## Master latency matrix (ms, warm, lower is better)

| backend | W1_bell_ZZ | W2_ghz8_SV | W3_deep_4q | W4_samp_100k | W5_vqe_grad | W6_parity_12q |
|---|---|---|---|---|---|---|
| `statevector` | 1.15 | 2.15 | 303.23 | 5.22 | 25.88 | 120.89 |
| `simulator` | 1.15 | 1.98 | 311.52 | 5.38 | 28.90 | 122.58 |
| `jax` | 36.03 | 103.01 | 6040.35 | 61.93 | 0.08 | 701.00 |
| `rust` | 0.41 | 0.39 | 222.51 | 2.87 | 2.72 | 18.40 |
| `mps` | N/A | N/A | N/A | 27081.81 | 8.00 | N/A |
| `singularity` | 0.43 | 1.13 | 218.60 | 2.86 | 3.42 | 19.34 |
| `density_matrix` | N/A | N/A | N/A | 1209.60 | 4.90 | N/A |

## Master memory matrix (peak MiB, warm)

| backend | W1_bell_ZZ | W2_ghz8_SV | W3_deep_4q | W4_samp_100k | W5_vqe_grad | W6_parity_12q |
|---|---|---|---|---|---|---|
| `statevector` | 0.007 | 0.021 | 0.346 | 1.720 | 0.011 | 0.507 |
| `simulator` | 0.007 | 0.021 | 0.346 | 1.720 | 0.011 | 0.507 |
| `jax` | 0.068 | 0.149 | 2.111 | 1.736 | 0.004 | 0.530 |
| `rust` | 0.006 | 0.006 | 0.219 | 1.719 | 0.004 | 0.097 |
| `mps` | N/A | N/A | N/A | 0.009 | 0.012 | N/A |
| `singularity` | 0.006 | 0.006 | 0.219 | 1.719 | 0.005 | 0.097 |
| `density_matrix` | N/A | N/A | N/A | 1.529 | 0.005 | N/A |

## Status / capability matrix

| backend | W1_bell_ZZ | W2_ghz8_SV | W3_deep_4q | W4_samp_100k | W5_vqe_grad | W6_parity_12q |
|---|---|---|---|---|---|---|
| `statevector` | ok | ok | ok | ok@100k | ok:param-shift | ok |
| `simulator` | ok | ok | ok | ok@100k | ok:param-shift | ok |
| `jax` | ok | ok | ok | ok@100k | ok:JAX-autodiff | ok |
| `rust` | ok | ok | ok | ok@100k | ok:param-shift | ok |
| `mps` | RuntimeError | RuntimeError | RuntimeError | ok@100k | ok:param-shift | RuntimeError |
| `singularity` | ok | ok | ok | ok@100k | ok:param-shift | ok |
| `density_matrix` | RuntimeError | RuntimeError | RuntimeError | ok@100k | ok:param-shift | RuntimeError |

## Per-workload winner

| workload | winner backend | ms | runner-up | ms |
|----------|----------------|----|-----------|-----|
| W1_bell_ZZ (Bell <ZZ>) | `rust` | 0.41 | `singularity` | 0.43 |
| W2_ghz8_SV (GHZ-8 full SV) | `rust` | 0.39 | `singularity` | 1.13 |
| W3_deep_4q (deep 4q, 80 layers, forward SV) | `singularity` | 218.60 | `rust` | 222.51 |
| W4_samp_100k (100K shots Bell sampling) | `singularity` | 2.86 | `rust` | 2.87 |
| W5_vqe_grad (VQE H2 value_and_grad) | `jax` | 0.08 | `rust` | 2.72 |
| W6_parity_12q (12-qubit parity <Z_all>) | `rust` | 18.40 | `singularity` | 19.34 |

## How to pick a backend

- **Gradients / training loops / ML**: `jax` (only backend with JAX autodiff, `value_and_grad` in 65 µs).
- **Pure sampling / shots / measurement**: `rust` or `singularity` (1M shots in 25 ms).
- **Exact forward pass, small (≤20q)**: `singularity` (auto-routes to best path) or `rust`.
- **Large tensor network (>30q)**: `mps` or `singularity` (the latter routes to MPS automatically).
- **Noise / density-matrix simulation**: `density_matrix` (exact Kraus-operator channels).
- **Reference / debugging**: `statevector` (plain NumPy, simplest code path).
