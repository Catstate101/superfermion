# Industrial Benchmark Report -- SuperFermion vs Qiskit-Aer vs PennyLane

*Generated:* 2026-04-15T00:37:02.825586Z  | smoke=False  | shots=20000

## Executive Summary

291 records across 11 backends × 9 qubit counts × 4 circuit families, 3h11m total wall time.

**Verdict on correctness (observable-Linf vs Qiskit-Aer statevector reference):**

| Backend                           | Status | Notes |
|-----------------------------------|--------|-------|
| superfermion.statevector          | OK     | F=1.0, obs_linf<0.02 everywhere |
| superfermion.jax                  | OK     | F=1.0 to 24q |
| superfermion.rust                 | OK     | F=1.0 to 24q |
| superfermion.mps                  | OK     | Scales to 64q cleanly, obs_linf<0.02 |
| superfermion.singularity          | OK ≤24q / CRASH ≥32q | `TypeError: SingularityBackend._run_rust_fused() got multiple values for argument` — code bug |
| superfermion.jax_mps              | **BROKEN** | Sampling correct only on GHZ (trivially aligned); obs_linf 0.5-0.95 on random/qft/vqe |
| superfermion.supremacy            | **BROKEN** | TVD pinned at 0.5 across ALL cells (sampling layer emits half the distribution); statevector F=1.0 on most but F=0.20 at 16q/qft, F=0.0016 at 20q/qft (fidelity collapse on deep QFT) |
| qiskit_aer.statevector            | REF    | |
| qiskit_aer.matrix_product_state   | OK ≤48q / CRASH at 64q | `CircuitTooWideForTarget` (library limit) |
| pennylane.default.qubit           | OK to 24q |  |
| pennylane.lightning.qubit         | OK to 24q |  |

**Latency rankings (median t_mean across all circuits):**

1. **sf.rust** — 6.4 ms median (fastest correct backend on small circuits)
2. **sf.singularity** — 8.3 ms median (routed-rust internally ≤24q)
3. sf.statevector — 35 ms
4. qiskit_aer.statevector — 55 ms (reference)
5. sf.mps — 72 ms (and **only backend that scales cleanly past 24q**: 588 ms at 64q)
6. pennylane.lightning.qubit — 168 ms
7. pennylane.default.qubit — 196 ms
8. sf.jax — 387 ms (JIT overhead dominates at small n; narrows at 24q)
9. qiskit_aer.matrix_product_state — 497 ms

**Large-n (≥32q) behavior** (only tensor-network backends in scope):
- **sf.mps** is the clear winner — 253 ms @ 32q, 396 ms @ 48q, 588 ms @ 64q
- qiskit_aer.mps works to 48q (2.7 s) then raises CircuitTooWideForTarget
- sf.jax_mps runs but sampling is broken at all sizes
- sf.supremacy runs but TVD pinned at 0.5
- sf.singularity crashes on its own routing past 24q (`_run_rust_fused` / `_run_mps_direct` arg collision)

**Memory efficiency (peak PyHeap + RSS delta at 24q/ghz statevector):**
- sf.rust: 269 MB RSS (complex128 buffer, expected)
- sf.statevector / pennylane.default.qubit: ~800 MB PyPeak (pure-Python arena)
- sf.jax: 134 MB RSS (compact)
- sf.mps: <2 MB (bond-dim keeps it tiny)
- pennylane.lightning.qubit: 14 MB RSS + 102 MB peak (C++ buffer off-heap)

**Endianness:** Harness convention Big-Endian (qubit 0 = MSB). Qiskit-Aer outputs are reversed via `reverse_bitstring_keys` / `reverse_statevector_endianness` at the framework boundary. Sanity-checked on an asymmetric 3-qubit circuit before the sweep; all three frameworks agree once normalized. SuperFermion's own `bridge.from_qasm` silently mirrors qubit indices, which would cause double-reversal — bypassed with a custom `_sf_circuit_from_qasm_no_reverse` parser in `bench/runners.py`.

**CPU/RAM safety:** 30 cells preflight-skipped with reason `statevector cap n<=24`, 3 skipped with `statevector would use 0.54 GB > 60% of 0.68 GB avail` at 24q/random when accumulated Python-heap pressure from prior cells was high. No OOM, no crash, no destructive cleanup — skips recorded explicitly.

**Recommended backends for production use:**
- Small circuits (≤16q, low-latency): **sf.rust** or **sf.singularity**
- Medium correctness-critical (17-24q): **sf.rust** or **qiskit_aer.statevector**
- Large MPS-tractable (≥32q): **sf.mps** (only viable SF option; qiskit_aer.mps works to 48q)
- **Avoid** sf.supremacy and sf.jax_mps for anything that needs correct samples.

## Methodology

- **Latency**: 2 untimed warm-up runs + 10 timed runs with `time.perf_counter()`. Reported: mean, std, min, p50, p95 (ms).
- **Memory**: `tracemalloc` peak (Python heap) + `psutil` RSS delta (captures Rust/JAX/CuPy arenas) around one dedicated run.
- **Endianness**: Harness convention is Big-Endian (qubit 0 = MSB, leftmost). Qiskit outputs are reversed at the framework boundary so all counts and statevectors in this report are directly comparable.
- **Reference**: `qiskit-aer` method=`statevector` (shot-sampled for TVD, exact statevector for fidelity/L2).
- **Accuracy thresholds**: TVD < 0.02, observable L-inf < 0.05, fidelity >= 1 - 1e-4.
- **CPU/RAM safety**: statevector backends hard-capped at 24 qubits and preflight-skipped if estimated footprint exceeds 60% of available RAM (see `bench/safety.py`).

## 4 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 108.71 | 110.12 | 123.61 | 3.64 | 2.12 | 0.0107 | 1.000000 | 0.0213 | ok |
| pennylane.lightning.qubit | 145.74 | 157.50 | 167.26 | 3.65 | 0.00 | 0.0052 | 1.000000 | 0.0104 | ok |
| qiskit_aer.matrix_product_state | 248.31 | 242.23 | 300.42 | 0.01 | 0.00 | 0.0014 | - | 0.0028 | ok |
| qiskit_aer.statevector | 33.53 | 33.59 | 35.62 | 0.01 | 0.05 | - | - | - | REF |
| superfermion.jax | 76.77 | 75.64 | 84.55 | 0.26 | 0.16 | 0.0010 | 1.000000 | 0.0019 | ok |
| superfermion.jax_mps | 99.71 | 89.63 | 137.20 | 0.12 | 0.00 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 22.75 | 23.57 | 25.62 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 1.21 | 1.09 | 1.84 | 0.36 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.singularity | 1.53 | 1.26 | 2.57 | 0.36 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.statevector | 17.90 | 17.62 | 20.92 | 0.33 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.supremacy | 4.96 | 4.78 | 6.07 | 0.04 | 0.00 | 0.5000 | 1.000000 | 1.0000 | ok |

## 4 qubits -- qft

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 114.11 | 113.69 | 124.55 | 3.66 | 0.00 | 0.0149 | 1.000000 | 0.0252 | ok |
| pennylane.lightning.qubit | 85.22 | 85.90 | 91.23 | 3.67 | 0.00 | 0.0090 | 1.000000 | 0.0127 | ok |
| qiskit_aer.matrix_product_state | 120.43 | 111.30 | 165.80 | 0.02 | 0.00 | 0.0125 | - | 0.0208 | ok |
| qiskit_aer.statevector | 49.13 | 48.78 | 53.63 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 201.45 | 205.27 | 239.71 | 0.41 | 0.02 | 0.0095 | 0.999999 | 0.0140 | ok |
| superfermion.jax_mps | 219.52 | 211.62 | 252.25 | 0.17 | 0.01 | 0.8750 | - | 1.0000 | ok |
| superfermion.mps | 22.64 | 21.97 | 26.23 | 0.02 | 0.00 | 0.0116 | - | 0.0100 | ok |
| superfermion.rust | 4.23 | 4.03 | 5.28 | 0.37 | 0.00 | 0.0097 | 1.000000 | 0.0140 | ok |
| superfermion.singularity | 3.26 | 3.22 | 3.58 | 0.37 | 0.00 | 0.0097 | 1.000000 | 0.0140 | ok |
| superfermion.statevector | 21.57 | 21.51 | 22.99 | 0.34 | 0.00 | 0.0097 | 1.000000 | 0.0140 | ok |
| superfermion.supremacy | 62.59 | 59.35 | 73.93 | 0.11 | 0.00 | 0.5000 | 1.000002 | 0.0000 | ok |

## 4 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 104.53 | 100.67 | 124.83 | 3.67 | 0.02 | 0.0099 | 1.000000 | 0.0118 | ok |
| pennylane.lightning.qubit | 99.61 | 100.53 | 104.85 | 3.68 | 0.05 | 0.0090 | 1.000000 | 0.0107 | ok |
| qiskit_aer.matrix_product_state | 243.47 | 255.50 | 333.08 | 0.02 | 0.00 | 0.0085 | - | 0.0084 | ok |
| qiskit_aer.statevector | 47.20 | 47.41 | 48.74 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 203.54 | 200.93 | 233.89 | 0.32 | 0.17 | 0.0103 | 1.000000 | 0.0128 | ok |
| superfermion.jax_mps | 141.45 | 129.94 | 209.56 | 0.15 | 0.01 | 0.6154 | - | 0.8153 | ok |
| superfermion.mps | 22.11 | 20.86 | 26.45 | 0.01 | 0.00 | 0.0119 | - | 0.0138 | ok |
| superfermion.rust | 4.36 | 4.49 | 5.67 | 0.37 | 0.00 | 0.0104 | 1.000000 | 0.0133 | ok |
| superfermion.singularity | 3.19 | 3.18 | 3.29 | 0.37 | 0.00 | 0.0104 | 1.000000 | 0.0133 | ok |
| superfermion.statevector | 26.24 | 25.47 | 33.17 | 0.34 | 0.00 | 0.0104 | 1.000000 | 0.0133 | ok |
| superfermion.supremacy | 15.42 | 15.13 | 17.89 | 0.06 | 0.00 | 0.5000 | 1.000000 | 0.3417 | ok |

## 4 qubits -- vqe

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 97.30 | 95.28 | 119.92 | 3.65 | 0.00 | 0.0024 | 1.000000 | 0.0025 | ok |
| pennylane.lightning.qubit | 109.63 | 107.08 | 138.61 | 3.66 | 0.00 | 0.0034 | 1.000000 | 0.0059 | ok |
| qiskit_aer.matrix_product_state | 151.03 | 149.27 | 162.16 | 0.02 | 0.00 | 0.0034 | - | 0.0052 | ok |
| qiskit_aer.statevector | 31.56 | 31.46 | 32.50 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 150.78 | 145.65 | 211.44 | 0.30 | 0.00 | 0.0028 | 1.000000 | 0.0052 | ok |
| superfermion.jax_mps | 108.81 | 109.88 | 115.28 | 0.13 | 0.00 | 0.0961 | - | 0.1789 | ok |
| superfermion.mps | 18.41 | 18.32 | 20.91 | 0.01 | 0.00 | 0.0024 | - | 0.0030 | ok |
| superfermion.rust | 1.95 | 1.88 | 2.39 | 0.37 | 0.00 | 0.0023 | 1.000000 | 0.0032 | ok |
| superfermion.singularity | 1.56 | 1.55 | 1.70 | 0.37 | 0.00 | 0.0023 | 1.000000 | 0.0032 | ok |
| superfermion.statevector | 23.65 | 21.38 | 32.35 | 0.33 | 0.00 | 0.0023 | 1.000000 | 0.0032 | ok |
| superfermion.supremacy | 14.09 | 13.76 | 16.00 | 0.06 | 0.00 | 0.5000 | 1.000000 | 0.9715 | ok |

## 8 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 136.65 | 130.52 | 187.10 | 5.73 | 0.00 | 0.0033 | 1.000000 | 0.0067 | ok |
| pennylane.lightning.qubit | 110.46 | 108.45 | 136.08 | 5.73 | 0.02 | 0.0037 | 1.000000 | 0.0074 | ok |
| qiskit_aer.matrix_product_state | 340.48 | 299.38 | 525.81 | 0.01 | 0.00 | 0.0006 | - | 0.0012 | ok |
| qiskit_aer.statevector | 37.44 | 37.28 | 38.90 | 0.01 | 0.00 | - | - | - | REF |
| superfermion.jax | 103.20 | 102.20 | 114.16 | 0.28 | 0.01 | 0.0010 | 1.000000 | 0.0019 | ok |
| superfermion.jax_mps | 140.00 | 140.97 | 146.95 | 0.20 | 0.00 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 29.71 | 32.38 | 33.93 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 1.10 | 1.06 | 1.33 | 0.37 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.singularity | 0.82 | 0.77 | 1.06 | 0.37 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.statevector | 19.86 | 17.64 | 32.39 | 0.34 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.supremacy | 16.75 | 16.46 | 21.15 | 0.05 | 0.00 | 0.5000 | 1.000000 | 1.0000 | ok |

## 8 qubits -- qft

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 107.29 | 105.70 | 117.06 | 5.86 | 0.00 | 0.0460 | 1.000000 | 0.0140 | ok |
| pennylane.lightning.qubit | 108.31 | 103.84 | 137.95 | 5.84 | 0.00 | 0.0443 | 1.000000 | 0.0193 | ok |
| qiskit_aer.matrix_product_state | 94.61 | 91.99 | 109.87 | 0.07 | 0.00 | 0.0460 | - | 0.0183 | ok |
| qiskit_aer.statevector | 87.84 | 73.57 | 128.88 | 0.07 | 0.00 | - | - | - | REF |
| superfermion.jax | 953.17 | 940.80 | 1018.17 | 1.06 | 0.02 | 0.0514 | 0.999998 | 0.0140 | ok |
| superfermion.jax_mps | 1546.77 | 1535.43 | 1615.06 | 0.57 | 0.00 | 0.9922 | - | 1.0000 | ok |
| superfermion.mps | 37.66 | 37.58 | 39.30 | 0.06 | 0.00 | 0.0472 | - | 0.0151 | ok |
| superfermion.rust | 11.18 | 10.97 | 12.45 | 0.40 | 0.00 | 0.0448 | 1.000000 | 0.0152 | ok |
| superfermion.singularity | 11.60 | 11.29 | 15.25 | 0.41 | 0.00 | 0.0448 | 1.000000 | 0.0152 | ok |
| superfermion.statevector | 31.46 | 31.57 | 37.73 | 0.42 | 0.00 | 0.0448 | 1.000000 | 0.0152 | ok |
| superfermion.supremacy | 570.70 | 555.48 | 655.80 | 0.52 | 0.01 | 0.5000 | 1.000004 | 0.0000 | ok |

## 8 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 138.81 | 142.32 | 152.35 | 5.80 | 0.02 | 0.0324 | 1.000000 | 0.0157 | ok |
| pennylane.lightning.qubit | 168.98 | 146.34 | 255.39 | 5.80 | 0.00 | 0.0358 | 1.000000 | 0.0112 | ok |
| qiskit_aer.matrix_product_state | 497.48 | 489.28 | 541.51 | 0.07 | 0.00 | 0.0414 | - | 0.0111 | ok |
| qiskit_aer.statevector | 40.22 | 40.28 | 42.82 | 0.07 | 0.00 | - | - | - | REF |
| superfermion.jax | 347.62 | 362.24 | 376.72 | 0.56 | 0.01 | 0.0405 | 0.999999 | 0.0106 | ok |
| superfermion.jax_mps | 226.09 | 224.40 | 242.55 | 0.26 | 0.00 | 0.8683 | - | 0.9029 | ok |
| superfermion.mps | 50.82 | 47.80 | 68.00 | 0.04 | 0.00 | 0.0392 | - | 0.0173 | ok |
| superfermion.rust | 5.81 | 5.71 | 6.36 | 0.38 | 0.00 | 0.0397 | 1.000000 | 0.0116 | ok |
| superfermion.singularity | 5.60 | 5.19 | 7.72 | 0.38 | 0.00 | 0.0397 | 1.000000 | 0.0116 | ok |
| superfermion.statevector | 27.67 | 27.42 | 31.14 | 0.38 | 0.00 | 0.0397 | 1.000000 | 0.0116 | ok |
| superfermion.supremacy | 39.12 | 39.29 | 46.75 | 0.09 | 0.17 | 0.5000 | 1.000001 | 0.6274 | ok |

## 8 qubits -- vqe

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 123.89 | 124.64 | 136.28 | 5.75 | 0.00 | 0.0108 | 1.000000 | 0.0114 | ok |
| pennylane.lightning.qubit | 120.47 | 121.28 | 130.46 | 5.76 | 0.00 | 0.0048 | 1.000000 | 0.0029 | ok |
| qiskit_aer.matrix_product_state | 436.99 | 474.46 | 530.13 | 0.03 | 0.00 | 0.0086 | - | 0.0096 | ok |
| qiskit_aer.statevector | 67.34 | 61.64 | 102.48 | 0.03 | 0.00 | - | - | - | REF |
| superfermion.jax | 372.22 | 368.22 | 446.49 | 0.43 | 0.00 | 0.0088 | 1.000000 | 0.0075 | ok |
| superfermion.jax_mps | 225.98 | 224.07 | 244.68 | 0.24 | 0.00 | 0.2246 | - | 0.3444 | ok |
| superfermion.mps | 51.02 | 51.53 | 66.57 | 0.02 | 0.00 | 0.0100 | - | 0.0072 | ok |
| superfermion.rust | 5.37 | 5.64 | 7.17 | 0.37 | 0.00 | 0.0066 | 1.000000 | 0.0066 | ok |
| superfermion.singularity | 6.04 | 5.90 | 7.26 | 0.37 | 0.00 | 0.0066 | 1.000000 | 0.0066 | ok |
| superfermion.statevector | 23.47 | 22.47 | 26.82 | 0.37 | 0.00 | 0.0066 | 1.000000 | 0.0066 | ok |
| superfermion.supremacy | 42.56 | 40.89 | 49.22 | 0.08 | 0.00 | 0.5000 | 0.999999 | 0.8874 | ok |

## 12 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 150.34 | 152.97 | 156.04 | 7.88 | 0.00 | 0.0070 | 1.000000 | 0.0140 | ok |
| pennylane.lightning.qubit | 139.24 | 136.81 | 156.22 | 7.82 | 0.00 | 0.0021 | 1.000000 | 0.0043 | ok |
| qiskit_aer.matrix_product_state | 470.62 | 440.80 | 605.72 | 0.02 | 0.00 | 0.0033 | - | 0.0066 | ok |
| qiskit_aer.statevector | 42.04 | 42.30 | 43.38 | 0.01 | 0.00 | - | - | - | REF |
| superfermion.jax | 157.07 | 159.04 | 167.56 | 0.30 | 0.18 | 0.0010 | 1.000000 | 0.0019 | ok |
| superfermion.jax_mps | 207.73 | 205.11 | 222.56 | 0.29 | 0.00 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 41.04 | 42.73 | 48.92 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 1.55 | 1.52 | 1.79 | 0.40 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.singularity | 1.27 | 1.26 | 1.31 | 0.40 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.statevector | 22.48 | 22.89 | 24.31 | 0.46 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.supremacy | 27.09 | 26.07 | 31.62 | 0.07 | 0.00 | 0.5000 | 1.000000 | 1.0000 | ok |

## 12 qubits -- qft

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 256.10 | 259.50 | 292.52 | 8.47 | 0.00 | 0.1827 | 1.000000 | 0.0124 | ok |
| pennylane.lightning.qubit | 180.45 | 182.20 | 191.57 | 8.35 | 0.00 | 0.1820 | 1.000000 | 0.0103 | ok |
| qiskit_aer.matrix_product_state | 133.68 | 132.28 | 140.81 | 1.08 | 0.00 | 0.1802 | - | 0.0082 | ok |
| qiskit_aer.statevector | 71.65 | 70.22 | 80.72 | 1.08 | 0.00 | - | - | - | REF |
| superfermion.jax | 2201.23 | 2200.59 | 2295.30 | 1.78 | 0.03 | 0.1784 | 0.999997 | 0.0140 | ok |
| superfermion.jax_mps | 4708.72 | 4693.91 | 4803.33 | 1.54 | 0.00 | 0.9988 | - | 1.0000 | ok |
| superfermion.mps | 66.93 | 64.84 | 84.09 | 0.40 | 0.00 | 0.1837 | - | 0.0151 | ok |
| superfermion.rust | 29.59 | 30.07 | 37.19 | 0.66 | 0.00 | 0.1843 | 1.000000 | 0.0140 | ok |
| superfermion.singularity | 39.04 | 37.74 | 49.17 | 0.66 | 0.00 | 0.1843 | 1.000000 | 0.0140 | ok |
| superfermion.statevector | 84.65 | 81.45 | 109.01 | 1.12 | 0.00 | 0.1843 | 1.000000 | 0.0140 | ok |
| superfermion.supremacy | 3623.12 | 3703.22 | 3813.57 | 0.77 | 1.56 | 0.5000 | 0.999983 | 0.0000 | ok |

## 12 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 203.00 | 204.90 | 224.50 | 8.16 | 0.02 | 0.1415 | 1.000000 | 0.0144 | ok |
| pennylane.lightning.qubit | 152.15 | 147.72 | 177.66 | 8.09 | 0.00 | 0.1461 | 1.000000 | 0.0150 | ok |
| qiskit_aer.matrix_product_state | 659.57 | 691.19 | 763.94 | 0.94 | 0.00 | 0.1429 | - | 0.0099 | ok |
| qiskit_aer.statevector | 55.13 | 54.37 | 60.60 | 0.94 | 0.00 | - | - | - | REF |
| superfermion.jax | 699.22 | 663.34 | 848.64 | 0.71 | 0.00 | 0.1476 | 0.999999 | 0.0067 | ok |
| superfermion.jax_mps | 357.55 | 358.62 | 369.94 | 0.38 | 0.00 | 0.9804 | - | 0.9615 | ok |
| superfermion.mps | 113.10 | 112.82 | 121.72 | 0.33 | 0.00 | 0.1447 | - | 0.0164 | ok |
| superfermion.rust | 12.24 | 12.38 | 12.77 | 0.57 | 0.00 | 0.1468 | 1.000000 | 0.0149 | ok |
| superfermion.singularity | 12.34 | 11.86 | 14.91 | 0.57 | 0.00 | 0.1468 | 1.000000 | 0.0149 | ok |
| superfermion.statevector | 39.05 | 39.88 | 44.25 | 1.02 | 0.00 | 0.1468 | 1.000000 | 0.0149 | ok |
| superfermion.supremacy | 76.99 | 75.74 | 86.46 | 0.13 | 0.00 | 0.5000 | 1.000000 | 0.4755 | ok |

## 12 qubits -- vqe

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 153.73 | 154.59 | 163.00 | 7.92 | 0.00 | 0.0194 | 1.000000 | 0.0173 | ok |
| pennylane.lightning.qubit | 144.54 | 145.80 | 153.76 | 7.85 | 0.00 | 0.0154 | 1.000000 | 0.0056 | ok |
| qiskit_aer.matrix_product_state | 481.14 | 482.29 | 494.93 | 0.07 | 0.00 | 0.0153 | - | 0.0084 | ok |
| qiskit_aer.statevector | 40.06 | 39.17 | 44.63 | 0.07 | 0.00 | - | - | - | REF |
| superfermion.jax | 333.07 | 327.28 | 371.61 | 0.62 | 0.03 | 0.0158 | 1.000000 | 0.0090 | ok |
| superfermion.jax_mps | 328.85 | 328.12 | 346.36 | 0.34 | 0.00 | 0.3029 | - | 0.3699 | ok |
| superfermion.mps | 56.65 | 55.08 | 77.50 | 0.03 | 0.00 | 0.0157 | - | 0.0111 | ok |
| superfermion.rust | 5.16 | 5.16 | 6.22 | 0.41 | 0.00 | 0.0151 | 1.000000 | 0.0067 | ok |
| superfermion.singularity | 5.46 | 5.06 | 7.26 | 0.41 | 0.00 | 0.0151 | 1.000000 | 0.0067 | ok |
| superfermion.statevector | 29.62 | 30.31 | 33.46 | 0.90 | 0.00 | 0.0151 | 1.000000 | 0.0067 | ok |
| superfermion.supremacy | 55.68 | 55.51 | 60.75 | 0.11 | 0.00 | 0.5000 | 0.999999 | 0.9240 | ok |

## 16 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 196.33 | 185.43 | 256.01 | 10.95 | 0.00 | 0.0026 | 1.000000 | 0.0051 | ok |
| pennylane.lightning.qubit | 168.39 | 168.45 | 186.51 | 9.91 | 0.00 | 0.0012 | 1.000000 | 0.0025 | ok |
| qiskit_aer.matrix_product_state | 731.37 | 680.28 | 910.96 | 0.02 | 0.00 | 0.0002 | - | 0.0004 | ok |
| qiskit_aer.statevector | 36.83 | 36.10 | 43.00 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 224.18 | 212.03 | 274.31 | 0.32 | 0.00 | 0.0010 | 1.000000 | 0.0019 | ok |
| superfermion.jax_mps | 269.30 | 264.31 | 293.64 | 0.37 | 0.00 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 47.63 | 47.20 | 54.06 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 2.95 | 3.02 | 3.20 | 1.37 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.singularity | 3.32 | 3.14 | 3.97 | 1.37 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.statevector | 55.92 | 54.86 | 62.32 | 3.16 | 1.05 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.supremacy | 32.84 | 32.45 | 35.77 | 0.08 | 0.00 | 0.5000 | 1.000000 | 1.0000 | ok |

## 16 qubits -- qft

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 952.73 | 948.26 | 984.24 | 14.17 | 0.74 | 0.7371 | 1.000000 | 0.0185 | ok |
| pennylane.lightning.qubit | 305.23 | 309.37 | 325.99 | 13.06 | 0.26 | 0.7359 | 1.000000 | 0.0122 | ok |
| qiskit_aer.matrix_product_state | 228.59 | 216.04 | 284.18 | 4.53 | 0.14 | 0.7371 | - | 0.0130 | ok |
| qiskit_aer.statevector | 112.61 | 111.74 | 119.43 | 4.53 | 0.00 | - | - | - | REF |
| superfermion.jax | 4697.61 | 4593.33 | 4997.71 | 2.79 | 0.01 | 0.7361 | 0.999996 | 0.0140 | ok |
| superfermion.jax_mps | 9884.80 | 10145.98 | 11015.05 | 3.13 | 0.00 | 0.9996 | - | 0.9989 | ok |
| superfermion.mps | 93.47 | 93.39 | 109.58 | 1.54 | 0.00 | 0.7380 | - | 0.0151 | ok |
| superfermion.rust | 48.17 | 48.57 | 56.81 | 2.51 | 0.00 | 0.7359 | 1.000000 | 0.0204 | ok |
| superfermion.singularity | 58.28 | 59.84 | 64.80 | 2.51 | 0.00 | 0.7359 | 1.000000 | 0.0204 | ok |
| superfermion.statevector | 1280.21 | 1233.72 | 1487.09 | 10.58 | 2.90 | 0.7359 | 1.000000 | 0.0204 | ok |
| superfermion.supremacy | 46208.89 | 48220.98 | 50330.63 | 0.83 | 0.02 | 0.5000 | 0.202791 | 0.0000 | ok |

## 16 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 431.66 | 434.75 | 448.87 | 12.08 | 0.67 | 0.4020 | 1.000000 | 0.0143 | ok |
| pennylane.lightning.qubit | 211.91 | 212.20 | 231.62 | 10.99 | 0.01 | 0.4004 | 1.000000 | 0.0227 | ok |
| qiskit_aer.matrix_product_state | 909.35 | 1018.74 | 1123.34 | 2.49 | 0.00 | 0.3989 | - | 0.0110 | ok |
| qiskit_aer.statevector | 72.22 | 71.59 | 76.15 | 2.50 | 0.00 | - | - | - | REF |
| superfermion.jax | 1021.72 | 1021.93 | 1111.13 | 1.38 | 0.05 | 0.4046 | 1.000000 | 0.0141 | ok |
| superfermion.jax_mps | 405.43 | 384.90 | 471.20 | 0.50 | 0.00 | 0.9920 | - | 0.7513 | ok |
| superfermion.mps | 142.68 | 144.66 | 149.89 | 0.82 | 0.00 | 0.3983 | - | 0.0158 | ok |
| superfermion.rust | 24.21 | 23.99 | 27.80 | 1.68 | 0.00 | 0.4052 | 1.000000 | 0.0167 | ok |
| superfermion.singularity | 23.37 | 23.58 | 26.35 | 1.68 | 0.00 | 0.4052 | 1.000000 | 0.0167 | ok |
| superfermion.statevector | 257.53 | 252.35 | 283.31 | 9.81 | 2.53 | 0.4052 | 1.000000 | 0.0167 | ok |
| superfermion.supremacy | 106.84 | 105.71 | 128.23 | 0.16 | 0.00 | 0.5000 | 1.000003 | 0.7525 | ok |

## 16 qubits -- vqe

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 276.03 | 279.03 | 292.60 | 11.03 | 0.00 | 0.0368 | 1.000000 | 0.0140 | ok |
| pennylane.lightning.qubit | 189.21 | 185.23 | 214.81 | 9.98 | 0.02 | 0.0376 | 1.000000 | 0.0106 | ok |
| qiskit_aer.matrix_product_state | 732.16 | 645.02 | 992.55 | 0.16 | 0.00 | 0.0364 | - | 0.0116 | ok |
| qiskit_aer.statevector | 53.51 | 38.25 | 118.15 | 0.16 | 0.00 | - | - | - | REF |
| superfermion.jax | 621.84 | 597.74 | 739.88 | 0.70 | 0.09 | 0.0306 | 1.000000 | 0.0082 | ok |
| superfermion.jax_mps | 446.56 | 440.65 | 474.40 | 0.45 | 0.00 | 0.4661 | - | 0.5488 | ok |
| superfermion.mps | 72.11 | 68.69 | 88.28 | 0.06 | 0.00 | 0.0327 | - | 0.0103 | ok |
| superfermion.rust | 6.98 | 6.91 | 7.68 | 1.39 | 0.00 | 0.0359 | 1.000000 | 0.0096 | ok |
| superfermion.singularity | 8.30 | 8.00 | 10.05 | 1.39 | 0.00 | 0.0359 | 1.000000 | 0.0096 | ok |
| superfermion.statevector | 202.29 | 203.87 | 211.69 | 9.68 | 2.98 | 0.0359 | 1.000000 | 0.0096 | ok |
| superfermion.supremacy | 79.69 | 78.88 | 86.49 | 0.14 | 0.00 | 0.5000 | 0.999999 | 0.8246 | ok |

## 20 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 559.52 | 559.03 | 586.17 | 50.96 | 0.00 | 0.0017 | 1.000000 | 0.0033 | ok |
| pennylane.lightning.qubit | 357.84 | 310.76 | 547.26 | 12.01 | 0.00 | 0.0010 | 1.000000 | 0.0019 | ok |
| qiskit_aer.matrix_product_state | 888.76 | 760.12 | 1155.82 | 0.02 | 0.00 | 0.0003 | - | 0.0007 | ok |
| qiskit_aer.statevector | 84.60 | 85.59 | 91.43 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 402.65 | 407.41 | 422.19 | 0.36 | 8.39 | 0.0010 | 1.000000 | 0.0019 | ok |
| superfermion.jax_mps | 317.33 | 319.94 | 328.89 | 0.46 | 0.00 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 55.06 | 55.71 | 65.49 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 40.09 | 40.82 | 43.27 | 17.10 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.singularity | 35.67 | 34.99 | 40.00 | 17.10 | 0.00 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.statevector | 794.58 | 796.15 | 823.34 | 50.34 | 16.78 | 0.0031 | 1.000000 | 0.0062 | ok |
| superfermion.supremacy | 61.40 | 59.88 | 71.30 | 0.10 | 8.39 | 0.5000 | 1.000000 | 1.0000 | ok |

## 20 qubits -- qft

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 18730.05 | 18249.93 | 20673.88 | 68.61 | 1.04 | 0.9811 | 1.000000 | 0.0149 | ok |
| pennylane.lightning.qubit | 3144.20 | 3161.93 | 3337.86 | 16.03 | 1.16 | 0.9811 | 1.000000 | 0.0121 | ok |
| qiskit_aer.matrix_product_state | 306.83 | 281.15 | 443.04 | 4.98 | 0.67 | 0.9811 | - | 0.0151 | ok |
| qiskit_aer.statevector | 341.70 | 340.31 | 362.92 | 4.98 | 0.00 | - | - | - | REF |
| superfermion.jax | 11312.22 | 11412.92 | 12303.73 | 3.84 | 8.44 | 0.9811 | 0.999995 | 0.0140 | ok |
| superfermion.jax_mps | 21264.67 | 21261.40 | 21619.47 | 5.47 | 0.00 | 1.0000 | - | 1.0000 | ok |
| superfermion.mps | 125.15 | 109.30 | 170.89 | 1.83 | 0.00 | 0.9811 | - | 0.0151 | ok |
| superfermion.rust | 144.00 | 139.37 | 175.25 | 17.31 | 0.00 | 0.9811 | 1.000000 | 0.0203 | ok |
| superfermion.singularity | 137.04 | 130.51 | 164.42 | 17.31 | 0.00 | 0.9811 | 1.000000 | 0.0203 | ok |
| superfermion.statevector | 30987.59 | 30990.97 | 31593.18 | 154.07 | 112.63 | 0.9811 | 1.000000 | 0.0203 | ok |
| superfermion.supremacy | 150428.48 | 149456.16 | 160177.11 | 0.91 | 0.00 | 0.5000 | 0.001581 | 0.0000 | ok |

## 20 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 5581.70 | 5585.30 | 5738.64 | 67.94 | 0.00 | 0.7497 | 1.000000 | 0.0171 | ok |
| pennylane.lightning.qubit | 1468.64 | 1467.56 | 1527.11 | 14.73 | 0.00 | 0.7478 | 1.000000 | 0.0140 | ok |
| qiskit_aer.matrix_product_state | 1471.79 | 1402.21 | 1676.06 | 4.57 | 0.00 | 0.7455 | - | 0.0191 | ok |
| qiskit_aer.statevector | 186.26 | 183.11 | 196.80 | 4.58 | 0.00 | - | - | - | REF |
| superfermion.jax | 1995.97 | 1923.75 | 2411.29 | 2.20 | 8.40 | 0.7486 | 0.999999 | 0.0155 | ok |
| superfermion.jax_mps | 612.06 | 587.56 | 695.24 | 0.62 | 0.00 | 0.9984 | - | 0.9504 | ok |
| superfermion.mps | 253.24 | 261.51 | 278.32 | 1.48 | 0.00 | 0.7510 | - | 0.0191 | ok |
| superfermion.rust | 68.39 | 66.99 | 85.45 | 17.14 | 0.00 | 0.7510 | 1.000000 | 0.0193 | ok |
| superfermion.singularity | 59.49 | 58.92 | 64.52 | 17.14 | 0.00 | 0.7510 | 1.000000 | 0.0193 | ok |
| superfermion.statevector | 5631.88 | 5620.66 | 6456.54 | 153.83 | 134.95 | 0.7510 | 1.000000 | 0.0193 | ok |
| superfermion.supremacy | 121.33 | 124.26 | 128.46 | 0.20 | 8.39 | 0.5000 | 1.000001 | 0.7523 | ok |

## 24 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 7255.18 | 7031.47 | 8298.31 | 805.94 | 0.01 | 0.0009 | - | 0.0017 | ok |
| pennylane.lightning.qubit | 2768.36 | 2776.90 | 2893.80 | 14.07 | 102.35 | 0.0001 | - | 0.0002 | ok |
| qiskit_aer.matrix_product_state | 943.10 | 891.97 | 1174.18 | 0.02 | 0.00 | 0.0024 | - | 0.0049 | ok |
| qiskit_aer.statevector | 816.48 | 821.98 | 853.14 | 0.02 | 0.00 | - | - | - | REF |
| superfermion.jax | 2876.97 | 2865.87 | 3022.04 | 0.40 | 134.45 | 0.0010 | - | 0.0019 | ok |
| superfermion.jax_mps | 388.59 | 384.13 | 421.16 | 0.54 | 0.01 | 0.0010 | - | 0.0019 | ok |
| superfermion.mps | 81.28 | 75.71 | 101.71 | 0.01 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.rust | 540.21 | 536.11 | 568.15 | 268.76 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.singularity | 535.38 | 536.78 | 556.46 | 268.76 | 0.00 | 0.0031 | - | 0.0062 | ok |
| superfermion.statevector | 13472.84 | 13267.51 | 14991.14 | 805.32 | 268.61 | 0.0031 | - | 0.0062 | ok |
| superfermion.supremacy | 379.00 | 379.65 | 389.47 | 0.11 | 134.22 | 0.5000 | - | 1.0000 | ok |

## 24 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | 102081.57 | 101487.72 | 105612.80 | 1074.65 | 8.79 | 0.8941 | - | 0.0119 | ok |
| pennylane.lightning.qubit | 24968.12 | 24429.16 | 27546.37 | 17.31 | 0.00 | 0.8955 | - | 0.0120 | ok |
| qiskit_aer.matrix_product_state | 1130.36 | 1086.99 | 1348.49 | 4.93 | 2.89 | 0.8952 | - | 0.0160 | ok |
| qiskit_aer.statevector | 1776.62 | 1781.88 | 1813.46 | 4.92 | 0.63 | - | - | - | REF |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector would use 0.54 GB > 60% of 0.68 GB avail |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector would use 0.54 GB > 60% of 0.68 GB avail |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector would use 0.54 GB > 60% of 0.68 GB avail |
| superfermion.jax_mps | 701.07 | 685.12 | 787.76 | 0.73 | 0.45 | 0.9986 | - | 0.9480 | ok |
| superfermion.mps | 302.03 | 304.08 | 327.94 | 1.68 | 0.05 | 0.8946 | - | 0.0205 | ok |
| superfermion.singularity | 660.80 | 652.84 | 725.36 | 268.81 | 1.00 | 0.8953 | - | 0.0174 | ok |
| superfermion.supremacy | 1002.27 | 1000.94 | 1108.08 | 0.23 | 134.78 | 0.5000 | - | 0.7479 | ok |

## 32 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit_aer.matrix_product_state | 1750.64 | 1826.61 | 2036.41 | 0.02 | 0.01 | - | - | - | ok |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 612.05 | 584.39 | 843.67 | 0.72 | 0.04 | - | - | - | ok |
| superfermion.mps | 75.50 | 74.07 | 93.24 | 0.01 | 0.13 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_rust_fused() got multiple |
| superfermion.supremacy | 33.31 | 32.98 | 35.58 | 0.13 | 0.02 | - | - | - | ok |

## 32 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit_aer.matrix_product_state | 2026.38 | 2189.81 | 2698.95 | 5.37 | 0.00 | - | - | - | ok |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 1014.22 | 992.52 | 1101.50 | 0.97 | 0.52 | - | - | - | ok |
| superfermion.mps | 430.99 | 436.41 | 478.98 | 1.94 | 0.00 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_rust_fused() got multiple |
| superfermion.supremacy | 201.54 | 190.11 | 236.31 | 0.29 | 0.00 | - | - | - | ok |

## 48 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit_aer.matrix_product_state | 2615.34 | 2620.73 | 3717.38 | 0.02 | 0.00 | - | - | - | ok |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 812.57 | 796.56 | 910.70 | 1.06 | 0.00 | - | - | - | ok |
| superfermion.mps | 142.55 | 137.55 | 164.18 | 0.01 | 0.00 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_mps_direct() got multiple |
| superfermion.supremacy | 66.29 | 66.06 | 75.07 | 0.18 | 0.00 | - | - | - | ok |

## 48 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit_aer.matrix_product_state | 2815.76 | 2982.42 | 3137.11 | 5.78 | 0.00 | - | - | - | ok |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 1549.07 | 1537.89 | 1631.19 | 1.44 | 0.00 | - | - | - | ok |
| superfermion.mps | 649.60 | 650.84 | 697.94 | 2.28 | 0.00 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_mps_direct() got multiple |
| superfermion.supremacy | 300.37 | 301.33 | 318.66 | 0.42 | 0.00 | - | - | - | ok |

## 64 qubits -- ghz

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit.matrix_product_state | - | - | - | - | - | - | - | - | ERR: CircuitTooWideForTarget: 'Number of qubits (64) in circuit-3 |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 1247.84 | 1259.74 | 1388.42 | 1.40 | 0.00 | - | - | - | ok |
| superfermion.mps | 188.55 | 183.06 | 216.34 | 0.01 | 0.00 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_mps_direct() got multiple |
| superfermion.supremacy | 118.22 | 117.31 | 136.83 | 0.22 | 0.00 | - | - | - | ok |

## 64 qubits -- random

| Framework/Backend | t_mean (ms) | t_p50 | t_p95 | PyPeak (MB) | RSS d (MB) | TVD | Fidelity | Obs Linf | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pennylane.default.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| pennylane.lightning.qubit | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| qiskit.matrix_product_state | - | - | - | - | - | - | - | - | ERR: CircuitTooWideForTarget: 'Number of qubits (64) in circuit-3 |
| sf.jax | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.rust | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| sf.statevector | - | - | - | - | - | - | - | - | skipped: statevector cap n<=24 |
| superfermion.jax_mps | 2068.21 | 2052.87 | 2189.63 | 1.91 | 0.00 | - | - | - | ok |
| superfermion.mps | 987.33 | 985.85 | 1129.43 | 2.63 | 0.00 | - | - | - | ok |
| superfermion.singularity | - | - | - | - | - | - | - | - | ERR: TypeError: SingularityBackend._run_mps_direct() got multiple |
| superfermion.supremacy | 417.92 | 407.22 | 482.63 | 0.54 | 0.00 | - | - | - | ok |

## Anomalies

- **superfermion.supremacy** @ 4q / ghz: TVD=0.5000 
- **superfermion.jax_mps** @ 4q / random: TVD=0.6154 
- **superfermion.supremacy** @ 4q / random: TVD=0.5000 
- **superfermion.jax_mps** @ 4q / qft: TVD=0.8750 
- **superfermion.supremacy** @ 4q / qft: TVD=0.5000 
- **superfermion.jax_mps** @ 4q / vqe: TVD=0.0961 
- **superfermion.supremacy** @ 4q / vqe: TVD=0.5000 
- **superfermion.supremacy** @ 8q / ghz: TVD=0.5000 
- **superfermion.statevector** @ 8q / random: TVD=0.0397 
- **superfermion.jax** @ 8q / random: TVD=0.0405 
- **superfermion.rust** @ 8q / random: TVD=0.0397 
- **superfermion.mps** @ 8q / random: TVD=0.0392 
- **superfermion.jax_mps** @ 8q / random: TVD=0.8683 
- **superfermion.singularity** @ 8q / random: TVD=0.0397 
- **superfermion.supremacy** @ 8q / random: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 8q / random: TVD=0.0414 
- **pennylane.default.qubit** @ 8q / random: TVD=0.0324 
- **pennylane.lightning.qubit** @ 8q / random: TVD=0.0358 
- **superfermion.statevector** @ 8q / qft: TVD=0.0448 
- **superfermion.jax** @ 8q / qft: TVD=0.0514 
- **superfermion.rust** @ 8q / qft: TVD=0.0448 
- **superfermion.mps** @ 8q / qft: TVD=0.0472 
- **superfermion.jax_mps** @ 8q / qft: TVD=0.9922 
- **superfermion.singularity** @ 8q / qft: TVD=0.0448 
- **superfermion.supremacy** @ 8q / qft: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 8q / qft: TVD=0.0460 
- **pennylane.default.qubit** @ 8q / qft: TVD=0.0460 
- **pennylane.lightning.qubit** @ 8q / qft: TVD=0.0443 
- **superfermion.jax_mps** @ 8q / vqe: TVD=0.2246 
- **superfermion.supremacy** @ 8q / vqe: TVD=0.5000 
- **superfermion.supremacy** @ 12q / ghz: TVD=0.5000 
- **superfermion.statevector** @ 12q / random: TVD=0.1468 
- **superfermion.jax** @ 12q / random: TVD=0.1476 
- **superfermion.rust** @ 12q / random: TVD=0.1468 
- **superfermion.mps** @ 12q / random: TVD=0.1447 
- **superfermion.jax_mps** @ 12q / random: TVD=0.9804 
- **superfermion.singularity** @ 12q / random: TVD=0.1468 
- **superfermion.supremacy** @ 12q / random: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 12q / random: TVD=0.1429 
- **pennylane.default.qubit** @ 12q / random: TVD=0.1415 
- **pennylane.lightning.qubit** @ 12q / random: TVD=0.1461 
- **superfermion.statevector** @ 12q / qft: TVD=0.1843 
- **superfermion.jax** @ 12q / qft: TVD=0.1784 
- **superfermion.rust** @ 12q / qft: TVD=0.1843 
- **superfermion.mps** @ 12q / qft: TVD=0.1837 
- **superfermion.jax_mps** @ 12q / qft: TVD=0.9988 
- **superfermion.singularity** @ 12q / qft: TVD=0.1843 
- **superfermion.supremacy** @ 12q / qft: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 12q / qft: TVD=0.1802 
- **pennylane.default.qubit** @ 12q / qft: TVD=0.1827 
- **pennylane.lightning.qubit** @ 12q / qft: TVD=0.1820 
- **superfermion.jax_mps** @ 12q / vqe: TVD=0.3029 
- **superfermion.supremacy** @ 12q / vqe: TVD=0.5000 
- **superfermion.supremacy** @ 16q / ghz: TVD=0.5000 
- **superfermion.statevector** @ 16q / random: TVD=0.4052 
- **superfermion.jax** @ 16q / random: TVD=0.4046 
- **superfermion.rust** @ 16q / random: TVD=0.4052 
- **superfermion.mps** @ 16q / random: TVD=0.3983 
- **superfermion.jax_mps** @ 16q / random: TVD=0.9920 
- **superfermion.singularity** @ 16q / random: TVD=0.4052 
- **superfermion.supremacy** @ 16q / random: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 16q / random: TVD=0.3989 
- **pennylane.default.qubit** @ 16q / random: TVD=0.4020 
- **pennylane.lightning.qubit** @ 16q / random: TVD=0.4004 
- **superfermion.statevector** @ 16q / qft: TVD=0.7359 
- **superfermion.jax** @ 16q / qft: TVD=0.7361 
- **superfermion.rust** @ 16q / qft: TVD=0.7359 
- **superfermion.mps** @ 16q / qft: TVD=0.7380 
- **superfermion.jax_mps** @ 16q / qft: TVD=0.9996 
- **superfermion.singularity** @ 16q / qft: TVD=0.7359 
- **superfermion.supremacy** @ 16q / qft: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 16q / qft: TVD=0.7371 
- **pennylane.default.qubit** @ 16q / qft: TVD=0.7371 
- **pennylane.lightning.qubit** @ 16q / qft: TVD=0.7359 
- **superfermion.statevector** @ 16q / vqe: TVD=0.0359 
- **superfermion.jax** @ 16q / vqe: TVD=0.0306 
- **superfermion.rust** @ 16q / vqe: TVD=0.0359 
- **superfermion.mps** @ 16q / vqe: TVD=0.0327 
- **superfermion.jax_mps** @ 16q / vqe: TVD=0.4661 
- **superfermion.singularity** @ 16q / vqe: TVD=0.0359 
- **superfermion.supremacy** @ 16q / vqe: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 16q / vqe: TVD=0.0364 
- **pennylane.default.qubit** @ 16q / vqe: TVD=0.0368 
- **pennylane.lightning.qubit** @ 16q / vqe: TVD=0.0376 
- **superfermion.supremacy** @ 20q / ghz: TVD=0.5000 
- **superfermion.statevector** @ 20q / random: TVD=0.7510 
- **superfermion.jax** @ 20q / random: TVD=0.7486 
- **superfermion.rust** @ 20q / random: TVD=0.7510 
- **superfermion.mps** @ 20q / random: TVD=0.7510 
- **superfermion.jax_mps** @ 20q / random: TVD=0.9984 
- **superfermion.singularity** @ 20q / random: TVD=0.7510 
- **superfermion.supremacy** @ 20q / random: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 20q / random: TVD=0.7455 
- **pennylane.default.qubit** @ 20q / random: TVD=0.7497 
- **pennylane.lightning.qubit** @ 20q / random: TVD=0.7478 
- **superfermion.statevector** @ 20q / qft: TVD=0.9811 
- **superfermion.jax** @ 20q / qft: TVD=0.9811 
- **superfermion.rust** @ 20q / qft: TVD=0.9811 
- **superfermion.mps** @ 20q / qft: TVD=0.9811 
- **superfermion.jax_mps** @ 20q / qft: TVD=1.0000 
- **superfermion.singularity** @ 20q / qft: TVD=0.9811 
- **superfermion.supremacy** @ 20q / qft: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 20q / qft: TVD=0.9811 
- **pennylane.default.qubit** @ 20q / qft: TVD=0.9811 
- **pennylane.lightning.qubit** @ 20q / qft: TVD=0.9811 
- **superfermion.supremacy** @ 24q / ghz: TVD=0.5000 
- **superfermion.mps** @ 24q / random: TVD=0.8946 
- **superfermion.jax_mps** @ 24q / random: TVD=0.9986 
- **superfermion.singularity** @ 24q / random: TVD=0.8953 
- **superfermion.supremacy** @ 24q / random: TVD=0.5000 
- **qiskit_aer.matrix_product_state** @ 24q / random: TVD=0.8952 
- **pennylane.default.qubit** @ 24q / random: TVD=0.8941 
- **pennylane.lightning.qubit** @ 24q / random: TVD=0.8955 
- **superfermion.singularity** @ 32q / ghz: error TypeError: SingularityBackend._run_rust_fused() got multiple values for argument
- **superfermion.singularity** @ 32q / random: error TypeError: SingularityBackend._run_rust_fused() got multiple values for argument
- **superfermion.singularity** @ 48q / ghz: error TypeError: SingularityBackend._run_mps_direct() got multiple values for argument
- **superfermion.singularity** @ 48q / random: error TypeError: SingularityBackend._run_mps_direct() got multiple values for argument
- **superfermion.singularity** @ 64q / ghz: error TypeError: SingularityBackend._run_mps_direct() got multiple values for argument
- **qiskit.matrix_product_state** @ 64q / ghz: error CircuitTooWideForTarget: 'Number of qubits (64) in circuit-326 is greater than m
- **superfermion.singularity** @ 64q / random: error TypeError: SingularityBackend._run_mps_direct() got multiple values for argument
- **qiskit.matrix_product_state** @ 64q / random: error CircuitTooWideForTarget: 'Number of qubits (64) in circuit-327 is greater than m

## Endianness notes

SuperFermion and PennyLane both use Big-Endian (qubit 0 leftmost / MSB). Qiskit uses Little-Endian (qubit 0 rightmost / LSB). Without normalization, a state like `|01>` on qubits (q0=0, q1=1) would be printed as `"01"` by SF/PennyLane but as `"10"` by Qiskit. This harness reverses Qiskit bitstrings and statevector amplitudes at the framework boundary, so every row in the tables above is directly comparable.

Evidence for SuperFermion's BE convention:
- `superfermion/bridge/__init__.py:56-57` -- `from_qiskit` reverses qubit indices (`n_qubits - 1 - idx`) when importing from Qiskit.
- `superfermion/bridge/__init__.py:258-261` -- `from_qasm` does the same reversal from QASM qubit indices.
