# INA Industry Quantum Benchmark Suite

**Generated**: 2026-03-17 14:46:48

Comparison of SuperFermion backends against Qiskit Aer for Industry Standard use cases.

## Hardware Safety & Efficiency Guardrails
- **Statevector Limit**: 26 qubits (to prevent system crash).
- **MPS Limit**: 200 qubits.
- **Shots**: 0 (Raw Engine Throughput).
- **Memory Tracking**: Measured via `tracemalloc` peak memory.

## VQE (Drug Discovery)

| N Qubits | Metric | singularity | rust | jax_mps | mps | Qiskit SV | Qiskit MPS |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 12 | Latency (ms) | 0.473 | 0.351 | 4.082 | 200.835 | 7.654 | 6.380 |
|  | Memory (MB) | 0.004 | 0.034 | 0.022 | 0.228 | 0.010 | 0.010 |
| 24 | Latency (ms) | 1.768 | 114.931 | 4.669 | 515.433 | 32.201 | 26.145 |
|  | Memory (MB) | 0.007 | 256.003 | 0.021 | 0.309 | 0.010 | 0.010 |
| 100 | Latency (ms) | 2.232 | 1.099 | 6.408 | 980.987 | Skip (SV Limit) | 52.318 |
|  | Memory (MB) | 0.008 | 0.006 | 0.021 | 0.539 | 0.000 | 0.020 |

## QAOA (Finance Optimization)

| N Qubits | Metric | singularity | rust | jax_mps | mps | Qiskit SV | Qiskit MPS |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 12 | Latency (ms) | 0.544 | 0.444 | 3.360 | 134.704 | 3.021 | 4.383 |
|  | Memory (MB) | 0.004 | 0.034 | 0.021 | 0.203 | 0.009 | 0.009 |
| 24 | Latency (ms) | 0.884 | 106.556 | 3.432 | 534.722 | 21.003 | 10.653 |
|  | Memory (MB) | 0.007 | 256.003 | 0.021 | 0.337 | 0.009 | 0.009 |
| 100 | Latency (ms) | 2.456 | 0.590 | 4.854 | 1043.219 | Skip (SV Limit) | 20.666 |
|  | Memory (MB) | 0.008 | 0.006 | 0.021 | 0.449 | 0.000 | 0.012 |

## QML (Quantum Machine Learning)

| N Qubits | Metric | singularity | rust | jax_mps | mps | Qiskit SV | Qiskit MPS |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 12 | Latency (ms) | 0.316 | 0.384 | 3.667 | 109.485 | 4.356 | 5.075 |
|  | Memory (MB) | 0.004 | 0.034 | 0.021 | 0.168 | 0.009 | 0.009 |
| 24 | Latency (ms) | 1.201 | 111.347 | 3.433 | 527.721 | 30.535 | 26.506 |
|  | Memory (MB) | 0.006 | 256.003 | 0.021 | 0.330 | 0.010 | 0.010 |
| 64 | Latency (ms) | 2.111 | 0.759 | 6.227 | 558.798 | Skip (SV Limit) | 18.364 |
|  | Memory (MB) | 0.008 | 0.006 | 0.022 | 0.545 | 0.000 | 0.013 |

## QFT (Fourier Transform)

| N Qubits | Metric | singularity | rust | jax_mps | mps | Qiskit SV | Qiskit MPS |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 12 | Latency (ms) | 1.269 | 0.351 | 5.571 | 466.197 | 19.766 | 18.221 |
|  | Memory (MB) | 0.007 | 0.034 | 0.021 | 0.299 | 0.009 | 0.009 |
| 24 | Latency (ms) | 1.289 | TIMEOUT | TIMEOUT | 1510.892 | 108.796 | 66.336 |
|  | Memory (MB) | 0.007 | 0.000 | 0.000 | 0.000 | 0.012 | 0.012 |
