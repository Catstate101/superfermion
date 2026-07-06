# SuperFermion Industrial MPS Showdown

Comparison of SuperFermion backends against Qiskit Aer and PennyLane using high qubit counts and MPS acceleration.

## VQE (Chemistry)

| Qubits | jax_mps | cuda_mps | mps | jax | Qiskit Aer (MPS) | PennyLane (MPS) |
|:---|:---|:---|:---|:---|:---|:---|
| 20 | 59.09 ms | 8.91 ms | 60.52 ms | 0.01 ms | 2.41 ms | Fail |
| 30 | 151.34 ms | 19.21 ms | 124.08 ms | Skip (Scale) | 2.80 ms | Fail |
| 40 | 178.10 ms | 31.09 ms | 174.19 ms | Skip (Scale) | 5.34 ms | Fail |
| 50 | 191.70 ms | 35.38 ms | 186.19 ms | Skip (Scale) | 3.78 ms | Fail |

## QML (Neural Nets)

| Qubits | jax_mps | cuda_mps | mps | jax | Qiskit Aer (MPS) | PennyLane (MPS) |
|:---|:---|:---|:---|:---|:---|:---|
| 20 | 62.11 ms | Fail | 92.04 ms | 0.01 ms | 3.11 ms | Fail |
| 30 | 137.09 ms | Fail | 152.80 ms | Skip (Scale) | 4.66 ms | Fail |
| 40 | 171.00 ms | Fail | 217.52 ms | Skip (Scale) | 10.79 ms | Fail |
| 50 | 266.84 ms | Fail | 220.24 ms | Skip (Scale) | 8.46 ms | Fail |

## QAOA (Finance)

| Qubits | jax_mps | cuda_mps | mps | jax | Qiskit Aer (MPS) | PennyLane (MPS) |
|:---|:---|:---|:---|:---|:---|:---|
| 20 | 86.72 ms | 6.10 ms | 108.05 ms | 0.02 ms | 5.83 ms | Fail |
| 30 | 206.93 ms | 22.74 ms | 188.15 ms | Skip (Scale) | 6.09 ms | Fail |
| 40 | 229.14 ms | 23.16 ms | 251.63 ms | Skip (Scale) | 9.49 ms | Fail |
| 50 | 280.02 ms | 41.36 ms | 238.22 ms | Skip (Scale) | 7.33 ms | Fail |

| 20 | 161.56 ms | Fail | 177.23 ms | 0.01 ms | 2.15 ms | Fail |
| 30 | 123.19 ms | Fail | 134.95 ms | Skip (Scale) | 3.01 ms | Fail |
| 40 | 189.33 ms | Fail | 178.29 ms | Skip (Scale) | 3.42 ms | Fail |
| 50 | 196.18 ms | Fail | 188.56 ms | Skip (Scale) | 4.10 ms | Fail |

## QML (Neural Nets)

| Qubits | jax_mps | cuda_mps | mps | jax | Qiskit Aer (MPS) | PennyLane (MPS) |
|:---|:---|:---|:---|:---|:---|:---|
| 20 | 71.98 ms | Fail | 90.54 ms | 0.01 ms | 2.62 ms | Fail |
| 30 | 105.46 ms | Fail | 97.77 ms | Skip (Scale) | 3.91 ms | Fail |
| 40 | 122.32 ms | Fail | 112.32 ms | Skip (Scale) | 4.82 ms | Fail |
| 50 | 145.45 ms | Fail | 143.21 ms | Skip (Scale) | 5.63 ms | Fail |

