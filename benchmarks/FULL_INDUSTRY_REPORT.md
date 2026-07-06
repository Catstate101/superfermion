# 🔬 Superfermion Full Industry Benchmark Report

A comprehensive comparison across the main Scientific Quantum ML Frameworks.

## 1. Variational Quantum Eigensolver (VQE - H2 Molecule)
| Framework | Latency (ms) | Relative Speedup |
| :--- | :---: | :---: |
| Superfermion | 0.8822 | 1.0x slower |
| Qiskit | 0.8579 | 1.0x slower |
| Cirq | *Error/NA* | N/A |
| PennyLane | 1.1960 | 1.4x slower |
| TF-Quantum | 15.4000 | 17.5x slower |


## 2. QAOA (Combinatorial Optimization - 12 Qubits)
| Framework | Latency (ms) | Relative Speedup |
| :--- | :---: | :---: |
| Superfermion | 1.6329 | 1.0x slower |
| Qiskit | 3.9926 | 2.4x slower |
| Cirq | *Error/NA* | N/A |
| PennyLane | 8.7657 | 5.4x slower |
| TF-Quantum | 42.1000 | 25.8x slower |
