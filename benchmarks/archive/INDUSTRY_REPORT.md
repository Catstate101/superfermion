# Superfermion Industry Benchmark Report

Comparison of Superfermion against latest Industry Kits on flagship problems.

| Benchmark Problem | Superfermion (ms) | Qiskit (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| VQE H2 | 0.0286 | 2.0385 | **71.4x** |
| QAOA Max-Cut | 0.5666 | 3.0773 | **5.4x** |


## Discussion
Superfermion's native JIT compilation and memory-efficient statevector handling allow it to dominate in scientific workloads like QAOA and VQE, where tight iteration loops are critical.