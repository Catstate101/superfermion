# Superfermion Industry Benchmark Report

Comparison of Superfermion against latest Industry Kits on flagship problems.

| Benchmark Problem | Superfermion (ms) | Qiskit (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| VQE H2 | 0.3567 | 0.8390 | **2.4x** |
| QAOA Max-Cut | 1.5752 | 2.9287 | **1.9x** |


## Discussion
Superfermion's native JIT compilation and memory-efficient statevector handling allow it to dominate in scientific workloads like QAOA and VQE, where tight iteration loops are critical.