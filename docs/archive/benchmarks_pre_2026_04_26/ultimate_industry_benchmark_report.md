# Ultimate Industry Benchmark Report

## Summary Statistics

- **Total Benchmarks Run:** 421
- **Successful:** 374
- **Failed:** 47

## Best Performers by Metric

- **Fastest Hot Run:** Superfermion-rust (0.0126ms)
- **Most Memory Efficient:** PennyLane-default.qubit (0.00MB)
- **Highest Fidelity:** Qiskit Aer-statevector (1.000000)

## Detailed Results by Circuit Type

### GHZ

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-jax | 207.78 | 0.02 | 0.00 | 1.000000 |
| 4 | Superfermion-rust | 3.62 | 0.03 | 0.00 | 1.000000 |
| 4 | Superfermion-singularity | 11.60 | 0.19 | 0.00 | 1.000000 |
| 4 | Qiskit Aer-statevector | 3.96 | 2.05 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 10.61 | 7.59 | 0.00 | 1.000000 |
| 4 | Superfermion-jax_mps | 1675.54 | 43.48 | 0.00 | -1.000000 |
| 4 | Superfermion-cuda_mps | 166.44 | 125.26 | 0.00 | -1.000000 |
| 8 | Superfermion-jax | 201.09 | 0.01 | 0.00 | 1.000000 |
| 8 | Superfermion-rust | 1.68 | 0.02 | 0.00 | 1.000000 |
| 8 | Superfermion-singularity | 6.61 | 0.19 | 0.00 | 1.000000 |
| 8 | Qiskit Aer-statevector | 1.99 | 3.06 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 11.36 | 11.91 | 0.00 | 1.000000 |
| 8 | Superfermion-jax_mps | 1035.31 | 281.51 | 0.01 | -1.000000 |
| 8 | Superfermion-cuda_mps | 502.31 | 492.80 | 0.00 | -1.000000 |
| 12 | Superfermion-jax | 251.82 | 0.02 | 0.00 | 1.000000 |
| 12 | Superfermion-rust | 4.37 | 0.03 | 0.00 | 1.000000 |
| 12 | Superfermion-singularity | 6.96 | 0.25 | 0.00 | 1.000000 |
| 12 | Qiskit Aer-statevector | 3.78 | 4.58 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 41.75 | 22.97 | 0.00 | 1.000000 |
| 12 | Superfermion-jax_mps | 809.93 | 750.50 | 0.02 | -1.000000 |
| 12 | Superfermion-cuda_mps | 1588.01 | 1060.76 | 0.00 | -1.000000 |
| 16 | Superfermion-jax | 411.28 | 0.02 | 0.00 | 1.000000 |
| 16 | Superfermion-rust | 18.32 | 0.02 | 0.00 | 1.000000 |
| 16 | Superfermion-singularity | 81.98 | 0.43 | 0.00 | 1.000000 |
| 16 | Qiskit Aer-statevector | 9.02 | 8.03 | 1.00 | 1.000000 |
| 16 | PennyLane-default.qubit | 59.23 | 42.08 | 1.00 | 1.000000 |
| 16 | Superfermion-jax_mps | 989.45 | 1575.27 | 0.01 | -1.000000 |
| 16 | Superfermion-cuda_mps | 2433.39 | 1735.24 | 0.00 | -1.000000 |
| 20 | Superfermion-rust | 454.20 | 0.03 | 0.00 | 1.000000 |
| 20 | Superfermion-jax | 538.49 | 0.04 | 0.00 | 1.000000 |
| 20 | Superfermion-singularity | 960.71 | 0.47 | 0.00 | 1.000000 |
| 20 | Qiskit Aer-statevector | 51.38 | 57.73 | 16.00 | 1.000000 |
| 20 | PennyLane-default.qubit | 349.23 | 358.04 | 16.00 | 1.000000 |
| 20 | Superfermion-jax_mps | 1007.59 | 2202.18 | 0.04 | -1.000000 |
| 20 | Superfermion-cuda_mps | 3352.68 | 2294.71 | 0.00 | -1.000000 |
| 24 | Superfermion-jax | 706.44 | 0.03 | 0.44 | -1.000000 |
| 24 | Superfermion-rust | 5525.13 | 0.06 | 0.00 | -1.000000 |
| 24 | Superfermion-singularity | 14140.96 | 0.35 | 0.02 | -1.000000 |
| 24 | Superfermion-mps | 1.44 | 0.53 | 0.00 | -1.000000 |
| 24 | Qiskit Aer-statevector | 983.34 | 991.96 | 256.06 | -1.000000 |
| 24 | Superfermion-jax_mps | 3417.96 | 3258.25 | 0.06 | -1.000000 |
| 24 | Superfermion-cuda_mps | 3720.95 | 3678.86 | 0.10 | -1.000000 |

### QAOA

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-jax | 912.78 | 0.02 | 0.00 | 1.000000 |
| 4 | Superfermion-rust | 16.97 | 0.02 | 0.00 | 0.001626 |
| 4 | Superfermion-jax | 674.97 | 0.02 | 0.00 | 1.000000 |
| 4 | Superfermion-rust | 13.64 | 0.03 | 0.00 | 0.001626 |
| 4 | Superfermion-singularity | 48.93 | 0.58 | 0.00 | 0.001626 |
| 4 | Superfermion-singularity | 57.54 | 0.63 | 0.00 | 0.001626 |
| 4 | Qiskit Aer-statevector | 4.45 | 4.22 | 0.01 | 1.000000 |
| 4 | Qiskit Aer-statevector | 6.18 | 5.52 | 0.01 | 1.000000 |
| 4 | PennyLane-default.qubit | 772.69 | 31.43 | 0.07 | 1.000000 |
| 4 | PennyLane-default.qubit | 42.00 | 40.96 | 0.06 | 1.000000 |
| 4 | Superfermion-jax_mps | 57.01 | 106.57 | 0.24 | -1.000000 |
| 4 | Superfermion-cuda_mps | 677.95 | 152.15 | 0.01 | -1.000000 |
| 4 | Superfermion-jax_mps | 884.03 | 189.66 | 0.03 | -1.000000 |
| 4 | Superfermion-cuda_mps | 260.61 | 210.45 | 5.19 | -1.000000 |
| 8 | Superfermion-jax | 1227.60 | 0.02 | 0.00 | 1.000000 |
| 8 | Superfermion-rust | 23.96 | 0.02 | 0.00 | 0.000000 |
| 8 | Superfermion-jax | 1305.16 | 0.02 | 0.00 | 1.000000 |
| 8 | Superfermion-rust | 40.85 | 0.02 | 0.00 | 0.000000 |
| 8 | Superfermion-singularity | 75.32 | 1.11 | 0.00 | 0.000000 |
| 8 | Superfermion-singularity | 95.74 | 1.15 | 0.00 | 0.000000 |
| 8 | Qiskit Aer-statevector | 7.03 | 6.41 | 0.00 | 1.000000 |
| 8 | Qiskit Aer-statevector | 11.64 | 11.35 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 52.21 | 60.06 | 0.05 | 1.000000 |
| 8 | PennyLane-default.qubit | 96.50 | 88.47 | 0.07 | 1.000000 |
| 8 | Superfermion-jax_mps | 920.94 | 299.28 | 0.09 | -1.000000 |
| 8 | Superfermion-cuda_mps | 366.91 | 372.23 | 0.01 | -1.000000 |
| 8 | Superfermion-cuda_mps | 613.44 | 451.01 | 0.02 | -1.000000 |
| 8 | Superfermion-jax_mps | 749.99 | 507.26 | 0.08 | -1.000000 |
| 12 | Superfermion-jax | 2060.25 | 0.02 | 0.00 | 1.000000 |
| 12 | Superfermion-jax | 2580.23 | 0.03 | 0.00 | 1.000000 |
| 12 | Superfermion-rust | 125.82 | 0.03 | 0.00 | 0.000000 |
| 12 | Superfermion-rust | 81.68 | 0.03 | 0.00 | 0.000000 |
| 12 | Superfermion-singularity | 141.43 | 1.56 | 0.00 | 0.000000 |
| 12 | Superfermion-singularity | 246.68 | 3.23 | 0.00 | 0.000000 |
| 12 | Qiskit Aer-statevector | 11.59 | 13.84 | 0.01 | 1.000000 |
| 12 | Qiskit Aer-statevector | 22.58 | 23.08 | 0.01 | 1.000000 |
| 12 | PennyLane-default.qubit | 112.89 | 107.34 | 0.06 | 1.000000 |
| 12 | PennyLane-default.qubit | 172.34 | 184.09 | 0.02 | 1.000000 |
| 12 | Superfermion-cuda_mps | 567.61 | 490.46 | 0.01 | -1.000000 |
| 12 | Superfermion-jax_mps | 751.74 | 518.78 | 0.10 | -1.000000 |
| 12 | Superfermion-jax_mps | 10098.33 | 760.96 | 0.15 | -1.000000 |
| 12 | Superfermion-cuda_mps | 1111.87 | 934.44 | 0.04 | -1.000000 |
| 16 | Superfermion-rust | 328.41 | 0.02 | 0.00 | 0.000000 |
| 16 | Superfermion-jax | 3245.57 | 0.03 | 0.00 | 0.999999 |
| 16 | Superfermion-rust | 170.66 | 0.03 | 0.00 | 0.000000 |
| 16 | Superfermion-jax | 2447.11 | 0.04 | 0.00 | 1.000000 |
| 16 | Superfermion-singularity | 563.98 | 2.01 | 0.00 | 0.000000 |
| 16 | Superfermion-singularity | 568.33 | 3.80 | 0.00 | 0.000000 |
| 16 | Qiskit Aer-statevector | 19.91 | 23.53 | 1.00 | 1.000000 |
| 16 | Qiskit Aer-statevector | 30.71 | 34.05 | 1.01 | 1.000000 |
| 16 | PennyLane-default.qubit | 291.84 | 296.37 | 1.05 | 1.000000 |
| 16 | PennyLane-default.qubit | 521.10 | 537.86 | 1.07 | 1.000000 |
| 16 | Superfermion-jax_mps | 747.69 | 700.77 | 0.09 | -1.000000 |
| 16 | Superfermion-cuda_mps | 836.93 | 704.32 | 0.00 | -1.000000 |
| 16 | Superfermion-cuda_mps | 1097.37 | 1005.72 | 0.00 | -1.000000 |
| 16 | Superfermion-jax_mps | 777.25 | 1073.37 | 0.18 | -1.000000 |
| 20 | Superfermion-rust | 2639.25 | 0.02 | 0.00 | 0.000000 |
| 20 | Superfermion-rust | 4925.09 | 0.04 | 0.00 | 0.000000 |
| 20 | Superfermion-jax | 5407.99 | 0.06 | 0.00 | 0.999999 |
| 20 | Superfermion-jax | 3768.04 | 0.06 | 0.00 | 1.000000 |
| 20 | Superfermion-singularity | 6942.14 | 1.82 | 0.00 | 0.000000 |
| 20 | Superfermion-singularity | 9764.50 | 3.44 | 0.00 | 0.000000 |
| 20 | Qiskit Aer-statevector | 135.40 | 126.64 | 16.01 | 1.000000 |
| 20 | Qiskit Aer-statevector | 190.85 | 237.83 | 16.01 | 1.000000 |
| 20 | Superfermion-jax_mps | 973.97 | 832.76 | 0.08 | -1.000000 |
| 20 | Superfermion-cuda_mps | 1009.34 | 1045.66 | 0.01 | -1.000000 |
| 20 | Superfermion-jax_mps | 796.62 | 1424.30 | 0.09 | -1.000000 |
| 20 | Superfermion-cuda_mps | 1288.97 | 1431.42 | 0.00 | -1.000000 |
| 20 | PennyLane-default.qubit | 4689.21 | 4253.45 | 16.62 | 1.000000 |
| 20 | PennyLane-default.qubit | 6714.63 | 6059.62 | 16.12 | 1.000000 |

### QFT

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-jax | 325.16 | 0.01 | 0.00 | 0.001055 |
| 4 | Superfermion-rust | 5.44 | 0.02 | 0.00 | 1.000000 |
| 4 | Superfermion-singularity | 21.66 | 0.48 | 0.00 | 0.001055 |
| 4 | Qiskit Aer-statevector | 35.94 | 3.20 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 41.94 | 23.60 | 0.01 | 1.000000 |
| 4 | Superfermion-jax_mps | 675.50 | 142.69 | 0.00 | -1.000000 |
| 4 | Superfermion-cuda_mps | 970.76 | 172.04 | 0.00 | -1.000000 |
| 8 | Superfermion-jax | 1268.30 | 0.02 | 0.00 | 0.000000 |
| 8 | Superfermion-rust | 32.54 | 0.04 | 0.00 | 0.000151 |
| 8 | Superfermion-singularity | 97.25 | 1.29 | 0.00 | 0.000032 |
| 8 | Qiskit Aer-statevector | 7.66 | 6.94 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 80.23 | 74.17 | 0.03 | 1.000000 |
| 8 | Superfermion-cuda_mps | 1601.91 | 1321.46 | 0.00 | -1.000000 |
| 8 | Superfermion-jax_mps | 881.06 | 1527.07 | 0.03 | -1.000000 |
| 12 | Superfermion-jax | 3063.89 | 0.02 | 0.00 | 0.000000 |
| 12 | Superfermion-rust | 66.68 | 0.02 | 0.00 | 0.000000 |
| 12 | Superfermion-singularity | 171.36 | 2.59 | 0.00 | 0.000000 |
| 12 | Qiskit Aer-statevector | 24.77 | 18.43 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 192.02 | 223.26 | 0.06 | 1.000000 |
| 12 | Superfermion-cuda_mps | 5269.91 | 3803.98 | 0.00 | -1.000000 |
| 12 | Superfermion-jax_mps | 1128.70 | 4190.26 | 0.18 | -1.000000 |
| 16 | Superfermion-jax | 5340.26 | 0.01 | 0.00 | 0.000000 |
| 16 | Superfermion-rust | 390.93 | 0.03 | 0.00 | 0.000000 |
| 16 | Superfermion-singularity | 1364.89 | 10.15 | 0.00 | 0.000000 |
| 16 | Qiskit Aer-statevector | 38.35 | 48.47 | 1.00 | 1.000000 |
| 16 | PennyLane-default.qubit | 1032.95 | 815.12 | 1.31 | 1.000000 |
| 16 | Superfermion-cuda_mps | 13382.10 | 10087.67 | 0.05 | -1.000000 |
| 16 | Superfermion-jax_mps | 1471.77 | 10972.08 | 0.14 | -1.000000 |

### QML

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-rust | 6.44 | 0.01 | 0.00 | 0.069119 |
| 4 | Superfermion-jax | 609.71 | 0.02 | 0.00 | 0.044589 |
| 4 | Superfermion-jax | 369.87 | 0.03 | 0.00 | 0.409349 |
| 4 | Superfermion-rust | 19.01 | 0.09 | 0.00 | 0.091328 |
| 4 | Superfermion-singularity | 20.67 | 0.52 | 0.00 | 0.409349 |
| 4 | Superfermion-singularity | 54.21 | 0.95 | 0.00 | 0.044589 |
| 4 | Qiskit Aer-statevector | 3.20 | 3.89 | 0.00 | 1.000000 |
| 4 | Qiskit Aer-statevector | 4.52 | 5.37 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 47.03 | 35.06 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 63.16 | 41.34 | 0.00 | 1.000000 |
| 4 | Superfermion-jax_mps | 40.85 | 61.44 | 0.01 | -1.000000 |
| 4 | Superfermion-jax_mps | 1148.57 | 143.03 | 0.01 | -1.000000 |
| 4 | Superfermion-cuda_mps | 183.15 | 210.59 | 0.00 | -1.000000 |
| 4 | Superfermion-cuda_mps | 364.29 | 405.22 | 0.00 | -1.000000 |
| 8 | Superfermion-jax | 1104.65 | 0.02 | 0.00 | 0.019531 |
| 8 | Superfermion-jax | 1220.03 | 0.03 | 0.00 | 0.201361 |
| 8 | Superfermion-rust | 30.73 | 0.04 | 0.00 | 0.002926 |
| 8 | Superfermion-rust | 45.86 | 0.04 | 0.00 | 0.050119 |
| 8 | Superfermion-singularity | 47.40 | 1.19 | 0.00 | 0.201361 |
| 8 | Superfermion-singularity | 80.12 | 1.54 | 0.00 | 0.019531 |
| 8 | Qiskit Aer-statevector | 4.93 | 6.27 | 0.00 | 1.000000 |
| 8 | Qiskit Aer-statevector | 7.10 | 9.72 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 60.57 | 48.81 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 97.26 | 79.73 | 0.00 | 1.000000 |
| 8 | Superfermion-jax_mps | 873.85 | 142.11 | 0.01 | -1.000000 |
| 8 | Superfermion-jax_mps | 962.70 | 267.21 | 0.01 | -1.000000 |
| 8 | Superfermion-cuda_mps | 351.47 | 474.24 | 0.00 | -1.000000 |
| 8 | Superfermion-cuda_mps | 745.66 | 543.53 | 0.00 | -1.000000 |
| 12 | Superfermion-jax | 2146.42 | 0.02 | 0.00 | 0.003460 |
| 12 | Superfermion-rust | 28.45 | 0.03 | 0.00 | 0.001178 |
| 12 | Superfermion-jax | 1753.32 | 0.03 | 0.00 | 0.044013 |
| 12 | Superfermion-rust | 40.60 | 0.05 | 0.00 | 0.000210 |
| 12 | Superfermion-singularity | 102.02 | 1.98 | 0.00 | 0.003460 |
| 12 | Superfermion-singularity | 82.32 | 2.31 | 0.00 | 0.044013 |
| 12 | Qiskit Aer-statevector | 11.13 | 11.20 | 0.00 | 1.000000 |
| 12 | Qiskit Aer-statevector | 37.15 | 12.51 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 103.90 | 101.25 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 139.01 | 120.67 | 0.01 | 1.000000 |
| 12 | Superfermion-jax_mps | 948.18 | 249.98 | 0.01 | -1.000000 |
| 12 | Superfermion-jax_mps | 878.33 | 397.40 | 0.02 | -1.000000 |
| 12 | Superfermion-cuda_mps | 617.74 | 595.06 | 0.00 | -1.000000 |
| 12 | Superfermion-cuda_mps | 750.56 | 787.71 | 0.00 | -1.000000 |
| 16 | Superfermion-jax | 3455.04 | 0.02 | 0.00 | 0.001880 |
| 16 | Superfermion-rust | 138.28 | 0.03 | 0.00 | 0.000014 |
| 16 | Superfermion-jax | 2547.71 | 0.04 | 0.00 | 0.018142 |
| 16 | Superfermion-rust | 154.47 | 0.04 | 0.00 | 0.000005 |
| 16 | Superfermion-singularity | 287.66 | 1.59 | 0.00 | 0.018142 |
| 16 | Superfermion-singularity | 314.14 | 1.89 | 0.00 | 0.001880 |
| 16 | Qiskit Aer-statevector | 15.13 | 18.03 | 1.00 | 1.000000 |
| 16 | Qiskit Aer-statevector | 37.97 | 37.72 | 1.00 | 1.000000 |
| 16 | Superfermion-jax_mps | 981.42 | 433.68 | 0.01 | -1.000000 |
| 16 | PennyLane-default.qubit | 406.80 | 438.50 | 1.01 | 1.000000 |
| 16 | PennyLane-default.qubit | 599.84 | 522.57 | 1.00 | 1.000000 |
| 16 | Superfermion-jax_mps | 872.61 | 536.32 | 0.01 | -1.000000 |
| 16 | Superfermion-cuda_mps | 861.49 | 730.29 | 0.00 | -1.000000 |
| 16 | Superfermion-cuda_mps | 837.79 | 935.34 | 0.00 | -1.000000 |

### RANDOM

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-jax | 287.38 | 0.02 | 0.00 | 0.000906 |
| 4 | Superfermion-rust | 15.47 | 0.02 | 0.00 | 0.130072 |
| 4 | Superfermion-jax | 331.83 | 0.02 | 0.00 | 0.187927 |
| 4 | Superfermion-rust | 14.77 | 0.03 | 0.00 | 0.640378 |
| 4 | Superfermion-singularity | 38.07 | 0.33 | 0.00 | 0.000906 |
| 4 | Superfermion-singularity | 40.19 | 0.52 | 0.00 | 0.152109 |
| 4 | Qiskit Aer-statevector | 1.89 | 1.46 | 0.00 | 1.000000 |
| 4 | Qiskit Aer-statevector | 2.41 | 2.34 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 28.90 | 21.37 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 34.32 | 27.27 | 0.00 | 1.000000 |
| 4 | Superfermion-jax_mps | 635.94 | 50.35 | 0.00 | -1.000000 |
| 4 | Superfermion-jax_mps | 37.52 | 61.71 | 0.02 | -1.000000 |
| 4 | Superfermion-cuda_mps | 102.92 | 75.97 | 0.00 | -1.000000 |
| 4 | Superfermion-cuda_mps | 120.74 | 106.64 | 0.00 | -1.000000 |
| 8 | Superfermion-jax | 810.42 | 0.02 | 0.00 | 0.001362 |
| 8 | Superfermion-jax | 452.30 | 0.02 | 0.00 | 0.000000 |
| 8 | Superfermion-rust | 19.53 | 0.03 | 0.00 | 0.374653 |
| 8 | Superfermion-rust | 31.53 | 0.03 | 0.00 | 0.000033 |
| 8 | Superfermion-singularity | 33.76 | 0.35 | 0.00 | 0.000000 |
| 8 | Superfermion-singularity | 83.22 | 0.75 | 0.00 | 0.001074 |
| 8 | Qiskit Aer-statevector | 2.91 | 2.24 | 0.00 | 1.000000 |
| 8 | Qiskit Aer-statevector | 3.10 | 3.17 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 47.99 | 34.49 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 64.97 | 62.79 | 0.01 | 1.000000 |
| 8 | Superfermion-jax_mps | 630.91 | 71.96 | 0.00 | -1.000000 |
| 8 | Superfermion-jax_mps | 71.70 | 142.60 | 0.16 | -1.000000 |
| 8 | Superfermion-cuda_mps | 173.58 | 142.89 | 0.00 | -1.000000 |
| 8 | Superfermion-cuda_mps | 235.11 | 232.80 | 0.00 | -1.000000 |
| 12 | Superfermion-jax | 773.33 | 0.01 | 0.00 | 0.000044 |
| 12 | Superfermion-jax | 1183.69 | 0.02 | 0.00 | 0.000024 |
| 12 | Superfermion-rust | 46.88 | 0.03 | 0.00 | 0.000002 |
| 12 | Superfermion-rust | 34.14 | 0.04 | 0.00 | 0.003853 |
| 12 | Superfermion-singularity | 64.96 | 0.74 | 0.00 | 0.000070 |
| 12 | Superfermion-singularity | 101.05 | 1.24 | 0.00 | 0.000002 |
| 12 | Qiskit Aer-statevector | 2.91 | 2.35 | 0.00 | 1.000000 |
| 12 | Qiskit Aer-statevector | 3.79 | 3.50 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 64.76 | 55.16 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 101.35 | 87.77 | 0.00 | 1.000000 |
| 12 | Superfermion-jax_mps | 613.96 | 90.05 | 0.02 | -1.000000 |
| 12 | Superfermion-jax_mps | 119.73 | 198.64 | 0.12 | -1.000000 |
| 12 | Superfermion-cuda_mps | 234.33 | 280.34 | 0.00 | -1.000000 |
| 12 | Superfermion-cuda_mps | 336.89 | 364.24 | 0.00 | -1.000000 |
| 16 | Superfermion-rust | 93.65 | 0.02 | 0.00 | 0.000001 |
| 16 | Superfermion-rust | 175.42 | 0.02 | 0.00 | 0.000000 |
| 16 | Superfermion-jax | 1132.49 | 0.03 | 0.00 | 0.000009 |
| 16 | Superfermion-jax | 2168.03 | 0.05 | 0.00 | 0.000000 |
| 16 | Superfermion-singularity | 186.08 | 0.99 | 0.00 | 0.000001 |
| 16 | Superfermion-singularity | 704.27 | 1.63 | 0.00 | 0.000000 |
| 16 | Qiskit Aer-statevector | 8.84 | 7.51 | 1.00 | 1.000000 |
| 16 | Qiskit Aer-statevector | 5.91 | 8.83 | 1.00 | 1.000000 |
| 16 | Superfermion-jax_mps | 647.02 | 135.19 | 0.00 | -1.000000 |
| 16 | PennyLane-default.qubit | 186.33 | 234.53 | 1.01 | 1.000000 |
| 16 | Superfermion-jax_mps | 145.17 | 307.85 | 0.15 | -1.000000 |
| 16 | Superfermion-cuda_mps | 360.14 | 362.50 | 0.00 | -1.000000 |
| 16 | PennyLane-default.qubit | 513.20 | 470.01 | 1.00 | 1.000000 |
| 16 | Superfermion-cuda_mps | 938.52 | 865.35 | 0.00 | -1.000000 |
| 20 | Superfermion-rust | 3442.20 | 0.03 | 0.00 | 0.000000 |
| 20 | Superfermion-jax | 2500.99 | 0.03 | 0.00 | 0.000000 |
| 20 | Superfermion-jax | 2959.94 | 0.04 | 0.00 | 0.000051 |
| 20 | Superfermion-rust | 1891.78 | 0.06 | 0.00 | 0.000000 |
| 20 | Superfermion-singularity | 3876.77 | 0.87 | 0.00 | 0.000000 |
| 20 | Superfermion-singularity | 6315.35 | 2.28 | 0.00 | 0.000000 |
| 20 | Qiskit Aer-statevector | 43.22 | 51.62 | 16.00 | 1.000000 |
| 20 | Qiskit Aer-statevector | 48.86 | 55.17 | 16.00 | 1.000000 |
| 20 | Superfermion-jax_mps | 932.92 | 173.66 | 0.01 | -1.000000 |
| 20 | Superfermion-jax_mps | 189.19 | 389.44 | 0.18 | -1.000000 |
| 20 | Superfermion-cuda_mps | 546.39 | 487.76 | 0.00 | -1.000000 |
| 20 | Superfermion-cuda_mps | 973.92 | 912.12 | 0.00 | -1.000000 |
| 20 | PennyLane-default.qubit | 2375.80 | 2143.94 | 16.00 | 1.000000 |
| 20 | PennyLane-default.qubit | 4884.28 | 4994.17 | 16.18 | 1.000000 |
| 30 | Superfermion-mps | 1.41 | 1.31 | 0.00 | -1.000000 |
| 30 | Superfermion-jax_mps | 658.95 | 208.24 | 0.10 | -1.000000 |
| 30 | Superfermion-cuda_mps | 693.85 | 753.50 | 0.00 | -1.000000 |
| 40 | Superfermion-mps | 2.23 | 2.84 | 0.00 | -1.000000 |
| 40 | Superfermion-jax_mps | 754.12 | 293.05 | 0.21 | -1.000000 |
| 40 | Superfermion-cuda_mps | 887.35 | 805.12 | 0.00 | -1.000000 |
| 50 | Superfermion-mps | 2.10 | 2.30 | 0.00 | -1.000000 |
| 50 | Superfermion-jax_mps | 929.20 | 394.45 | 0.15 | -1.000000 |
| 50 | Superfermion-cuda_mps | 1199.79 | 1023.59 | 0.00 | -1.000000 |
| 100 | Superfermion-mps | 7.25 | 6.74 | 0.09 | -1.000000 |
| 100 | Superfermion-jax_mps | 968.07 | 764.91 | 0.45 | -1.000000 |
| 100 | Superfermion-cuda_mps | 2351.12 | 1945.70 | 0.02 | -1.000000 |

### VQE

| Qubits | Backend | Cold (ms) | Hot (ms) | Memory (MB) | Fidelity |
|--------|---------|-----------|----------|-------------|----------|
| 4 | Superfermion-rust | 9.50 | 0.01 | 0.00 | 0.148397 |
| 4 | Superfermion-jax | 2279.36 | 0.02 | 0.00 | 0.000019 |
| 4 | Superfermion-jax | 430.29 | 0.02 | 0.00 | 0.019075 |
| 4 | Superfermion-rust | 29.89 | 0.09 | 0.00 | 0.036074 |
| 4 | Superfermion-singularity | 8.49 | 0.26 | 0.00 | 0.000019 |
| 4 | Superfermion-singularity | 45.63 | 0.63 | 0.00 | 0.019075 |
| 4 | Qiskit Aer-statevector | 6.18 | 3.34 | 0.00 | 1.000000 |
| 4 | Qiskit Aer-statevector | 4.29 | 4.51 | 0.00 | 1.000000 |
| 4 | PennyLane-default.qubit | 41.30 | 23.28 | 0.00 | 1.000000 |
| 4 | Superfermion-jax_mps | 1526.60 | 30.39 | 0.65 | -1.000000 |
| 4 | PennyLane-default.qubit | 31.88 | 31.86 | 0.00 | 1.000000 |
| 4 | Superfermion-jax_mps | 769.25 | 70.04 | 0.04 | -1.000000 |
| 4 | Superfermion-cuda_mps | 3576.88 | 113.38 | 0.04 | -1.000000 |
| 4 | Superfermion-cuda_mps | 228.06 | 118.97 | 0.03 | -1.000000 |
| 8 | Superfermion-rust | 12.06 | 0.02 | 0.00 | 0.007403 |
| 8 | Superfermion-jax | 966.45 | 0.02 | 0.00 | 0.002985 |
| 8 | Superfermion-jax | 751.87 | 0.02 | 0.00 | 0.000007 |
| 8 | Superfermion-rust | 24.16 | 0.02 | 0.00 | 0.000975 |
| 8 | Superfermion-singularity | 36.22 | 0.56 | 0.00 | 0.000007 |
| 8 | Superfermion-singularity | 45.64 | 0.90 | 0.00 | 0.002985 |
| 8 | Qiskit Aer-statevector | 5.80 | 6.27 | 0.00 | 1.000000 |
| 8 | Qiskit Aer-statevector | 5.75 | 6.29 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 44.34 | 44.61 | 0.00 | 1.000000 |
| 8 | PennyLane-default.qubit | 68.21 | 63.41 | 0.00 | 1.000000 |
| 8 | Superfermion-jax_mps | 1099.94 | 65.63 | 0.01 | -1.000000 |
| 8 | Superfermion-jax_mps | 823.56 | 121.67 | 0.02 | -1.000000 |
| 8 | Superfermion-cuda_mps | 181.26 | 171.11 | 0.00 | -1.000000 |
| 8 | Superfermion-cuda_mps | 263.66 | 247.76 | 0.00 | -1.000000 |
| 12 | Superfermion-jax | 1220.81 | 0.02 | 0.00 | 0.000007 |
| 12 | Superfermion-jax | 1395.98 | 0.02 | 0.00 | 0.000098 |
| 12 | Superfermion-rust | 13.69 | 0.02 | 0.00 | 0.000686 |
| 12 | Superfermion-rust | 33.03 | 0.03 | 0.00 | 0.000061 |
| 12 | Superfermion-singularity | 45.49 | 0.91 | 0.00 | 0.000098 |
| 12 | Superfermion-singularity | 69.41 | 1.94 | 0.00 | 0.000007 |
| 12 | Qiskit Aer-statevector | 7.80 | 6.35 | 0.00 | 1.000000 |
| 12 | Qiskit Aer-statevector | 9.07 | 10.52 | 0.00 | 1.000000 |
| 12 | Superfermion-jax_mps | 900.82 | 66.23 | 0.00 | -1.000000 |
| 12 | PennyLane-default.qubit | 70.58 | 70.30 | 0.00 | 1.000000 |
| 12 | PennyLane-default.qubit | 101.51 | 101.19 | 0.00 | 1.000000 |
| 12 | Superfermion-jax_mps | 845.27 | 149.85 | 0.01 | -1.000000 |
| 12 | Superfermion-cuda_mps | 251.91 | 227.56 | 0.00 | -1.000000 |
| 12 | Superfermion-cuda_mps | 326.62 | 316.99 | 0.00 | -1.000000 |
| 16 | Superfermion-rust | 62.38 | 0.02 | 0.00 | 0.000030 |
| 16 | Superfermion-rust | 101.35 | 0.02 | 0.00 | 0.000006 |
| 16 | Superfermion-jax | 2089.34 | 0.03 | 0.00 | 0.000000 |
| 16 | Superfermion-jax | 1495.93 | 0.05 | 0.00 | 0.000000 |
| 16 | Superfermion-singularity | 175.96 | 1.08 | 0.00 | 0.000000 |
| 16 | Superfermion-singularity | 260.28 | 2.03 | 0.00 | 0.000000 |
| 16 | Qiskit Aer-statevector | 10.37 | 13.55 | 1.00 | 1.000000 |
| 16 | Qiskit Aer-statevector | 24.97 | 18.87 | 1.00 | 1.000000 |
| 16 | Superfermion-jax_mps | 884.09 | 84.84 | 0.00 | -1.000000 |
| 16 | Superfermion-jax_mps | 722.18 | 168.57 | 0.02 | -1.000000 |
| 16 | PennyLane-default.qubit | 205.97 | 208.99 | 1.01 | 1.000000 |
| 16 | Superfermion-cuda_mps | 292.32 | 283.26 | 0.00 | -1.000000 |
| 16 | PennyLane-default.qubit | 289.67 | 298.02 | 1.00 | 1.000000 |
| 16 | Superfermion-cuda_mps | 606.66 | 422.37 | 0.00 | -1.000000 |
| 20 | Superfermion-jax | 2091.08 | 0.03 | 0.50 | 0.000000 |
| 20 | Superfermion-rust | 1238.11 | 0.03 | 0.00 | 0.000000 |
| 20 | Superfermion-jax | 2923.17 | 0.03 | 0.44 | 0.000000 |
| 20 | Superfermion-rust | 830.96 | 0.03 | 0.00 | 0.000004 |
| 20 | Superfermion-singularity | 2192.26 | 1.61 | 0.00 | 0.000000 |
| 20 | Superfermion-singularity | 3209.35 | 2.02 | 0.00 | 0.000000 |
| 20 | Qiskit Aer-statevector | 42.45 | 43.79 | 16.00 | 1.000000 |
| 20 | Qiskit Aer-statevector | 44.26 | 47.32 | 16.00 | 1.000000 |
| 20 | Superfermion-jax_mps | 1189.80 | 101.95 | 0.01 | -1.000000 |
| 20 | Superfermion-jax_mps | 905.26 | 233.92 | 0.00 | -1.000000 |
| 20 | Superfermion-cuda_mps | 385.51 | 380.17 | 0.00 | -1.000000 |
| 20 | Superfermion-cuda_mps | 559.16 | 519.91 | 0.00 | -1.000000 |
| 20 | PennyLane-default.qubit | 2462.64 | 2479.55 | 16.00 | 1.000000 |
| 20 | PennyLane-default.qubit | 3300.88 | 3443.51 | 16.01 | 1.000000 |
| 24 | Superfermion-jax | 7562.16 | 0.04 | 0.00 | -1.000000 |
| 24 | Superfermion-rust | 21646.80 | 0.04 | 0.00 | -1.000000 |
| 24 | Superfermion-rust | 14158.50 | 0.04 | 0.00 | -1.000000 |
| 24 | Superfermion-jax | 3933.43 | 0.08 | 0.00 | -1.000000 |
| 24 | Superfermion-mps | 1.48 | 1.54 | 0.00 | -1.000000 |
| 24 | Superfermion-mps | 2.13 | 1.72 | 0.00 | -1.000000 |
| 24 | Superfermion-singularity | 52916.64 | 2.60 | 0.00 | -1.000000 |
| 24 | Superfermion-singularity | 38306.46 | 5.66 | 0.20 | -1.000000 |
| 24 | Superfermion-jax_mps | 7830.68 | 179.90 | 0.00 | -1.000000 |
| 24 | Superfermion-jax_mps | 18363.11 | 268.09 | 0.64 | -1.000000 |
| 24 | Superfermion-cuda_mps | 799.17 | 465.46 | 3.07 | -1.000000 |
| 24 | Qiskit Aer-statevector | 796.99 | 598.32 | 256.01 | -1.000000 |
| 24 | Superfermion-cuda_mps | 617.13 | 662.51 | 0.42 | -1.000000 |
| 24 | Qiskit Aer-statevector | 673.45 | 669.91 | 257.34 | -1.000000 |
| 30 | Superfermion-mps | 1.59 | 2.33 | 0.02 | -1.000000 |
| 30 | Superfermion-jax_mps | 1668.37 | 184.74 | 1.17 | -1.000000 |
| 30 | Superfermion-cuda_mps | 691.65 | 597.02 | 0.00 | -1.000000 |
| 40 | Superfermion-mps | 4.26 | 2.94 | 0.00 | -1.000000 |
| 40 | Superfermion-jax_mps | 1391.15 | 217.24 | 0.61 | -1.000000 |
| 40 | Superfermion-cuda_mps | 901.89 | 805.22 | 0.00 | -1.000000 |
| 50 | Superfermion-mps | 5.75 | 3.67 | 0.00 | -1.000000 |
| 50 | Superfermion-jax_mps | 1237.40 | 213.89 | 0.63 | -1.000000 |
| 50 | Superfermion-cuda_mps | 1149.40 | 1020.54 | 0.01 | -1.000000 |
| 100 | Superfermion-mps | 10.46 | 9.06 | 0.08 | -1.000000 |
| 100 | Superfermion-jax_mps | 1659.43 | 611.36 | 1.10 | -1.000000 |
| 100 | Superfermion-cuda_mps | 2008.64 | 2128.54 | 0.41 | -1.000000 |
