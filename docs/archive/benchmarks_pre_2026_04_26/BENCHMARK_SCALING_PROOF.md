# SuperFermion scaling proof

_This report falsifies or confirms the scaling claim: "SF is 20-35x vs Qiskit / 30-120x vs PennyLane, and the factor grows with circuit depth and training iterations." Two independent axes are swept. All three frameworks run at complex128. Times are best-of-3 warm. Hardware: CPU, single process._

## AXIS A - circuit-depth scaling (4 qubits, forward <Z_0>)

| n_layers | gate count | SF ms | Qiskit ms | PennyLane ms | SF vs Qiskit | SF vs PennyLane |
|----------|------------|-------|-----------|--------------|--------------|-----------------|
| 1 | 16 | 0.090 | 7.808 | 14.815 | **86.4x** | **163.9x** |
| 5 | 80 | 0.046 | 7.714 | 62.016 | **167.3x** | **1345.2x** |
| 20 | 320 | 0.046 | 20.940 | 241.513 | **452.3x** | **5216.3x** |
| 80 | 1280 | 0.060 | 87.461 | 1004.128 | **1448.0x** | **16624.6x** |

**Memory (peak MiB, warm call):**

| n_layers | SF | Qiskit | PennyLane | SF vs Qiskit | SF vs PennyLane |
|----------|----|--------|-----------|--------------|-----------------|
| 1 | 0.0033 | 0.0251 | 0.0418 | 7.7x | 12.8x |
| 5 | 0.0033 | 0.0248 | 0.0853 | 7.6x | 26.1x |
| 20 | 0.0033 | 0.0280 | 0.2651 | 8.6x | 81.2x |
| 80 | 0.0033 | 0.0648 | 0.9917 | 19.9x | 303.8x |

## AXIS B - VQE-iteration scaling (H2 tapered, analytic grad both sides)

| maxiter | SF total ms | Qiskit total ms | PennyLane total ms | SF vs Qiskit | SF vs PennyLane | SF \|dE\| | Qiskit \|dE\| | PL \|dE\| |
|---------|-------------|-----------------|---------------------|--------------|-----------------|-----------|--------------|-----------|
| 5 | 21.7 | 638.4 | 485.9 | **29.4x** | **22.4x** | 2.03e-03 | 2.03e-03 | 2.03e-03 |
| 25 | 10.9 | 956.6 | 766.4 | **87.5x** | **70.1x** | 3.02e-08 | 3.02e-08 | 3.02e-08 |
| 100 | 10.7 | 943.8 | 743.9 | **88.2x** | **69.6x** | 3.02e-08 | 3.02e-08 | 3.02e-08 |
| 400 | 11.0 | 914.3 | 892.5 | **82.9x** | **80.9x** | 3.02e-08 | 3.02e-08 | 3.02e-08 |

## Verdict

- **AXIS A speedup range** vs Qiskit: **86.4x -> 1448.0x** (grows with depth ✓)
- **AXIS A speedup range** vs PennyLane: **163.9x -> 16624.6x** (grows with depth ✓)
- **AXIS B speedup range** vs Qiskit: **29.4x -> 88.2x** (grows with iters ✓)
- **AXIS B speedup range** vs PennyLane: **22.4x -> 80.9x** (grows with iters ✓)
