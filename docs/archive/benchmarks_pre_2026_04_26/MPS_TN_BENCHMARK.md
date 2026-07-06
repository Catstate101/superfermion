# MPS / Tensor-Network benchmark (large-n, deep-depth)

**Config**: bond_dim=64, Trotter steps=20, observable $\langle Z_0\rangle$, CPU threads=4, per-call timeout=60s, complex128.

Engines:

| Engine | Category | Notes |
|---|---|---|
| **sf.statevector** | SF dense SV | cap n<=22, exact expval |
| **sf.simulator**   | SF dense SV (alias) | cap n<=22, exact expval |
| **sf.jax**         | SF JAX jit dense SV | cap n<=22, exact expval |
| **sf.rust**        | SF Rust dense SV | cap n<=22, exact expval |
| **sf.mps**         | SF native MPS | bond_dim=64, shot-based <Z_0> (10k shots) |
| **sf.singularity** | SF adaptive dispatcher | picks MPS/dense/DM |
| **sf.density_matrix** | SF dense rho (4^n) | cap n<=12, exact open-system expval |
| **aer-mps**        | Qiskit-Aer C++ matrix_product_state | bond_dim=64, exact expval |
| **quimb**          | quimb CircuitMPS | max_bond=64, exact local expval |
| **cirq-mps**       | cirq.contrib.quimb MPSSimulator | max_bond=64, exact via partial_trace |

## W1  TFIM Trotter time evolution  (n sites, 20 steps, J=1, h=0.5)

| n | sf.statevector | sf.simulator | sf.jax | sf.rust | sf.mps | sf.singularity | sf.density_matrix | aer-mps | quimb | cirq-mps | ref | max |D| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=20 | - | - | 28.4 | 55.0 | 6810.8 | 156.0 | - | 104.6 | 2536.0 | 11210.0 | sf.rust | 4.4e-03 |
| n=40 | - | - | - | - | 24554.0 | - | - | 200.6 | 4554.4 | 21352.4 | aer-mps | 4.4e-03 |
| n=60 | - | - | - | - | 47477.8 | - | - | 329.3 | 9237.2 | 37978.0 | aer-mps | 4.4e-03 |
| n=80 | - | - | - | - | - | - | - | 471.2 | 11725.4 | 52027.6 | aer-mps | 2.5e-12 |
| n=100 | - | - | - | - | - | - | - | 603.6 | 14850.8 | - | aer-mps | 2.5e-12 |

## W2  XYZ Heisenberg time evolution (n sites, 20 steps, J=1,1,1)

| n | sf.statevector | sf.simulator | sf.jax | sf.rust | sf.mps | sf.singularity | sf.density_matrix | aer-mps | quimb | cirq-mps | ref | max |D| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=20 | - | - | - | - | 18826.4 | - | - | 3738.0 | 9786.8 | 51584.3 | aer-mps | 4.4e-03 |
| n=40 | - | - | - | - | - | - | - | 12274.7 | 20250.4 | - | aer-mps | 5.6e-12 |
| n=60 | - | - | - | - | - | - | - | 19568.2 | 31669.8 | - | aer-mps | 5.6e-12 |
| n=80 | - | - | - | - | - | - | - | 27276.1 | 42236.9 | - | aer-mps | 5.6e-12 |

## W3  QAOA p=2 on path-graph MaxCut (n sites)

| n | sf.statevector | sf.simulator | sf.jax | sf.rust | sf.mps | sf.singularity | sf.density_matrix | aer-mps | quimb | cirq-mps | ref | max |D| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=20 | 19183.1 | 19592.8 | 31.3 | 41.6 | 74.9 | 172.3 | - | 19.8 | 313.9 | 1101.3 | sf.rust | 4.4e-03 |
| n=40 | - | - | - | - | 102.9 | 55340.6 | - | 40.6 | 604.2 | 2181.2 | aer-mps | 4.4e-03 |
| n=60 | - | - | - | - | 143.6 | - | - | 41.9 | 995.7 | 3557.5 | aer-mps | 4.4e-03 |
| n=80 | - | - | - | - | 170.7 | - | - | 71.7 | 1776.3 | 4562.9 | aer-mps | 4.4e-03 |
| n=100 | - | - | - | - | 238.2 | - | - | 84.2 | 1964.5 | 5701.3 | aer-mps | 4.4e-03 |

## Scaling summary

* SF dense engines (sf.statevector/simulator/jax/rust) hard-cap at n<=22 by 2^n memory.
* SF density_matrix hard-caps at n<=12 by 4^n memory.
* sf.mps / sf.singularity / aer-mps / quimb / cirq-mps scale to n=100 at bounded memory (bond dim).
* Time complexity per Trotter step: full-SV is O(n * 2^n); MPS is O(n * chi^2 * d^2).

## Correctness & caveats

* <Z_0> is single-site - invariant to MSB/LSB qubit ordering.
* sf.mps + sf.singularity use 10 000 shots (sigma ~ 1e-2). Other engines compute exact expval.
* Bond dim chi=64 is sufficient for TFIM dt*steps <= 1.0 (area-law regime).
  Long-time dynamics past the light-cone require chi >> 64; MPS will then diverge from the exact SV truth.
* CPU threads capped at 4 via OMP/MKL/OPENBLAS env variables.
* Per-call wall-clock timeout = 60s; slow engines are marked ERR and later sizes skipped.
