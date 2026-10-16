# Prepared CPU execution

The opt-in performance API preserves the existing simulation API and uses
complex128 CPU statevector arithmetic. It snapshots a circuit, caches its gate
ordering and constant matrices, and rebuilds numerical fusion plans for every
parameter set. It does not cache parameter-dependent values across executions.

```python
import numpy as np
import superfermion as sf

x = sf.param("x")
circuit = sf.Circuit(2).ry(x, 0).cx(0, 1)
prepared = sf.prepare(circuit)
rows = [{"x": value} for value in (0.1, 0.7, 1.2)]

states = prepared.states_batch(rows)
means = prepared.mean_z_batch(rows)
cost = sf.DiagonalObservable([0.0, 1.0, 2.0, 0.5])
energies = prepared.diagonal_batch(rows, cost)

state = prepared.simulate({"x": 0.1})
assert abs(state.norm_squared() - 1.0) < 1e-12
```

`states_batch` returns rows in the same little-endian order as the existing raw
Rust statevector API, with qubit q at bit q. This does not change bridge-specific
ordering conventions. Batch state banks are limited to 1 GiB; split larger
inputs. The NumPy bank owns its buffer and may be modified without changing the
prepared circuit. `State.numpy()` continues to return an independent copy.

`mean_z_batch` and `diagonal_batch` keep states in Rust and export only scalars.
Create the immutable diagonal observable once; its weights are copied only at
construction. Batch calls release the Python GIL and execute circuits serially;
within-circuit parallelism uses the existing Rayon kernels.

All free parameters must be present and finite; extra parameters are rejected.
Prepared execution rejects measurements, resets, opaque custom operations and
classical conditions. It supports CPU/f64 only, independently of SF_F32.

For `sf.run(..., shots=0, return_statevector=False)`, the modified device returns
the Rust state without a NumPy export and builds its probability dictionary
only if requested. The default behavior is unchanged. CPU statevector neural
layers also avoid unnecessary sampling and exports and remove the throwaway
simulation before calling the adjoint routine. Adjoint differentiation still
performs its own forward pass; this change does not reuse the autograd forward
state or alter the adjoint memory policy.

General statevector Pauli expectations compile flip/sign masks once per term.
They retain complex coefficients and correct Y phases. Native mean-Z uses
popcount; diagonal expectations scan amplitudes directly without allocating a
probability vector. Purity retains its existing meaning; use norm_squared for
actual vector normalization checks.

`superfermion.kernels.angle_fidelity_kernel(X, Y)` implements the exact product
of squared cosines for product RY encoding followed by a fixed unitary. It is
an analytical shortcut, not a faster general quantum simulator. It is invalid
for data-dependent entanglers or arbitrary multi-layer maps.

No new speedup is asserted before running the supplied Colab benchmark.
