# Superfermion API Reference

All public API lives under `import superfermion as sf`. Application modules are importable but not promoted to the `sf.*` namespace.

---

## Core Surface (`sf.*`)

### Circuit Construction

**`sf.Circuit(n_qubits: int)`** — Create a quantum circuit.

```python
qc = sf.Circuit(2).h(0).cx(0, 1)
```

Fluent API — each gate returns the circuit for chaining. Supports: `h`, `x`, `y`, `z`, `s`, `t`, `cx`/`cnot`, `cz`, `swap`, `rx`, `ry`, `rz`, `u3`, `ccx`/`toffoli`, `measure`, and more.

**`sf.param(name: str)`** — Create a symbolic parameter for parameterized circuits.

```python
theta = sf.param("theta")
qc = sf.Circuit(1).ry(theta, 0)
bound = qc.bind({"theta": 0.5})
```

### Execution

**`sf.run(circuit, *, device="cpu", method="statevector", shots=1024, **kwargs) → RunResult`**

Execute a circuit and return results.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `circuit` | `Circuit` | required | The circuit to execute |
| `device` | `str \| DeviceExecutor` | `"cpu"` | Device to run on |
| `method` | `str` | `"statevector"` | Simulation method |
| `shots` | `int` | `1024` | Number of measurement shots (0 = exact) |
| `noise_model` | `NoiseModel \| None` | `None` | Noise model (density_matrix method) |
| `bond_dim` | `int` | `64` | MPS bond dimension |

**`sf.simulate(circuit, *, device="cpu", method="statevector", **kwargs) → State`**

Shorthand for exact simulation. Returns `sf.State` directly (equivalent to `sf.run(..., shots=0).state`).

### State

**`sf.State`** — Rust-native quantum state handle.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `expectation` | `(obs: list) → float` | `float` | Pauli expectation value |
| `grad` | `(obs, dag, params) → dict` | `dict[str, float]` | Parameter gradients |
| `sample` | `(shots: int) → dict` | `dict[str, int]` | Measurement counts |
| `numpy` | `() → np.ndarray` | `ndarray` | State as NumPy array |
| `entropy` | `() → float` | `float` | Von Neumann entropy |
| `purity` | `() → float` | `float` | State purity |
| `fidelity` | `(other: State) → float` | `float` | Fidelity with another state |
| `partial_trace` | `(qubits: list) → np.ndarray` | `ndarray` | Partial trace |
| `qfim` | `(dag, params) → np.ndarray` | `ndarray` | Quantum Fisher Information Matrix |

Observable format for `expectation`/`grad`: `[(pauli_indices, coeff_real, coeff_imag), ...]` where `pauli_indices` is a `list[int]` with `0=I, 1=X, 2=Y, 3=Z`.

### Results

**`sf.RunResult`** — Container for execution results.

| Attribute | Type | Description |
|---|---|---|
| `counts` | `dict[str, int]` | Measurement counts (empty if shots=0) |
| `state` | `State \| None` | Quantum state (simulators only) |
| `statevector` | `np.ndarray \| None` | Raw statevector (convenience) |
| `metadata` | `dict` | Method-specific data (density_matrix, etc.) |

### Errors

**`sf.MethodError`** — Raised when an operation is not supported by the current simulation method (e.g., calling `grad` on a stabilizer state).

### Protocols

| Protocol | Description |
|---|---|
| `sf.DeviceExecutor` | Interface for anything that executes circuits |
| `sf.DeviceCapabilities` | Describes device capabilities |
| `sf.Provider` | Supplies device executors |
| `sf.Job` | Async result handle |
| `sf.Algorithm` | Configurable algorithm protocol |
| `sf.AlgorithmResult` | Algorithm output |

### Noise

**`sf.NoiseModel`** — Noise model for density matrix simulation.

```python
nm = sf.NoiseModel()
nm.add_depolarizing(p=0.01)
nm.add_amplitude_damping(gamma=0.05)
nm.add_phase_damping(gamma=0.02)
nm.add_readout_error(p=0.01)
```

### Observables

**`sf.PauliString(pauli_str, coeff=1.0)`** — Single Pauli term.

**`sf.SparsePauliOp`** — Sparse sum of Pauli strings.

```python
obs = sf.SparsePauliOp.from_dict({"ZZ": 1.0, "XI": 0.5})
```

**`sf.Hamiltonian`** — Collection of Pauli terms representing a Hamiltonian.

**`sf.expval(statevector, observable)`** — Compute expectation value from a statevector and observable.

---

## Application Modules

### `superfermion.nn` — ML Framework Layers

```python
from superfermion.nn.quantum_layer import QuantumLayer     # Flax (JAX)
from superfermion.nn.torch_layer import TorchQuantumLayer  # PyTorch
from superfermion.nn.tf_layer import TFQuantumLayer        # TensorFlow
```

All layers accept `(circuit, observable, device, method)` and expose trainable quantum parameters. Backward passes use `sf.State.grad()`.

### `superfermion.qml.gradient` — Gradient Methods

| Module | Function | Description |
|---|---|---|
| `adjoint` | `adjoint_grad_vector()` | Adjoint differentiation (fastest) |
| `parameter_shift` | `parameter_shift_gradient()` | Analytic parameter-shift |
| `spsa` | `spsa_gradient()` | Stochastic approximation |
| `qng` | `qng_step()` | Quantum Natural Gradient |
| `riemannian` | `riemannian_gradient()` | Riemannian (natural) gradient |
| `stochastic_reconfig` | `sr_update()`, `sr_step()` | Stochastic Reconfiguration |

### `superfermion.algorithms` — Quantum Algorithms

| Module | Class | Description |
|---|---|---|
| `vqe` | `VQE` | Variational Quantum Eigensolver |
| `qaoa` | `QAOA` | Quantum Approximate Optimization |
| `qsvm` | `QSVM` | Quantum Support Vector Machine |
| `qrl` | `QuantumREINFORCE` | Quantum Reinforcement Learning |
| `qbm` | `QBM` | Quantum Boltzmann Machine |

### `superfermion.qec` — Quantum Error Correction

10 codes: Repetition, Shor, Steane, Bacon-Shor, Surface, Toric, Color, Honeycomb, Hypercube, CSS.

4 decoders: MWPM, Union-Find, BP+OSD, Neural.

### `superfermion.chemistry` — Quantum Chemistry

Jordan-Wigner and Bravyi-Kitaev transformations, UCCSD ansatz, PySCF bridge, molecular Hamiltonian library.

### `superfermion.compiler` — Circuit Compilation

`sf.compile(circuit, target=...)` — Gate decomposition, rotation merging, SABRE routing. Targets: IBM, Rigetti, IonQ, IQM gate sets.

### `superfermion.bridge` — Cross-Framework Interop

```python
from superfermion.bridge import from_qiskit, to_qiskit, from_pennylane, to_cirq
```

### `superfermion.viz` — Visualization

Bloch sphere plots, state bar charts, circuit drawing (requires matplotlib).
