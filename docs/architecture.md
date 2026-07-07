# Superfermion Architecture

## Design Principle: Python is the API, Rust Does the Work

Superfermion follows a strict two-layer architecture. Python owns the user-facing API surface — circuit construction, configuration, experiment tracking, ML framework bridges. Rust owns all performance-critical computation — statevector simulation, gate application, sampling, gradients, tensor operations.

JAX is reserved for a single purpose: thin custom-gradient bridges in `superfermion.nn` that connect `sf.State.grad()` to framework-native autograd (JAX `custom_vjp`, PyTorch `autograd.Function`, TF `custom_gradient`). It is never used for simulation, linear algebra, or observable computation.

---

## Core Concepts

### `sf.Circuit`

The fluent circuit builder. Immutable after construction; returns new circuit on each gate call.

```python
qc = sf.Circuit(2).h(0).cx(0, 1)
```

### `sf.State`

A Rust-native quantum state handle exposed via PyO3. This is the first-class object for all post-simulation operations:

| Method | Description |
|---|---|
| `state.expectation(obs)` | Pauli expectation value |
| `state.grad(obs, dag, params)` | Parameter-shift gradient |
| `state.sample(shots)` | Measurement sampling |
| `state.numpy()` | Export as NumPy array |
| `state.entropy()` | Von Neumann entropy |
| `state.purity()` | State purity |
| `state.fidelity(other)` | Fidelity with another state |
| `state.partial_trace(qubits)` | Partial trace |
| `state.qfim(dag, params)` | Quantum Fisher Information Matrix |

All methods dispatch to Rust. Python never touches raw state data.

### `sf.run()` and `sf.simulate()`

Two entry points for circuit execution:

```python
result = sf.run(circuit, device="cpu", method="statevector", shots=1024)
# result.counts, result.state, result.metadata

state = sf.simulate(circuit, device="cpu", method="statevector")
# Returns sf.State directly (shots=0 implied)
```

`sf.run()` returns a `RunResult` with counts, state, and metadata. `sf.simulate()` is shorthand for zero-shot execution that returns the `sf.State` directly.

---

## Module Map

```
superfermion/
├── __init__.py          Circuit, run, simulate, State, MethodError, ...
├── circuit.py           Circuit builder, parameter binding
├── runner.py            sf.run() / sf.simulate() dispatch
├── results.py           RunResult dataclass
├── parameters.py        param() symbolic parameter factory
├── _sf_core             Rust PyO3 extension (State, QuantumDAG, ...)
│
├── devices/
│   ├── rust_device.py   RustDevice — routes to Rust simulation methods
│   ├── ibm.py           IBM Quantum provider adapter
│   ├── ionq.py          IonQ provider adapter
│   ├── braket.py        AWS Braket provider adapter
│   └── openquantum.py   OpenQuantum provider adapter
│
├── observables/
│   └── core.py          PauliString, SparsePauliOp, Hamiltonian, expval
│
├── noise/
│   └── __init__.py      NoiseModel, noise channels (depolarizing, damping, ...)
│
├── nn/
│   ├── quantum_layer.py Flax QuantumLayer (jax.custom_vjp → sf.State.grad)
│   ├── torch_layer.py   TorchQuantumLayer (autograd.Function → sf.State.grad)
│   └── tf_layer.py      TFQuantumLayer (tf.custom_gradient → sf.State.grad)
│
├── qml/
│   ├── gradient/
│   │   ├── adjoint.py       Adjoint differentiation
│   │   ├── parameter_shift.py Parameter-shift rule
│   │   ├── spsa.py          SPSA stochastic gradient
│   │   ├── qng.py           Quantum Natural Gradient
│   │   ├── riemannian.py    Riemannian gradient (natural gradient via QFIM)
│   │   └── stochastic_reconfig.py  Stochastic Reconfiguration
│   └── encoding/        Angle, amplitude, IQP encoding
│
├── algorithms/
│   ├── vqe.py           Variational Quantum Eigensolver
│   ├── qaoa.py          QAOA
│   ├── qsvm.py          Quantum SVM
│   ├── qrl.py           Quantum Reinforcement Learning
│   └── qbm.py           Quantum Boltzmann Machine
│
├── chemistry/           JW/BK transforms, UCCSD, PySCF bridge
├── qec/                 10 codes + 4 decoders
├── compiler/            Gate decomposition, rotation merge, SABRE routing
├── bridge/              Qiskit, Cirq, PennyLane, QASM interop
├── mitigation/          ZNE, readout error mitigation
├── pulse/               Waveforms, schedules, gate calibration
├── viz/                 Circuit drawing, Bloch sphere, histograms
└── experiment/          Experiment tracking protocols
```

---

## Rust Workspace

```
crates/
├── sf-ir/           QuantumDAG, QuantumStateImpl trait, simulation engines
│                    (statevector, MPS, stabilizer, density matrix)
├── sf-compiler/     Pass manager, gate cancellation, rotation merge, twirl
├── sf-router/       SABRE routing, hardware topology
├── sf-pulse/        Waveforms, schedules
├── sf-qec/          Stabilizer codes, MWPM/UnionFind decoders
├── sf-gpu/          CUDA statevector simulation (cudarc, sm_75+)
└── sf-bindings/     PyO3 FFI — exposes State, QuantumDAG, compile, etc.
```

### `QuantumStateImpl` Trait

The Rust trait that all simulation methods implement:

```rust
pub trait QuantumStateImpl {
    fn expectation(&self, observable: &[(Vec<u8>, f64, f64)]) -> (f64, f64);
    fn sample(&self, shots: usize) -> Vec<u64>;
    fn numpy(&self) -> Vec<Complex64>;
    fn entropy(&self) -> f64;
    fn purity(&self) -> f64;
    fn fidelity(&self, other: &Self) -> f64;
    fn partial_trace(&self, qubits: &[usize]) -> Vec<Complex64>;
}
```

The Python `sf.State` wraps whichever `QuantumStateImpl` the simulation produced, dispatching method calls to the Rust implementation.

---

## Execution Flow

```
sf.run(circuit, device="cpu", method="statevector", shots=1024)
    │
    ▼
runner.py — resolve device, bind parameters, optional compile
    │
    ▼
RustDevice.execute(dag, method, shots, **kwargs)
    │
    ├── statevector  → dag.simulate()           [Rust, Rayon-parallel]
    ├── mps          → dag.simulate_mps()       [Rust, faer QR/SVD]
    ├── stabilizer   → dag.simulate_stabilizer() [Rust, word-packed tableau]
    ├── density_matrix → dag.simulate_dm_noisy() [Rust, Kraus channels]
    └── gpu          → dag.simulate_gpu()       [CUDA via cudarc]
    │
    ▼
sf.State (Rust-native handle)
    │
    ▼
RunResult(counts, state, metadata)
```

---

## Simulation Methods

| Method | Engine | Max Qubits | Best For |
|---|---|---|---|
| `statevector` | Rust (Rayon) | ~25 CPU, ~30 GPU | Exact sim, gradients, QML |
| `mps` | Rust (faer) | 200+ | Low-entanglement circuits |
| `stabilizer` | Rust (word-packed) | ~1000 | Clifford circuits, QEC |
| `density_matrix` | Rust (Kraus) | ~12 | Noisy simulation |

---

## ML Framework Integration

Each ML layer is ~20 lines wrapping `sf.State.grad()` with the framework's custom gradient mechanism:

| Framework | Layer | Mechanism |
|---|---|---|
| Flax (JAX) | `QuantumLayer` | `jax.custom_vjp` |
| PyTorch | `TorchQuantumLayer` | `torch.autograd.Function` |
| TensorFlow | `TFQuantumLayer` | `tf.custom_gradient` |

All three call `sf.State.grad()` (Rust parameter-shift) in the backward pass.

---

## Provider / Device Model

Superfermion defines protocols; platform layers implement them:

| Protocol | Purpose |
|---|---|
| `DeviceExecutor` | Anything that can execute a circuit |
| `DeviceCapabilities` | What a device supports (methods, max qubits) |
| `Provider` | Supplies devices (IBM, IonQ, Catstate, ...) |
| `Job` | Async result handle for QPU execution |
| `Algorithm` | Configurable algorithm (VQE, QAOA, ...) |
| `AlgorithmResult` | Algorithm output |

Local simulation uses `RustDevice` (implements `DeviceExecutor`). Cloud providers implement the same protocol.
