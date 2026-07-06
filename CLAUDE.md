# SuperFermion — CLAUDE.md (Project Context for Claude Code)

> **Token-efficient context map** auto-loaded at session start.
> Pointers to deeper docs in `docs/` are loaded on-demand only.

---

## 1. 10-second pitch

Superfermion is a hardware-agnostic quantum computation framework — think
PyTorch for quantum circuits. **12+ simulation backends** (statevector, rust, mps,
jax, cuda, stabilizer, density-matrix, jax_mps, cupy, cuda_mps, dwave, cluster),
**native adjoint differentiation** (15-28x faster than PennyLane Lightning for
VQE/QML gradients), **MPS tensor networks** scaling to 200+ qubits, built-in
**QEC** (topological/color/linear/honeycomb codes + 4 decoders), multi-framework
ML (Flax QuantumLayer, TorchQuantumLayer, TFQuantumLayer), compiled Rust SIMD
core (AVX-2/FMA) with PyO3 bindings. Protocol-driven device adapters for IBM,
IonQ, AWS Braket, and OpenQuantum.

---

## 2. Files that matter (read in this order if you're new)

| Where | What | Why you'd open it |
|---|---|---|
| `pyproject.toml` | Package config, deps, pytest settings | "What version? What deps?" |
| `Cargo.toml` | Rust workspace (6 crates) | "What Rust crates exist?" |
| `superfermion/__init__.py` | Public API surface + lazy loading | "What does `sf` export?" |
| `superfermion/runner.py` | `sf.run()` — single execution entry point | "How does execution work?" |
| `superfermion/circuit.py` | `Circuit` class (~770 lines) | "How do I build circuits?" |
| `superfermion/devices/__init__.py` | `DeviceExecutor` protocol, `DeviceCapabilities` | "How are devices abstracted?" |
| `superfermion/backends/factory.py` | Backend registry + lazy loading | "Which backends exist?" |
| `superfermion/backends/singularity.py` | Auto-router (master backend) | "How does dispatch work?" |
| `superfermion/experiment/protocols.py` | `TrackerProtocol` for experiment tracking | "How does tracking work?" |
| `crates/sf-bindings/src/lib.rs` | PyO3 FFI bridge | "What Rust functions does Python call?" |
| `crates/sf-ir/src/dag.rs` | Core QuantumDAG + simulation engine | "How does simulation work in Rust?" |

---

## 3. THE ONE THING you must remember

**`maturin develop --release` does NOT update `superfermion/_sf_core.pyd`.**
After every Rust rebuild, copy the artefact into the package:

```bash
cp .venv/lib/python3.*/site-packages/_sf_core/_sf_core*.so superfermion/
```

---

## 4. Execution flow

```
sf.run(circuit, device="cpu", shots=1000)
        |
        v
  Parameter validation (unbound params -> error)
        |
        v
  _resolve_device(device)
    - string -> _resolve_builtin() -> LocalDeviceExecutor wrapping a backend
    - DeviceExecutor object -> use directly
        |
        v
  Hardware-aware compilation (if target= set)
        |
        v
  Gate fusion (unless device.capabilities().skip_fusion)
        |
        v
  TrackerProtocol lifecycle (on_run_start -> execute -> on_run_complete/on_run_error)
        |
        v
  executor.execute(circuit, shots) -> RunResult
```

The **SingularityBackend** auto-routes internally:
n <= 10 -> numpy turbo, 11-32 -> Rust dense, >32 -> MPS,
expval -> MPS via Rust QR boundary contraction, grad -> adjoint.

---

## 5. Python package map (superfermion/)

| Module | Key contents | When to touch |
|---|---|---|
| `circuit.py` | `Circuit` class, `GateRecord`, ~40 gate types | Adding gates or circuit features |
| `runner.py` | `sf.run()` — main execution entry point | Changing dispatch logic |
| `parameters.py` | `SymbolicParameter`, `param()` | Changing parameter system |
| `results.py` | `RunResult` dataclass | Changing result format |
| `primitives.py` | `SFEstimator`, `SFSampler` (Qiskit v2 compatible) | Changing primitives |
| `devices/__init__.py` | `DeviceExecutor` protocol, `DeviceCapabilities` | Changing device abstraction |
| `devices/local.py` | `LocalDeviceExecutor` wrapping simulation backends | Changing local device |
| `devices/ibm.py` | `IBMDeviceExecutor`, `IBMDevice` factory | Changing IBM QPU integration |
| `devices/ionq.py` | `IonQDeviceExecutor`, `IonQDevice` factory | Changing IonQ integration |
| `devices/braket.py` | `BraketDeviceExecutor`, `BraketDevice` factory | Changing AWS Braket integration |
| `devices/openquantum.py` | `OpenQuantumDeviceExecutor`, `OpenQuantumDevice` factory | Changing OpenQuantum integration |
| `experiment/protocols.py` | `TrackerProtocol` | Changing experiment tracking interface |
| `experiment/context.py` | `sf.experiment()` context manager (contextvars) | Changing experiment scope |
| `experiment/local_tracker.py` | `LocalTracker` (JSON to disk) | Changing local tracking |
| `backends/factory.py` | `get_backend()`, `register_backend()`, `list_backends()` | Changing backend registration |
| `backends/singularity.py` | Auto-router with topology caching | Changing dispatch strategy |
| `backends/rust_sim.py` | RustBackend wrapper | Changing Rust-Python bridge |
| `backends/mps.py` | MPSSimulatorBackend (lazy SWAP, SVD truncation) | Changing MPS behaviour |
| `backends/stabilizer.py` | StabilizerBackend + Clifford detection | Changing Clifford handling |
| `backends/density_matrix.py` | DensityMatrixBackend + NoiseModel | Changing noise models |
| `backends/simulator.py` | StatevectorBackend (pure numpy) | Changing numpy sim |
| `backends/turbo.py` | Gate fusion, decompose_for_rust, sampling | Changing optimisation |
| `qml/gradient/adjoint.py` | `adjoint_grad_vector` | Changing adjoint diff |
| `qml/gradient/parameter_shift.py` | Parameter-shift + finite-diff | Changing gradient methods |
| `qml/gradient/qng.py` | Quantum Natural Gradient | Changing QNG |
| `qml/gradient/riemannian.py` | Riemannian optimisation | Changing riemannian |
| `qml/gradient/spsa.py` | SPSA gradient | Changing SPSA |
| `qml/gradient/core.py` | JAX custom primitives for circuits | Changing JAX grad plumbing |
| `qml/templates.py` | AngleEmbedding, ZZFeatureMap, ansatze | Changing QML templates |
| `nn/quantum_layer.py` | `QuantumLayer` (Flax) | Changing Flax integration |
| `nn/torch_layer.py` | `TorchQuantumLayer` (PyTorch) | Changing PyTorch integration |
| `nn/tf_layer.py` | `TFQuantumLayer` (TF/Keras) | Changing TF integration |
| `nn/activation.py` | `QAct` — quantum activation layer | Changing quantum non-linearities |
| `nn/linear.py`, `nn/conv.py` | Hybrid QML layers with QuantumLayer | Changing hybrid layers |
| `algorithms/variational.py` | VQE, QAOA (scipy-based) | Changing variational algorithms |
| `algorithms/grover.py` | Grover's search | Changing Grover |
| `algorithms/qpe.py` | Quantum Phase Estimation | Changing QPE |
| `algorithms/hhl.py` | HHL linear systems | Changing HHL |
| `chemistry/hamiltonians.py` | FermionicOperator, jordan_wigner(), bravyi_kitaev() | Changing fermion mappings |
| `chemistry/ansatz.py` | `uccsd_ansatz()` function | Changing chemistry ansatz |
| `chemistry/pyscf_bridge.py` | PySCF integration | Changing chem backend |
| `qec/codes/topological.py` | Surface2D, Toric2D, Hypercube4D | Changing topological codes |
| `qec/codes/color.py` | ColorCode | Changing color codes |
| `qec/codes/linear.py` | Repetition, Shor, Steane, BaconShor, GenericCSS | Changing linear codes |
| `qec/decoders/__init__.py` | MWPM, UnionFind, BP+OSD, Neural | Changing decoders |
| `compiler/manager.py` | `compile()` — compilation pipeline | Changing compilation |
| `compiler/passes.py` | GateCancellation, RotationMerging, ConstantFolding | Changing compiler passes |
| `compiler/specs.py` | Hardware specs (basis gates, topology) | Adding hardware targets |
| `bridge/__init__.py` | from/to: Qiskit, Cirq, PennyLane, Braket, IonQ, QASM | Changing framework bridges |
| `observables/core.py` | PauliString, SparsePauliOp, Hamiltonian, expval | Changing observables |
| `serialization/` | Circuit JSON, QASM roundtrip, reproducibility manifest | Changing serialization |
| `utils/exceptions.py` | Structured error hierarchy | Changing exceptions |
| `viz/core.py` | draw_mpl, plot_histogram, plot_bloch, plot_state_city | Changing visualization |
| `pulse/` | Waveforms, schedules, gate calibration | Changing pulse-level |
| `noise/__init__.py` | NoiseChannel, NoiseModel | Changing noise models |
| `mitigation/__init__.py` | ZNE, measurement error mitigation | Changing mitigation |

---

## 6. Rust crate map (crates/)

| Crate | Purpose |
|---|---|
| `sf-ir` | Quantum IR: DAG, MPS tensor network (faer QR/SVD), stabilizer tableau, gate sequence (SOA), QASM 2.0 parser, density matrix |
| `sf-compiler` | Compiler: PassManager, gate cancellation, superconducting decomposition, rotation merging, Pauli twirling |
| `sf-router` | Router: SABRE algorithm, coupling maps (IBM Eagle/Heron, Rigetti Ankaa, IonQ Forte, IQM Garnet), noise-aware layout |
| `sf-pulse` | Pulse-level control: Gaussian/DRAG/Square envelopes, pulse schedules, gate calibration |
| `sf-qec` | QEC: StabilizerCode trait, repetition/surface/steane codes, MWPM/UnionFind/Lookup decoders, syndrome extraction |
| `sf-bindings` | PyO3 bridge: wraps all 5 crates as Python classes |

### Key Rust performance features

- **Ping-pong statevector buffers**: two pre-allocated buffers, no per-gate malloc
- **Hand-flattened f64 kernels**: gate matrix elements extracted as individual
  f64 values so LLVM autovectorizes with AVX-2 + FMA
- **Diagonal fast paths**: pure scaling for Z, S, T, Rz, P, CZ, Rzz
- **Permutation fast paths**: pure bit-flip for X, CNOT, SWAP, CCX
- **faer BLAS-quality matmul**: ~8 GFLOPS at bond dim 64
- **Word-packed stabilizer**: 64 qubits per u64 word
- **GateSequence SOA layout**: ~56 bytes/gate in Rust vs ~200 bytes/gate as Python GateRecord

---

## 7. Test suite layout

```
tests/
  conftest.py          -- shared fixtures (bell_circuit, ghz_circuit, MockTracker, etc.)
  unit/                -- fast, isolated tests
    test_circuit.py, test_devices.py, test_experiment.py, test_factory.py,
    test_gates.py, test_observables.py, test_parameters.py, test_results.py,
    test_runner.py
  integration/         -- cross-module tests
    test_multi_backend.py, test_provider_adapters.py, test_run_pipeline.py,
    test_serialization_roundtrip.py, test_tracked_experiment.py
  backends/            -- backend correctness
    test_correctness.py, test_scaling.py, test_turbo.py
  domain/              -- algorithm/feature tests
    test_algorithms.py, test_bridges.py, test_compiler.py,
    test_gradients.py, test_qec.py
```

---

## 8. Quick-reference commands

```bash
# Unit tests (~30s)
python -m pytest tests/unit/ -q

# Full test suite
python -m pytest tests/ -q --timeout=120

# Rust-only tests
cargo test --lib -p sf-ir -p sf-compiler -p sf-router -p sf-qec -p sf-pulse

# Build Rust extension
cd crates/sf-bindings && maturin develop --release && cd ../..
```

---

## 9. Coding conventions

- **Always read before edit**
- **SF endianness is q0=MSB** everywhere. Qiskit + Rust core are q0=LSB.
  Bridge functions reverse: `n - 1 - q`.
- **Gate names are uppercase internally** in `Circuit`
- **Python 3.10-3.13** supported; Rust edition 2021
- **Formatting**: black (line-length=100), ruff (E, F, I, N, W, UP)
- **Tests**: pytest with `testpaths = ["tests"]`; CI skips tests requiring
  Qiskit/PennyLane/Aer
- **Git**: don't auto-commit unless asked

---

## 10. Endianness reference

```
SF (q0 = MSB):           |q0>|q1>|q2>  ->  qubit 0 is the most significant bit
Qiskit (q0 = LSB):       |q2>|q1>|q0>  ->  qubit 0 is the least significant bit
Rust core (q0 = LSB):    same as Qiskit

Conversions:
  sf.bridge.from_qiskit()  ->  reverses qubit indices (n-1-q)
  sf.bridge.to_qiskit()    ->  reverses qubit indices (n-1-q)
  RustBackend.run()        ->  flips axes on result
  Rust QASM parser         ->  n-1-idx when reading qubit references
```
