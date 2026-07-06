# SuperFermion — CLAUDE.md (Project Context for Claude Code)

> **Token-efficient context map** auto-loaded at session start.
> Pointers to deeper docs in `docs/` are loaded on-demand only.

---

## 1. 10-second pitch

Superfermion is a quantum-circuit simulator with **12+ backends** (statevector,
rust, mps, jax, cuda, stabilizer, density-matrix, jax_mps, cupy, cuda_mps,
dwave, cluster). Features **native adjoint differentiation** (15–28× faster than
PennyLane Lightning for VQE/QML gradients), **MPS tensor networks** scaling to
200+ qubits, and built-in **QEC** (11 codes + 4 decoders). Multi-framework ML:
Flax QuantumLayer, TorchQuantumLayer, TFQuantumLayer. Compiled Rust SIMD core
(AVX-2/FMA) with PyO3 bindings. FastAPI backend.

---

## 2. Files that matter (read in this order if you're new)

| Where | What | Why you'd open it |
|---|---|---|
| `docs/architecture.md` | system architecture guide | "How does X talk to Y?" |
| `pyproject.toml` | Python package config, deps, pytest settings | "What version? What deps?" |
| `Cargo.toml` | Rust workspace (6 crates) | "What Rust crates exist?" |
| `superfermion/__init__.py` | public API surface | "What does `sf` export?" |
| `superfermion/backends/registry.py` | backend registry + lazy loading | "Which backends exist?" |
| `superfermion/backends/singularity.py` | auto-router (master backend) | "How does dispatch work?" |
| `superfermion/circuit.py` | Circuit class (~870 lines) | "How do I build circuits?" |
| `crates/sf-bindings/src/lib.rs` | PyO3 FFI bridge (1107 lines) | "What Rust functions does Python call?" |
| `crates/sf-ir/src/dag.rs` | core QuantumDAG + simulation engine | "How does simulation work in Rust?" |
| `docs/benchmarks.md` | canonical benchmark scoreboard | "How fast is SF?" |

---

## 3. THE ONE THING you must remember

**`maturin develop --release` does NOT update `superfermion/_sf_core.pyd`.**
After every Rust rebuild, copy the artefact into the package or your changes
silently won't take effect:

```bash
cp .venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd \
   superfermion/_sf_core.pyd
```

One-liner combo:
```bash
cd crates/sf-bindings && maturin develop --release && \
cp ../../.venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd \
   ../../superfermion/_sf_core.pyd
```

On Linux, the extension is `.so` instead of `.pyd` and lives at a different path.

---

## 4. Backend dispatch decision tree

```
user calls backend.run(circuit, shots)
                │
                ▼
      ┌──────────────────────┐
      │ is_clifford_circuit? │
      └──────────────────────┘
          │ yes      │ no
          ▼          ▼
 StabilizerBackend   normal path of the chosen backend
```

Every backend has Clifford auto-dispatch. For non-Clifford circuits the
backends use their native paths:

- **`singularity`** (auto-router): n ≤ 10 → numpy turbo · 11–32 → Rust dense ·
  >32 → MPS · expval → MPS via Rust QR boundary contraction · grad → adjoint.
  Zero-overhead Fast Parameter Injection path for variational loops
  (`dag.update_parameters()` + `run_dag()`).
- **`rust`** — Rust dense statevector with AVX-2/FMA SIMD; ping-pong buffers;
  diagonal-1q + diagonal-2q + permutation specialisations. Supports symbolic
  parameters. 32-qubit max.
- **`mps`** — Python wrapper around Rust MPS (faer QR for high bond). Default
  bond dim 64. Lazy SWAP routing. 200+ qubit scaling.
- **`stabilizer`** — Word-packed Aaronson–Gottesman tableau (poly-time, ~1000
  qubits). Clifford detection + canonical circuit synthesis.
- **`density_matrix`** — Exact density matrix with Kraus noise operators.
  NoiseModel class with depolarizing, amplitude damping, phase damping,
  bit/phase flip, readout error. Max 12 qubits (density matrix is 4^n).
- **`jax`** — JAX-native scan-based statevector. JIT-compiled, GPU-ready.
- **`cuda`** — NVIDIA cuQuantum GPU statevector.
- **`cupy`** — CuPy GPU statevector.
- **`jax_mps`** — JAX-based MPS.
- **`cuda_mps`** — GPU-accelerated MPS.
- **`dwave`** — D-Wave quantum annealer interface.
- **`cluster`** — Distributed JAX across multiple nodes.

For QML / VQE call **`singularity.grad(...)`** directly: it routes to
`adjoint_grad_vector` which is 15–28× faster than PennyLane Lightning's
adjoint at n=4–10 with 8–20 params.

---

## 5. Python package map (superfermion/)

| Module | Key contents | When to touch |
|---|---|---|
| `circuit.py` | `Circuit` class, `GateRecord`, ~40 gate types | Adding gates or circuit features |
| `runner.py` | `run()` — main execution entry point | Changing dispatch logic |
| `parameters.py` | `SymbolicParameter`, `param()` | Changing parameter system |
| `results.py` | `RunResult` dataclass | Changing result format |
| `train.py` | `train()` — JAX training loop | Changing training pipeline |
| `pipeline.py` | `Pipeline` class | Changing ML pipeline |
| `primitives.py` | `SFEstimator`, `SFSampler` (Qiskit v2 compatible) | Changing primitives |
| `backends/singularity.py` | auto-router with topology caching | Changing dispatch strategy |
| `backends/rust_sim.py` | RustBackend wrapper | Changing Rust-Python bridge |
| `backends/mps.py` | MPSSimulatorBackend (lazy SWAP, SVD truncation) | Changing MPS behaviour |
| `backends/stabilizer.py` | StabilizerBackend + Clifford detection | Changing Clifford handling |
| `backends/density_matrix.py` | DensityMatrixBackend + NoiseModel | Changing noise models |
| `backends/simulator.py` | StatevectorBackend (pure numpy) | Changing numpy sim |
| `backends/turbo.py` | gate fusion, decompose_for_rust, sampling | Changing optimisation |
| `backends/jax_sim.py` | JAXBackend | Changing JAX backend |
| `backends/cuda.py` | CUSimulatorBackend | Changing cuQuantum backend |
| `backends/cupy_sim.py` | CupyBackend | Changing CuPy backend |
| `backends/dwave.py` | DWaveBackend | Changing annealer interface |
| `backends/cluster.py` | DistributedJAXBackend | Changing distributed backend |
| `backends/supremacy_core.py` | SupremacyBackend | Changing supremacy benchmarking |
| `qml/gradient/adjoint.py` | `adjoint_grad_vector` | Changing adjoint diff |
| `qml/gradient/parameter_shift.py` | parameter-shift + finite-diff | Changing gradient methods |
| `qml/gradient/qng.py` | Quantum Natural Gradient | Changing QNG |
| `qml/gradient/riemannian.py` | Riemannian optimisation | Changing riemannian |
| `qml/gradient/spsa.py` | SPSA gradient | Changing SPSA |
| `qml/gradient/core.py` | JAX custom primitives for circuits | Changing JAX grad plumbing |
| `qml/templates.py` | AngleEmbedding, ZZFeatureMap, ansatze | Changing QML templates |
| `nn/quantum_layer.py` | `QuantumLayer` (Flax, 313 lines) | Changing Flax integration |
| `nn/torch_layer.py` | `TorchQuantumLayer` (PyTorch, 313 lines) | Changing PyTorch integration |
| `nn/tf_layer.py` | `TFQuantumLayer` (TF/Keras, 296 lines) | Changing TF integration |
| `nn/*.py` | attention, conv, dropout, embedding, norm, recurrent, transformer | Changing NN building blocks |
| `algorithms/variational.py` | VQE, QAOA (scipy-based) | Changing variational algorithms |
| `algorithms/vqe.py` | VQE — DEPRECATED (JAX/Optax) | Legacy compatibility |
| `algorithms/qaoa.py` | QAOA — DEPRECATED (JAX/Optax) | Legacy compatibility |
| `algorithms/grover.py` | Grover's search | Changing Grover |
| `algorithms/qpe.py` | Quantum Phase Estimation | Changing QPE |
| `algorithms/hhl.py` | HHL linear systems | Changing HHL |
| `algorithms/amplitude_estimation.py` | Amplitude estimation | Changing amplitude est |
| `algorithms/qsvm.py` | Quantum SVM | Changing QSVM |
| `algorithms/qrl.py` | Quantum RL | Changing QRL |
| `algorithms/qbm.py` | Quantum Boltzmann Machine | Changing QBM |
| `chemistry/hamiltonians.py` | Jordan-Wigner + Bravyi-Kitaev transforms | Changing fermion mappings |
| `chemistry/ansatz.py` | UCCSD ansatz | Changing chemistry ansatz |
| `chemistry/pyscf_bridge.py` | PySCF integration | Changing chem backend |
| `chemistry/library.py` | Molecular Hamiltonian library | Adding molecules |
| `qec/codes/linear.py` | Repetition, Shor, Steane, BaconShor, GenericCSS | Changing linear codes |
| `qec/codes/surface.py` | SurfaceCode | Changing surface code |
| `qec/codes/topological.py` | Surface2D, Toric2D, Hypercube4D | Changing topological codes |
| `qec/codes/color.py` | ColorCode | Changing color codes |
| `qec/codes/honeycomb.py` | HoneycombCode | Changing honeycomb code |
| `qec/codes/ldpc.py` | LDPC, BivariateBicycle, GKP | Changing LDPC codes |
| `qec/decoders/__init__.py` | MWPM, UnionFind, BP+OSD, Neural | Changing decoders |
| `compiler/manager.py` | `compile()` — compilation pipeline | Changing compilation |
| `compiler/passes.py` | GateCancellation, RotationMerging, ConstantFolding | Changing compiler passes |
| `compiler/advanced_passes.py` | Commutation, KAK, PauliTwirling, DynamicalDecoupling | Changing advanced passes |
| `runtime/arbiter.py` | `RuntimeArbiter` — best-backend selection | Changing auto-selection |
| `runtime/orchestrator.py` | `JobOrchestrator` | Changing job orchestration |
| `runtime/scheduler.py` | `CloudScheduler` | Changing scheduling |
| `runtime/providers/ibm.py` | IBM Quantum provider | Changing IBM integration |
| `runtime/providers/aws.py` | AWS Braket provider | Changing AWS integration |
| `runtime/providers/ionq.py` | IonQ provider | Changing IonQ integration |
| `runtime/providers/openquantum.py` | OpenQuantum provider | Changing OpenQuantum integration |
| `bridge/__init__.py` | from/to: Qiskit, Cirq, PennyLane, Braket, IonQ, QASM | Changing framework bridges |
| `viz/core.py` | draw_mpl, plot_histogram, plot_bloch, plot_state_city | Changing visualisation |
| `serve/app.py` | FastAPI server (code execution + CLI endpoints) | Changing API server |
| `utils/exceptions.py` | 15-class SFError hierarchy | Changing exceptions |
| `utils/validation.py` | Input validation | Changing validation |
| `serialization/circuit_format.py` | Circuit JSON serialization | Changing serial format |
| `serialization/qasm_roundtrip.py` | QASM round-trip | Changing QASM support |
| `telemetry/` | structured logging, tracing, metrics | Changing observability |
| `pulse/` | waveforms, schedules, gate calibration | Changing pulse-level |
| `noise/__init__.py` | NoiseChannel, NoiseModel | Changing noise models |
| `mitigation/__init__.py` | ZNE, measurement error mitigation | Changing mitigation |
| `plugins/__init__.py` | plugin registration system | Changing plugin system |
| `security/` | audit, credentials, sanitize, TLS, tokens | Changing security |
| `intelligence/` | SuperpositionalAgent, QNSCore, EntangledBus — DEPRECATED | Do not use |
| `qdl/` | QResNetBlock, QuantumSelfAttention | Changing QDL layers |
| `qllm/` | QuantumTransformerBlock, QuantumGPT | Changing QLLM layers |
| `benchmarks/` | `suite.py`, `full_industry_suite.py`, `industry_standard_qml.py` | Changing benchmarks |

---

## 6. Rust crate map (crates/)

| Crate | Source files | Purpose |
|---|---|---|
| `sf-ir` | `dag.rs`, `ops.rs`, `mps.rs`, `stabilizer.rs`, `gate_list.rs`, `qasm.rs`, `dm.rs`, `qubits.rs`, `classical.rs`, `serialize.rs` | Quantum IR: DAG representation, MPS tensor network (faer QR/SVD), stabilizer tableau, gate sequence (SOA), QASM 2.0 parser, density matrix, qubit mapping |
| `sf-compiler` | `lib.rs`, `passes.rs`, `decompose/mod.rs`, `decompose/superconducting.rs`, `rotation_merge.rs`, `twirl.rs` | Compiler: PassManager, gate cancellation, superconducting decomposition (IBM basis), rotation merging, Pauli twirling |
| `sf-router` | `lib.rs`, `topology.rs`, `layout.rs`, `sabre.rs`, `token_swap.rs` | Router: SABRE algorithm, coupling maps (IBM Eagle 127q, Heron 133q, Rigetti Ankaa 84q, IonQ Forte 36q, IQM Garnet 20q), noise-aware layout |
| `sf-pulse` | `lib.rs`, `waveforms.rs`, `schedule.rs`, `calibration.rs` | Pulse-level control: Gaussian/DRAG/Square envelopes, pulse schedules per channel, gate calibration DB |
| `sf-qec` | `lib.rs`, `codes.rs`, `codes/ldpc.rs`, `codes/bivariate.rs`, `decoders.rs`, `syndrome.rs`, `logical_ops.rs` | QEC: StabilizerCode trait, repetition/surface/steane codes, MWPM/UnionFind/Lookup decoders, syndrome extraction circuit builder |
| `sf-bindings` | `lib.rs` (1107 lines) | PyO3 bridge: wraps all 5 crates as Python classes (QuantumDAG, MPSState, GateSequence, StabilizerTableau, Compiler, Router, CouplingMap, PulseSchedule, QECCode, MWPMDecoder, UnionFindDecoder) |

### Key Rust performance features

- **Ping-pong statevector buffers** (`dag.rs:588-609`): two pre-allocated buffers,
  no per-gate malloc
- **Hand-flattened f64 kernels** (`dag.rs:709-751`): gate matrix elements
  extracted as individual f64 values so LLVM autovectorizes with AVX-2 + FMA
- **Diagonal fast paths**: pure scaling for Z, S, T, Rz, P, CZ, Rzz — 1 read +
  1 write per amplitude
- **Permutation fast paths**: pure bit-flip for X, CNOT, SWAP, CCX — no FMAs
- **faer BLAS-quality matmul** (`mps.rs`): ~8 GFLOPS at bond dim 64 vs ~1
  GFLOPS for nalgebra; cache-blocked AVX-2 microkernels for QR + matmul
- **Word-packed stabilizer** (`stabilizer.rs`): 64 qubits per u64 word;
  `count_ones()` for phase computation (~64x faster than per-bit loop)
- **GateSequence SOA layout** (`gate_list.rs`): ~56 bytes/gate in Rust vs
  ~200 bytes/gate as Python GateRecord
- **`.cargo/config.toml`**: `rustflags = ["-C", "target-cpu=native"]` for AVX-2/FMA

---

## 7. Areas needing testing / tuning

- **SABRE routing**: implemented in `crates/sf-router/src/sabre.rs` but needs
  more testing on real hardware topologies
- **Dynamical decoupling**: DD passes in `compiler/advanced_passes.py` but not
  yet integrated into the full compile pipeline
- **Algorithm coverage**: Grover, QPE, HHL, Amplitude Estimation are implemented
  in `algorithms/` but need more test coverage and performance tuning
- **Job orchestrator**: `runtime/orchestrator.py` exists but multi-provider
  race/fanout is minimal
- **Plugin ecosystem**: `plugins/__init__.py` has registration scaffolding but
  no third-party plugins exist yet
- **Active space chemistry**: JW/BK + PySCF bridge done, but no integral
  computation or active space selection
- **QFT/MPS at high entanglement**: QFT n=14,18 and RandomUniversal n=20 push
  MPS bond past 64 — Aer-mps wins ~2.5x. Fix needs deeper MPS rewrite (move
  gate-sweep loop entirely into Rust, canonicalise lazily, cache m_matrix
  workspace)

---

## 8. Quick-reference commands

```bash
# Smoke test (~30 s) — run before every commit
python -m pytest tests/test_rust_kernel_correctness.py \
                 tests/test_singularity_routing_and_expval.py \
                 tests/test_stabilizer.py \
                 tests/test_adjoint_grad.py -q

# Full test suite (~7 min, skips cross-framework comparisons)
python -m pytest tests/ -q --ignore=tests/test_algo_comparison.py

# Cross-framework validation tests (requires Qiskit + PennyLane)
python -m pytest tests/test_accuracy_vs_qiskit.py \
                 tests/test_cross_framework_vqe_qaoa.py \
                 tests/test_standalone_features.py -q

# Benchpress latency + memory benchmarks
python -m pytest tests/benchpress/ -q

# Industry benchmark (3 min)
python benchmarks/full_industry_suite.py

# QML benchmark (90 s)
python benchmarks/industry_standard_qml.py

# Rust-only tests
cargo test --lib -p sf-ir -p sf-compiler -p sf-router -p sf-qec -p sf-pulse

# Build Rust extension
cd crates/sf-bindings && maturin develop --release && cd ../..

# Full stack (Docker)
docker compose up
```

---

## 9. Coding conventions

- **Always read before edit** (`Edit` tool requires it)
- **No emojis in bench scripts** (Windows cp1252 console). Reports may use them
- **Heavy unicode** (═ ⟨ ⟩ Δ) breaks `print()` on Windows; use
  `PYTHONIOENCODING=utf-8` or stick to ASCII
- **SF endianness is q0=MSB** everywhere. Qiskit + Rust core are q0=LSB. The
  `to_qiskit` bridge reverses qubit indices; `RustBackend.run` flips axes.
  The Rust QASM parser at `crates/sf-ir/src/qasm.rs:71` reverses:
  `n - 1 - idx`.
- **Disk is tight** (~1-3 GB free). Don't add Rust deps without checking
  `target/` size; use `cargo clean` + temp cleanup if pinch-points
- **Gate names are uppercase internally** in `Circuit` (the constructor
  normalizes them)
- **Python 3.10-3.13** supported; Rust edition 2021
- **Formatting**: black (line-length=100), ruff (E, F, I, N, W, UP), mypy
  (warn_return_any, warn_unused_configs)
- **Tests**: pytest with `testpaths = ["tests"]` and `python_files = ["test_*.py"]`
  from `pyproject.toml`. CI skips tests requiring Qiskit/PennyLane/Aer and the
  Rust kernel correctness test (needs compiled `_sf_core` extension)
- **Git**: fast-iteration codebase — don't auto-commit unless asked.

---

## 10. Endianness reference

```
SF (q0 = MSB):           |q0⟩|q1⟩|q2⟩  →  qubit 0 is the most significant bit
Qiskit (q0 = LSB):       |q2⟩|q1⟩|q0⟩  →  qubit 0 is the least significant bit
Rust core (q0 = LSB):    same as Qiskit

Conversions:
  sf.bridge.from_qiskit()  →  reverses qubit indices (n-1-q)
  sf.bridge.to_qiskit()    →  reverses qubit indices (n-1-q)
  RustBackend.run()        →  flips axes on result
  Rust QASM parser         →  n-1-idx when reading qubit references
```

---

## 11. When the user says "everything", they mean:

1. SF correct at machine epsilon (≤ 1e-12 for dense, ≤ 1e-7 for MPS at bond=64)
2. SF beats Aer + PennyLane on as many cells as possible
3. **No wrappers around Aer or PL** — pure SF wins
4. The win sticks: tests pass, CLAUDE.md / notes are kept current
