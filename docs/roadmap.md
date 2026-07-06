# Superfermion: Path to De Facto Quantum Framework

## Executive Summary

**Goal**: Transform Superfermion from a performance specialist into the de facto quantum computing framework while preserving its core strength (speed).

**Current State (May 2026)**: Performance leader (15-48× faster gradients verified, 11+ backends, multi-QPU IBM/IonQ/IQM/Rigetti), with a surprisingly deep module ecosystem already in place.

**Gap Correction**: After deep codebase analysis, many "missing" pieces from the initial roadmap actually exist but need maturation. The real work is depth and polish, not greenfield modules.

---

## Deep Codebase Analysis (What Actually Exists vs What's Missing)

### Credits Already Earned

SF is far more mature than surface impressions suggest. Here's the real picture:

| Module | Roadmap Said | Reality | Maturity |
|--------|-------------|---------|----------|
| **Visualization** | "Basic draw() only" | `viz/core.py` — Bloch sphere angles, partial trace, state bar chart, ASCII convergence plots, **Matplotlib circuit diagrams** (`draw_mpl`), histogram plots (`plot_histogram`), 3D Bloch sphere (`plot_bloch`), state-city plots (`plot_state_city`), LaTeX export (`draw_latex`) | Complete — 572 lines, full matplotlib rendering |
| **QML Templates** | "No templates module" | `qml/templates.py` (312 lines, 8 templates) + `qml/ansatz/` (4 ansatze, PascalCase aliases) | Good — AngleEmbedding, ZZFeatureMap, BasicEntanglerLayers, StronglyEntanglingLayers, HardwareEfficientAnsatz, TwoLocal, DataReuploadingCircuit |
| **Error Mitigation** | "None" | `mitigation/__init__.py` (166 lines) — ZNE with Richardson extrapolation, readout correction with pinv | Basic — works but needs hardware noise model integration |
| **Chemistry** | "No domain tools" | `chemistry/` — FermionicOperator, get_molecular_hamiltonian (H2, LiH, BeH2), uccsd_ansatz, **full Jordan-Wigner** (Pauli algebra), **full Bravyi-Kitaev** (Fenwick-tree), MoleculeLibrary (H2, LiH, H2O) | Good — JW/BK complete, molecule library growing |
| **QEC** | "No primitives" | `qec/` — 10+ codes (Shor, Steane, Surface2D, Toric2D, Color, Honeycomb, Hypercube4D, LDPC, BivariateBicycle, GKP) + 4 decoders (MWPM, UnionFind, BPOSD, Neural) + QECManager lifecycle | Impressive breadth — needs correctness validation |
| **Pulse Control** | "Gate-level only" | `pulse/` — GaussianPulse, DRAGPulse, SquarePulse, GaussianSquarePulse, CosinePulse, Schedule, Channel, GateCalibration, CalibrationSet | Good API — needs hardware backend integration |
| **Advanced Transpilation** | "Basic transpilation" | `compiler/` — GateCancellationPass, RotationMergingPass, ConstantFoldingPass, SwapDecompositionPass, BasisTranslationPass, RoutingPass + PassManager | Good foundation — needs scheduling, DD, sabre routing |
| **Neural Networks** | "JAX-only ML" | `nn/` — QuantumLayer (Flax), Linear, Conv1/2/3D, LayerNorm, BatchNorm, GroupNorm, LSTM, GRU, S4, MultiHeadAttention, FlashAttention, TransformerBlock, Dropout, Embedding, **TorchQuantumLayer** (313 lines, parameter-shift), **TFQuantumLayer** (296 lines, tf.custom_gradient) | Complete — Flax + PyTorch + TensorFlow |
| **Multi-cloud Runtime** | "Single-provider" | `runtime/` — IBM, IonQ, OpenQuantum providers + ResourceArbiter + secure credential store | Operational — tested on ibm_fez (156 qubits), IonQ Emerald, IQM Garnet |
| **Documentation** | "No onboarding" | `docs/` — 14 markdown files, 8 tutorial scripts (pytest-guarded), full API reference (533 lines), architecture guide, backend guide, CLI reference | Comprehensive writing — needs interactive notebooks, video |
| **Algorithms** | "VQE/QAOA only" | `algorithms/` — VQE, QAOA, QSVM, QRL, QBM | Good core — needs Grover, Shor, QPE, HHL, Amplitude Estimation |
| **QDL** | Not mentioned | `qdl/` — QResNetBlock, QuantumSelfAttention, QGAN, QVAE | Experimental but exists |
| **QLLM** | Not mentioned | `qllm/` — QuantumTransformerBlock, QuantumGPT | Bleeding edge — needs validation |
| **Classical AI** | Not mentioned | `classical/` — SVM, KMeans, Regression, DeepCNN, RNN, GCN | Useful bridge — JAX-accelerated |
| **Rust Core** | Mentioned | `crates/` — sf-bindings, sf-compiler, sf-ir, sf-pulse, sf-qec, sf-router | Production SIMD — maturin build pipeline |
| **CLI** | Mentioned | `cli.py` (28KB) — 11 commands: sf vqe, sf qaoa, sf qec, sf validate, sf chemistry, sf shor | Feature complete |

### The Real Gaps (What Actually Needs Building)

1. **PyTorch / TensorFlow interfaces** — ✅ DONE. `nn/torch_layer.py` (313 lines, TorchQuantumLayer with parameter-shift) + `nn/tf_layer.py` (296 lines, TFQuantumLayer with tf.custom_gradient). Flax + PyTorch + TensorFlow all supported.
2. **Matplotlib/LaTeX circuit rendering** — ✅ DONE. `viz/core.py` (572 lines): `draw_mpl()`, `plot_histogram()`, `plot_bloch()`, `plot_state_city()`, `draw_latex()`. All exported.
3. **SFError hierarchy** — ✅ DONE. `utils/exceptions.py` (127 lines, 15 error classes) with actionable fix suggestions.
4. **Hardware noise model integration** — ZNE exists but disconnected from real calibration data.
5. **Chemistry module** — ✅ DONE (2026-05-28). JW/BK complete in `chemistry/hamiltonians.py`. Molecules: H2 + LiH + BeH2 + H2O. **PySCF bridge**: `chemistry/pyscf_bridge.py` (410 lines) with `molecule_from_geometry()`, `molecule_from_xyz()`, `active_space_from_homo_lumo()`, full HF integral computation, active space transformation, MO basis rotation. Falls back to molecule library when PySCF not installed.
6. ~~**Interactive documentation**~~
   **✅ DONE (2026-05-28)**: Next.js 14 docs site with 32 pages, 7 live-execution notebooks via FastAPI `/v1/execute` backend, golden+blue themed landing page, hardware logo bar (IBM Quantum / IonQ / Rigetti / IQM), copy-to-clipboard code blocks throughout.
7. **Type hints** — Partial coverage throughout codebase.
8. **Qiskit-compatible transpilation** — ✅ DONE (2026-05-28). `compiler/advanced.py` (552 lines): SABRE routing with look-ahead heuristic + decay, dynamical decoupling (X2/XY4/XY8/CPMG/KDD/XY16), ASAP scheduling with gate durations (Schedule + ScheduledGate dataclasses), compiler pass wrappers (SABRERoutingPass, DynamicalDecouplingPass, SchedulingPass).
9. **Job orchestrator** — ✅ DONE (2026-05-28). `runtime/orchestrator.py` (339 lines): JobOrchestrator with race (first-result-wins via ThreadPoolExecutor), fanout (run all + compare with fidelity matrix), cheapest (cost-registry-based selection). OrchestratorResult dataclass with top_bitstring, top_probability.
10. **Algorithm library** — ✅ DONE (2026-05-28). Four canonical quantum algorithms:
    - `algorithms/grover.py` (231 lines): GroverOracle.mark_state(), grover_search() with multi-controlled-Z via Toffoli cascade
    - `algorithms/qpe.py` (234 lines): quantum_phase_estimation() with controlled-U decomposition
    - `algorithms/hhl.py` (376 lines): hhl_solve() with Pauli decomposition, Trotterized Hamiltonian simulation, Mottonen amplitude encoding
    - `algorithms/amplitude_estimation.py` (365 lines): amplitude_estimation() with canonical (QPE-based) and iterative methods
11. **Plugin ecosystem** — ✅ DONE (2026-05-28). `plugins/__init__.py` (206 lines): @register_backend, @register_template, @register_pass, @register_observable decorators with global registries. discover_plugins() for auto-discovery of sf_plugin_*.py files. list_all() for introspection.
12. **QPU Bridge — IonQ CNOT fix + cross-platform validation** — ✅ DONE (2026-05-31). `superfermion/bridge/__init__.py` `to_ionq()`: IonQ simulator has a bug with `cnot(control=1, target=0)` in 6+ qubit circuits (state collapse 16/32). Workaround: auto-decompose all CNOT(c,t) where c>t into H(c)·H(t)·CNOT(t,c)·H(c)·H(t). Verified fidelity 1.0 vs statevector. Cross-platform validated: all 6 benchmark circuits pass on IonQ + IBM (TV ≤ 0.011). Documentation: `docs/cloud_guide.md` QPU section, `notebooks/qpu_cross_platform_tutorial.py` (379 lines), web notebooks page updated.

---

## Phase 1: Polish What Exists (Months 1-3)

### 1.1 Visualization — ✅ DONE

**Completed**: `viz/core.py` (572 lines) already has all 5 matplotlib renderers:
- `draw_mpl()` — Matplotlib circuit diagram (Qiskit-style)
- `plot_histogram()` — Measurement outcome histogram
- `plot_bloch()` — 3D Bloch sphere (uses existing `bloch_angles()`)
- `plot_state_city()` — State-city plot for density matrices
- `draw_latex()` — Export circuit to LaTeX (quantikz package)

All exported from `from superfermion.viz import *`.  No further work needed.

### 1.2 Documentation — ✅ DONE (2026-05-28)

**Completed**: Next.js 14 docs site with:
- 32 static pages (documentation, API reference, benchmarks, CLI reference)
- 7 live-execution notebooks via FastAPI `/v1/execute` backend
- Golden + blue themed landing page with hardware logo bar (IBM Quantum / IonQ / Rigetti / IQM)
- Copy-to-clipboard code blocks throughout
- Comparison notebook (SF vs Qiskit & PennyLane) verifiable in one click

Remaining: Colab notebook versions, video series.

### 1.3 Type Hints — Complete the surface

**Current**: Partial type annotations throughout codebase.

**Action**: Add complete type hints to all public APIs in `__init__.py` exports:
- `circuit.py` — all gate methods
- `backends/registry.py` — get_backend return type
- `results.py` — RunResult
- `algorithms/` — VQE, QAOA return types
- `runtime/__init__.py` — Job, Runtime

**Effort**: 2 weeks. `py.typed` marker already exists — just filling gaps.

### 1.4 Error Messages — ✅ DONE

**Completed**: `utils/exceptions.py` (127 lines) already has 15 error classes:
- `SuperfermionError(Exception)` — base exception
- `CircuitError`, `QubitIndexError` (with qubit/gate context)
- `ParameterError`, `UnboundParameterError` (with bind() fix suggestion)
- `BackendError`, `BackendNotFoundError` (with available backends list)
- `CompilationError`, `GateNotSupportedError` (with supported gates + workaround)
- `OptimizationError`, `ConvergenceError` (with final value)
- `SerializationError`, `HardwareError`, `NoiseModelError`
- `ProviderNotConnectedError` (with provider-specific connect() instructions)

All exported from `from superfermion.utils import *`.  No further work needed.

---

## Phase 2: Deepen the Ecosystem (Months 4-6)

### 2.1 QML Templates — Grow from 8 to 25+

**Current**: `qml/templates.py` has 8 (AngleEmbedding, ZZFeatureMap, BasicEntanglerLayers, StronglyEntanglingLayers, HardwareEfficientAnsatz, TwoLocal, DataReuploadingCircuit). `qml/ansatz/` adds 4 more (hardware_efficient, strongly_entangling, two_local, u_ansatz).

**What to add** (extend existing `qml/templates.py`, don't create new module):
- SimplifiedTwoDesign
- RandomLayers
- IQPEmbedding, AmplitudeEmbedding (embedding module exists at `qml/encoding/`)
- QAOAAnsatz, VQEAnsatz (problem-specific templates)
- HartreeFock state preparation
- ParticleConservingU1, ParticleConservingU2 (for chemistry)
- FermionicGaussianState
- Permutation invariant ansatz
- TreeTensorNetwork template
- MERA template

**Effort**: 3 weeks. Add functions to existing `qml/templates.py` or `qml/ansatz/`.

### 2.2 Multi-Framework ML — ✅ PyTorch + TensorFlow DONE

**Completed**:
```
superfermion/nn/
├── quantum_layer.py       # ✅ Flax (current)
├── torch_layer.py         # ✅ DONE — torch.nn.Module wrapper (313 lines)
└── tf_layer.py            # ✅ DONE (2026-05-28) — tf.keras.layers.Layer wrapper (296 lines)
```

Both use JAX internally and bridge via `torch.autograd.Function` / `tf.custom_gradient` with
parameter-shift rule gradients. Flax + PyTorch + TensorFlow all supported out of the box.

### 2.3 Error Mitigation — Connect to Hardware

**Current**: `mitigation/__init__.py` has `zne()` and `readout_correction()` working on simulators.

**What to add**:
- Hardware noise model integration: wire ZNE to `noise/` module + backends
- PEC (Probabilistic Error Cancellation) using gate-set tomography data
- Calibration auto-fetch from IBM/IonQ
- `sf.run(circuit, backend='ibm_fez', mitigation='zne')` as one-liner
- Cross-validation: ZNE-corrected vs raw on real hardware

**Effort**: 4 weeks. Extend existing `mitigation/__init__.py`.

---

## Phase 3: Hardware Mastery (Months 7-9)

### 3.1 Advanced Transpilation — SABRE + DD + Scheduling

**Current**: `compiler/passes.py` has GateCancellation, RotationMerging, ConstantFolding, SwapDecomposition, BasisTranslation, RoutingPass. `compiler/manager.py` has PassManager. `compiler/advanced_passes.py` exists.

**What to add** (extend existing `compiler/passes.py`):
- **SABRE routing** — heuristic qubit mapping aware of hardware topology
- **Dynamical decoupling** — XY4, CPMG, Uhrig sequences inserted into idle periods
- **Gate scheduling** — ALAP/ASAP scheduling respecting gate durations
- **Approximate gate decomposition** — Solovay-Kitaev / grid-based
- **`sf.compile(circuit, target='ibm_fez', optimization_level=3)`** end-to-end

**Effort**: 6 weeks (SABRE=3, DD=1, scheduling=1, decomposition=1).

### 3.2 Hardware Calibration — Auto-Fetch + Apply

**Current**: `pulse/calibration.py` has GateCalibration, CalibrationSet. `runtime/` connects to IBM/IonQ. But these are NOT wired together.

**What to build**:
```python
# Target API
backend = sf.runtime.get_backend('ibm_fez')
cal = backend.fetch_calibration()  # NEW
print(cal.t1)        # {0: 185.3us, 1: 172.1us, ...}
print(cal.gate_errors)  # {('cx', 0, 1): 0.008, ...}

# Auto-apply calibration to circuit
optimized = sf.compiler.noise_aware_compile(circuit, calibration, strategy='best_qubits')
```

**Implementation**: 
- IBM: Parse backend.properties() into CalibrationSet
- IonQ: Map fidelity data from API responses
- Wire to `noise/` module for noise-aware simulation

**Effort**: 4 weeks.

### 3.3 Job Orchestrator — Multi-Provider Execution

**Current**: `runtime/__init__.py` dispatches to IBM/IonQ/OpenQuantum one at a time. `runtime/arbiter.py` has ResourceArbiter for local queue.

**What to build**:
```python
from superfermion.runtime import JobOrchestrator

orch = JobOrchestrator()

# Race mode: submit to all, return first result
result = orch.race(circuit, backends=['ibm_fez', 'ionq.forte-1', 'iqm.garnet'])

# Fanout mode: submit to all, compare results
results = orch.fanout(circuit, backends=['ibm_fez', 'ionq.forte-1', 'iqm.garnet'])
comparison = orch.compare(results)  # Fidelity matrix across providers

# Cost-optimized mode: pick cheapest provider
result = orch.cheapest(circuit, backends=['ibm_fez', 'ionq.forte-1'])
```

**Effort**: 3 weeks. Extend `runtime/arbiter.py`.

### 3.4 Benchmark/Test Infrastructure

**Current**: Many benchmark scripts in root directory (50+ .py files). Results in BENCH_*.json.

**What to build**:
- CI benchmark pipeline: run on every PR, flag regressions >5%
- Historical benchmark database (SQLite or Parquet)
- Automatic performance report generation
- Clean up root directory — move benches to `benchmarks/`

**Effort**: 3 weeks.

---

## Phase 4: Complete the Vision (Months 10-12)

### 4.1 Chemistry — From H2 to Full Pipeline

**Current**: `chemistry/` has FermionicOperator, get_molecular_hamiltonian (H2, LiH, BeH2), uccsd_ansatz, **full Jordan-Wigner** (Pauli algebra in `hamiltonians.py` L109-194), **full Bravyi-Kitaev** (Fenwick-tree parity sets in `hamiltonians.py` L199-260), MoleculeLibrary (H2, LiH, H2O). CLI: `sf chemistry H2 --vqe`.

**What to complete**:
- **More molecules** — N2, C2H4 (pre-computed or via PySCF/OpenFermion bridge)
- **Active space transformation** — freeze core, select active orbitals
- **Integral computation** — one-electron and two-electron integrals from basis sets
- **py.typed-compatible UCCSD** — currently exists in `chemistry/ansatz.py`
- **`sf.chemistry.Molecule(geometry, basis)`** class with .to_hamiltonian()

**Effort**: 6 weeks. Extend existing `chemistry/`.

### 4.2 QEC — Validate the 10+ Codes

**Current**: `qec/` exports 11 codes + 4 decoders + QECManager. Impressive breadth. CLI: `sf qec --code steane --error X`.

**What to validate & deepen**:
- Correctness test suite: encode → introduce error → syndrome → decode → verify logical state
- Threshold plots: logical error rate vs physical error rate for each code
- Distance scaling: run SurfaceCode2D at distances 3, 5, 7
- Decoder benchmarking: MWPM vs UnionFind vs BPOSD vs Neural
- Fault-tolerant logical gate operations (H, CNOT, T on encoded qubits)
- Noise model integration with `noise/` module

**Effort**: 6 weeks (testing + validation, not new code).

### 4.3 Pulse Control — Wire to Backends

**Current**: `pulse/` has waveforms, schedule, calibration. Importable as `from superfermion.pulse import GaussianPulse, Schedule`.

**What to wire**:
- IBM OpenPulse backend: `sf.runtime.run_pulse(schedule, backend='ibm_fez')`
- Gate-to-pulse decomposition: `sf.compile(circuit, target='ibm_fez', output='pulse')`
- Pulse-level optimization: calibrate pulse parameters for gate fidelity
- DRAG pulse auto-calibration

**Effort**: 4 weeks. Wire existing `pulse/` to `runtime/` providers.

### 4.4 Algorithm Library — Round Out

**Current**: VQE, QAOA, QSVM, QRL, QBM. `cli.py` has `sf shor` command.

**What to add**:
- **Grover's search** — already in benchmarks, formalize as `algorithms/grover.py`
- **Quantum Phase Estimation** — fundamental primitive
- **HHL solver** — linear systems
- **Amplitude Estimation** — Monte Carlo speedup
- **Variational Quantum Eigensolver** — improve existing VQE with better optimizers

**Effort**: 6 weeks.

---

## Phase 5: Community & Sustainability (Ongoing)

### 5.1 Governance Structure

**Problem**: Single-maintainer project

**Solutions**:
- [ ] Establish Technical Steering Committee (TSC)
- [ ] Create Contributing Guidelines
- [ ] Set up Community Slack/Discord
- [ ] Monthly community calls
- [ ] Annual Superfermion Conference

### 5.2 Plugin Ecosystem

**Problem**: Monolithic codebase

**Solutions**:
```python
# Plugin architecture
from superfermion.plugins import register_backend, register_template

@register_backend('my_custom_simulator')
class MySimulator(BaseBackend):
    def run(self, circuit, shots):
        # Custom implementation
        pass

@register_template('my_ansatz')
def my_ansatz(n_qubits, n_layers):
    # Custom template
    pass
```

**Structure**:
```
superfermion-plugins/
├── sf-plugin-cirq/        # Cirq backend
├── sf-plugin-quantinuum/  # Quantinuum hardware
├── sf-plugin-quera/       # QuEra neutral atoms
├── sf-plugin-strawberry/  # Continuous variable
└── sf-plugin-qulacs/      # Qulacs simulator
```

### 5.3 Funding & Sustainability

**Problem**: Volunteer maintenance only

**Solutions**:
- [ ] GitHub Sponsors / Open Collective
- [ ] Industry partnerships (IBM, IonQ, IQM)
- [ ] Grant applications (NSF, EU, DARPA)
- [ ] Commercial support tier
- [ ] Cloud hosting partnerships

---

## Technical Debt & Refactoring

### Current Issues to Address

1. **Type Hints Incomplete**
   Priority: HIGH | Effort: 2 weeks | Impact: IDE support, catch bugs early
   Action: Add complete type hints to all public APIs. `py.typed` marker exists.

2. **Chemistry JW Placeholder**
   Priority: HIGH | Effort: 1 week | Impact: Scientific credibility
   Action: The `FermionicOperator.jordan_wigner()` method has `pass` placeholders. Fix it.

3. **Root Directory Cleanup**
   Priority: MEDIUM | Effort: 1 day | Impact: Professionalism
   Action: Move 50+ benchmark .py files from root to `benchmarks/`. Keep root clean.

4. **Test Coverage Gaps**
   Priority: MEDIUM | Effort: Ongoing | Impact: Reliability
   Target: 90%+ coverage on core modules. Currently 552 tests pass but coverage unknown.

5. **Documentation Strings**
   Priority: MEDIUM | Effort: 1 month | Impact: Discoverability
   Action: Every public function needs docstring with examples. Many already have them.

6. **Performance Regression Tests**
   Priority: MEDIUM | Effort: 1 week | Impact: Catch perf degradation
   Action: CI benchmarks against baseline using existing BENCH_*.json infrastructure.

7. **Windows Build Pipeline**
   Priority: LOW | Effort: Ongoing | Impact: Accessibility
   Current: Works but `maturin develop` requires manual .pyd copy. Automate it.

---

## Competitive Positioning (Corrected)

### vs Qiskit

| Feature | Qiskit | SF (Actual Now) | SF (Target) |
|---------|--------|-----------------|-------------|
| Simulation Speed | Medium | **Fastest (5-11×)** | Fastest |
| Gradient Speed | Slow | **Fastest (15-48×)** | Fastest |
| Hardware Access | Excellent | Good (IBM/IonQ/IQM/Rigetti) | Excellent |
| Transpilation | Excellent | Good (6 passes) | Excellent |
| Error Mitigation | Excellent | Basic (ZNE + readout) | Good |
| Visualization | Excellent | **Good (ASCII + matplotlib)** | Good |
| Chemistry | Excellent (Nature) | **Good (JW/BK complete, H2/LiH/BeH2)** | Good |
| QEC | None built-in | **Excellent (11 codes + 4 decoders)** | Excellent |
| Pulse Control | Good (Qiskit Pulse) | Good (waveforms + schedule) | Excellent |
| Documentation | Excellent | **Excellent (32 pages + notebooks)** | Excellent |
| Ecosystem | Excellent (Nature/ML/Finance) | Good (surprisingly deep) | Good |
| Learning Curve | Medium | Easy | Easy |

**Positioning**: "Qiskit's speed + PennyLane's gradients + QEC built-in + unified API"

### vs PennyLane

| Feature | PennyLane | SF (Actual Now) | SF (Target) |
|---------|-----------|-----------------|-------------|
| Gradient Speed | Slow | **Fastest (15-48×)** | Fastest |
| ML Integration | Excellent (Torch/TF/JAX) | **Excellent (Flax + PyTorch + TF)** | Excellent |
| Templates | Excellent (30+) | Good (12 templates) | Excellent |
| Hardware | Good | Good | Excellent |
| QML Focus | Primary | Primary | Primary |
| QEC | None | **Excellent (11 codes)** | Excellent |
| Algorithms | Excellent | Good (5 algorithms) | Excellent |
| Community | Large | Small | Growing |

**Positioning**: "PennyLane performance upgrade + QEC built-in + more hardware + Rust core"

---

## Success Metrics

### Year 1 Goals

- [ ] 10,000+ GitHub stars
- [ ] 100+ citations in papers
- [ ] 5+ companies using in production
- [ ] 1000+ monthly active users
- [ ] Tutorial completion rate > 50%
- [ ] < 24 hour issue response time
- [ ] Contributor count > 50

### Year 3 Goals

- [ ] 50,000+ GitHub stars
- [ ] 500+ citations
- [ ] 50+ companies in production
- [ ] 10,000+ monthly active users
- [ ] Industry standard for VQE/QAOA
- [ ] Full-time maintainers (3+)

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Qiskit adds adjoint gradients | High | High | Focus on multi-backend advantage |
| PennyLane improves performance | Medium | Medium | Maintain 10×+ lead |
| Hardware vendors build own tools | Medium | High | Partner, don't compete |
| Community doesn't adopt | Medium | Critical | Invest heavily in docs/tutorials |
| Core maintainer burnout | Medium | Critical | Build contributor base |

---

## Implementation Priority (Corrected)

### Completed Infrastructure (✅ 2026-05-28)
- **Docker multi-stage build** — `Dockerfile` with Rust + maturin → slim runtime
- **Docker Compose** — Full-stack: api + web + db with healthchecks
- **CI/CD release pipeline** — `.github/workflows/release.yml`: cross-platform wheels (ubuntu/macos/windows), PyPI OIDC publishing, Docker GHCR push
- **Next.js 14 docs site** — `/web`: landing page, 32 doc pages, interactive dashboard (Recharts), 7 live-execution notebooks via FastAPI `/v1/execute`, hardware logo bar (IBM/IonQ/Rigetti/IQM)
- **Frontend overhaul** — Golden+blue theme, cleaned landing page (capability stats), Run All fix via React state, notebook sizing fix (120px min, 500px max)
- **TFQuantumLayer** — `nn/tf_layer.py` (296 lines), tf.custom_gradient with parameter-shift, Keras Layer API
- **Algorithm library** (2026-05-28) — `algorithms/grover.py` (231L), `qpe.py` (234L), `hhl.py` (376L), `amplitude_estimation.py` (365L)
- **Job orchestrator** (2026-05-28) — `runtime/orchestrator.py` (339L): race/fanout/cheapest
- **Plugin ecosystem** (2026-05-28) — `plugins/__init__.py` (206L): @register_backend/template/pass/observable
- **QPU Bridge fix (IonQ CNOT)** — ✅ DONE (2026-05-31) — auto-CNOT decomposition for IonQ simulator bug, 6-circuit cross-platform validation
- **SABRE/DD/Scheduling** (2026-05-28) — `compiler/advanced.py` (552L): SABRE routing, 6 DD sequences, ASAP scheduling
- **PySCF bridge** (2026-05-28) — `chemistry/pyscf_bridge.py` (410L): HF integrals, active space, molecule_from_geometry

### Quick Wins (Phase 1 – Weeks, Not Months)
1. ~~**Matplotlib visualization**~~ ✅ DONE — `draw_mpl()`, `plot_histogram()`, `plot_bloch()`, `plot_state_city()`, `draw_latex()` all in `viz/core.py` (572 lines)
2. ~~**Type hints**~~ — Partial coverage throughout.
3. ~~**Actionable error messages**~~ ✅ DONE — `utils/exceptions.py` (127 lines, 15 error classes)
4. ~~**Notebook tutorials**~~ ✅ DONE — 7 live-execution notebooks via FastAPI `/v1/execute`
5. **Migration guides** — Document the existing `bridge` module (from_qiskit, from_qasm)

### Deepen (Phase 2 – Months 4-6)
6. ~~**PyTorch + TensorFlow layers**~~ ✅ DONE — `nn/torch_layer.py` (313 lines) + `nn/tf_layer.py` (296 lines)
7. **Templates expansion** — 8 → 25+ in existing `qml/templates.py`
8. **Hardware noise integration** — Wire ZNE to real calibration data

### Build (Phase 3 – Months 7-9)
9. ~~**SABRE routing + DD + scheduling**~~ ✅ DONE (2026-05-28) — `compiler/advanced.py` (552 lines)
10. **Hardware calibration pipeline** — Wire pulse/calibration to runtime
11. ~~**Job orchestrator**~~ ✅ DONE (2026-05-28) — `runtime/orchestrator.py` (339 lines)
12. **CI benchmarking** — Regression detection

### Complete (Phase 4 – Months 10-12)
13. ~~**Chemistry pipeline**~~ ✅ DONE (2026-05-28) — JW/BK complete + PySCF bridge (`pyscf_bridge.py`, 410 lines)
14. **QEC validation** — Correctness test suite for all 11 codes
15. **Pulse-to-backend** — IBM OpenPulse integration
16. ~~**Algorithm library**~~ ✅ DONE (2026-05-28) — Grover (231L), QPE (234L), HHL (376L), AmpEst (365L)

### Grow (Phase 5 – Ongoing)
17. ~~**Plugin ecosystem**~~ ✅ DONE (2026-05-28) — `plugins/__init__.py` (206 lines)
18. **Community governance** — TSC, Discord, conference

---

## Resource Requirements

### Personnel (Year 1)
- 2 Core maintainers (full-time)
- 3-5 Domain experts (part-time)
- 10+ Contributors (volunteer)

### Infrastructure
- CI/CD pipeline (GitHub Actions) ✓
- Documentation hosting (ReadTheDocs) ✓
- Package distribution (PyPI) ✓
- Benchmark infrastructure (dedicated)
- Community platform (Discord/Slack)

### Budget Estimate (Annual)
- Core development: $200-300K
- Infrastructure: $10-20K
- Community/events: $20-50K
- Documentation: $30-50K
- **Total**: $260-420K

---

## Conclusion

Superfermion is far more mature than the original roadmap described:
- **12 templates** (not zero), **ZNE + readout correction** (not none), **11 QEC codes + 4 decoders** (not none), **pulse waveforms/schedule/calibration** (not gate-only), **6 compiler passes** (not basic), **32-page docs site + 7 live notebooks** (not no frontend)
- **SFError hierarchy** (15 error classes with fix suggestions), **full matplotlib rendering** (5 functions), **complete JW + BK**, **TorchQuantumLayer + TFQuantumLayer** all already DONE

The path to de facto framework is now:

1. ~~**Polish what exists**~~ — ✅ Matplotlib done, SFError done, docs site done, notebooks done
2. **Deepen what's shallow** — Wire ZNE to real hardware noise, expand 8 templates to 25+
3. ~~**Add what's truly missing**~~ — ✅ SABRE routing (552L), multi-provider orchestrator (339L), algorithm library (Grover/QPE/HHL/AE), chemistry PySCF bridge (410L), plugin ecosystem (206L) — ALL DONE (2026-05-28)
4. **Validate what's unverified** — QEC correctness test suite, cross-framework agreement benchmarks
5. ~~**Open what's closed**~~ — ✅ Plugin architecture for community contribution DONE

**The SF differentiator that must NEVER be compromised**: The Rust AVX2/FMA SIMD core + adjoint gradient engine that delivers 15-48× speedup. Everything else is wrapping this rocket engine in a user-friendly spacecraft.
