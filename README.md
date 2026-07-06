# Superfermion

A high-performance quantum-circuit simulator with **12+ backends**, native
adjoint differentiation, MPS tensor networks scaling to **200+ qubits**,
built-in quantum error correction (11 codes + 4 decoders), multi-framework
ML integration (Flax / PyTorch / TensorFlow), and a Rust SIMD acceleration
core with PyO3 bindings.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://rust-lang.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## What is Superfermion?

Superfermion is a quantum computing framework that combines a Python-native API
with a Rust SIMD acceleration core. It provides:

- **12+ simulator backends** — statevector, Rust (AVX-2/FMA), MPS, stabilizer
  (Aaronson-Gottesman tableau), density matrix, JAX, JAX-MPS, CUDA (cuQuantum),
  CuPy, CuPy-MPS, D-Wave annealer, distributed JAX cluster
- **Singularity auto-router** — automatically dispatches circuits to the
  optimal backend based on qubit count, gate set, and memory constraints
- **Adjoint differentiation** — 1 forward + 1 backward pass regardless of
  parameter count; 15-28x faster than PennyLane Lightning for VQE/QML gradients
  at n=4-10 with 8-20 parameters
- **MPS tensor networks** — Python wrapper around Rust MPS with faer-based QR
  decomposition; lazy SWAP routing; scales to 200+ qubits for low-entanglement
  circuits
- **Stabilizer simulator** — word-packed Aaronson-Gottesman tableau; Clifford
  auto-dispatch from every backend; poly-time simulation to ~1000 qubits
- **Quantum Error Correction** — 11 codes (Repetition, Shor, Steane, Bacon-Shor,
  Surface, Toric, Color, Honeycomb, LDPC, Bivariate Bicycle, GKP) + 4 decoders
  (MWPM, Union-Find, BP+OSD, Neural)
- **Multi-framework ML** — `QuantumLayer` (Flax), `TorchQuantumLayer` (PyTorch),
  `TFQuantumLayer` (TensorFlow); 5 gradient methods (adjoint, parameter-shift,
  SPSA, QNG, Riemannian)
- **Quantum algorithms** — VQE, QAOA, QSVM, QRL, QBM, Grover, QPE, HHL,
  Amplitude Estimation
- **Chemistry module** — Jordan-Wigner + Bravyi-Kitaev transformations, UCCSD
  ansatz, PySCF bridge, molecular Hamiltonian library
- **Hardware compilation** — gate decomposition, rotation merging, SABRE qubit
  routing, Pauli twirling, dynamical decoupling; targets IBM, Rigetti, IonQ, IQM
- **QPU providers** — IBM Quantum, IonQ, AWS Braket, OpenQuantum
- **Cross-framework bridges** — Qiskit, Cirq, PennyLane, Braket, IonQ, OpenQASM 2/3
- **CLI** — 26+ commands (`sf vqe`, `sf qaoa`, `sf qec`, `sf validate`, `sf convert`, …)
- **FastAPI backend** — code execution server, Docker Compose stack
  with PostgreSQL

---

## Installation

```bash
git clone https://github.com/superfermion/superfermion.git
cd superfermion
pip install -e .

# Build the Rust extension (one-time)
cd crates/sf-bindings && maturin develop --release && cd ../..

# CRITICAL: maturin develop writes to .venv/Lib/site-packages/_sf_core/,
# but Python imports from superfermion/_sf_core.pyd. Without this copy,
# Rust changes silently don't take effect:
cp .venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd \
   superfermion/_sf_core.pyd
```

Requirements: Python 3.10–3.13, Rust 1.75+, ~3 GB free disk for the Rust build.

Optional dependency groups:
```bash
pip install -e ".[dev]"        # pytest, ruff, mypy, black
pip install -e ".[gpu]"        # JAX with CUDA 12
pip install -e ".[qpu]"        # IBM + AWS Braket SDKs
pip install -e ".[benchmarks]" # PennyLane, Qiskit Aer, pandas, matplotlib
pip install -e ".[chemistry]"  # PySCF
pip install -e ".[viz]"        # matplotlib
pip install -e ".[all]"        # everything
```

---

## Quick Start

```python
import superfermion as sf

# Bell state
qc = sf.Circuit(2)
qc.h(0)
qc.cx(0, 1)
result = sf.run(qc, backend="rust", shots=1024)
print(result.counts)  # {'00': ~512, '11': ~512}

# Expectation value via the auto-routing backend
sing = sf.get_backend("singularity")
print(sing.expval(qc, "ZZ"))  # 1.0

# VQE-style gradient via adjoint differentiation
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.gradient.adjoint import adjoint_grad_vector

qc = sf.Circuit(4)
names = []
for q in range(4):
    nm = f"t{q}"
    qc.ry(sf.param(nm), q)
    names.append(nm)
for i in range(3):
    qc.cx(i, i + 1)

obs = SparsePauliOp.from_dict({"ZZZZ": 1.0, "XIII": 0.5})
theta = [0.1, 0.2, 0.3, 0.4]
grad = adjoint_grad_vector(qc, obs, names, theta)
```

---

## Architecture

```
Python API (superfermion/)
    |-- Circuit, run(), compile(), param(), train(), Pipeline
    |-- backends/     12+ simulator backends + auto-router
    |-- qml/          templates, gradients (adjoint, param-shift, SPSA, QNG)
    |-- nn/           Flax/PyTorch/TF quantum layers + classical NN building blocks
    |-- algorithms/   VQE, QAOA, QSVM, QRL, QBM, Grover, QPE, HHL, AmplitudeEst
    |-- chemistry/    JW/BK transforms, UCCSD ansatz, PySCF bridge
    |-- qec/          11 codes + 4 decoders
    |-- compiler/     gate decomposition, rotation merge, SABRE routing, twirling
    |-- runtime/      job orchestration, cloud providers (IBM, IonQ, AWS, OpenQuantum)
    |-- pulse/        waveforms, schedules, gate calibration
    |-- bridge/       Qiskit, Cirq, PennyLane, Braket, IonQ, QASM interop
    |-- viz/          matplotlib circuit drawings, Bloch sphere, histograms
    |-- serve/        FastAPI backend
    |-- telemetry/   structured logging, tracing, metrics
    |-- security/    credential management, TLS, tokens
    |-- utils/        exceptions (15-class hierarchy), validation, logging
    |
    +-- _sf_core.pyd  (Rust PyO3 extension)
         |-- QuantumDAG, MPSState, GateSequence, StabilizerTableau
         |-- Compiler, Router, CouplingMap, PulseSchedule
         |-- QECCode, MWPMDecoder, UnionFindDecoder

Rust workspace (crates/)
    |-- sf-ir/        Quantum IR: DAG representation, MPS, stabilizer, QASM parser
    |-- sf-compiler/  Pass manager, gate cancellation, rotation merge, twirl
    |-- sf-router/    SABRE routing, topology (IBM Eagle/Heron, Rigetti, IonQ, IQM)
    |-- sf-pulse/     Waveforms (Gaussian, DRAG, square), schedules, calibration
    |-- sf-qec/       Stabilizer codes, MWPM/UnionFind decoders, syndrome extraction
    |-- sf-bindings/  PyO3 FFI bridge (_sf_core Python extension)

    |-- Landing page, docs (22 pages), dashboard, notebooks (8), CLI terminal
    |-- Code execution via FastAPI backend at localhost:8000
```

### Backend dispatch flow

```
sf.run(circuit, backend=..., shots=N)
    |
    v
Validate parameters -> optional hardware compile -> auto-select backend
    |
    v
Gate fusion (turbo engine) -- SKIP for stabilizer
    |
    v
Dispatch to resolved backend:
    |
    +-- SingularityBackend (auto-router):
    |     n <= 10:      numpy turbo
    |     10 < n <= 32: Rust dense (AVX-2/FMA)
    |     n > 22 + Clifford: stabilizer tableau
    |     n > 32:       MPS (bond dim 64)
    |
    +-- RustBackend:          fusion -> Rust IR -> ping-pong statevector
    +-- StabilizerBackend:    Rust tableau evolution -> sample
    +-- MPSSimulatorBackend:  Rust MPS fast path or Python sweep
    +-- DensityMatrixBackend: Kraus noise + Rust DM path
    +-- JAXBackend:           JAX-native scan-based kernel
    +-- CUSimulatorBackend:   NVIDIA cuQuantum GPU
    +-- CupyBackend:          CuPy GPU
    +-- DistributedJAXBackend: multi-node JAX
    +-- DWaveBackend:         D-Wave quantum annealer
```

---

## Backends

| Backend | Alias(es) | Max Qubits | Best For |
|---|---|---|---|
| `statevector` | | ~25 | Small exact circuits |
| `rust` | | ~32 | Dense simulation with SIMD |
| `jax` | | ~30 | JIT-compiled, GPU-ready |
| `mps` | | 200+ | Low-entanglement, QAOA, VQE |
| `jax_mps` | | 200+ | JAX-based MPS |
| `stabilizer` | | ~1000 | Clifford-only circuits |
| `density_matrix` | | ~12 | Noisy simulation, Kraus ops |
| `cuda` | | ~30 | NVIDIA GPU (cuQuantum) |
| `cupy` | `cupy_sim` | ~30 | CuPy GPU |
| `cuda_mps` | | 200+ | GPU-accelerated MPS |
| `singularity` | `god`, `lightning`, `omnipotent` | auto | Auto-routing master backend |
| `supremacy` | | auto | Supremacy benchmark core |
| `dwave` | | — | D-Wave quantum annealer |
| `cluster` | | auto | Distributed JAX across nodes |

Use `sf.list_backends()` to see available backends and `sf.get_backend(name)` to
get a backend instance.

---

## Benchmarks

Performance measured against Qiskit-Aer 0.17 and PennyLane 0.42 across
industry-standard workloads.

### Speed advantages

| Workload | Best SF backend | vs best non-SF | Speedup |
|---|---|---|---|
| Clifford circuits (any n) | `stabilizer` | aer-stabilizer | 2.3-3.5x |
| QAOA (n=10-40) | `mps` | aer-mps | 5-11x |
| Gradient (n=4-10, 8-20 params) | adjoint | pl.lightning | 15-28x |
| GHZ states (n=10-22) | `mps` | aer-stabilizer | 5-7x |
| Grover search | `stabilizer` / `singularity` | aer-stabilizer | 3-4x |
| Bernstein-Vazirani | `singularity` | aer-stabilizer | 4-7x |
| VQE-H2 (full SGD loop) | `statevector` | pennylane.default | 8.5x |
| RandomUniversal n=20 | `mps` | aer-mps | 2.7x |

### Test results

```
Full test suite:     552 passed
Smoke (core):        113 passed (~30s runtime)
Cross-framework:     78/78 accuracy checks pass
Industry benchmark:  23/23 cells SF fastest
QML benchmark:       13/13 cells SF fastest
Scientific v2:       16/16 accuracy checks pass
Heavy-workload:      25/25 cells SF fastest
HCQ scaling:         29/29 cells SF fastest
```

Full per-cell tables in [`docs/benchmarks.md`](docs/benchmarks.md).

---

## CLI

```bash
sf info                                   # system info
sf version                                # sf + dependency versions
sf validate                               # full installation audit
sf backends                               # list available backends
sf benchmark --qubits 10 --iterations 50  # quick performance sweep
sf run circuit.json --shots 4096          # execute circuit from JSON
sf vqe --hamiltonian H2                   # VQE on H2 / LiH / BeH2 / TFIM
sf qaoa --graph ring6 --p-layers 3        # QAOA MaxCut
sf chemistry H2 --vqe                     # quantum chemistry workflow
sf qec --code steane --error X            # logical qubit lifecycle
sf plugin list                            # list registered plugins
sf auth login --provider ibm --token XXX  # configure provider credentials
sf auth status                            # check credential status
sf convert circuit.json circuit.qasm      # convert between formats
sf estimate circuit.json --backend ibm    # estimate QPU execution cost
sf compare circuit.json --backends statevector,jax  # compare backends
sf jobs list --provider ibm               # manage quantum jobs
sf qpu list --provider ibm                # list available QPUs
```

Full reference: [`docs/cli.md`](docs/cli.md).

---

## Key Modules

### Gradients

| Method | File | Description |
|---|---|---|
| Adjoint | `qml/gradient/adjoint.py` | 1 forward + 1 backward pass; fastest for most QML |
| Parameter-shift | `qml/gradient/parameter_shift.py` | 2 forward passes per param; analytic |
| Finite difference | `qml/gradient/parameter_shift.py` | Centered difference fallback |
| SPSA | `qml/gradient/spsa.py` | Stochastic approximation; noisy-hardware friendly |
| Quantum Natural | `qml/gradient/qng.py` | Fubini-Study metric; faster convergence |
| Riemannian | `qml/gradient/riemannian.py` | Manifold-constrained optimization |

### Machine Learning Layers

```python
# Flax (JAX)
from superfermion.nn.quantum_layer import QuantumLayer
layer = QuantumLayer(n_qubits=4, ansatz="hardware_efficient", backend="singularity")

# PyTorch
from superfermion.nn.torch_layer import TorchQuantumLayer
layer = TorchQuantumLayer(n_qubits=4, n_layers=2, backend="singularity")

# TensorFlow / Keras
from superfermion.nn.tf_layer import TFQuantumLayer
layer = TFQuantumLayer(n_qubits=4, n_layers=2, backend="singularity")
```

### Quantum Error Correction

```python
from superfermion.qec import SurfaceCode, MWPMDecoder, QECManager

code = SurfaceCode(distance=3)
decoder = MWPMDecoder(code)
manager = QECManager(code, decoder)
result = manager.run_logical_cycle()
```

### Cross-Framework Bridge

```python
from superfermion.bridge import from_qiskit, to_qiskit, from_pennylane, to_cirq

sf_circuit = from_qiskit(qiskit_circuit)     # Qiskit -> SF
qiskit_circuit = to_qiskit(sf_circuit)       # SF -> Qiskit
sf_circuit = from_pennylane(pl_tape)         # PennyLane -> SF
cirq_circuit = to_cirq(sf_circuit)           # SF -> Cirq
```

---

## Documentation

| Document | Content |
|---|---|
| [`docs/getting_started.md`](docs/getting_started.md) | First-time install + 5-minute tour |
| [`docs/user_manual.md`](docs/user_manual.md) | Linear read-first reference |
| [`docs/architecture.md`](docs/architecture.md) | Python + Rust module map |
| [`docs/api_reference.md`](docs/api_reference.md) | Full API surface |
| [`docs/backends.md`](docs/backends.md) | Backend selection guide |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Canonical benchmark scoreboard |
| [`docs/cli.md`](docs/cli.md) | CLI reference (26 commands) |
| [`docs/tutorials/`](docs/tutorials/) | 8 runnable examples |
| [`CLAUDE.md`](CLAUDE.md) | Developer context for AI-assisted work |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guide |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
| [`SECURITY.md`](SECURITY.md) | Security policy |

---

## Citing

```bibtex
@misc{superfermion-2026,
  title  = {SuperFermion: a high-performance quantum-circuit simulator
            with native adjoint differentiation},
  author = {SuperFermion Team},
  year   = {2026},
  url    = {https://github.com/superfermion/superfermion}
}
```

## License

[Apache License 2.0](LICENSE).
