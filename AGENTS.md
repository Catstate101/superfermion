# AGENTS.md — AI Coding Agent Instructions

> **What this is**: project-level instructions for AI coding agents working on
> the Superfermion codebase. Read this before making any changes.

---

## Project Overview

Superfermion is a hardware-agnostic quantum computation framework with a Python
API and a Rust SIMD acceleration core. Think PyTorch for quantum circuits:
differentiable, compilable to any hardware, trainable end-to-end.

- **Python package** (`superfermion/`) — public API, 12+ simulation backends,
  quantum algorithms, QML gradient methods, neural network layers, QEC, chemistry,
  compiler, device adapters, experiment tracking
- **Rust workspace** (`crates/`, 6 crates) — SIMD statevector engine, MPS tensor
  network, stabilizer tableau, compiler passes, SABRE router, pulse-level control,
  QEC codes/decoders, PyO3 bindings
- **Tests** (`tests/`) — pytest suite organized as `unit/`, `integration/`,
  `backends/`, `domain/`
- **Benchmarks** (`benchmarks/`) — industry comparison scripts
- **Docs** (`docs/`) — user + developer documentation

## Critical Rules

### 1. Rust-Python Bridge
After any Rust change, you MUST copy the built extension:
```bash
# Linux
cp .venv/lib/python3.*/site-packages/_sf_core/_sf_core*.so superfermion/

# Windows
cp .venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd superfermion/_sf_core.pyd
```
`maturin develop --release` alone is NOT enough — Python imports the file from
`superfermion/`, not from `.venv/`.

### 2. Endianness
Superfermion uses **q0=MSB** (big-endian). Qiskit and the Rust core use
**q0=LSB** (little-endian). All bridge functions must reverse qubit indices:
`n - 1 - q`. See `superfermion/bridge/__init__.py` for the conversion functions.

### 3. Testing Before Changes
Run the test suite before committing:
```bash
# Quick smoke test
python -m pytest tests/unit/ -q --timeout=60

# Full suite
python -m pytest tests/ -q --timeout=120

# Rust-only tests
cargo test --lib -p sf-ir -p sf-compiler -p sf-router -p sf-qec -p sf-pulse
```

### 4. No Auto-Commit
Do not create git commits unless explicitly asked.

### 5. Read Before Edit
Always read a file before editing it.

---

## Development Workflow

### Setup
```bash
pip install -e ".[dev]"
cd crates/sf-bindings && maturin develop --release && cd ../..
# Copy the built extension (see Critical Rules above)
```

### Running Tests
```bash
# Unit tests (~30s)
python -m pytest tests/unit/ -q

# Integration tests
python -m pytest tests/integration/ -q

# Backend correctness tests
python -m pytest tests/backends/ -q

# Domain tests (algorithms, QEC, gradients, compiler, bridges)
python -m pytest tests/domain/ -q

# Full suite
python -m pytest tests/ -q --timeout=120

# Rust-only tests
cargo test --lib -p sf-ir -p sf-compiler -p sf-router -p sf-qec -p sf-pulse
```

### Code Quality
```bash
ruff check superfermion/ tests/
black --check superfermion/ tests/
mypy superfermion/
```

---

## Architecture Rules

### Adding a New Backend
1. Create `superfermion/backends/your_backend.py` extending `Backend` (from `base.py`)
2. Implement `run(circuit, shots)` and properties `n_qubits`, `supported_gates`
3. Register in `superfermion/backends/factory.py` via `register_backend()`
4. Add Clifford auto-dispatch using `maybe_clifford_dispatch()` if applicable
5. Add tests in `tests/backends/`
6. Document in `docs/backends.md`

### Adding a New Device Adapter
1. Create `superfermion/devices/your_device.py` implementing `DeviceExecutor` protocol
2. The protocol requires `execute(circuit, shots, **kwargs) -> RunResult` and `capabilities() -> DeviceCapabilities`
3. Add tests in `tests/unit/test_devices.py` and `tests/integration/test_provider_adapters.py`

### Adding a New Gate to Circuit
1. Add a method to `Circuit` in `superfermion/circuit.py`
2. Add the gate name mapping in `GateRecord.to_unitary()`
3. Add to `OpType` enum in `crates/sf-ir/src/ops.rs`
4. Add the unitary matrix in `ops.rs:to_unitary_matrix()`
5. Add the gate parsing in `crates/sf-bindings/src/lib.rs:parse_gate()`
6. Add a simulation path in `crates/sf-ir/src/dag.rs:simulate()` if needed
7. Add the gate name to the Turbo decomposition logic in `backends/turbo.py`
8. Add tests: single-gate correctness, multi-gate interaction

---

## Key Files Quick Reference

| Task | Read These Files First |
|---|---|
| Add a backend | `backends/base.py`, `backends/factory.py`, `backends/singularity.py` |
| Add a device adapter | `devices/__init__.py`, `devices/local.py`, `devices/ibm.py` |
| Add a gate | `circuit.py`, `crates/sf-ir/src/ops.rs`, `crates/sf-bindings/src/lib.rs` |
| Fix a simulation bug | `backends/turbo.py`, `crates/sf-ir/src/dag.rs` |
| Add a QEC code | `qec/codes/`, `crates/sf-qec/src/codes.rs` |
| Add a gradient method | `qml/gradient/adjoint.py`, `qml/gradient/core.py` |
| Add an algorithm | `algorithms/variational.py`, `algorithms/grover.py` |
| Change compilation | `compiler/manager.py`, `crates/sf-compiler/src/lib.rs` |
| Change execution flow | `runner.py`, `devices/__init__.py` |
| Add framework bridge | `bridge/__init__.py` |
| Change experiment tracking | `experiment/protocols.py`, `experiment/context.py` |

---

## Conventions

### Python
- Line length: 100 (black, ruff)
- Target: Python 3.10
- Ruff rules: E, F, I, N, W, UP
- Mypy: warn_return_any, warn_unused_configs
- Docstrings: not required on existing code; add only for complex new functions

### Rust
- Edition: 2021
- Use workspace dependencies defined in root `Cargo.toml`
- SIMD: via LLVM autovectorization, not explicit intrinsics
- Use `faer` for matrix operations in performance-critical paths
- Use `rayon` for parallel iteration
- Use `thiserror` for error types
- Use `petgraph` for graph operations

### General
- No emojis in benchmark scripts or stdout output (Windows cp1252)
- Gate names are normalized to uppercase internally
- Don't add features, refactors, or "improvements" beyond what was asked
- Keep solutions simple — three similar lines are better than a premature abstraction

---

## Project-Specific Gotchas

1. **Statevector caching**: SingularityBackend caches statevectors by MD5
   fingerprint for n <= 24 qubits. If you change gate implementations, the cache
   will return stale results — clear it or bump the fingerprint.

2. **Turbo gate fusion**: `fuse_single_qubit_gates()` is applied by default in
   `runner.py` but SKIPPED if the device signals `skip_fusion` via
   `DeviceCapabilities`. If adding a new backend/device, decide whether fusion
   applies.

3. **MPS bond dimension**: Default is 64. The SVD truncation threshold is
   `1e-12 * max(singular_values)`. High-entanglement circuits (QFT, Random)
   can blow past this limit.

4. **Clifford detection**: `is_clifford_circuit()` checks every gate against a
   whitelist. If you add a gate, update the whitelist in `backends/stabilizer.py`.

5. **Rust QASM parser**: The hand-rolled parser in `crates/sf-ir/src/qasm.rs`
   is 6x faster than Qiskit's but supports OpenQASM 2.0 only, not 3.0.

6. **Memory**: `DensityMatrixBackend` uses O(4^n) memory — capped at 12 qubits.
   `StabilizerBackend` uses O(n^2) — works to ~1000 qubits.
   `MPS` uses O(n * D^2) where D is bond dim — works to 200+ qubits for low
   entanglement.

---

## CI Pipeline

Runs on push/PR to main/master:
- **Rust checks**: `cargo test --lib` for sf-ir, sf-compiler, sf-router, sf-qec, sf-pulse
- **Python checks**: ruff lint + pytest (skipping tests that need Qiskit/PennyLane/Aer)
- **Benchmark smoke**: quick import + circuit execution
