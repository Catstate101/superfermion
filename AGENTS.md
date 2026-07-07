# AGENTS.md — AI Coding Agent Instructions

> **What this is**: project-level instructions for AI coding agents working on
> the Superfermion codebase. Read this before making any changes.

---

## Project Overview

Superfermion is a quantum computing framework with a Python API and a
Rust simulation core. Python is the API, Rust does the work.

- **Python package** (`superfermion/`) — public API, simulation device layer,
  quantum algorithms, QML gradient methods, neural network layers, QEC,
  chemistry, compiler, device adapters, experiment tracking
- **Rust workspace** (`crates/`, 7 crates) — in-place statevector engine, MPS
  tensor network, stabilizer tableau, density matrix, compiler passes, SABRE
  router, pulse-level control, QEC codes/decoders, PyO3 bindings
- **Tests** (`tests/`) — pytest suite organized as `unit/`, `integration/`,
  `backends/`, `domain/`, `e2e/`
- **Docs** (`docs/`) — user + developer documentation

## Critical Rules

### 1. Rust-Python Bridge
After any Rust change, rebuild:
```bash
cd crates/sf-bindings && maturin develop --release && cd ../..
# Copy built extension into package (Linux):
cp target/release/lib_sf_core.so superfermion/_sf_core.so
```

### 2. Endianness
Superfermion uses **q0=MSB** (big-endian). Qiskit and the Rust core use
**q0=LSB** (little-endian). All bridge functions must reverse qubit indices:
`n - 1 - q`. See `superfermion/bridge/__init__.py`.

### 3. Testing Before Changes
```bash
# Quick smoke test
python -m pytest tests/unit/ -q --timeout=60

# Full suite
python -m pytest tests/ -q --timeout=120

# Rust-only tests
cargo test --workspace
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
pip install maturin
cd crates/sf-bindings && maturin develop --release && cd ../..
cp target/release/lib_sf_core.so superfermion/_sf_core.so  # Linux
```

### Running Tests
```bash
python -m pytest tests/unit/ -q           # Unit tests (~30s)
python -m pytest tests/integration/ -q    # Integration tests
python -m pytest tests/backends/ -q       # Backend correctness
python -m pytest tests/domain/ -q         # Algorithms, QEC, gradients
python -m pytest tests/e2e/ -q            # End-to-end pipelines
python -m pytest tests/ -q --timeout=120  # Full suite

cargo test --workspace                    # All Rust tests
```

### Code Quality
```bash
ruff check superfermion/ tests/
ruff format --check superfermion/ tests/
mypy superfermion/
```

---

## Architecture Rules

### Adding a New Device Adapter
1. Create `superfermion/devices/your_device.py` implementing `DeviceExecutor` protocol
2. The protocol requires `execute(circuit, shots, **kwargs) -> RunResult` and `capabilities() -> DeviceCapabilities`
3. Add tests in `tests/unit/test_devices.py`

### Adding a New Gate to Circuit
1. Add a method to `Circuit` in `superfermion/circuit.py`
2. Add the gate name mapping in `GateRecord.to_unitary()`
3. Add to `OpType` enum in `crates/sf-ir/src/ops.rs`
4. Add the unitary matrix in `ops.rs:to_unitary_matrix()`
5. Add the gate parsing in `crates/sf-bindings/src/lib.rs:parse_gate()`
6. Add tests: single-gate correctness, multi-gate interaction

---

## Key Files Quick Reference

| Task | Read These Files First |
|---|---|
| Add a device adapter | `devices/__init__.py`, `devices/rust_device.py` |
| Add a gate | `circuit.py`, `crates/sf-ir/src/ops.rs`, `crates/sf-bindings/src/lib.rs` |
| Fix a simulation bug | `crates/sf-ir/src/dag.rs` |
| Add a QEC code | `qec/codes/`, `crates/sf-qec/src/codes.rs` |
| Add a gradient method | `qml/gradient/adjoint.py`, `qml/gradient/core.py` |
| Add an algorithm | `algorithms/variational.py`, `algorithms/grover.py` |
| Change compilation | `compiler/manager.py`, `crates/sf-compiler/src/lib.rs` |
| Change execution flow | `runner.py`, `devices/__init__.py` |
| Add framework bridge | `bridge/__init__.py` |

---

## Conventions

### Python
- Line length: 100 (ruff)
- Target: Python 3.10
- Ruff rules: E, F, I, N, W, UP

### Rust
- Edition: 2021
- Use workspace dependencies defined in root `Cargo.toml`
- Use `faer` for matrix operations in performance-critical paths
- Use `rayon` for parallel iteration

### General
- No emojis in benchmark scripts or stdout output (Windows cp1252)
- Gate names are normalized to uppercase internally
- Don't add features, refactors, or "improvements" beyond what was asked
- Keep solutions simple

---

## Project-Specific Gotchas

1. **MPS bond dimension**: Default is 64. High-entanglement circuits (QFT,
   Random) can exceed this limit.

2. **Clifford detection**: `is_clifford_circuit()` checks every gate against a
   whitelist. If you add a gate, update the whitelist.

3. **Rust QASM parser**: Supports OpenQASM 2.0 only, not 3.0.

4. **Memory**: `density_matrix` uses O(4^n) — capped at 12 qubits.
   `stabilizer` uses O(n^2) — works to ~1000 qubits.
   `mps` uses O(n * D^2) where D is bond dim.

---

## CI Pipeline

Runs on push/PR to main/master:
- **Rust**: `cargo test --workspace`
- **Python**: ruff lint + `maturin develop --release` + pytest
- **Smoke**: quick import + circuit execution
