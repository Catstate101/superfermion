# AGENTS.md — AI Coding Agent Instructions

> **What this is**: project-level instructions for AI coding agents working on
> the Superfermion codebase. Read this before making any changes.

---

## Project Overview

Superfermion is a high-performance quantum circuit simulator with a Python API
and a Rust SIMD acceleration core. The project is organized as:

- **Python package** (`superfermion/`, 89 files) — public API, 12+ backends,
  quantum algorithms, ML layers, QEC, chemistry, compiler, runtime, visualization
- **Rust workspace** (`crates/`, 6 crates, 33 source files) — SIMD statevector
  engine, MPS tensor network, stabilizer tableau, compiler passes, SABRE router,
  pulse-level control, QEC codes/decoders, PyO3 bindings
- **Tests** (`tests/`) — pytest-based test suite
- **Benchmarks** (`benchmarks/`) — industry comparison scripts
- **Docs** (`docs/`) — user + developer docs

## Critical Rules

### 1. Rust-Python Bridge
After any Rust change, you MUST copy the built extension:
```bash
# Windows
cp .venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd superfermion/_sf_core.pyd

# Linux
cp .venv/lib/python3.*/site-packages/_sf_core/_sf_core*.so superfermion/
```
`maturin develop --release` alone is NOT enough — Python imports the file from
`superfermion/`, not from `.venv/`.

### 2. Endianness
Superfermion uses **q0=MSB** (big-endian). Qiskit and the Rust core use
**q0=LSB** (little-endian). All bridge functions must reverse qubit indices:
`n - 1 - q`. See `superfermion/bridge/__init__.py` for the conversion functions.

### 3. Testing Before Changes
Always run the smoke test suite before committing:
```bash
python -m pytest tests/test_rust_kernel_correctness.py \
                 tests/test_singularity_routing_and_expval.py \
                 tests/test_stabilizer.py \
                 tests/test_adjoint_grad.py -q
```

### 4. No Auto-Commit
Do not create git commits unless explicitly asked. This is a fast-iteration
codebase and the user controls when changes are committed.

### 5. Read Before Edit
Always read a file before editing it. This is enforced by the Edit tool.

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
# Core smoke test (30s)
python -m pytest tests/test_rust_kernel_correctness.py \
                 tests/test_singularity_routing_and_expval.py \
                 tests/test_stabilizer.py \
                 tests/test_adjoint_grad.py -q

# Full suite (7 min, skips cross-framework tests)
python -m pytest tests/ -q --ignore=tests/test_algo_comparison.py

# Cross-framework tests (requires Qiskit + PennyLane)
python -m pytest tests/test_accuracy_vs_qiskit.py \
                 tests/test_cross_framework_vqe_qaoa.py \
                 tests/test_standalone_features.py -q

# Benchpress (latency + memory benchmarks)
python -m pytest tests/benchpress/ -q

# Rust-only tests
cargo test --lib -p sf-ir -p sf-compiler -p sf-router -p sf-qec -p sf-pulse
```

### Code Quality
```bash
ruff check superfermion/ tests/
black --check superfermion/ tests/
mypy superfermion/
```

### Full Stack (Docker)
```bash
docker compose up   # API on :8000, PostgreSQL on :5432
```

---

## Architecture Rules

### Adding a New Backend
1. Create `superfermion/backends/your_backend.py` extending `Backend` (from `base.py`)
2. Implement `run(circuit, shots)` and properties `n_qubits`, `supported_gates`
3. Register in `superfermion/backends/registry.py` in the `_lazy_init` method
4. Add alias(es) if desired (e.g., `"singularity"` also registers as `"god"`)
5. If applicable, add Clifford auto-dispatch using `maybe_clifford_dispatch()`
6. Add tests in `tests/test_backends.py`
7. Document in `docs/backends.md` and `SF_ALL_BACKENDS.md`

### Adding a New Gate to Circuit
1. Add a method to `Circuit` in `superfermion/circuit.py`
2. Add the gate name mapping in `GateRecord.to_unitary()` (line ~450)
3. Add to `OpType` enum in `crates/sf-ir/src/ops.rs`
4. Add the unitary matrix in `ops.rs:to_unitary_matrix()`
5. Add the gate parsing in `crates/sf-bindings/src/lib.rs:parse_gate()`
6. Add a simulation path in `crates/sf-ir/src/dag.rs:simulate()` if needed
7. Add the gate name to the Turbo decomposition logic in `backends/turbo.py`
8. Add tests: single-gate correctness, multi-gate interaction

### Adding a Rust Dependency
- Check `target/` size first — disk is tight (~1–3 GB free)
- Add to workspace `Cargo.toml` `[workspace.dependencies]`
- Then reference in the specific crate's `Cargo.toml`
- Run `cargo clean` if target grows too large
- Document in `.claude/notes/cargo-deps.md`

---

## Key Files Quick Reference

| Task | Read These Files First |
|---|---|
| Add a backend | `backends/base.py`, `backends/registry.py`, `backends/singularity.py` |
| Add a gate | `circuit.py`, `crates/sf-ir/src/ops.rs`, `crates/sf-bindings/src/lib.rs` |
| Fix a simulation bug | `backends/turbo.py`, `crates/sf-ir/src/dag.rs` |
| Add a QEC code | `qec/codes/`, `crates/sf-qec/src/codes.rs` |
| Add a gradient method | `qml/gradient/adjoint.py`, `qml/gradient/core.py` |
| Add an algorithm | `algorithms/variational.py`, `algorithms/grover.py` |
| Change compilation | `compiler/manager.py`, `crates/sf-compiler/src/lib.rs` |
| Add a provider | `runtime/providers/` |
| Add framework bridge | `bridge/__init__.py` |
| Add a visualization | `viz/core.py` |
| Change CLI | `cli.py` |
| Change API endpoint | `serve/app.py` |

---

## Conventions

### Python
- Line length: 100 (black, ruff)
- Target: Python 3.10
- Ruff rules: E, F, I, N, W, UP
- Mypy: warn_return_any, warn_unused_configs (but pyright type-checking disabled)
- Docstrings: not required on existing code; add only for complex new functions

### Rust
- Edition: 2021
- Use workspace dependencies defined in root `Cargo.toml`
- SIMD: via LLVM autovectorization, not explicit intrinsics
- Use `faer` for matrix operations (not nalgebra) in performance-critical paths
- Use `rayon` for parallel iteration
- Use `thiserror` for error types
- Use `petgraph` for graph operations

### General
- No emojis in benchmark scripts or stdout output (Windows cp1252)
- Heavy unicode (═ ⟨ ⟩ Δ) breaks `print()` on Windows
- Gate names are normalized to uppercase internally
- Don't add features, refactors, or "improvements" beyond what was asked
- Don't add error handling for scenarios that can't happen
- Keep solutions simple — three similar lines are better than a premature
  abstraction

---

## Project-Specific Gotchas

1. **Statevector caching**: SingularityBackend caches statevectors by MD5
   fingerprint for n ≤ 24 qubits. If you change gate implementations, the cache
   will return stale results — clear it or bump the fingerprint.

2. **Turbo gate fusion**: `fuse_single_qubit_gates()` is applied by default in
   `runner.py` but SKIPPED for the stabilizer backend. If adding a new backend,
   decide whether fusion applies.

3. **MPS bond dimension**: Default is 64. The SVD truncation threshold is
   `1e-12 * max(singular_values)`. High-entanglement circuits (QFT, Random)
   can blow past this limit.

4. **Clifford detection**: `is_clifford_circuit()` checks every gate against a
   whitelist. If you add a gate, update the whitelist in
   `backends/stabilizer.py`.

5. **Rust QASM parser**: The hand-rolled parser in `crates/sf-ir/src/qasm.rs`
   is 6x faster than Qiskit's but supports OpenQASM 2.0 only, not 3.0. The
   Python fallback handles qelib1.inc includes.

6. **pyproject.toml testpaths**: pytest discovers from `["."]`, not `["tests/"]`.
   This means Python files at the repo root could be collected as tests.

7. **Memory**: `DensityMatrixBackend` uses O(4^n) memory — capped at 12 qubits.
   `StabilizerBackend` uses O(n^2) — works to ~1000 qubits.
   `MPS` uses O(n * D^2) where D is bond dim — works to 200+ qubits for low
   entanglement.

---

## CI Pipeline

Runs on push/PR to main/master:
- **Rust checks**: `cargo test --lib` for sf-ir, sf-compiler, sf-router, sf-qec,
  sf-pulse (not sf-bindings — requires Python)
- **Python checks**: ruff lint + pytest (skipping tests that need Qiskit/
  PennyLane/Aer and `test_rust_kernel_correctness.py`)
- **Benchmark smoke**: quick import + GHZ-8 circuit execution
- **Docker build**: validates Dockerfile

Release workflow (on `v*.*.*` tags):
- Build wheels cross-platform (Linux/macOS/Windows) via maturin
- Publish to PyPI
- Build + push Docker image

---

## Getting Help

- `/help` — get help with AI coding agent features
- Report issues at the project's GitHub repository
- For questions about the AI coding tool itself (not Superfermion), check the
  tool's documentation
