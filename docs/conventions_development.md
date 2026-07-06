# Superfermion Development Conventions

This document outlines the coding standards, architectural principles, and best practices for the Superfermion project.

## 1. Architectural Principles

- **Single Source of Truth**: The Rust IR (`sf-ir`) is the canonical representation of a quantum circuit. All other representations (Python `Circuit`, OpenQASM, hardware JSON) must be translatable to/from this IR.
- **Hardware Agnosticism**: Core logic (optimizations, high-level IR) must remain hardware-neutral. Hardware-specific logic belongs in `sf-compiler/decompose` or backend-specific crates.
- **Performance First**: Performance-critical paths (compilation passes, simulation kernels) must be implemented in Rust.
- **Differentiability**: All quantum operations must support gradient computation (e.g., via the parameter-shift rule) to enable integration with JAX.
- **Modular Design**: Every module (`qml`, `qdl`, `qllm`, `qec`, `chemistry`, `intelligence`) must be independently importable and testable.

## 2. Rust Conventions

- **Safety**: Prefer safe Rust. Use `unsafe` only for performance-critical FFI or SIMD, and always document why it's safe.
- **Error Handling**: Use `thiserror` for crate-internal errors and `anyhow` for applications/CLI. Never `unwrap()` in library code; use `.expect()` with a descriptive message if a failure is truly impossible.
- **Types**: Use `SmallVec` for qubit lists and attributes to avoid heap allocations for small circuits. Use `petgraph` for DAG manipulations.
- **Testing**: Every new feature must have unit tests in a `mod tests` block at the bottom of the file.
- **Crate Structure**: The Cargo workspace hosts: `sf-ir`, `sf-compiler`, `sf-router`, `sf-qec`, `sf-pulse`, `sf-bindings`.

## 3. Python Conventions

- **Type Hinting**: All Python code must be strictly typed. Run `mypy` before submitting.
- **API Design**: Use fluent, chainable APIs for the `Circuit` class (e.g., `c.h(0).cx(0, 1)`).
- **Docstrings**: Use Google-style docstrings. Include examples for all public methods.
- **JAX Integration**: Ensure all user-facing functions are compatible with JAX transformations (`jit`, `grad`, `vmap`).
- **Imports**: Use `from __future__ import annotations` in every file for forward reference support.
- **Naming**: Use `snake_case` for functions/variables, `PascalCase` for classes. Circuit gates are lowercase methods (`.h()`, `.cx()`, `.ry()`).

## 4. Module Structure

### Core Modules (always available)
| Module | Purpose | Example |
|--------|---------|---------|
| `sf.Circuit` | Circuit construction | `sf.Circuit(2).h(0).cx(0, 1)` |
| `sf.run()` | Execution | `sf.run(circuit, shots=1000)` |
| `sf.compile()` | Compilation | `sf.compile(circuit, target=spec)` |
| `sf.param()` | Parameter creation | `sf.param("theta")` |
| `sf.qml` | JAX bridge | `sf.qml.circuit_to_jax(c)` |

### Algorithm & Frontier AI Modules
| Module | Purpose |
|--------|---------|
| `sf.algorithms.vqe` | Variational Quantum Eigensolver |
| `sf.algorithms.qaoa` | Quantum Approximate Optimization |
| `sf.algorithms.qsvm` | Quantum Support Vector Machine |
| `sf.algorithms.qrl` | Quantum Reinforcement Learning |
| `sf.algorithms.qbm` | Quantum Boltzmann Machine |
| `sf.qml.quantum_ai` | QNN, QGNN, QGAN, QVAE, QNLP engine |
| `sf.classical` | JAX-accelerated Classical AI (GNN, CNN, ML) |

### Deep Learning Modules
| Module | Purpose |
|--------|---------|
| `sf.nn.QuantumLayer` | Flax-compatible quantum layer |
| `sf.nn.Linear` | Hybrid quantum-classical dense layer |
| `sf.qdl.resnet` | QResNet blocks |
| `sf.qdl.attention` | Quantum self-attention |
| `sf.qllm.transformer` | QuantumGPT (quantum LLM) |

### Infrastructure Modules
| Module | Purpose |
|--------|---------|
| `sf.chemistry` | Molecular Hamiltonians, UCCSD |
| `sf.qec` | QEC Manager, Linear/Topological/LDPC Codes |
| `sf.qec.decoders` | MWPM, Union-Find, BP+OSD, Neural decoders |
| `sf.qml.measurements` | Entropy, Purity, Fidelity, Adv. metrics |
| `sf.noise` | Noise models (IBM Eagle, etc.) |
| `sf.mitigation` | Error mitigation (ZNE, PEC) |
| `sf.bridge` | Qiskit/PennyLane interop |
| `sf.serve` | FastAPI REST gateway |
| `sf.intelligence` | Superpositional agents, QNS |

## 5. Security & Multi-Tenancy

- **Auth**: API keys via `X-SF-API-KEY` header. Tiers: `free` (12 qubits), `pro` (28 qubits), `unlimited` (127 qubits).
- **DoS Protection**: Circuit qubit limit enforced by `ResourceArbiter.validate_security()`. Max 40 qubits on local simulation without explicit override.
- **Input Sanitization**: All QASM inputs must be validated before parsing.
- **Quotas**: Per-user credit quotas tracked in the VAULT. Rate limiting on all cloud endpoints.

## 6. Cloud & Hardware Routing

- **Arbiter Logic**: `<=20q -> JAX local`, `21-35q -> cluster/JAX`, `>35q -> quantum cloud (IBM Eagle)`.
- **Supported Devices**: IBM Eagle (127q), Rigetti Aspen-M3 (80q), IonQ Aria (25q), Linear-5 (test).
- **Providers**: `IBMProvider` (Qiskit Runtime), `BraketProvider` (AWS).
- **Compilation**: SWAP decomposition, gate cancellation, basis translation applied automatically.

## 7. Testing Standards

- **Unit Tests**: Every module must have corresponding `tests/test_*.py` files.
- **Naming**: Test functions prefixed with `test_`. Descriptive names: `test_vqe_h2_convergence`.
- **Assertions**: Use physical assertions where possible (norm=1, energy<0, fidelity>0.99).
- **CI**: All tests must pass before merge. Run with `python -m pytest tests/`.
- **Validation**: The full suite includes: `test_industry_validation.py` (57 tests, 19 domains), `test_extended_validation.py` (16 tests), `test_complex_workflow.py`.

## 8. Documentation

- **Master Plan**: All major changes must be reflected in `superfermion_master_plan.md`.
- **Implementation Tracker**: The `superfermion_implementation_tracker.md` must be updated at the end of every development session.
- **API Reference**: `docs/api_reference.md` must document all public functions with examples.
- **Getting Started**: `docs/getting_started.md` provides a 5-minute intro.
- **Notebooks**: `notebooks/superfermion_demo.ipynb` serves as an interactive tutorial.

## 9. Toolchain

- **Rust**: Use the latest stable toolchain.
- **Python**: Use `uv` for dependency management. Target Python 3.12+.
- **Linting**: 
    - Rust: `cargo fmt`, `cargo clippy`
    - Python: `ruff check`, `ruff format`
- **CLI**: `python -m superfermion.cli [info|validate|backends|version|run|benchmark]`

## 10. Physics Conventions

- **Qubit Ordering**: Big-endian — qubit 0 is the most significant bit. `|01>` means qubit 0 = 0, qubit 1 = 1.
- **Gate Phase**: Standard textbook conventions (e.g., `RX(theta) = exp(-i * theta/2 * X)`).
- **Measurement**: Computational basis (Z-basis) by default. Post-measurement state collapse.
- **Normalization**: Statevectors must always satisfy `sum(|a_i|^2) = 1` within tolerance 1e-6.
- **Parameter Shift**: Gradient rule: `df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2`.
