# Superfermion Coding Conventions

> Guidelines for maintaining quality and performance in the Superfermion codebase.

---

## Python Style

### 1. Differentiable First
Logic inside algorithms should be JAX-compatible where gradients are needed.
- Use `jnp` instead of `np` inside JAX-transformed code.
- Avoid side effects in simulation logic.

### 2. Properties Over Methods
Static properties use `@property`, not getter methods.
- **Good**: `circuit.n_qubits`
- **Bad**: `circuit.get_n_qubits()`

### 3. Fluent API (Chainable)
Gate methods return `self` for chaining:
- **Example**: `sf.Circuit(2).h(0).cnot(0, 1).draw()`

### 4. Logging
Use `superfermion.utils.logging` helpers for internal messages.

---

## Rust Style

### 1. Performance Critical
Implement in Rust only on the critical path (simulation kernels, routing, IR).

### 2. Minimal Bindings
Keep the PyO3 interface simple. Wrap `_sf_core` in high-level Python classes.

---

## Testing Guidelines

### 1. Pytest Suite
Run `python -m pytest tests/unit/ -q` for fast smoke tests.
Full suite: `python -m pytest tests/ -q --timeout=120`.

### 2. Hardware Mocking
QPU adapter tests use mocks when live credentials are unavailable.

---

## Component Ownership

| Module | Responsible For | Primary Language |
|--------|-----------------|------------------|
| `sf-ir` | Gate definitions & DAG | Rust |
| `qml` | Gradients & templates | Python |
| `algorithms` | VQE, QAOA, QSVM | Python |
| `devices` | DeviceExecutor protocol, QPU adapters | Python |
| `qec` | Fault-tolerance codes & decoders | Python |
| `compiler` | Transpilation passes | Python + Rust |

---

## API Principles

- **No hidden global state** — pass device objects explicitly
- **Pass objects, not magic strings** — `IBMDevice(token=...)` not hidden registries
- **Scope is visible** — tracking via `with sf.experiment(...)` context manager
