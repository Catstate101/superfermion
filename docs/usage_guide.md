# Superfermion Usage Guide

Canonical reference for how the SF framework is used. Every code example here reflects the actual implemented API surface. If an API changes, this document must be updated to match.

Last verified against: `superfermion/__init__.py`, `runner.py`, `circuit.py`, `compiler/manager.py`, `experiment/context.py`, `devices/__init__.py`, `results.py`, `observables/core.py`

---

## First-Class Citizens

```
sf.Circuit(n)          Build circuits
sf.run(circuit, ...)   Execute anywhere
sf.simulate(circuit)   Execute and return sf.State directly
sf.compile(circuit)    Compile for hardware
sf.State               Rust-native quantum state handle
sf.experiment(name)    Scoped experiment tracking
sf.param(name)         Symbolic parameters
sf.Hamiltonian(...)    Observable construction
```

---

## 1. Circuit Construction

`sf.Circuit` is a fluent builder. Each gate call returns the circuit for chaining.

```python
import superfermion as sf

qc = sf.Circuit(2).h(0).cnot(0, 1)

print(qc.n_qubits)     # 2
print(qc.gate_count)    # 2
print(qc.depth)         # 2
print(qc.draw())        # text diagram
print(qc.draw(ascii_only=True))   # force `-`, `o`, `+` (cp1252-safe)
```

`draw()` picks the Unicode box drawing when the target stream can encode it and
falls back to plain ASCII when it cannot (a cp1252 Windows console,
`PYTHONIOENCODING=ascii`, a redirected stream), so `print(qc.draw())` does not
raise `UnicodeEncodeError`.

### Available Gates

| Category | Gates |
|----------|-------|
| 1Q Clifford | `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `sx`, `id` |
| 1Q Parametric | `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)`, `p(phi, q)`, `u(theta, phi, lam, q)` |
| 2Q | `cnot(c, t)` / `cx(c, t)`, `cz`, `cy`, `swap`, `iswap`, `ecr` |
| 2Q Parametric | `cp`, `cu`, `rzz`, `rxx`, `ryy` |
| 3Q | `ccx` / `toffoli`, `cswap` / `fredkin` |
| Special | `measure(q)`, `measure_all()`, `barrier(*qubits)`, `reset(q)` |

---

## 2. Execution — `sf.run()`

`sf.run()` is the single universal entry point. It accepts any device (local or QPU) and any simulation method.

```python
import superfermion as sf

qc = sf.Circuit(2).h(0).cnot(0, 1)

result = sf.run(qc, shots=4096)
print(result.counts)        # {'00': 2048, '11': 2048}
print(result.state)         # sf.State (Rust handle)
print(result.shots)         # 4096
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Circuit` | required | The circuit to execute |
| `device` | `str` or `DeviceExecutor` | `"cpu"` | Where to run |
| `shots` | `int` | `1000` | Number of measurement shots |
| `method` | `str` or `None` | `"statevector"` | Simulation algorithm |
| `target` | `str` or `None` | `None` | Hardware target for auto-compilation |
| `tracker` | `TrackerProtocol` or `None` | `None` | Explicit tracker |
| `params` | `dict` or `None` | `None` | Auto-bind symbolic parameters |

### Device Resolution

```python
# Local simulation (builtin shorthands)
sf.run(qc, device="cpu")                  # Rust CPU (Rayon + AVX)
sf.run(qc, device="gpu")                  # Rust GPU (CUDA)

# QPU via provider objects (protocol-based, no registries)
from superfermion.devices.ibm import IBMDevice
ibm = IBMDevice(token="...")
sf.run(qc, device=ibm("ibm_brisbane"))    # explicit object, not magic string

from superfermion.devices.ionq import IonQDevice
ionq = IonQDevice(api_key="...")
sf.run(qc, device=ionq("aria-1"))         # same pattern, different provider
```

### Auto-Bind Parameters

```python
ansatz = sf.Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)

# These are equivalent:
result = sf.run(ansatz.bind({"t": 0.5}), shots=1000)
result = sf.run(ansatz, params={"t": 0.5}, shots=1000)
```

---

## 3. Exact Simulation — `sf.simulate()`

Shortcut that returns `sf.State` directly instead of `RunResult`.

```python
state = sf.simulate(qc)

# sf.State is Rust-native — all methods dispatch to Rust
print(state.n_qubits)           # 2
print(state.entropy())          # Von Neumann entropy
print(state.purity())           # state purity
print(state.fidelity(state))    # 1.0

sv = state.numpy()              # export to numpy
samples = state.sample(1000)    # fast Rust sampling
```

---

## 4. Simulation Methods

All methods use the same `sf.run()` / `sf.simulate()` API. Only the `method=` parameter changes.

### Statevector (default)

Exact simulation. Supports all `sf.State` methods including gradients. Up to ~25 qubits on CPU, ~30 on GPU.

```python
result = sf.run(qc, method="statevector", shots=1000)
```

### MPS (Tensor Network)

For weakly entangled circuits. Scales to 200+ qubits.

```python
result = sf.run(qc, method="mps", bond_dim=64, shots=1000)
```

### Stabilizer

For Clifford-only circuits. Polynomial time, scales to ~1000 qubits.

```python
clifford = sf.Circuit(100).h(0)
for i in range(99):
    clifford = clifford.cnot(i, i + 1)

result = sf.run(clifford, method="stabilizer", shots=1000)
```

### Density Matrix

For noisy simulation with Kraus channels.

```python
noise = sf.NoiseModel().add_depolarizing(0.01)
result = sf.run(qc, method="density_matrix", noise_model=noise, shots=0)
```

---

## 5. Compilation — `sf.compile()`

Compile circuits for hardware targets. Handles basis translation, gate optimization, and qubit routing.

```python
import superfermion as sf
from superfermion.compiler.specs import HardwareSpec

qc = sf.Circuit(5).h(0).cnot(0, 3).cnot(1, 4).swap(2, 3)

# Optimization only (no hardware target)
optimized = sf.compile(qc, level=1)

# Compile for a specific topology and gate set
target = HardwareSpec(
    name="my_device",
    n_qubits=5,
    native_gates=["rz", "sx", "x", "cx"],
    coupling_map=[(0,1), (1,2), (2,3), (3,4)],
)
compiled = sf.compile(qc, level=2, target=target)
```

### Optimization Levels

| Level | What it does |
|-------|-------------|
| 0 | No optimization (passthrough) |
| 1 | SWAP decomposition, gate cancellation, rotation merging, constant folding |
| 2 | Level 1 + repeated cancellation, Pauli twirling, dynamical decoupling (if target set) |

### Auto-Compilation via `sf.run()`

```python
result = sf.run(qc, target="linear_5", shots=1000)
```

---

## 6. Observables and Expectation Values

```python
import superfermion as sf

# Build observables
H = sf.Hamiltonian([
    sf.PauliString("ZZ", coeff=0.5),
    sf.PauliString("XI", coeff=0.3),
    sf.PauliString("IZ", coeff=0.2),
])

# Exact expectation from state
state = sf.simulate(sf.Circuit(2).h(0).cnot(0, 1))
energy = state.expectation(H.to_sparse_list())

# Estimate from measurement counts
result = sf.run(sf.Circuit(2).h(0).cnot(0, 1), shots=10000)
energy = result.expectation(H.to_sparse_list())
```

Observable format for `state.expectation()` and `state.grad()`:

```python
# List of (pauli_indices, coeff_real, coeff_imag)
# Pauli encoding: 0=I, 1=X, 2=Y, 3=Z
zz_obs = [([3, 3], 1.0, 0.0)]     # ZZ with coefficient 1.0
xz_obs = [([1, 3], 0.5, 0.0)]     # 0.5 * XZ
```

---

## 7. Parametric Circuits and Gradients

```python
import superfermion as sf

theta = sf.param("theta")
phi = sf.param("phi")

ansatz = (sf.Circuit(2)
    .ry(theta, 0)
    .ry(phi, 1)
    .cnot(0, 1)
    .rz(theta, 1))

print(ansatz.n_parameters)   # 2
print(ansatz.parameters)     # ['theta', 'phi']

# Bind and compute gradients
params = {"theta": 0.5, "phi": 1.2}
state = sf.simulate(ansatz, params=params)

obs = [([3, 3], 1.0, 0.0)]  # ZZ
dag = ansatz.bind(params).to_ir()
grads = state.grad(obs, dag, params)
print(grads)  # {"theta": -0.23..., "phi": 0.41...}
```

### Supported Gradient Gates

Adjoint differentiation (Rust, `O(M+N)·2ⁿ`) is the default on statevector; the
parameter-shift rule covers every device and every method. Both are exact and
both are anchored to central finite differences in
`tests/domain/test_gradient_regressions.py`:

| Gate | Parameters | Adjoint | Parameter-shift | Buildable from `Circuit` |
|---|---|---|---|---|
| `rx`, `ry`, `rz`, `rxx`, `ryy`, `rzz` | θ | ✓ | ½[f(θ+π/2) − f(θ−π/2)] | yes |
| `p`, `r1`, `cp` | φ | ✓ | same (generator spectrum {0, ±1}) | yes |
| `u` / `u3` | θ, φ, λ | ✓ (each slot) | same, per slot | yes |
| `cu` | θ, φ, λ, γ | ✓ (each slot) | θ needs the extra π shift; φ, λ, γ use the π/2 rule | yes |
| `crx`, `cry`, `crz` | θ | ✓ | π/2 plus π shift for the controlled rotations | yes |

Two conveniences, both exact rather than approximate: `r1(φ)` is the same
matrix as `p(φ)` and exists so that QASM or Qiskit-style code ports over
unchanged; `cry(θ)` has no dedicated IR opcode and is emitted as
`cu(θ, 0, 0)`, which is *exactly* `CRy` because `U3(θ,0,0) == Ry(θ)` and a
controlled block keeps the phase.

Cross-framework status: for every gate in the table, the adjoint
gradient agrees with PennyLane's reverse-mode autodiff and with its
parameter-shift to **2.8e-16**, and with central finite differences on Qiskit's
`Statevector` to **1.7e-11** (the finite-difference floor), with the expectation
value itself matching to **2.2e-16** across all three frameworks. The
`r1`/`crx`/`cry`/`crz` builders specifically were checked the same way: 0.0
against Qiskit's `Statevector` and ≤4.4e-16 against PennyLane on ⟨O⟩, and
≤8.3e-17 against PennyLane's gradient.

Three rules keep the numbers honest:

- **A name reused across gates is one term per occurrence.** `ry(t); ry(t)`
  returns the full derivative (both gates move), not half of it.
- **Nothing is silently zero.** A gate outside the shift-rule table raises
  `ValueError` naming it, and a parametric gate without a generator raises
  from the Rust kernel instead of contributing 0.
- **Pass the unbound circuit.** `adjoint_grad(circuit.bind(params), …)` raises
  with a hint: binding folds the symbols into constants, so the DAG can no
  longer differentiate them.

Adjoint is statevector-only. Density-matrix, MPS and stabilizer states raise
`MethodError` for adjoint and use parameter-shift, which works everywhere
(`method="density_matrix"` included).

---

## 8. VQE Optimization Loop

```python
import superfermion as sf
import numpy as np

H = sf.Hamiltonian([
    sf.PauliString("II", coeff=-1.0523),
    sf.PauliString("IZ", coeff=0.3979),
    sf.PauliString("ZI", coeff=-0.3979),
    sf.PauliString("ZZ", coeff=-0.0112),
    sf.PauliString("XX", coeff=0.1809),
])

ansatz = (sf.Circuit(2)
    .ry(sf.param("t0"), 0)
    .ry(sf.param("t1"), 1)
    .cnot(0, 1)
    .ry(sf.param("t2"), 0)
    .ry(sf.param("t3"), 1))

params = {f"t{i}": np.random.uniform(0, np.pi) for i in range(4)}
obs = H.to_sparse_list()
lr = 0.1

for step in range(50):
    state = sf.simulate(ansatz, params=params)
    energy = state.expectation(obs)

    dag = ansatz.bind(params).to_ir()
    grads = state.grad(obs, dag, params)

    for key in params:
        params[key] -= lr * grads.get(key, 0.0)

    if step % 10 == 0:
        print(f"Step {step}: energy = {energy:.6f}")
```

---

## 9. Experiment Tracking

Tracking is scoped via context manager using `contextvars` (thread-safe, no global state).

```python
import superfermion as sf

# Local tracking (no server needed)
with sf.experiment("bell-study") as tracker:
    r1 = sf.run(sf.Circuit(2).h(0).cnot(0, 1), shots=1000)
    r2 = sf.run(sf.Circuit(3).h(0).cnot(0,1).cnot(1,2), shots=1000)

# Runs saved to ~/.superfermion/runs/bell-study/
print(tracker.runs)  # list of run metadata dicts

# With Catstate platform (enhanced — same code pattern)
# import catstate
# cs = catstate.connect()
# with cs.experiment("bell-study"):
#     sf.run(qc, device=cs("best"), shots=4096)
```

---

## 10. QPU Providers — Protocol-Based

SF defines `DeviceExecutor` as a Python protocol. Any object satisfying it works with `sf.run()`. No registries, no magic strings for cloud devices.

```python
# Standalone provider usage (no platform needed)
from superfermion.devices.ibm import IBMDevice
ibm = IBMDevice(token="...")
result = sf.run(qc, device=ibm("ibm_brisbane"), shots=8192)

# With Catstate platform (same sf.run, different device object)
# cs = catstate.connect()
# result = sf.run(qc, device=cs("ibm:brisbane"), shots=8192)   # BYOA
# result = sf.run(qc, device=cs("best"), shots=8192)           # managed
```

### Device Access Tiers

| What you pass | Credentials | Tracking |
|---------------|-------------|----------|
| `device="cpu"` | None | Optional |
| `device="gpu"` | None | Optional |
| `device=ibm("brisbane")` | User's IBM token (local) | Optional |
| `device=cs("ibm:brisbane")` | User's IBM token (stored on Catstate) | Via platform |
| `device=cs("best")` | Catstate's accounts | Via platform |

---

## 11. ML Framework Integration

Each layer is a thin wrapper (~20 lines) around `sf.State.grad()`.

### PyTorch

```python
import torch
from superfermion.nn.torch_layer import TorchQuantumLayer

layer = TorchQuantumLayer(circuit, observable, device="cpu")
output = layer(params_tensor)    # forward: sf.simulate + expectation
output.backward()                # backward: sf.State.grad()
```

### Flax (JAX)

```python
from superfermion.nn.quantum_layer import QuantumLayer
import jax

layer = QuantumLayer(circuit, observable, device="cpu")
params = layer.init(jax.random.PRNGKey(0))
output = layer.apply(params)     # custom_vjp routes to sf.State.grad()
```

### TensorFlow

```python
from superfermion.nn.tf_layer import TFQuantumLayer

layer = TFQuantumLayer(circuit, observable, device="cpu")
with tf.GradientTape() as tape:
    output = layer(params_tensor)
grads = tape.gradient(output, params_tensor)
```

---

## 12. Cross-Framework Interop

```python
from superfermion.bridge import from_qiskit, to_qiskit, to_qasm, from_qasm
from superfermion.serialization import to_qasm3, from_qasm3

# Qiskit interop
qiskit_circuit = to_qiskit(sf_circuit)
sf_circuit = from_qiskit(qiskit_circuit)

# OpenQASM 2.0
qasm_str = to_qasm(sf_circuit)
sf_circuit = from_qasm(qasm_str)      # Rust-accelerated parser

# OpenQASM 3.0
qasm3_str = sf_circuit.to_qasm3()
sf_circuit = from_qasm3(qasm3_str)

# ``from_qasm3`` is strict: a statement it cannot represent raises ValueError
# naming it, instead of being skipped. The old skip behaviour is opt-in:
lossy = from_qasm3(qasm3_str, strict=False)   # may be missing operations

# ``from_qasm`` takes both versions: QASM 3 input the native parser rejects is
# routed to ``from_qasm3``, so this works too.
sf_circuit = from_qasm(sf_circuit.to_qasm3())

# JSON round-trip
json_str = sf_circuit.to_json()
sf_circuit = sf.Circuit.from_json(json_str)
```

---

## 13. Serialization and Export

```python
# Unitary matrix (small circuits only, exponential cost)
U = qc.to_unitary()   # little-endian: qubit 0 is the least significant bit

# Gate list
gates = qc.to_gate_list()   # [{"name": "h", "qubits": [0], "params": []}, ...]

# Rust IR (for gradient computation)
dag = qc.to_ir()            # QuantumDAG object
```

Ordering matters when you compare matrices: `qc.to_unitary()` puts qubit 0 in the
least significant bit (the Qiskit `Operator` convention, the same order the
statevector uses), while a single gate record, `qc._gates[i].to_unitary()`,
returns the textbook matrix with the record's first qubit *most* significant.
Bit-reverse the record to compare the two:

```python
import numpy as np

record = qc._gates[0]
k = len(record.qubits)
perm = [int(format(i, f"0{k}b")[::-1], 2) for i in range(1 << k)]
local = record.to_unitary()[np.ix_(perm, perm)]   # same order as qc.to_unitary()
```

---

## API Map

```
sf.Circuit(n)                     Circuit builder
  .h(q) .x(q) .cnot(c,t) ...     Gate methods (fluent, chainable)
  .rx(theta, q) .ry .rz .p .u    Parametric gates
  .measure(q) .measure_all()      Measurement
  .bind(params) -> Circuit        Parameter binding
  .n_qubits -> int                Qubit count
  .gate_count -> int              Total gate count
  .depth -> int                   Circuit depth
  .n_parameters -> int            Number of unbound parameters
  .parameters -> list[str]        Parameter names
  .to_qasm3() -> str              Export to OpenQASM 3.0
  .to_unitary() -> ndarray        Export to unitary matrix
  .to_ir() -> QuantumDAG          Export to Rust IR
  .to_json() -> str               Export to JSON
  .to_gate_list() -> list         Export to gate list
  .draw() -> str                  ASCII visualization

sf.run(circuit, ...) -> RunResult   Universal execution
  device="cpu" | "gpu" | executor   Where to run
  method="statevector"|"mps"|...    How to simulate
  shots=1000                        Measurement repetitions
  params={...}                      Auto-bind parameters
  target="device_name"              Auto-compile for hardware
  tracker=tracker_obj               Explicit tracking

sf.simulate(circuit, ...) -> State  Shortcut for sf.run().state

sf.compile(circuit, ...) -> Circuit Hardware compilation
  level=0|1|2                       Optimization aggressiveness
  target=HardwareSpec               Target device specification

sf.State                            Rust-native quantum state
  .expectation(obs) -> float        Exact expectation value
  .grad(obs, dag, params) -> dict   Adjoint differentiation
  .sample(shots) -> list            Measurement sampling
  .numpy() -> ndarray               Export to numpy
  .entropy() -> float               Von Neumann entropy
  .purity() -> float                State purity
  .fidelity(other) -> float         State fidelity
  .partial_trace(qubits) -> ndarray Partial trace
  .qfim(dag, params) -> ndarray     Quantum Fisher Information Matrix
  .n_qubits -> int                  Qubit count
  .method -> str                    Simulation method used

sf.experiment(name, tracker=None)   Context manager for tracking
sf.param(name) -> SymbolicParameter Symbolic parameter factory
sf.Hamiltonian(terms)               Observable from PauliString list
sf.PauliString(str, coeff=1.0)      Single Pauli term
sf.SparsePauliOp                    Sparse sum of Pauli terms
sf.expval(statevector, observable)  Compute expectation value
sf.NoiseModel()                     Noise model for density matrix sim
sf.RunResult                        Execution result container
sf.DeviceExecutor                   Protocol for device targets
sf.LocalTracker                     File-based experiment tracker
```
