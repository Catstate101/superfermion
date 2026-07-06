# Superfermion API Reference

> Complete reference for every public class, function, and module.

---

## `superfermion` (Top Level)

```python
import superfermion as sf
```

| Symbol | Type | Description |
|--------|------|-------------|
| `sf.Circuit(n_qubits)` | Class | Create a quantum circuit |
| `sf.run(circuit, backend, shots)` | Function | Execute a circuit |
| `sf.compile(circuit, target)` | Function | Compile for hardware |
| `sf.param(name)` | Function | Create a symbolic parameter |
| `sf.list_backends()` | Function | List available backends (see [backends.md](backends.md)) |
| `sf.get_backend(name)` | Function | Get a backend instance |
| `sf.estimate_cost(circuit)` | Function | Static gate/depth/memory estimate |
| `sf.benchmark(circuit, backend, iterations)` | Function | One-shot runtime micro-benchmark |
| `sf.VQE`, `sf.QAOA` | Class | Top-level variational solvers (scipy path) |
| `sf.SFEstimator`, `sf.SFSampler` | Class | Qiskit-v2-compatible primitives |
| `sf.PauliString`, `sf.SparsePauliOp`, `sf.Hamiltonian`, `sf.expval` | — | Observable DSL |
| `sf.Pipeline` | Class | Declarative preprocess → encode → model chain |
| `sf.train` | Function | JAX/Flax training loop |
| `sf.qml` | Module | JAX/ML integration |
| `sf.classical` | Module | JAX-accelerated Classical AI |
| `sf.qec` | Module | Fault-Tolerant QEC Manager |
| `sf.nn.Linear` | Class | Hybrid quantum-classical dense layer |
| `sf.chemistry` | Module | Molecular Hamiltonians and UCCSD |
| `sf.bridge`, `sf.serialization`, `sf.compiler`, `sf.mitigation`, `sf.noise`, `sf.pulse`, `sf.experiment`, `sf.runtime`, `sf.security`, `sf.telemetry`, `sf.data`, `sf.config`, `sf.serve`, `sf.algorithms` | Module | See [Advanced modules](#advanced-modules) below |
| `sf.__version__` | String | Framework version |

---

## `sf.Circuit`

### Constructor

```python
c = sf.Circuit(n_qubits: int, n_cbits: int = None, name: str = None)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `c.n_qubits` | `int` | Number of qubits |
| `c.n_cbits` | `int` | Number of classical bits |
| `c.depth` | `int` | Circuit depth (critical path) |
| `c.gate_count` | `int` | Total gates |
| `c.parameters` | `list[str]` | Free parameter names |
| `c.n_parameters` | `int` | Number of free parameters |

### Single-Qubit Gates

| Method | Matrix | Params |
|--------|--------|--------|
| `c.h(q)` | $(X+Z)/\sqrt{2}$ | — |
| `c.x(q)` | Pauli-X | — |
| `c.y(q)` | Pauli-Y | — |
| `c.z(q)` | Pauli-Z | — |
| `c.s(q)` | $\sqrt{Z}$ | — |
| `c.sdg(q)` | $S^\dagger$ | — |
| `c.t(q)` | $\sqrt{S}$ | — |
| `c.tdg(q)` | $T^\dagger$ | — |
| `c.sx(q)` | $\sqrt{X}$ | — |
| `c.id(q)` | Identity | — |
| `c.rx(theta, q)` | $e^{-i\theta X/2}$ | `theta` |
| `c.ry(theta, q)` | $e^{-i\theta Y/2}$ | `theta` |
| `c.rz(theta, q)` | $e^{-i\theta Z/2}$ | `theta` |
| `c.p(phi, q)` | $\text{diag}(1, e^{i\phi})$ | `phi` |
| `c.u(theta, phi, lam, q)` / `c.u3(...)` | General U3 | `theta, phi, lam` |

### Two-Qubit Gates

| Method | Description |
|--------|-------------|
| `c.cx(ctrl, tgt)` / `c.cnot(ctrl, tgt)` | Controlled-X |
| `c.cz(q1, q2)` | Controlled-Z |
| `c.cy(ctrl, tgt)` | Controlled-Y |
| `c.cp(phi, ctrl, tgt)` | Controlled-phase |
| `c.swap(q1, q2)` | SWAP |
| `c.iswap(q1, q2)` | iSWAP |
| `c.ecr(q1, q2)` | Echoed cross-resonance (IBM hardware-native) |
| `c.rzz(theta, q1, q2)` | $e^{-i\theta ZZ/2}$ |
| `c.rxx(theta, q1, q2)` | $e^{-i\theta XX/2}$ |
| `c.ryy(theta, q1, q2)` | $e^{-i\theta YY/2}$ |

### Three-Qubit Gates

| Method | Description |
|--------|-------------|
| `c.ccx(c1, c2, tgt)` / `c.toffoli(...)` | Toffoli |
| `c.cswap(ctrl, q1, q2)` / `c.fredkin(...)` | Fredkin |

### Other Operations

| Method | Description |
|--------|-------------|
| `c.measure(q, cbit)` | Measure qubit into classical bit |
| `c.measure_all()` | Measure every qubit into the matching classical bit |
| `c.barrier(*qubits)` | Insert barrier |
| `c.reset(q)` | Reset qubit to $\|0\rangle$ |

### Serialization & Export

| Method | Returns | Description |
|--------|---------|-------------|
| `c.to_qasm3()` | `str` | Export to OpenQASM 3.0 |
| `c.to_json()` | `str` | Export to JSON |
| `Circuit.from_json(s)` | `Circuit` | Import from JSON |
| `c.to_gate_list()` | `list[dict]` | Ordered gate records (name, qubits, params) |
| `c.to_ir()` | `Any` | Internal IR handle (for compiler passes) |
| `c.to_unitary()` | `jnp.ndarray` | Dense unitary (small circuits only) |
| `c.draw()` | `str` | ASCII circuit diagram |
| `c.bind(values: dict)` | `Circuit` | Bind parameters to values |

---

## `sf.qml` — JAX Integration

```python
import superfermion as sf

f = sf.qml.circuit_to_jax(circuit, backend="jax")
statevector = f(*params)
```

| Function | Description |
|----------|-------------|
| `sf.qml.circuit_to_jax(circuit, backend)` | Convert circuit to JAX function |

### Frontier Quantum AI (`sf.qml.quantum_ai`)

| Class | Type | Description |
|-------|------|-------------|
| `QuantumCircuitLayer` | Flax Module | Universal VQC layer for hybrid NNs |
| `QuantumGNNLayer` | Flax Module | Quantum Graph Neural Network layer |
| `QuantumGAN` | Flax Module | Generative Adversarial Network (Q-Gen) |
| `QuantumVAE` | Flax Module | Variational Autoencoder (Q-Latent) |
| `QuantumNLP` | Flax Module | QNLP with syntactic circuit embeddings |

### Advanced Measurements (`sf.qml.measurements`)

| Function | Description |
|----------|-------------|
| `von_neumann_entropy(state)` | Compute S = -tr(rho log rho) |
| `purity(state)` | Compute tr(rho^2) |
| `fidelity(state1, state2)` | Compute state overlap |
| `expectation_value(state, op)` | Compute <psi|O|psi> |
| `compute_all_metrics(state)` | Summary of all quantum statistics |

---

## `superfermion.nn.quantum_layer.QuantumLayer`

Flax `nn.Module` wrapping a quantum circuit.

```python
from superfermion.nn.quantum_layer import QuantumLayer

model = QuantumLayer(circuit, backend="jax")
params = model.init(key)
output = model.apply(params)
```

---

## `superfermion.observables.core`

| Class | Description |
|-------|-------------|
| `PauliString(pauli_str, coeffs)` | Tensor product of Paulis, e.g., `"XIZ"` |
| `Hamiltonian(terms: list[PauliString])` | Linear combination of PauliStrings |

```python
h = Hamiltonian([
    PauliString("ZI", coeffs=1.0),
    PauliString("IX", coeffs=0.5),
])
energy = h.expectation(statevector)  # <psi|H|psi>
```

---

## `superfermion.algorithms`

### VQE

```python
from superfermion.algorithms.variational import VQE

vqe = VQE(ansatz, hamiltonian)
result = vqe.minimize()
# result.optimal_value, result.history
```

### QAOA

```python
from superfermion.algorithms.variational import QAOA

qaoa = QAOA(n_qubits=4, edges=[(0,1),(1,2),(2,3)], p_layers=2)
result = qaoa.minimize()
```

### QSVM

```python
from superfermion.algorithms.qsvm import QSVM

qsvm = QSVM(ansatz, num_classes=2, optimizer=None)
result = qsvm.fit(x_train, y_train, iterations=100)
preds = qsvm.predict(result.optimal_params, x_test)
```

### QRL (Quantum Reinforcement Learning)

```python
from superfermion.algorithms.qrl import QuantumREINFORCE

agent = QuantumREINFORCE(ansatz, num_actions=2)
action, probs = agent.select_action(params, state, key)
new_params, new_opt_state = agent.update(params, opt_state, trajectories)
```

### QBM (Quantum Boltzmann Machine)

```python
from superfermion.algorithms.qbm import QBM

model = QBM(n_qubits=3)
energies = model.apply(params, x)             # (batch,) energies
Z = model.get_partition_function(params)       # Partition function
```

---

## `superfermion.qdl` — Quantum Deep Learning

```python
from superfermion.qdl import QResNetBlock, QuantumSelfAttention

# QResNet: y = x + Q(x) with automatic dimensional projection
block = QResNetBlock(circuit)

# Quantum Attention: Q, K, V transformed via quantum circuits
attn = QuantumSelfAttention(circuit, dim=4, num_heads=1)
```

---

## `sf.classical` — JAX Classical AI

High-performance classical components accelerated by the Superfermion JAX engine.

### Classical Machine Learning (`sf.classical.ml`)
- `SVM`: JAX-accelerated Support Vector Machine.
- `KMeans`: JIT-compatible clustering.
- `JAX_Regression`: Automated linear regression via autograd.

### Classical Neural Networks (`sf.classical.nn`)
- `DeepCNN`: Residual-based convolutional networks.
- `RNN`: LSTM/GRU sequential models.
- `GCN`: Graph Convolutional Networks.

### Mathematical Physics (`sf.classical.math`)
- `solve_heat_equation`: 2D PDE solver with JIT.
- `sv.simulate_classical_vibration`: Time-evolution of dynamical states.

---

## `sf.qec` — Fault-Tolerance

### QEC Manager
```python
from superfermion.qec import QECManager
manager = QECManager()
res = manager.run_logical_lifecycle("steane", error_type="X")
```

### Codes & Decoders
| Symbol | Type | Description |
|--------|------|-------------|
| `SurfaceCode2D` | Code | Standard 2D rotated surface code |
| `HypercubeCode4D` | Code | Frontier 4D topological protection |
| `BivariateBicycleCode`| Code | High-rate LDPC code |
| `MWPMDecoder` | Decoder| Minimum Weight Perfect Matching |
| `UnionFindDecoder` | Decoder| High-speed Union-Find decoder |

---

## `superfermion.qllm` — Quantum LLMs

```python
from superfermion.qllm import QuantumTransformerBlock, QuantumGPT

# Single transformer block with quantum attention
block = QuantumTransformerBlock(circuit, dim=4, num_heads=1)

# Full GPT model with interleaved classical/quantum layers
model = QuantumGPT(
    vocab_size=1000, dim=4, n_layers=4,
    n_heads=1, seq_len=128, q_circuit=circuit
)
```

---

## `superfermion.algorithms.core.AlgorithmResult`

```python
@dataclass
class AlgorithmResult:
    optimal_value: float
    optimal_params: Dict[str, Any]
    history: List[float]
    fidelity_history: Optional[List[float]] = None
    final_fidelity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## `superfermion.qml.fidelity`

```python
from superfermion.qml.fidelity import state_fidelity

f = state_fidelity(state_a, state_b)  # |<a|b>|^2
```

---

## `superfermion.chemistry` — Molecular Simulation

| Function | Description |
|----------|-------------|
| `get_molecular_hamiltonian("H2")` | Get Hamiltonian for small molecules |
| `uccsd_ansatz(n_qubits, n_electrons)` | Unitary Coupled Cluster ansatz |
| `FermionicOperator(terms)` | Base class for fermionic ops |

```python
from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz

H = get_molecular_hamiltonian("H2")
ansatz = uccsd_ansatz(n_qubits=2, n_electrons=2)
```

---

## `superfermion.primitives` — Qiskit-v2 compatible primitives

```python
from superfermion.primitives import SFEstimator, SFSampler
from superfermion.observables.core import SparsePauliOp

H = SparsePauliOp.from_dict({"ZZ": 1.0, "XI": 0.5})

est = SFEstimator(backend="statevector")
[result] = est.run([(circuit, H)]).result()
print(result.data.evs)           # <psi|H|psi>

samp = SFSampler(backend="statevector")
[result] = samp.run([circuit], shots=1000).result()
print(result.data.meas.counts)   # {'00': 520, '11': 480}
```

`SparsePauliOp` supports `.from_dict`, `.from_list`, `+`, `*`, and
`.expectation(statevector)`. `expval(statevector, observable)` is
re-exported at `sf.expval` for convenience.

---

## `superfermion.Pipeline` — declarative workflows

```python
from superfermion import Pipeline
from superfermion.data.preprocessing import min_max_scale

pipe = Pipeline([
    ("scale",   min_max_scale),
    ("quantum", QuantumLayer(circuit=my_ansatz)),
])
out = pipe.execute(X, params=params)
```

Also exposes `sf.make_pipeline(*steps)` for unnamed steps.

---

## `superfermion.train` — JAX training loop

```python
import optax
import superfermion as sf

state, history = sf.train(
    model,
    train_loader,
    optimizer = optax.adam(1e-3),
    loss_fn   = my_loss,
    epochs    = 10,
    val_loader = val_loader,
    jit       = True,
)
```

Returns a Flax `TrainState` plus `{"train_loss": [...], "val_loss": [...], "epoch_time": [...]}`.

---

## `superfermion.utils.analytics`

| Function | Description |
|----------|-------------|
| `estimate_cost(circuit)` | Return `{gate_count, depth, qubits, memory_bytes}` without running. |
| `benchmark(circuit, backend="jax", iterations=10)` | Time-averaged ms/iteration. |

---

## Advanced modules

These top-level subpackages are exported from `superfermion` and
importable directly. They're documented here as the surface most users
touch; see the source for full signatures.

### `sf.bridge` — cross-framework import/export
```python
from superfermion.bridge import (
    from_qiskit, from_qasm, from_cirq,    # import to SF
    to_qiskit, to_qasm3, to_cirq, to_pennylane,  # export from SF
    to_braket, to_ionq                     # cloud provider export
)

# Import from other frameworks
sf_circ = from_qasm(qasm_str)      # OpenQASM 2.0 / 3.0
sf_circ = from_qiskit(qc)          # qiskit.QuantumCircuit
sf_circ = from_cirq(cirq_circ)     # cirq.Circuit (NEW)

# Export to other frameworks
qasm    = to_qasm3(sf_circ)
qiskit  = to_qiskit(sf_circ)
cirq    = to_cirq(sf_circ)         # (NEW)
qfunc   = to_pennylane(sf_circ)    # returns callable for qml.qnode (NEW)
braket  = to_braket(sf_circ)       # braket.Circuit
ionq    = to_ionq(sf_circ)         # IonQ JSON format
```

| Function | Description |
|----------|-------------|
| `from_qiskit(qc)` | Convert Qiskit QuantumCircuit to SF |
| `from_qasm(qasm_str)` | Parse OpenQASM 2.0/3.0 to SF |
| `from_cirq(cirq_circuit)` | Convert Cirq Circuit to SF (NEW) |
| `to_qiskit(circuit)` | Convert SF Circuit to Qiskit |
| `to_qasm3(circuit)` | Export SF Circuit to OpenQASM 3.0 |
| `to_cirq(circuit)` | Convert SF Circuit to Cirq (NEW) |
| `to_pennylane(circuit)` | Convert SF Circuit to PennyLane qfunc (NEW) |
| `to_braket(circuit)` | Convert SF Circuit to AWS Braket |
| `to_ionq(circuit)` | Convert SF Circuit to IonQ JSON |

> Endianness note: the bridge silently reverses qubit indices. Do not
> double-reverse.

> Cirq note: Rotation gates (Rx, Ry, Rz, XPowGate, etc.) are converted with
> angle extraction. CXPowGate and CZPowGate map to controlled-X/Z.

### `sf.compiler` — passes & transpilation
```python
import superfermion as sf
from superfermion.compiler import RotationMergingPass, PassManager

compiled = sf.compile(circuit, target="ibm_eagle")
# or build your own pipeline
pm = PassManager([RotationMergingPass()])
compiled = pm.run(circuit)
```

### `sf.mitigation` — error mitigation
```python
from superfermion.mitigation import zne, zne_with_calibration, readout_correction, calibration_based_noise_model

# Basic ZNE (circuit folding + Richardson extrapolation)
corrected = zne(circuit, observable_fn, scale_factors=[1, 2, 3], backend="jax")

# Calibration-driven ZNE (uses real device gate fidelities)
from superfermion.pulse.calibration import CalibrationSet
cals = CalibrationSet("ibm_brisbane", dt=0.222)
cals.add_default_single_qubit(0)
cals.add_default_two_qubit(0, 1)
result = zne_with_calibration(circuit, observable_fn, cals, scale_factors=[1, 2, 3])
# result["zne_value"] — extrapolated expectation
# result["raw_values"] — per-scale-factor values
# result["noise_params"] — calibration-derived parameters

# Build NoiseModel from backend name
nm = calibration_based_noise_model("ibm_eagle")
```

| Function | Description |
|----------|-------------|
| `zne(circuit, observable_fn, scale_factors, backend)` | Zero-Noise Extrapolation via circuit folding |
| `zne_with_calibration(circuit, observable_fn, calibration, scale_factors, backend, apply_readout_correction)` | ZNE driven by CalibrationSet or NoiseModel |
| `readout_correction(counts, calibration_matrix, n_qubits)` | Invert measurement-error confusion matrix |
| `calibration_based_noise_model(backend_name, dt)` | Build NoiseModel from backend calibration defaults |

### `sf.noise` — noise channels & models
```python
from superfermion.noise import NoiseModel, depolarizing_channel, amplitude_damping
nm = NoiseModel().add_gate_noise(depolarizing_channel(0.01), gate="CX")
```
Pair with the `density_matrix` backend (see [backends.md](backends.md)).

### `sf.pulse` — pulse-level control
```python
from superfermion.pulse import Schedule, GaussianPulse, DRAGPulse, CalibrationSet
sched = Schedule()
sched.play(GaussianPulse(duration=160, sigma=40, amp=0.2), channel="q0")

# Calibration-driven noise modeling (ZNE bridge)
cals = CalibrationSet("ibm_brisbane", dt=0.222)
cals.add_default_single_qubit(0)
cals.add_default_two_qubit(0, 1)
params = cals.extract_noise_params()   # {depolarizing_1q, readout_error, ...}
nm = cals.to_noise_model()             # NoiseModel from calibration
```

| Class / Method | Description |
|----------------|-------------|
| `Schedule()` | Multi-channel pulse schedule |
| `GaussianPulse(duration, sigma, amp)` | Gaussian waveform |
| `DRAGPulse(duration, sigma, amp, beta)` | DRAG pulse (transmon leakage suppression) |
| `CalibrationSet(backend_name, dt)` | Gate calibration manager |
| `CalibrationSet.extract_noise_params()` | Extract noise parameters from gate fidelities |
| `CalibrationSet.to_noise_model()` | Build NoiseModel from calibration data |

### `sf.experiment` — experiment tracking
```python
from superfermion.experiment import Tracker, ExperimentRun, ModelRegistry
with Tracker("vqe-h2") as run:
    run.log_metric("energy", -1.137)
    run.log_params({"layers": 2})
```

### `sf.runtime` — resource arbiter / job scheduler
```python
from superfermion.runtime import ResourceArbiter
arbiter = ResourceArbiter()
job = arbiter.submit(circuit, backend="mps", shots=1024)
result = job.result()

# ── Cloud Job Scheduler ─────────────────────────
from superfermion.runtime.scheduler import CloudScheduler, JobPriority, SchedulingPolicy

# Create scheduler with cost-aware routing
scheduler = CloudScheduler(max_workers=8, policy=SchedulingPolicy.PRIORITY)
scheduler.register_backend("ibm_brisbane", provider="ibm", max_concurrent=4)
scheduler.register_backend("jax", provider="local")

# Submit single job
job_id = scheduler.submit(circuit, backend="jax", priority=JobPriority.HIGH)
result = scheduler.wait_for(job_id, timeout=30)

# Batch submission
batch = scheduler.submit_batch([c1, c2, c3], backend="jax")

# Dependency chains (B runs after A)
jid_a = scheduler.submit(circuit_a)
jid_b = scheduler.submit(circuit_b, dependencies=[jid_a])

# Scheduler metrics
metrics = scheduler.metrics()
# {"total_jobs": 5, "completed": 4, "running": 1, "avg_wait_ms": 2.3, ...}

# Context manager
with CloudScheduler(max_workers=4) as sched:
    jid = sched.submit(circuit)
    result = sched.wait_for(jid)
```

| Class / Enum | Description |
|--------------|-------------|
| `CloudScheduler(max_workers, policy, poll_interval)` | Priority-queue job dispatcher |
| `SchedulerJob` | Job dataclass (id, circuit, backend, priority, status, dependencies) |
| `JobPriority` | Enum: CRITICAL(5) > HIGH(4) > NORMAL(3) > LOW(2) > BATCH(1) |
| `SchedulingPolicy` | Enum: PRIORITY, COST_AWARE, ROUND_ROBIN, LATENCY_FIRST |
| `BackendRegistration` | Backend metadata (provider, max_concurrent, cost, latency) |
| `BatchResult` | Aggregated batch result (job_ids, results, errors) |
| `get_scheduler()` | Module-level singleton scheduler |

### `sf.security` — credentials, audit, transport
```python
from superfermion.security import CredentialStore, AuditLog, TLSConfig, TokenManager
creds = CredentialStore.from_env()
token = TokenManager().mint(scope=["run"])
```

### `sf.telemetry` — structured logs, traces, metrics
```python
from superfermion.telemetry import StructuredLogger, Tracer, MetricsCollector
log   = StructuredLogger("vqe")
tracer = Tracer("compile")
with tracer.span("decompose"):
    ...
```

### `sf.data` — datasets & preprocessing
```python
from superfermion.data import Dataset, DataLoader
from superfermion.data.preprocessing import min_max_scale, angle_encoding_transform
loader = DataLoader(Dataset(X, y), batch_size=32, shuffle=True)
```

### `sf.serialization` — portable snapshots
```python
from superfermion.serialization import save_circuit, load_circuit, save_params, load_params
save_circuit(circuit, "ansatz.sfc")
restored = load_circuit("ansatz.sfc")
```

### `sf.config` — runtime settings
```python
from superfermion.config import get_config, set_default_backend
set_default_backend("jax")
```

### `sf.serve` — REST / websocket gateway *(experimental)*
```bash
python -m superfermion.serve.gateway --port 8000
```

### `sf.intelligence` — Superpositional agents (DEPRECATED)

> This module is deprecated and will be removed. See [intelligence.md](intelligence.md).
`SuperpositionalAgent`, `QNSCore`, `EntangledBus`
(alias `QuantumIntelligenceBus`).
