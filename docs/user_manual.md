# SuperFermion User Manual

> Single linear "read this first" reference.  Should take ~30 minutes to
> work through.  See [`docs/api_reference.md`](api_reference.md) for the
> full API surface and [`docs/benchmarks.md`](benchmarks.md) for the
> canonical performance scoreboard.

## Table of contents

1. [Install](#1-install)
2. [Five-minute tour](#2-five-minute-tour)
3. [Backends — which to pick](#3-backends--which-to-pick)
4. [Building circuits](#4-building-circuits)
5. [Observables and expectation values](#5-observables-and-expectation-values)
6. [Gradients and training](#6-gradients-and-training)
7. [Algorithm presets (VQE, QAOA, QSVM, QEC)](#7-algorithm-presets)
8. [Noise modeling & mitigation](#8-noise-modeling--mitigation)
9. [Cloud job scheduling](#9-cloud-job-scheduling)
10. [CLI](#10-cli)
11. [Performance tips](#11-performance-tips)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Install

### Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.10–3.13 | runtime |
| Rust | 1.75+ | compiles the `_sf_core` extension |
| `maturin` | 1.0+ | builds the Rust extension as a Python wheel |
| ~3 GB free disk | — | cargo build artefacts |

### Step 1 — clone and Python install

```bash
git clone https://github.com/superfermion/superfermion.git
cd superfermion
pip install -e .          # or: pip install -e ".[dev]" for test deps
```

### Step 2 — build the Rust extension (one-time)

```bash
cd crates/sf-bindings
maturin develop --release
cd ../..
```

### Step 3 — copy the freshly built `.pyd` into the package (Windows)

**This is non-obvious and bites people.**  `maturin develop` writes the
new artefact to `.venv/Lib/site-packages/_sf_core/`, but Python imports
SuperFermion's Rust extension from `superfermion/_sf_core.pyd` (inside
the package).  Without this copy, your `pip install -e .` install will
keep loading the old binary.

```bash
cp .venv/Lib/site-packages/_sf_core/_sf_core.cp313-win_amd64.pyd \
   superfermion/_sf_core.pyd
```

(Adjust the wheel filename for your Python version.)

### Step 4 — verify

```bash
python -m pytest tests/test_rust_kernel_correctness.py \
                 tests/test_singularity_routing_and_expval.py \
                 tests/test_stabilizer.py \
                 tests/test_adjoint_grad.py -q
# expect: 113 passed
```

---

## 2. Five-minute tour

```python
import superfermion as sf
from superfermion.backends.registry import get_backend

# 1) Build a circuit fluently.  q0 is MSB.
qc = sf.Circuit(3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)         # GHZ on 3 qubits

# 2) Run on any backend.  "singularity" auto-routes to the fastest path
#    given the circuit (Clifford -> stabilizer; entangled -> MPS;
#    dense small -> Rust SV).
result = get_backend("singularity").run(qc, shots=1024)
print(result.counts)        # {'000': 521, '111': 503}
print(result.statevector)   # numpy array of length 8 (complex128)

# 3) Compute Pauli expectations directly (no SV materialisation).
print(get_backend("singularity").expval(qc, "ZZZ"))   # +1.0 (GHZ)
print(get_backend("singularity").expval(qc, "ZZI"))   # +1.0
print(get_backend("singularity").expval(qc, "XII"))   #  0.0

# 4) Take a gradient via adjoint differentiation (1 fwd + 1 bwd pass).
import numpy as np
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.gradient.adjoint import adjoint_grad_vector

p_qc = sf.Circuit(2)
p_qc.ry(sf.param("a"), 0)
p_qc.ry(sf.param("b"), 1)
p_qc.cx(0, 1)
obs = SparsePauliOp.from_dict({"ZZ": 1.0})
grad = adjoint_grad_vector(p_qc, obs, ["a", "b"], np.array([0.3, 0.7]))
# grad = numpy array shape (2,)
```

That's the whole API surface in one screen.

---

## 3. Backends — which to pick

| Backend | Best for | Cost model | Notes |
|---|---|---|---|
| **`singularity`** | "I just want it fast" | auto-routes | recommended default |
| `stabilizer` | Clifford-only circuits | O(n²) per gate, O(n²) per Pauli | beats Aer-stabilizer 2-4× |
| `mps` | entanglement-bounded circuits, Pauli expvals | O(n·χ³) per gate | beats aer-mps 5-11× on QAOA / Trotter at moderate bond |
| `rust` | dense statevector you need to read | O(2ⁿ) per gate, AVX-2 SIMD | use when you must inspect `.statevector` |
| `statevector` | tiny n (≤8), debugging | pure-Python | slowest by 5-10× of `rust` |
| `density_matrix` | noisy circuits, ≤12 qubits | O(4ⁿ) memory | not used for noiseless workloads |
| `jax` / `jax_mps` | JAX gradient interop | JIT-compiled | for differentiable workflows that integrate JAX directly |
| `cuda` / `cuda_mps` | CUDA GPU acceleration | needs CuPy + GPU | rarely faster than Rust on small n; CUDA cost amortises at n≥18 |
| `supremacy` | random-circuit benchmarks | — | research-only |

**`get_backend("singularity")` is the right answer 90 % of the time.**

Routing inside `singularity.run(circuit, shots)`:

```
n > 22 and is_clifford   → stabilizer  (poly-time, exact)
n ≤ 10                   → numpy-tensordot (fast cold-start)
n ≤ 32 (RAM permits)     → Rust dense statevector (AVX-2 + ping-pong)
n > 32                   → MPS (Rust QR boundary contraction)
```

`singularity.expval(circuit, observable)` and
`singularity.grad(circuit, obs, names, values)` add their own routing on
top: Clifford → stabilizer; otherwise `sf.mps.expval` for `expval`, and
`adjoint_grad_vector` for `grad`.

---

## 4. Building circuits

### Gates

```python
qc = sf.Circuit(n)

# 1-qubit gates
qc.h(q); qc.x(q); qc.y(q); qc.z(q)
qc.s(q); qc.sdg(q); qc.t(q); qc.tdg(q); qc.sx(q)
qc.rx(theta, q); qc.ry(theta, q); qc.rz(theta, q)
qc.p(phi, q)                              # phase gate
qc.u(theta, phi, lam, q)                  # general 1q

# 2-qubit gates
qc.cx(c, t); qc.cnot(c, t)                # CNOT
qc.cz(a, b); qc.cy(c, t)
qc.swap(a, b); qc.iswap(a, b)
qc.cp(phi, c, t)                          # controlled phase
qc.crx(theta, c, t); qc.cry(theta, c, t); qc.crz(theta, c, t)
qc.rxx(theta, a, b); qc.ryy(theta, a, b); qc.rzz(theta, a, b)
qc.ch(c, t)
qc.ecr(a, b)                              # echoed cross-resonance

# 3-qubit
qc.ccx(c1, c2, t)                         # Toffoli
qc.cswap(c, t1, t2)                       # Fredkin
```

All methods support fluent chaining: `qc.h(0).cx(0, 1).rz(0.7, 1)`.

### Symbolic parameters

```python
qc = sf.Circuit(2)
qc.ry(sf.param("theta"), 0)
qc.cx(0, 1)
qc.rz(sf.param("phi"), 1)

bound = qc.bind({"theta": 0.5, "phi": 0.7})
# bound is a new Circuit with concrete float params
```

### Endianness

**SuperFermion uses q0 = MSB everywhere.**  String position 0 of a Pauli
operator string corresponds to qubit 0:

```python
"ZZII"   # Z on q0, Z on q1
```

This is the **opposite** of Qiskit (q0 = LSB).  The
`superfermion.bridge.to_qiskit` helper transparently reverses qubit
indices when interoperating with Qiskit.

---

## 5. Observables and expectation values

### Pauli strings and `SparsePauliOp`

```python
from superfermion.observables.core import SparsePauliOp

# Single Pauli
op1 = SparsePauliOp.from_dict({"ZZII": 1.0})

# Linear combination
op2 = SparsePauliOp.from_dict({
    "ZIIZ":  0.5,
    "XIIIX": 0.3,
    "YYII":  0.2,
})
```

### Computing `<O>`

```python
sing = get_backend("singularity")
val = sing.expval(qc, op2)               # SparsePauliOp accepted
val = sing.expval(qc, "ZZII")            # raw string accepted
val = sing.expval(qc, {"ZZII": 1.0})     # dict accepted
```

The routing automatically picks the fastest backend:
- Clifford circuit → `stabilizer.expval` (machine-eps exact)
- otherwise → `mps.expval` (boundary contraction in Rust, no 2ⁿ SV)

### Computing `<O>` via dense path

If you need the full statevector for some other reason:

```python
sv = get_backend("rust").run(qc, shots=0).statevector
import numpy as np
val = float(np.real(op2._fast_expval(np.asarray(sv, dtype=np.complex128))))
```

---

## 6. Gradients and training

### Adjoint differentiation (recommended)

```python
from superfermion.qml.gradient.adjoint import adjoint_grad_vector

grad = adjoint_grad_vector(circuit, observable, param_names, param_values)
# Returns: numpy array of shape (len(param_names),)
```

Cost: **1 forward + 1 backward pass = O(M·2ⁿ) regardless of N.**
At n=10 with 20 params, **15-28× faster than PennyLane Lightning's
adjoint**.

### Parameter-shift rule (when you need shot-noise modelling)

```python
from superfermion.qml.gradient.parameter_shift import parameter_shift_grad_vector

grad = parameter_shift_grad_vector(circuit, observable, param_names,
                                    param_values, backend="statevector",
                                    shots=0)
# Cost: 2N forward passes
```

### Training loop

```python
import numpy as np
theta = np.random.uniform(-np.pi, np.pi, len(param_names))
for step in range(100):
    g = adjoint_grad_vector(circuit, observable, param_names, theta)
    theta -= 0.05 * g
```

VQE-H₂ (4 qubits, 20-step SGD) runs in **~30 ms** with `sf.adjoint` vs
580 ms with PennyLane Lightning — see
[`docs/benchmarks.md`](benchmarks.md).

---

## 7. Algorithm presets

The `superfermion.algorithms` module ships ready-to-run drivers for
common variational algorithms.

```python
# VQE
from superfermion.algorithms.variational import VQE
from superfermion.observables.core import Hamiltonian, PauliString

ansatz = sf.Circuit(2).ry(sf.param("t"), 0).cx(0, 1)
H = Hamiltonian([PauliString("ZZ", coeffs=1.0)])
result = VQE(ansatz, H).minimize()
print(result.optimal_value)

# QAOA on a ring graph
from superfermion.algorithms.variational import QAOA
qaoa = QAOA(n_qubits=6, edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], p_layers=3)
optimal = qaoa.minimize()

# QEC lifecycle (Steane code)
from superfermion.algorithms.qec import SteaneLifecycle
lifecycle = SteaneLifecycle(error="X")
results = lifecycle.run()
```

For runnable end-to-end examples see
[`docs/tutorials/`](tutorials/) — 8 scripts covering VQE-H₂, QAOA-MaxCut,
QSVM, QRL, QBM, QLLM-GPT, QEC, hardware compile.

---

## 8. Noise modeling & mitigation

Superfermion includes a full noise simulation and error mitigation stack
that can be driven by real device calibration data.

### Building a noise model

```python
from superfermion.noise import NoiseModel, ibm_eagle_noise, ideal_noise

# Preset: IBM Eagle (127-qubit)
nm = ibm_eagle_noise()
# adds: 0.1% 1Q depolarizing, 1% 2Q depolarizing,
#       0.05% amplitude damping, 0.1% phase damping,
#       1% readout error

# Preset: no noise (ideal)
nm = ideal_noise()

# Custom
nm = (NoiseModel()
    .add_depolarizing(0.01)           # 1% 1Q
    .add_depolarizing(0.05, n_qubits=2)  # 5% 2Q
    .add_amplitude_damping(0.005)
    .add_readout_error(0.02))
```

### Device-calibrated noise models

The `CalibrationSet` class stores gate fidelities (from hardware
calibration data) and can automatically construct a `NoiseModel`:

```python
from superfermion.pulse.calibration import CalibrationSet

# Create calibration with default gate parameters
cals = CalibrationSet("ibm_brisbane", dt=0.222)
cals.add_default_single_qubit(0)
cals.add_default_single_qubit(1)
cals.add_default_two_qubit(0, 1)

# Extract noise parameters from gate fidelities
params = cals.extract_noise_params()
# {"avg_1q_fidelity": 0.999767, "avg_2q_fidelity": 0.995,
#  "depolarizing_1q": 0.000311, "readout_error": 0.000117, ...}

# Build a NoiseModel directly from calibration
nm = cals.to_noise_model()
```

### Zero-Noise Extrapolation (ZNE)

#### Basic ZNE (circuit folding)

```python
from superfermion.mitigation import zne

c = sf.Circuit(2).h(0).cx(0, 1)

def observable(sv):
    return float(sv[0].real)  # expectation of Z on q0

mitigated = zne(c, observable, scale_factors=[1, 2, 3], backend="jax")
```

#### Calibration-driven ZNE (uses real device data)

```python
from superfermion.mitigation import zne_with_calibration

result = zne_with_calibration(c, observable, cals, scale_factors=[1, 2, 3])

print(result["zne_value"])       # zero-noise extrapolated expectation
print(result["raw_values"])      # per-scale-factor expectations
print(result["noise_params"])    # calibration-derived parameters
```

#### Quick calibrated model from backend name

```python
from superfermion.mitigation import calibration_based_noise_model

nm = calibration_based_noise_model("ibm_eagle")
# Returns a fully-populated NoiseModel
```

### Readout error correction

```python
from superfermion.mitigation import readout_correction
import jax.numpy as jnp

cal_matrix = jnp.eye(4)  # ideal calibration
counts = {"00": 250, "01": 250, "10": 250, "11": 250}
corrected = readout_correction(counts, cal_matrix, n_qubits=2)
```

---

## 9. Cloud job scheduling

The `CloudScheduler` provides enterprise-grade distributed execution
with priority queues, batch submission, and dependency chains.

### Quick start

```python
from superfermion.runtime.scheduler import CloudScheduler, JobPriority

scheduler = CloudScheduler(max_workers=8)
scheduler.register_backend("jax", provider="local")

job_id = scheduler.submit(circuit, backend="jax", priority=JobPriority.HIGH)
result = scheduler.wait_for(job_id, timeout=30)
```

### Priority levels (CRITICAL → BATCH)

```python
from superfermion.runtime.scheduler import JobPriority

scheduler.submit(circuit, priority=JobPriority.CRITICAL)  # value=5
scheduler.submit(circuit, priority=JobPriority.HIGH)      # value=4
scheduler.submit(circuit, priority=JobPriority.NORMAL)    # value=3 (default)
scheduler.submit(circuit, priority=JobPriority.LOW)       # value=2
scheduler.submit(circuit, priority=JobPriority.BATCH)     # value=1
```

### Batch submission

```python
circuits = [sf.Circuit(2).h(0).cx(0, 1) for _ in range(5)]
batch = scheduler.submit_batch(circuits, backend="jax")

for jid in batch.job_ids:
    result = scheduler.wait_for(jid, timeout=30)
```

### Job dependencies

```python
# job_b only runs after job_a completes
jid_a = scheduler.submit(preprocess_circuit)
jid_b = scheduler.submit(main_circuit, dependencies=[jid_a])
```

### Scheduling policies

```python
from superfermion.runtime.scheduler import SchedulingPolicy

# Default: strictly by priority
scheduler = CloudScheduler(policy=SchedulingPolicy.PRIORITY)

# Run on cheapest backend first
scheduler = CloudScheduler(policy=SchedulingPolicy.COST_AWARE)

# Fair distribution
scheduler = CloudScheduler(policy=SchedulingPolicy.ROUND_ROBIN)

# Fastest (local) backends first
scheduler = CloudScheduler(policy=SchedulingPolicy.LATENCY_FIRST)
```

### Metrics & monitoring

```python
metrics = scheduler.metrics()
# {"total_jobs": 50, "completed": 48, "running": 2,
#  "failed": 0, "avg_wait_ms": 3.2, "backend_loads": {"jax": 2}}
```

### Context manager

```python
with CloudScheduler(max_workers=4) as sched:
    sched.register_backend("jax", provider="local")
    jid = sched.submit(circuit)
    result = sched.wait_for(jid)
# Scheduler auto-stops on exit
```

---

## 10. CLI

11 first-class commands, all with `--help`:

```bash
sf info               # System diagnostic
sf version            # SF + dep versions
sf validate           # 9-point install audit
sf backends           # list 11 simulator backends
sf benchmark          # quick n-qubit performance sweep
sf run circuit.json   # run a JSON-encoded circuit
sf vqe                # VQE on H2 / LiH / BeH2 / TFIM presets
sf qaoa               # QAOA on preset graphs
sf chemistry          # quantum-chemistry workflow (operator inspection + VQE)
sf qec                # logical-qubit lifecycle on Steane / Surface
sf shor               # Shor's measurement post-processing
```

See [`docs/cli.md`](cli.md) for full argument reference and the
circuit-JSON schema.

---

## 11. Performance tips

| Use case | Recipe |
|---|---|
| Largest possible Clifford circuit | `get_backend("stabilizer").expval(qc, pauli_str)` — works to ~1000 qubits |
| Repeated runs of the same circuit | `singularity.pre_bake(circuit)` then `.run(...)` — topology cache makes warm runs microseconds |
| QAOA / VQE expvals | `get_backend("singularity").expval(...)` routes to MPS automatically |
| Large dense circuits (n=20+) | use `sf.rust` directly; AVX-2 ping-pong gives ~5× over `sf.statevector` |
| Heisenberg-style highly-entangled | currently SF's weakness; `aer-mps` wins.  Use it for now and watch for the BLAS-MPS update |
| Shot-noise simulation | `singularity.run(qc, shots=N)` — for Clifford circuits we route through tableau sampling, otherwise the dense path |

**The single most important tip:**  After every `maturin develop --release`,
remember to copy the `.pyd` into the package (see step 3 of install).
The stale-binary trap silently makes Rust changes look like no-ops.

---

## 12. Troubleshooting

### "My Rust changes aren't taking effect"
Almost certainly the [`.pyd`-copy step](#step-3--copy-the-freshly-built-pyd-into-the-package-windows).
Check `os.path.getmtime("superfermion/_sf_core.pyd")` — should be recent.

### "`<ZZII>` returns 0 when I expected 1"
SF uses **q0 = MSB** (string position 0 = qubit 0).  Double-check your
Pauli-string convention.  See [Endianness](#endianness).

### "AttributeError: 'list' object has no attribute 'items'"
You're probably calling `op._terms.items()` on a `SparsePauliOp`.
`._terms` is a list of `(str, complex)` tuples, not a dict.  Use:
```python
for pauli_str, coef in op._terms: ...
```

### "Tests fail with `cannot reshape array of size N into shape (1,)`"
You called a backend that auto-dispatches to stabilizer at large n
(returns `statevector=None`) but the calling code expected a populated
`.statevector`.  Use `get_backend("singularity")` for n≤22 (it falls
through to the dense path and populates SV); use `sf.stabilizer`
explicitly for large-n Clifford expvals.

### "Build fails with `os error 112` (no space on disk)"
The cargo target dir is huge.  Free disk:
```bash
cargo clean                   # ~1 GB
rm -rf "$LOCALAPPDATA/Temp/.tmp"*
rm -rf "$LOCALAPPDATA/Temp/cargo"*
pip cache purge
```

### "PennyLane gradient says `argnum`/`argnums`"
PennyLane 0.42+ renamed `argnum` to `argnums`.  Our adjoint API doesn't
care about either; if you're using `qml.grad` directly, pin the
spelling to your installed version.

### "Qiskit `EstimatorV2` returns a wrong gradient"
Qiskit binds parameters in **lexicographic** name order: `t0, t1, t10,
t11, t2, t3, …`.  Use zero-padded names (`t00, t01, …`) to keep the
binding numeric.

---

For deeper material:
- [`CLAUDE.md`](../CLAUDE.md) and [`.claude/notes/`](../.claude/notes/) — developer context
- [`docs/architecture.md`](architecture.md) — full module map
- [`docs/api_reference.md`](api_reference.md) — every public function
- [`docs/benchmarks.md`](benchmarks.md) — canonical scoreboard
- `docs/archive/benchmarks_pre_2026_04_26/` — historical benchmark
  reports (kept for regression diff; understate current SF by ~5-30×)
