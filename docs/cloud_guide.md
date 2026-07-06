# Cloud & Distributed Scaling Guide

> **Superfermion Cloud** — From individual research to global enterprise deployment.

---

## ⚡ QPU Execution via Bridge Layer

Superfermion can transpile circuits directly to native QPU formats and
submit them to real quantum hardware. The bridge layer handles all
provider-specific serialization, including known hardware quirks.

### Supported QPU Providers

| Provider | Backend Key | Bridge Function | Native Format |
|----------|------------|----------------|---------------|
| **IBM Quantum** | `ibm_brisbane` | `sf.runtime.connect('ibm', ...)` | OpenQASM 3 |
| **IonQ** | `ionq.simulator` / `ionq.qpu` | `sf.bridge.to_ionq(circuit)` | IonQ JSON v0.3 |
| **AWS Braket** | via `sf.bridge.to_braket(circuit)` | Amazon Braket SDK | Braket IR |
| **Rigetti** | via OpenQuantum | `sf.runtime.connect('openquantum', ...)` | Quil |
| **IQM** | via OpenQuantum | `sf.runtime.connect('openquantum', ...)` | IQM JSON |

### Quick Example: Run on Real Hardware

```python
import superfermion as sf

# Build a circuit
c = sf.Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all()

# Run locally for ground truth
ground_truth = sf.run(c, backend="statevector")

# Submit to IonQ simulator
sf.runtime.connect('ionq', api_key="your-ionq-key")
ionq_job = sf.runtime.run(c, backend="ionq.simulator", shots=1024)
print(f"IonQ job: {ionq_job.job_id}")

# Submit to IBM
sf.runtime.connect('ibm', token="your-ibm-token")
ibm_job = sf.runtime.run(c, backend="ibm_brisbane", shots=1024)
print(f"IBM job: {ibm_job.job_id}")
```

### IonQ Bridge: Known Quirks & Workarounds

The `sf.bridge.to_ionq()` function converts Superfermion circuits to IonQ's
native JSON gate format. It includes automatic workarounds for known IonQ
simulator issues:

- **CNOT ordering**: IonQ's simulator has a bug where `cnot(control=1, target=0)`
  in 6+ qubit circuits collapses the state space. The bridge automatically
  decomposes every reversed CNOT (control > target) using the identity:
  \(\text{CNOT}(c,t) = H_c \cdot H_t \cdot \text{CNOT}(t,c) \cdot H_c \cdot H_t\)
  This uses only forward CNOT (control < target) and avoids the bug entirely.
  Fidelity verified at 1.0 against statevector ground truth.

- **Bit ordering**: IonQ uses LSB-first encoding; the bridge handles reversal
  automatically so SF's MSB-first convention is preserved.

### Cross-Platform Validation Results (2026-05-31)

All 6 standard benchmark circuits pass on both IonQ and IBM simulators:

| Circuit | Qubits | IonQ States | IBM States | TV(I-G) |
|---------|--------|------------|------------|---------|
| Bell | 2 | 2 | 4 | 0.0079 |
| GHZ | 5 | 2 | 17 | 0.00075 |
| BV | 8 | 2 | 36 | 0.0043 |
| DJ Balanced | 6 | 2 | 59 | 0.00165 |
| QAOA | 4 | 16 | 16 | 0.0097 |
| Random Clifford | 6 | **32** | 64 | **0.0109** |

---

## ☁️ sf-serve: The API Gateway

Superfermion includes a production-ready FastAPI gateway that exposes the platform's capabilities via REST and WebSockets.

### Starting the Server
```bash
uvicorn superfermion.serve.app:app --host 0.0.0.0 --port 8000
```

### API Reference (v1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/run` | Submit a circuit (JSON/QASM) with auth. |
| `GET` | `/v1/jobs/{id}` | Poll result or status. |
| `POST` | `/v1/intelligence/think` | DEPRECATED — Ask an agent to process an observation. |
| `WS` | `/v1/monitor` | Real-time telemetry (telemetry/load). |

---

## 🛡️ Security & Multi-Tenancy

The gateway is secured via the **X-SF-API-KEY** header.

```python
import httpx

headers = {"X-SF-API-KEY": "your-secret-key"}
payload = {"qasm": "...", "backend": "jax"}

response = httpx.post("http://localhost:8000/v1/run", json=payload, headers=headers)
```

### Usage Quotas
Superfermion enforces strict resource limits to ensure stability:
- **Free Tier**: 12 Qubits max.
- **Pro Tier**: 28 Qubits max.
- **Enterprise**: Custom limits (127+ Qubits).

---

## 🏗️ GPU Cluster Orchestration

For large-scale simulations (20+ qubits), Superfermion leverages **Distributed JAX** via the `cluster` backend.

### How it Works
1.  **Mesh Discovery**: The `ClusterManager` detects all local and network GPUs.
2.  **Sharding**: The $2^N$ statevector is sliced into "shards" across devices.
3.  **Parallel Execution**: JAX executes the gates in parallel on every shard simultaneously.

### Usage
```python
# The system automatically handles sharding and communication
result = sf.run(large_circuit, backend="cluster")
```

---

## 🤖 Resource Arbiter: The Routing Logic

The **Arbiter** is the intelligent dispatcher that decides where to run your code:

| Qubit Count | Decision | Rationale |
|-------------|----------|-----------|
| 1 - 20 | **Local JAX** | Low latency, fast simulation. |
| 21 - 35 | **Distributed Cluster** | Scales memory across many GPUs. |
| 36+ | **Quantum Cloud (QPU)** | Computational complexity exceeds classical memory. |

You can override this by specifying a `target` explicitly:
```python
sf.run(circuit, target="ibm_eagle") # Forces IBM Cloud
```

---

## ⚡ Cloud Job Scheduler

The `CloudScheduler` provides enterprise-grade distributed job execution
with priority queues, dependency chains, and multi-provider dispatch.

### Quick Start
```python
from superfermion.runtime.scheduler import CloudScheduler, JobPriority, SchedulingPolicy

# Create scheduler
scheduler = CloudScheduler(max_workers=8, policy=SchedulingPolicy.PRIORITY)

# Register backends
scheduler.register_backend("ibm_brisbane", provider="ibm", max_concurrent=4)
scheduler.register_backend("ionq_forte", provider="ionq", max_concurrent=1)
scheduler.register_backend("jax", provider="local")

# Submit a job
job_id = scheduler.submit(circuit, backend="jax", priority=JobPriority.HIGH)
result = scheduler.wait_for(job_id, timeout=30)
```

### Priority Levels
| Level | Value | Use Case |
|-------|-------|----------|
| `CRITICAL` | 5 | Production inference, SLA-bound tasks |
| `HIGH` | 4 | User-facing requests |
| `NORMAL` | 3 | Default development jobs |
| `LOW` | 2 | Background validation |
| `BATCH` | 1 | Overnight sweeps, hyperparameter tuning |

### Scheduling Policies
| Policy | Behavior |
|--------|----------|
| `PRIORITY` | Strictly by JobPriority (default) |
| `COST_AWARE` | Cheapest backend first |
| `ROUND_ROBIN` | Fair distribution across all backends |
| `LATENCY_FIRST` | Fastest (local) backends first |

### Batch Submission
```python
circuits = [make_ansatz(i) for i in range(10)]
batch = scheduler.submit_batch(circuits, backend="jax", priority=JobPriority.BATCH)

for jid in batch.job_ids:
    result = scheduler.wait_for(jid)
```

### Job Dependencies
```python
# job_b will only run after job_a completes
jid_a = scheduler.submit(preprocess_circuit)
jid_b = scheduler.submit(main_circuit, dependencies=[jid_a])
```

### Metrics
```python
metrics = scheduler.metrics()
# {
#   "total_jobs": 150,
#   "queued": 5,
#   "running": 3,
#   "completed": 140,
#   "failed": 2,
#   "avg_wait_ms": 12.5,
#   "avg_duration_ms": 45.3,
#   "backend_loads": {"jax": 3, "ibm_brisbane": 2}
# }
```

### Context Manager
```python
with CloudScheduler(max_workers=4) as scheduler:
    scheduler.register_backend("jax", provider="local")
    jid = scheduler.submit(circuit)
    result = scheduler.wait_for(jid)
# Scheduler auto-stops on exit
```
