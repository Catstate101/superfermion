# Getting Started with Superfermion

> **Superfermion** — The World's Most Advanced Quantum AI Platform.
> Differentiable, Hardware-Agnostic, and Scalable to the Singularity.

---

## Installation

```bash
# Install core dependencies
pip install jax jaxlib flax optax fastapi uvicorn httpx rich

# Clone and Export Path
git clone https://github.com/superfermion/superfermion.git
cd superfermion
$env:PYTHONPATH = "." # Windows
export PYTHONPATH="." # Linux/macOS
```

---

## 🚀 The Three-Pillar Workflow

### 1. Build and Run (Local Simulation)
Create circuits with an intuitive fluent API and execute them on JAX-accelerated backends.

```python
import superfermion as sf

# 1. Build
c = sf.Circuit(2).h(0).cx(0, 1)

# 2. Run (Auto-uses JAX for high speed)
result = sf.run(c, shots=1000)
print(f"Result counts: {result.counts}")
```

### 2. Connect to the World (Hardware-Aware)
Optimize and transpile your circuit for any QPU in the world with one line of code.

```python
# Optimize for IBM Eagle (127 Qubits)
# This handles Basis Translation and Routing automatically!
job = sf.run(c, target="ibm_eagle")

print(f"Job Status: {job.status}")
print(f"Optimal Backend: {job.metadata['backend']}")
```

### 3. Move to the Singularity (Autonomous AI)
Deploy an autonomous agent that "thinks in superposition" and evolves its own brain.
> **Note:** The intelligence module is deprecated and will be removed.

```python
# DEPRECATED — do not use in new code
from superfermion.intelligence import SuperpositionalAgent

# Create an agent with 4 entangled qubits for its internal policy
agent = SuperpositionalAgent(n_qubits=4)

# Ask the agent to 'think' about sensor data
observation = jnp.array([1.2, -0.5, 0.0, 3.14])
thought = agent.think(observation)

print(f"Superposition Dimension: {len(thought)}") # 16 states explored at once
```

---

## 🛠 Advanced Features

### Distributed GPU Clusters
For circuits > 20 qubits, Superfermion automatically shards the statevector across your GPU cluster.

```python
# This uses 'DistributedJAXBackend' under the hood
# 28 qubits = 268 Million Amplitudes distributed across your GPUs
result = sf.run(sf.Circuit(28), backend="cluster")
```

### Secured REST Gateway
Launch the Superfermion Server to expose your quantum resources to the web.

```bash
# Start the production gateway
uvicorn superfermion.serve.app:app --host 0.0.0.0 --port 8000
```

---

## 🛡️ Security & Resource Arbiter
Superfermion includes a built-in **Resource Arbiter** that protects your hardware.
- **Free Tier**: Limited to 12 qubits to prevent memory exhaustion.
- **Depth Guards**: Rejects circuits with > 10,000 gates to prevent infinite execution.
- **Auto-Routing**: If you try running 40 qubits, the system will automatically block local execution and suggest a QPU cloud target.

---

## 📚 Next Steps
- [Architecture Guide](architecture.md): Deep dive into the framework internals.
- [Intelligence Guide](intelligence.md): Learn how the Quantum Neural Singularity (QNS) works.
- [Cloud & Cluster Guide](cloud_guide.md): Scaling your quantum algorithms to datacenters.
- [API Reference](api_reference.md): Complete list of gates, algorithms, and options.
- [CLI Reference](cli.md): 26 CLI commands for the de facto quantum CLI.

---

## 🖥️ CLI Quick Reference

Superfermion provides a comprehensive CLI with 26 commands:

```bash
# Core commands
sf info                    # System info
sf validate                # Installation audit
sf backends                # List 11 simulator backends

# Circuit execution
sf run circuit.json        # Execute circuit
sf benchmark --qubits 10   # Performance sweep

# Algorithms
sf vqe --hamiltonian H2    # VQE optimization
sf qaoa --graph ring6      # QAOA MaxCut
sf chemistry H2 --vqe      # Quantum chemistry
sf qec --code steane       # Error correction

# New de facto CLI commands
sf plugin list             # Manage plugins
sf auth login --provider ibm --token XXX  # Configure credentials
sf convert circuit.json circuit.qasm     # Format conversion
sf estimate circuit.json --backend ibm   # QPU cost estimation
sf compare circuit.json --backends statevector,jax  # Backend comparison
sf jobs list --provider ibm              # Job management
```
