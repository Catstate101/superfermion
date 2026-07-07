# Getting Started with Superfermion

## Installation

```bash
git clone https://github.com/superfermion/superfermion.git
cd superfermion
pip install -e .

# Build the Rust extension (required for simulation)
pip install maturin
cd crates/sf-bindings && maturin develop --release && cd ../..
```

Requirements: Python 3.10+, Rust 1.75+.

---

## Your First Circuit

```python
import superfermion as sf

# Create a Bell state
qc = sf.Circuit(2).h(0).cx(0, 1)

# Run with measurement shots
result = sf.run(qc, device="cpu", shots=1024)
print(result.counts)  # {'00': ~512, '11': ~512}
```

---

## Exact Simulation with `sf.simulate()`

For exact statevector access without sampling:

```python
state = sf.simulate(qc, device="cpu")

# State is a Rust-native handle with rich methods
print(state.numpy())        # [0.707+0j, 0, 0, 0.707+0j]
print(state.entropy())      # 0.0 (pure state)
print(state.purity())       # 1.0
```

---

## Expectation Values

Compute expectation values directly on the quantum state:

```python
# Observable as Rust-format tuples: (pauli_indices, coeff_real, coeff_imag)
# Pauli encoding: 0=I, 1=X, 2=Y, 3=Z
zz_obs = [([3, 3], 1.0, 0.0)]  # ZZ with coefficient 1.0

state = sf.simulate(qc, device="cpu")
print(state.expectation(zz_obs))  # 1.0 for Bell state
```

Or use the Python observable classes:

```python
obs = sf.SparsePauliOp.from_dict({"ZZ": 1.0})
print(sf.expval(state.numpy(), obs))
```

---

## Parameterized Circuits and Gradients

```python
theta = sf.param("theta")
qc = sf.Circuit(1).ry(theta, 0)

# Bind parameters and simulate
bound = qc.bind({"theta": 0.5})
state = sf.simulate(bound, device="cpu")

# Compute gradient via parameter-shift (Rust-native)
obs = [([3], 1.0, 0.0)]  # Z observable
dag = qc.to_ir()
grads = state.grad(obs, dag, {"theta": 0.5})
print(grads)  # {"theta": -0.479...}
```

---

## Simulation Methods

### Statevector (default)

Exact simulation, supports all `sf.State` methods including gradients.

```python
result = sf.run(circuit, device="cpu", method="statevector", shots=1024)
```

### MPS (Tensor Network)

For low-entanglement circuits. Scales to 200+ qubits.

```python
result = sf.run(circuit, device="cpu", method="mps", shots=1024, bond_dim=64)
```

### Stabilizer

For Clifford-only circuits. Polynomial time up to ~1000 qubits.

```python
# Clifford circuit (H, S, CNOT, etc.)
clifford = sf.Circuit(100).h(0)
for i in range(99):
    clifford.cx(i, i + 1)

result = sf.run(clifford, device="cpu", method="stabilizer", shots=10000)
```

### Density Matrix

For noisy simulation with Kraus channels.

```python
from superfermion.noise import NoiseModel

noise = NoiseModel().add_depolarizing(0.01)
result = sf.run(circuit, device="cpu", method="density_matrix",
                shots=0, noise_model=noise)
rho = result.metadata["density_matrix"]  # Full density matrix
```

---

## VQE Example

```python
from superfermion.algorithms.vqe import VQE

# H2 Hamiltonian
hamiltonian = sf.SparsePauliOp.from_dict({
    "II": -0.81,
    "ZI": 0.17,
    "IZ": -0.23,
    "ZZ": 0.12,
    "XX": 0.04,
})

# Ansatz circuit
ansatz = sf.Circuit(2)
ansatz.ry(sf.param("t0"), 0)
ansatz.ry(sf.param("t1"), 1)
ansatz.cx(0, 1)

vqe = VQE(ansatz, hamiltonian, device="cpu")
result = vqe.minimize()
print(f"Ground state energy: {result.optimal_value:.4f}")
```

---

## ML Framework Integration

### PyTorch

```python
from superfermion.nn.torch_layer import TorchQuantumLayer

layer = TorchQuantumLayer(circuit, observable, device="cpu")
output = layer()       # Forward pass
output.backward()      # Backward uses sf.State.grad()
```

### Flax (JAX)

```python
from superfermion.nn.quantum_layer import QuantumLayer
import jax

layer = QuantumLayer(circuit, observable, device="cpu")
params = layer.init(jax.random.PRNGKey(0))
output = layer.apply(params)
```

### TensorFlow

```python
from superfermion.nn.tf_layer import TFQuantumLayer

layer = TFQuantumLayer(circuit, observable, device="cpu")
output = layer(None)  # Forward pass, gradients via tf.GradientTape
```

---

## GPU Simulation

```python
# Requires CUDA-capable GPU and sf-gpu crate
result = sf.run(circuit, device="gpu", shots=0)
```

---

## Next Steps

- [Architecture](architecture.md) — How Superfermion is built
- [API Reference](api_reference.md) — Full API surface
- [Tutorials](tutorials/) — Runnable examples
