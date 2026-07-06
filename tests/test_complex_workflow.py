"""
Complex End-to-End Workflow: Drug-inspired Molecular Ground State
================================================================

This test simulates a real quantum chemistry research workflow:

1. Define a molecular Hamiltonian (H2-inspired)
2. Build a hardware-efficient ansatz
3. Run VQE with real-time fidelity tracking
4. Run QAOA on a related optimization problem
5. Classify simulation results using QSVM
6. Compile and benchmark the pipeline
7. Log all results as a structured experiment

This is the kind of workflow Superfermion is built to power.
"""

from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import superfermion as sf
from superfermion.algorithms.variational import VQE, QAOA
from superfermion.algorithms.qsvm import QSVM
from superfermion.observables.core import Hamiltonian, PauliString
from superfermion.qml.fidelity import state_fidelity
from superfermion.algorithms.core import AlgorithmResult

print("=" * 70)
print("  SUPERFERMION COMPLEX WORKFLOW: Molecular Ground State Pipeline")
print("=" * 70)
t0 = time.time()

# =====================================================================
# PHASE 1: Define Molecular Hamiltonian (H2-inspired, 2 qubits)
# =====================================================================
print("\n[Phase 1] Defining H2-inspired Hamiltonian...")

# H = -1.05 * II + 0.39 * IZ - 0.39 * ZI - 0.01 * ZZ + 0.18 * XX
# Simplified for 2-qubit demo:
h2_hamiltonian = Hamiltonian([
    PauliString("ZI", coeffs=-0.4),
    PauliString("IZ", coeffs=0.4),
    PauliString("ZZ", coeffs=-0.2),
    PauliString("XX", coeffs=0.18),
])
print("  Hamiltonian terms: 4 (ZI, IZ, ZZ, XX)")

# Ground state energy calculation (exact diagonalization for reference)
from superfermion.backends.jax_sim import JAXBackend
sim = JAXBackend()

# Build the full Hamiltonian matrix for reference
n = 4  # 2^2 = 4 basis states
H_matrix = jnp.zeros((n, n), dtype=jnp.complex64)
for basis_idx in range(n):
    basis = jnp.zeros(n, dtype=jnp.complex64).at[basis_idx].set(1.0)
    col = jnp.zeros(n, dtype=jnp.complex64)
    for term in h2_hamiltonian.terms:
        col = col + term._apply(basis)
    H_matrix = H_matrix.at[:, basis_idx].set(col)

eigenvalues = jnp.linalg.eigvalsh(H_matrix)
exact_ground = float(jnp.min(eigenvalues))
print(f"  Exact Ground State Energy: {exact_ground:.6f}")


# =====================================================================
# PHASE 2: VQE with Fidelity Tracking
# =====================================================================
print("\n[Phase 2] Running VQE with fidelity tracking...")

# Build ansatz
ansatz = sf.Circuit(2)
ansatz.ry(sf.param("a0"), 0)
ansatz.ry(sf.param("a1"), 1)
ansatz.cx(0, 1)
ansatz.ry(sf.param("a2"), 0)
ansatz.ry(sf.param("a3"), 1)

# Compute exact ground state vector for fidelity tracking
eigvals, eigvecs = jnp.linalg.eigh(H_matrix)
ground_state = eigvecs[:, jnp.argmin(eigvals)]

vqe = VQE(ansatz, h2_hamiltonian, optimizer="L-BFGS-B")
vqe_result = vqe.minimize(iterations=150)

# Compute fidelity manually using the optimized parameters
optimized_circuit = ansatz.bind(vqe_result.optimal_params)
final_sv = sim.run(optimized_circuit, shots=0).statevector
fidelity = float(jnp.abs(jnp.dot(jnp.conjugate(final_sv.ravel()), ground_state.ravel()))**2)

print(f"  VQE Energy:        {vqe_result.optimal_value:.6f}")
print(f"  Exact Energy:      {exact_ground:.6f}")
print(f"  Error:             {abs(vqe_result.optimal_value - exact_ground):.6f}")
print(f"  Final Fidelity:    {fidelity:.6f}")

vqe_converged = abs(vqe_result.optimal_value - exact_ground) < 0.1
print(f"  Converged: {'YES' if vqe_converged else 'NO'}")

# =====================================================================
# PHASE 3: QAOA on Related MaxCut
# =====================================================================
print("\n[Phase 3] Running QAOA on related graph problem...")

qaoa = QAOA(n_qubits=2, edges=[(0, 1)], p_layers=2)
qaoa_result = qaoa.minimize(iterations=100)

print(f"  QAOA Max Cut: {qaoa_result.optimal_value:.6f}")
print(f"  Converged: {'YES' if qaoa_result.optimal_value > 0.8 else 'NO'}")

# =====================================================================
# PHASE 4: QSVM Classification of Simulation Results
# =====================================================================
print("\n[Phase 4] Training QSVM classifier on simulation data...")

# Generate synthetic data: "good" vs "bad" parameter configs
key = jax.random.PRNGKey(42)
# Class 0: params near ground state, Class 1: random params
x_good = jax.random.normal(key, (8, 4)) * 0.1
x_bad = jax.random.uniform(jax.random.PRNGKey(1), (8, 4)) * jnp.pi
x_train = jnp.concatenate([x_good, x_bad])
y_train = jnp.concatenate([jnp.zeros(8, dtype=jnp.int32), jnp.ones(8, dtype=jnp.int32)])

qsvm_circuit = sf.Circuit(4)
for i in range(4):
    qsvm_circuit.ry(sf.param(f"q{i}"), i)

qsvm = QSVM(qsvm_circuit, num_classes=2)
qsvm_result = qsvm.fit(x_train, y_train, iterations=80)

preds = qsvm.predict(qsvm_result.optimal_params, x_train)
accuracy = float(jnp.mean(preds == y_train))
print(f"  QSVM Training Accuracy: {accuracy*100:.1f}%")

# =====================================================================
# =====================================================================
# PHASE 5: Summary
# =====================================================================
print("\n[Phase 5] Workflow summary...")

prediction = None
top_token = None

# =====================================================================
# SUMMARY
# =====================================================================
total_time = time.time() - t0
print("\n" + "=" * 70)
print("  WORKFLOW COMPLETE")
print("=" * 70)
print(f"  Total Time:           {total_time:.2f}s")
print(f"  VQE Ground Energy:    {vqe_result.optimal_value:.6f} (exact: {exact_ground:.6f})")
print(f"  VQE Fidelity:         {fidelity:.6f}")
print(f"  QAOA Optimum:         {qaoa_result.optimal_value:.6f}")
print(f"  QSVM Accuracy:        {accuracy*100:.1f}%")
print(f"  QDL Predictions:      N/A (deprecated)")
print(f"  QLLM Next Token:      N/A (deprecated)")
print("=" * 70)
