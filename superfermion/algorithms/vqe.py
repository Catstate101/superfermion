"""
Variational Quantum Eigensolver — legacy JAX/Optax implementation.

.. deprecated::
    Superseded by :mod:`superfermion.algorithms.variational`.
    The new VQE uses scipy.optimize, works with every SF backend, and has
    no JAX-tracing restrictions::

        from superfermion.algorithms.variational import VQE  # recommended

    This file is kept for backward compatibility only.

VQE is a hybrid quantum-classical algorithm used to find the minimum eigenvalue
(ground state energy) of a given Hamiltonian.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "superfermion.algorithms.vqe is deprecated. "
    "Use superfermion.algorithms.variational instead: "
    "from superfermion.algorithms.variational import VQE",
    DeprecationWarning,
    stacklevel=2,
)

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, Optional, Tuple

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
from superfermion.observables.core import Hamiltonian
from superfermion.qml.fidelity import state_fidelity
from superfermion.algorithms.core import AlgorithmResult


class VQE:
    """Variational Quantum Eigensolver manager.
    
    Coordinates the circuit ansatz, Hamiltonian observable, and classical optimizer.
    """

    def __init__(
        self, 
        ansatz: sf.Circuit, 
        hamiltonian: Hamiltonian,
        optimizer: Optional[optax.GradientTransformation] = None
    ):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer or optax.adam(0.1)
        
        # Wrap the ansatz in a differentiable Flax layer
        self.model = QuantumLayer(n_qubits=ansatz.n_qubits, ansatz=ansatz, backend="jax")

    def get_loss_fn(self):
        """Returns a JAX-compatible loss function (expectation value)."""
        
        @jax.jit
        def loss_fn(params):
            from superfermion.backends.jax_sim import JAXBackend
            sim = JAXBackend()
            
            # Reconstruct numeric parameters from the pytree
            flat_params = params['params']['weights']
            
            # Bind parameters to the ansatz
            p_names = self.ansatz.parameters
            p_dict = {name: val for name, val in zip(p_names, flat_params)}
            bound_ansatz = self.ansatz.bind(p_dict)
            
            state = sim.simulate(bound_ansatz)
            
            energy = self.hamiltonian.expectation(state)
            return jnp.real(energy)
            
        return loss_fn

    def minimize(self, iterations: int = 100, seed: int = 42, target_state: Optional[jnp.ndarray] = None, tol: float = 1e-6) -> AlgorithmResult:
        """Run the VQE optimization loop.
        
        Args:
            iterations: Number of optimization steps.
            seed: Random seed for initialization.
            target_state: Optional "gold" statevector to track fidelity against.
            tol: Tolerance for early stopping.
        """
        from superfermion.utils import info, debug
        info(f"Starting VQE optimization: {iterations} iterations, seed={seed}")
        
        key = jax.random.PRNGKey(seed)
        params = self.model.init(key)
        opt_state = self.optimizer.init(params)
        
        loss_fn = self.get_loss_fn()
        
        @jax.jit
        def step(p, opt_st):
            # Compute loss and gradients
            loss, grads = jax.value_and_grad(loss_fn)(p)
            
            # Update optimizer
            updates, new_opt_st = self.optimizer.update(grads, opt_st, p)
            new_p = optax.apply_updates(p, updates)
            
            # Optional: Return current state if we need fidelity
            current_fidelity = jnp.array(0.0)
            if target_state is not None:
                from superfermion.backends.jax_sim import JAXBackend
                sim = JAXBackend()
                state = sim.simulate(self.ansatz, new_p['params']['weights'])
                current_fidelity = state_fidelity(state, target_state)
            
            return new_p, new_opt_st, loss, current_fidelity

        history = []
        fidelity_history = []
        last_loss = float('inf')
        
        for i in range(iterations):
            params, opt_state, loss, fidelity = step(params, opt_state)
            current_loss = float(loss)
            history.append(current_loss)
            
            if target_state is not None:
                fidelity_history.append(float(fidelity))
                
            if i % 10 == 0:
                fid_str = f", fidelity={fidelity:.6f}" if target_state is not None else ""
                debug(f"  Iteration {i:3d}: Loss = {current_loss:10.6f}{fid_str}")
                
            # Early stopping
            if abs(last_loss - current_loss) < tol:
                info(f"VQE converged at iteration {i} (tolerance {tol})")
                break
            last_loss = current_loss
            
        info(f"VQE optimization complete. Final Loss: {current_loss:.6f}")
            
        return AlgorithmResult(
            optimal_value=current_loss,
            optimal_params=params,
            history=history,
            fidelity_history=fidelity_history if target_state is not None else None,
            final_fidelity=float(fidelity) if target_state is not None else None,
            metadata={"seed": seed, "iterations": iterations, "converged_at": i}
        )
