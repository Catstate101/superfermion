"""
Quantum Approximate Optimization Algorithm — legacy JAX/Optax implementation.

.. deprecated::
    Superseded by :mod:`superfermion.algorithms.variational`.
    The new QAOA uses scipy.optimize, works with every SF backend, and has
    no JAX-tracing restrictions::

        from superfermion.algorithms.variational import QAOA  # recommended

    This file is kept for backward compatibility only.

QAOA is used for solving combinatorial optimization problems.
It consists of alternating applications of a cost Hamiltonian and a mixer Hamiltonian.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "superfermion.algorithms.qaoa is deprecated. "
    "Use superfermion.algorithms.variational instead: "
    "from superfermion.algorithms.variational import QAOA",
    DeprecationWarning,
    stacklevel=2,
)

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, List, Optional, Tuple

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
from superfermion.observables.core import Hamiltonian, PauliString
from superfermion.qml.fidelity import state_fidelity
from superfermion.algorithms.core import AlgorithmResult


class QAOA:
    """QAOA manager.
    
    Coordinates the cost Hamiltonian, mixer Hamiltonian, and optimization steps.
    """

    def __init__(
        self, 
        n_qubits: int,
        cost_hamiltonian: Hamiltonian,
        p_layers: int = 1,
        optimizer: Optional[optax.GradientTransformation] = None
    ):
        self.n_qubits = n_qubits
        self.cost_hamiltonian = cost_hamiltonian
        self.p_layers = p_layers
        self.optimizer = optimizer or optax.adam(0.01)
        
        # Build the QAOA circuit ansatz
        self.ansatz = self._build_ansatz()
        self.model = QuantumLayer(n_qubits=self.n_qubits, ansatz=self.ansatz, backend="jax")

    def _build_ansatz(self) -> sf.Circuit:
        """Create the QAOA circuit with alternating layers."""
        c = sf.Circuit(self.n_qubits)
        
        # Initial state: Uniform superposition
        for i in range(self.n_qubits):
            c.h(i)
            
        # Layers of Cost and Mixer
        for p in range(self.p_layers):
            # 1. Cost Hamiltonian Layer: e^{-i * gamma * H_cost}
            # For H_cost = sum Z_i Z_j, we need RZZ gates
            gamma = sf.param(f"gamma_{p}")
            for term in self.cost_hamiltonian.terms:
                if term.pauli_str.count('Z') == 2:
                    # Find indices of Z
                    idx = [i for i, char in enumerate(term.pauli_str) if char == 'Z']
                    # RZZ(2 * theta * coeff) because RZZ(theta) is e^{-i*theta/2 * Z1Z2}
                    # We want e^{-i * gamma * coeff * Z1Z2}
                    # So theta = 2 * gamma * coeff
                    # Since sf.param doesn't support math yet in the IR, 
                    # we handle scaling in the logic if needed, or 
                    # just define gamma as the effective rotation.
                    c.rzz(gamma, idx[0], idx[1])
                elif term.pauli_str.count('Z') == 1:
                    idx = term.pauli_str.find('Z')
                    c.rz(gamma, idx)

            # 2. Mixer Hamiltonian Layer: e^{-i * beta * H_mixer}
            # H_mixer = sum X_i, so we need RX gates
            beta = sf.param(f"beta_{p}")
            for i in range(self.n_qubits):
                c.rx(beta, i)
                
        return c

    def get_loss_fn(self):
        """Returns a JAX-compatible loss function (expectation of cost H)."""
        
        @jax.jit
        def loss_fn(params):
            from superfermion.backends.jax_sim import JAXBackend
            sim = JAXBackend()
            flat_params = params['params']['weights']
            
            p_names = self.ansatz.parameters
            p_dict = {name: val for name, val in zip(p_names, flat_params)}
            bound_ansatz = self.ansatz.bind(p_dict)
            
            state = sim.simulate(bound_ansatz)
            energy = self.cost_hamiltonian.expectation(state)
            return jnp.real(energy)
            
        return loss_fn

    def optimize(self, iterations: int = 100, seed: int = 42, target_state: Optional[jnp.ndarray] = None) -> AlgorithmResult:
        """Run the QAOA optimization loop."""
        key = jax.random.PRNGKey(seed)
        params = self.model.init(key)
        opt_state = self.optimizer.init(params)
        
        loss_fn = self.get_loss_fn()
        
        @jax.jit
        def step(p, opt_st):
            loss, grads = jax.value_and_grad(loss_fn)(p)
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
        for i in range(iterations):
            params, opt_state, loss, fidelity = step(params, opt_state)
            history.append(float(loss))
            if target_state is not None:
                fidelity_history.append(float(fidelity))
            
        return AlgorithmResult(
            optimal_value=float(loss),
            optimal_params=params,
            history=history,
            fidelity_history=fidelity_history if target_state is not None else None,
            final_fidelity=float(fidelity) if target_state is not None else None,
            metadata={"seed": seed, "iterations": iterations}
        )
