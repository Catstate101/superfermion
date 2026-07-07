"""
QuantumLayer — JAX/Flax ↔ Superfermion bridge.

Wraps a parameterized quantum circuit as a Flax ``nn.Module``.
Gradients flow via ``jax.custom_vjp`` calling ``sf.State.grad()``.

Requires: ``pip install jax flax``
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

import superfermion as sf


_PAULI_MAP = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


def _obs_to_rust(observable) -> list:
    """Convert a SF observable to the Rust [(paulis, re, im)] format.

    Python Pauli strings use MSB-first ordering (e.g. "ZI" = Z on last qubit),
    while Rust expects LSB-indexed (paulis[0] = qubit 0), so we reverse.
    """
    from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian
    terms = []
    if isinstance(observable, PauliString):
        paulis = [_PAULI_MAP[c] for c in reversed(observable.pauli_str)]
        terms.append((paulis, float(observable.coeffs.real), float(observable.coeffs.imag)))
    elif isinstance(observable, SparsePauliOp):
        for ps, coeff in observable._terms:
            paulis = [_PAULI_MAP[c] for c in reversed(ps)]
            terms.append((paulis, float(complex(coeff).real), float(complex(coeff).imag)))
    elif isinstance(observable, Hamiltonian):
        for t in observable.terms:
            paulis = [_PAULI_MAP[c] for c in reversed(t.pauli_str)]
            terms.append((paulis, float(t.coeffs.real), float(t.coeffs.imag)))
    elif isinstance(observable, list):
        return observable
    else:
        raise TypeError(f"Unsupported observable type: {type(observable)}")
    return terms


def _make_quantum_fn(circuit, observable, device_str, method):
    """Create a JAX-compatible function for a specific circuit/observable."""
    param_names = list(circuit.parameters) if circuit.parameters else []
    rust_obs = _obs_to_rust(observable)

    @jax.custom_vjp
    def _quantum_expectation(params_array):
        p_dict = dict(zip(param_names, np.asarray(params_array).tolist()))
        bound = circuit.bind(p_dict)
        state = sf.simulate(bound, device=device_str, method=method)
        return jnp.array(float(state.expectation(rust_obs)))

    def _fwd(params_array):
        val = _quantum_expectation(params_array)
        return val, params_array

    def _bwd(params_array, g):
        p_dict = dict(zip(param_names, np.asarray(params_array).tolist()))
        bound = circuit.bind(p_dict)
        state = sf.simulate(bound, device=device_str, method=method)
        dag = circuit.to_ir()
        grads = state.grad(rust_obs, dag, p_dict)
        grad_np = np.array([grads.get(k, 0.0) for k in param_names])
        return (jnp.array(grad_np) * g,)

    _quantum_expectation.defvjp(_fwd, _bwd)
    return _quantum_expectation


class QuantumLayer(nn.Module):
    """Flax module wrapping a Superfermion variational quantum circuit.

    Args:
        circuit: Parameterized ``sf.Circuit``.
        observable: Observable for expectation value measurement.
        device: Simulation device.
        method: Simulation method.
    """
    circuit: sf.Circuit
    observable: Any
    device: str = "cpu"
    method: str = "statevector"

    @nn.compact
    def __call__(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        param_names = list(self.circuit.parameters) if self.circuit.parameters else []
        n_params = len(param_names)

        if n_params > 0:
            weights = self.param(
                "weights",
                jax.nn.initializers.uniform(scale=2 * jnp.pi),
                (n_params,),
            )
        else:
            weights = jnp.array([])

        fn = _make_quantum_fn(self.circuit, self.observable, self.device, self.method)
        return fn(weights)
