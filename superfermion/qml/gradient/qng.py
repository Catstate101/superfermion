"""
Quantum Natural Gradient (QNG) — Optimization using the Fubini-Study metric.

Uses sf.State.qfim() (Rust-native) for the Quantum Fisher Information Matrix
and sf.State.grad() for the gradient. No JAX dependency.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List


def qng_step(
    state,
    dag,
    observable: list,
    param_names: List[str],
    param_values: Dict[str, float],
    learning_rate: float = 0.01,
    regularization: float = 1e-6,
) -> Dict[str, float]:
    """Perform one step of Quantum Natural Gradient descent.

    theta = theta - eta * G^{-1} * grad(L)

    Args:
        state: sf.State from simulating the circuit at current params.
        dag: QuantumDAG (unbound) for gradient/QFIM computation.
        observable: Pauli observable terms [(paulis, coef_re, coef_im), ...].
        param_names: Ordered list of parameter names.
        param_values: Dict mapping parameter names to current values.
        learning_rate: Step size.
        regularization: Tikhonov regularization for QFIM inversion.

    Returns:
        Updated parameter values dict.
    """
    grad_dict = state.grad(observable, dag, param_values)
    grad_vec = np.array([grad_dict.get(n, 0.0) for n in param_names])

    qfim = np.array(state.qfim(dag, param_values))

    g_reg = qfim + regularization * np.eye(len(param_names))
    update = np.linalg.solve(g_reg, grad_vec)

    new_params = {}
    for i, name in enumerate(param_names):
        new_params[name] = param_values[name] - learning_rate * update[i]
    return new_params
