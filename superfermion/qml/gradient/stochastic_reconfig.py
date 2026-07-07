"""
Stochastic Reconfiguration (SR) — High-order optimization for variational circuits.

Uses sf.State.qfim() (Rust-native) for the metric tensor and sf.State.grad()
for the energy gradient. No JAX dependency.
"""

import numpy as np
from typing import Dict, List


def sr_update(
    params: np.ndarray,
    grad: np.ndarray,
    qfim: np.ndarray,
    learning_rate: float,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Stochastic Reconfiguration parameter update.

    Args:
        params: Current parameters (1-D array).
        grad: Energy gradient (1-D array).
        qfim: Quantum Fisher Information Matrix (2-D array).
        learning_rate: Step size.
        regularization: Tikhonov regularization.

    Returns:
        Updated parameters.
    """
    identity = np.eye(qfim.shape[0])
    metric_inv = np.linalg.pinv(qfim + regularization * identity)
    update = metric_inv @ grad
    return params - learning_rate * update


def sr_step(
    state,
    dag,
    observable: list,
    param_names: List[str],
    param_values: Dict[str, float],
    learning_rate: float = 0.01,
    regularization: float = 1e-3,
) -> Dict[str, float]:
    """Perform one SR step using sf.State.

    Args:
        state: sf.State from simulating the circuit.
        dag: QuantumDAG (unbound) for gradient/QFIM computation.
        observable: Pauli observable terms.
        param_names: Ordered parameter names.
        param_values: Current parameter values.
        learning_rate: Step size.
        regularization: Tikhonov regularization.

    Returns:
        Updated parameter values dict.
    """
    grad_dict = state.grad(observable, dag, param_values)
    grad_vec = np.array([grad_dict.get(n, 0.0) for n in param_names])

    qfim = np.array(state.qfim(dag, param_values))
    params_vec = np.array([param_values[n] for n in param_names])

    new_vec = sr_update(params_vec, grad_vec, qfim, learning_rate, regularization)

    return {name: float(val) for name, val in zip(param_names, new_vec)}
