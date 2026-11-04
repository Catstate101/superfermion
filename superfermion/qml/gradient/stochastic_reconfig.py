"""
Stochastic Reconfiguration (SR) — High-order optimization for variational circuits.

Uses sf.State.qfim() (Rust-native) for the metric tensor and sf.State.grad()
for the energy gradient. No JAX dependency.

Parameter order
---------------
``State.qfim`` is positional in the *dag's* order
(``QuantumDAG.parameter_names()``), while ``param_names`` here is whatever
order the caller likes.  The gradient arrives as a dict and needs no
permutation; the metric is a matrix and does (F/V-10).  Until 0.1.25
``sr_step`` solved the metric in the dag's basis against a gradient and a
parameter vector in the caller's basis, which produced a silently wrong step
whenever the two differed (measured difference 0.024420 on a three-parameter
circuit).
"""

import numpy as np
from typing import Dict, List

from superfermion.qml.gradient._observables import normalize_for_gradient
from superfermion.qml.gradient._params import (
    check_param_names,
    check_regularization,
    dag_n_qubits,
    metric_in_param_order,
)


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
        regularization: Tikhonov regularization, added to the diagonal before
            the Moore-Penrose inverse; must be >= 0.

    Returns:
        Updated parameters.

    Raises:
        ValueError: ``regularization`` is negative, or the three arrays do not
            agree in size.
    """
    reg = check_regularization(regularization, "sr_update")
    metric = np.asarray(qfim, dtype=float)
    gradient = np.asarray(grad, dtype=float)
    values = np.asarray(params, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError(f"sr_update: qfim must be square, got {metric.shape}.")
    if gradient.shape != (metric.shape[0],) or values.shape != gradient.shape:
        raise ValueError(
            f"sr_update: qfim is {metric.shape} but grad is {gradient.shape} "
            f"and params is {values.shape}; all three must agree in size."
        )
    identity = np.eye(metric.shape[0])
    metric_inv = np.linalg.pinv(metric + reg * identity)
    update = metric_inv @ gradient
    return values - learning_rate * update


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
        observable: Any SF observable form (str, dict, SparsePauliOp,
            PauliString, list of labels/terms, or the engine term list).
            A non-Hermitian or wrong-width operator is refused rather than
            differentiated.
        param_names: Ordered parameter names. Any order, any subset of the
            dag's parameters; the metric is permuted to match.
        param_values: Current parameter values.
        learning_rate: Step size.
        regularization: Tikhonov regularization; must be >= 0.

    Returns:
        Updated parameter values dict, keyed by ``param_names``.

    Raises:
        ValueError: a name in ``param_names`` is not a dag parameter, the
            metric has the wrong shape, or ``regularization`` is negative.
    """
    check_param_names(dag, param_names, what="sr_step")
    check_regularization(regularization, "sr_step")
    terms = normalize_for_gradient(
        observable, dag_n_qubits(dag), what="sr_step observable"
    )

    grad_dict = state.grad(terms, dag, param_values)
    grad_vec = np.array([grad_dict[name] for name in param_names], dtype=float)

    qfim = metric_in_param_order(state.qfim(dag, param_values), dag, param_names)
    params_vec = np.array([param_values[n] for n in param_names], dtype=float)

    new_vec = sr_update(params_vec, grad_vec, qfim, learning_rate, regularization)

    return {name: float(val) for name, val in zip(param_names, new_vec)}
