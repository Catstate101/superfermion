"""
Quantum Natural Gradient (QNG) — Optimization using the Fubini-Study metric.

Uses sf.State.qfim() (Rust-native) for the Quantum Fisher Information Matrix
and sf.State.grad() for the gradient. No JAX dependency.

Parameter order
---------------
``State.grad`` and ``State.qfim`` are positional in the *dag's* order
(``QuantumDAG.parameter_names()``), while ``param_names`` here is whatever
order the caller likes.  The gradient is a dict and needs no permutation; the
metric is a matrix and does (F/V-10).  Until 0.1.25 the two were mixed, so a
sorted or filtered ``param_names`` produced a step that solved ``G`` in one
basis against ``g`` in another -- measured to differ from the correct step by
0.024420 on a three-parameter circuit, with no error and no warning.

Degenerate metrics
------------------
``np.linalg.solve`` does not raise on a merely *near*-singular matrix: on
``ry(a); ry(b)`` (metric ``[[1, 1], [1, 1]]`` stored to ~1e-11) it returned a
step of |delta theta| = 532 at ``learning_rate=0.05`` (F/V-11).  The rank is
probed first, and a rank-deficient metric takes the Moore-Penrose inverse --
the minimum-norm natural-gradient step, which is what
``superfermion.qml.gradient.riemannian_gradient`` and ``sr_update`` already
use.  At any positive regularization the two paths agree to 1e-12.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from superfermion.qml.gradient._observables import normalize_for_gradient
from superfermion.qml.gradient._params import (
    check_param_names,
    check_regularization,
    dag_n_qubits,
    metric_in_param_order,
)


def _natural_gradient(metric: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """``G^{-1} g``, falling back to ``G^{+} g`` when ``G`` is rank-deficient.

    The rank probe is what makes the fallback reliable: ``np.linalg.solve``
    succeeds on a matrix whose condition number is ~1e11 and returns a step
    built from that amplified inverse (F/V-11).
    """
    if np.linalg.matrix_rank(metric) < metric.shape[0]:
        return np.linalg.pinv(metric) @ gradient
    return np.linalg.solve(metric, gradient)


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
        observable: Any SF observable form (str, dict, SparsePauliOp,
            PauliString, list of labels/terms, or the engine term list).
            A non-Hermitian or wrong-width operator is refused rather than
            differentiated.
        param_names: Ordered list of parameter names. Any order, any subset of
            the dag's parameters; the metric is permuted to match.
        param_values: Dict mapping parameter names to current values.
        learning_rate: Step size.
        regularization: Tikhonov regularization for QFIM inversion.

    Returns:
        Updated parameter values dict, keyed by ``param_names``.

    Raises:
        ValueError: a name in ``param_names`` is not a dag parameter, the
            metric has the wrong shape, or ``regularization`` is negative.
    """
    check_param_names(dag, param_names, what="qng_step")
    reg = check_regularization(regularization, "qng_step")
    terms = normalize_for_gradient(
        observable, dag_n_qubits(dag), what="qng_step observable"
    )

    grad_dict = state.grad(terms, dag, param_values)
    grad_vec = np.array([grad_dict[name] for name in param_names], dtype=float)

    qfim = metric_in_param_order(state.qfim(dag, param_values), dag, param_names)
    g_reg = qfim + reg * np.eye(len(param_names))
    update = _natural_gradient(g_reg, grad_vec)

    return {
        name: param_values[name] - learning_rate * update[i]
        for i, name in enumerate(param_names)
    }
