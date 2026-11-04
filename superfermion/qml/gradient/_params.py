"""Align a caller's parameter order with the DAG's own order (F/V-10).

``State.grad`` and ``State.qfim`` are positional in the *DAG's* parameter
order (``QuantumDAG.parameter_names()``), while ``qng_step`` and ``sr_step``
are documented to take an arbitrary ``param_names`` list.  Before this
module the two were mixed: the gradient was assembled from the caller's order
and the metric was taken in the DAG's order, so

    qng_step(state, dag, obs, ["mm", "aa", "zz"], {...})

on a circuit whose DAG declared ``["zz", "aa", "mm"]`` solved ``G`` in one
basis against ``g`` in another.  Measured: the step matched the inconsistent
solve to 0.0e+00 and differed from the consistent one by 0.024420, silently.

A metric tensor is a matrix of coordinates, so the fix is a permutation of
rows *and* columns into the caller's order -- never a permutation of the
gradient alone.  A subset of the DAG's parameters is allowed and means the
sub-matrix for those coordinates, which is the Riemannian metric restricted
to them; a name the DAG does not know is an error rather than a silent 0.0.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def dag_parameter_names(dag: Any) -> list[str]:
    """The order ``State.grad`` / ``State.qfim`` answer in, or ``[]``.

    ``QuantumDAG.parameter_names()`` is the only place that order is
    published; a duck-typed object without it returns ``[]`` so callers can
    fall back instead of crashing on introspection.
    """
    try:
        names = dag.parameter_names()
    except AttributeError:
        return []
    if callable(names):
        names = names()
    try:
        return [str(name) for name in names]
    except TypeError:
        return []


def check_param_names(
    dag: Any, param_names: Sequence[str], what: str = "qng_step"
) -> list[str]:
    """Reject names the DAG does not know, and return the DAG's own order.

    ``grad_dict.get(name, 0.0)`` used to turn a typo (or a name that only
    exists in a different circuit) into a zero gradient, which reads as a
    converged parameter rather than as an error.
    """
    names = list(param_names)
    dag_names = dag_parameter_names(dag)
    if not dag_names:
        return names
    missing = [name for name in names if name not in dag_names]
    if missing:
        raise ValueError(
            f"{what}: {', '.join(repr(m) for m in missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not a parameter of this "
            f"dag (dag parameters: {dag_names}). Pass the unbound circuit's "
            f"dag (circuit.to_ir()) and the names it declares."
        )
    return dag_names


def dag_n_qubits(dag: Any) -> int:
    """The dag's qubit count for a width check; 0 means "unknown, skip"."""
    getter = getattr(dag, "n_qubits", None)
    if getter is None:
        return 0
    try:
        return int(getter() if callable(getter) else getter)
    except (TypeError, ValueError):
        return 0


def check_regularization(regularization: float, what: str) -> float:
    """A negative Tikhonov term makes the metric *less* invertible, never more."""
    value = float(regularization)
    if value < 0.0:
        raise ValueError(
            f"{what}: regularization must be >= 0, got {regularization!r}. "
            f"A negative value subtracts from the diagonal and makes a "
            f"degenerate metric harder to invert, not easier."
        )
    return value


def metric_in_param_order(
    qfim: Any,
    dag: Any,
    param_names: Sequence[str],
    what: str = "qng_step",
) -> np.ndarray:
    """Permute ``qfim`` out of the DAG's order and into ``param_names`` order.

    Returns ``M`` with ``M[i, j] == qfim[dag_index(names[i]),
    dag_index(names[j])]``.  When the DAG cannot be introspected the matrix is
    returned unchanged if its size already matches, and a shape mismatch is
    raised rather than broadcast.
    """
    metric = np.asarray(qfim, dtype=float)
    names = list(param_names)
    k = len(names)
    if metric.shape != (k, k):
        raise ValueError(
            f"{what}: the metric tensor is {metric.shape} but {k} parameter "
            f"name(s) were given. State.qfim() answers in the dag's own order "
            f"({dag_parameter_names(dag) or 'unknown'}), so the two must have "
            f"the same length."
        )
    dag_names = dag_parameter_names(dag)
    if not dag_names:
        # Nothing to permute against: the size check above is all we can do.
        return metric
    positions: Optional[list[int]] = []
    for name in names:
        try:
            positions.append(dag_names.index(name))
        except ValueError:
            raise ValueError(
                f"{what}: {name!r} is not a parameter of this dag "
                f"(dag parameters: {dag_names})."
            ) from None
    if positions == list(range(k)):
        return metric
    return metric[np.ix_(positions, positions)]
