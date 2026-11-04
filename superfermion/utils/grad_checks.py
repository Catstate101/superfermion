"""Diagnostics for gradient calls with bound/parameterless DAGs (SUP-6).

``State.grad()`` and ``State.qfim()`` return an empty result when the DAG
carries no symbolic parameters (e.g. a DAG built from a ``.bind()``-ed
circuit) — silently, with no error or warning. This module emits a
``UserWarning`` for that case so callers are told the result will be empty
and how to fix it (keep the DAG symbolic: ``dag = circuit.to_ir()``).

The check used to return early when ``param_values`` was empty, which is
exactly the case a fully bound circuit produces — ``sf.run(c.bind({...}))``
records no values, so ``State.grad`` answered ``{}`` with no warning at all
(F/V-6).  The warning now depends only on the DAG.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping


def _dag_parameter_names(dag: Any) -> "list | None":
    """``parameter_names()`` as a list, or ``None`` when not introspectable."""
    names = getattr(dag, "parameter_names", None)
    if names is None:
        return None
    try:
        names = names() if callable(names) else names
        return [str(name) for name in names]
    except (AttributeError, TypeError):
        return None


def warn_if_bound_dag(dag: Any, param_values: Any, what: str = "gradient") -> None:
    """Warn when ``dag`` carries no symbolic parameters to differentiate.

    A DAG without symbolic parameters can only yield an empty result from
    the Rust adjoint engine. The usual cause is passing a bound DAG
    (``circuit.bind(...).to_ir()``) — ``.bind()`` erases parameters. A
    parameterized DAG with missing values instead raises a catchable
    ``MethodError`` from the Rust engine, so no warning is needed there.

    Args:
        dag: QuantumDAG (or anything exposing ``parameter_names()``).
        param_values: dict of parameter name -> value the caller expects.
        what: API name for the message ("gradient", "QFIM", ...).
    """
    dag_params = _dag_parameter_names(dag)
    if dag_params is None:
        return  # cannot introspect — never crash the caller
    if dag_params:
        # Parameterized dag: missing values raise a MethodError from the
        # Rust adjoint engine; complete values evaluate normally.
        return
    if isinstance(param_values, Mapping):
        requested = list(param_values)
    elif isinstance(param_values, (list, tuple)):
        requested = list(param_values)
    else:
        requested = []
    warnings.warn(
        f"{what} with the dag has no symbolic parameters at all "
        f"(it may come from a bound circuit); requested "
        f"{requested!r} will return an empty result. "
        ".bind() erases parameters — keep the dag symbolic: "
        "dag = circuit.to_ir() (unbound), and pass concrete values via "
        "param_values=.",
        UserWarning,
        stacklevel=3,
    )
