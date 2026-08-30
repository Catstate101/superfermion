"""Diagnostics for gradient calls with bound/parameterless DAGs (SUP-6).

``State.grad()`` and ``State.qfim()`` return an empty result when the DAG
carries no symbolic parameters (e.g. a DAG built from a ``.bind()``-ed
circuit) — silently, with no error or warning. This module emits a
``UserWarning`` for that case so callers are told the result will be empty
and how to fix it (keep the DAG symbolic: ``dag = circuit.to_ir()``).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping


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
    if not isinstance(param_values, Mapping) or not param_values:
        return
    try:
        dag_params = list(dag.parameter_names())
    except AttributeError:
        return  # cannot introspect — never crash the caller
    if dag_params:
        # Parameterized dag: missing values raise a MethodError from the
        # Rust adjoint engine; complete values evaluate normally.
        return
    warnings.warn(
        f"{what} with the dag has no symbolic parameters at all "
        f"(it may come from a bound circuit); requested "
        f"{list(param_values)!r} will return an empty result. "
        ".bind() erases parameters — keep the dag symbolic: "
        "dag = circuit.to_ir() (unbound), and pass concrete values via "
        "param_values=.",
        UserWarning,
        stacklevel=3,
    )
