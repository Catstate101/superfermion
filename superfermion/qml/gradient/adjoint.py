"""Adjoint differentiation — fast gradient for parameterised quantum circuits.

Uses the Rust sf-ir adjoint engine (Jones & Gacon 2020) via PyO3 bindings.
Cost: O((M + N) * 2^n) — one forward pass + one backward pass, regardless
of the number of parameters.  For N=20 parameters this is a 40x reduction
vs parameter-shift.

Public API:
    adjoint_grad_vector(circuit, observable, param_names, param_values)
    adjoint_grad(circuit, observable, params_dict)

Observable convention (verified to ~1e-16 vs parameter-shift):
    Pauli strings are MSB-first: the leftmost character acts on qubit 0,
    matching PennyLane (``qml.Z(0)``) and Qiskit wire ordering.  E.g. on a
    4-qubit circuit ``"ZIII"`` measures Z on qubit 0.  ``params_dict`` must
    bind *every* circuit parameter (trainable + embedding values alike).

Supported gates: every parametric gate in the IR — ``rx``, ``ry``, ``rz``,
    ``rxx``, ``ryy``, ``rzz``, ``r1``/``p``, ``cp``, ``crx``, ``crz``,
    ``u``/``u3`` (all three angles) and ``cu`` (all four).  The generators
    live in one table in ``crates/sf-ir/src/adjoint.rs``; a parametric gate
    missing from it raises ``MethodError`` rather than returning 0.0.  All of
    them are reachable from the Python fluent API: ``r1``, ``crx`` and ``crz``
    map to their own opcodes, and ``cry`` is emitted as the exact equivalent
    ``cu(θ, 0, 0)`` (the IR has no dedicated ``CRy`` opcode).

Example:
    >>> from superfermion.qml.gradient.adjoint import adjoint_grad
    >>> g = adjoint_grad(circuit, "ZIII", {"t0": 0.5, "x0": 0.2, ...})
    >>> g["t0"]  # d<Z0>/d(t0)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from superfermion.circuit import Circuit
from superfermion.parameters import SymbolicParameter  # noqa: F401 - public re-export

from superfermion.qml.gradient._observables import (
    PAULI_CODES as _PAULI_MAP,  # noqa: F401 - historical name, kept importable
    normalize_for_gradient,
    observable_terms as _observable_terms,
    pauli_indices as _pauli_indices,  # noqa: F401 - historical name
)


def _observable_to_rust_terms(obs, n_qubits: int) -> list[tuple]:
    """Convert an observable to Rust's [(paulis_u8, coef_re, coef_im)] format.

    The spelling table lives in ``superfermion.qml.gradient._observables`` so
    that this module, the parameter-shift/finite-difference pair, and
    ``RunResult.grad`` all accept exactly the same forms -- they used to
    differ, and the differences are recorded as E/V-3 and E/V-4.

    Supports:
      - str: single Pauli string like "ZZI"
      - Sequence[str]: several Pauli strings, summed with unit weight
      - dict: {"ZI": 0.5, "IX": 0.3}
      - Hamiltonian / SparsePauliOp with .terms attribute
      - PauliString: a single weighted Pauli string (CH-7)
      - Sequence[PauliString | SparsePauliOp | str]: the same sum written as a
        list of terms
      - the engine form [(paulis, coef_re, coef_im)], including numpy integer
        code arrays

    No validation is applied here: this is the bare converter that
    ``RunResult.expectation`` also uses, where a non-Hermitian operator is a
    perfectly well-defined expectation value.  Gradient entry points call
    :func:`normalize_for_gradient`, which adds the width and Hermiticity
    checks the derivative engines rely on.
    """
    return _observable_terms(obs, n_qubits)


def _check_param_names(circuit: Circuit, param_names: Sequence[str]) -> None:
    """Reject names that are not (still) parameters of ``circuit``.

    A circuit that has already been through ``Circuit.bind(...)`` no longer
    knows its parameter names: every ``SymbolicParameter`` became a float,
    so the DAG reports no parameters and every derivative is structurally
    zero.  That silently returned all-zeros before 0.1.24 — this guard is
    what turns it into an error naming the cause.
    """
    known = set(circuit.parameters)
    missing = [nm for nm in param_names if nm not in known]
    if not missing:
        return
    raise ValueError(
        "adjoint_grad: "
        + ", ".join(repr(m) for m in missing)
        + (" is" if len(missing) == 1 else " are")
        + " not a parameter of this circuit (circuit parameters: "
        + (", ".join(repr(p) for p in sorted(known)) or "none")
        + "). Pass the unbound circuit (circuit.to_ir()), not "
        "circuit.bind(...): binding folds the parameters into constants, "
        "so the DAG no longer knows them and every derivative is 0."
    )


def adjoint_grad_vector(
    circuit: Circuit,
    observable,
    param_names: Sequence[str],
    param_values: np.ndarray,
) -> np.ndarray:
    """Adjoint differentiation gradient via the Rust sf-ir engine.

    Drop-in replacement for ``parameter_shift_grad_vector``.

    Args:
        circuit: Parametric ``sf.Circuit`` (built with ``sf.param``).
        observable: str ("ZIII", MSB-first: leftmost char = qubit 0),
            dict, SparsePauliOp, or Hamiltonian-like.
        param_names: trainable parameter names to differentiate.
        param_values: values aligned with ``param_names``. NOTE: the
            underlying DAG call requires *all* circuit parameters bound —
            use ``adjoint_grad`` with a full dict when the circuit also
            has embedding/feature parameters.

    Returns an array of shape ``(len(param_names),)`` with d<O>/d(theta_k).
    """
    params_dict = {nm: float(v) for nm, v in zip(param_names, param_values)}
    _check_param_names(circuit, param_names)
    # Width and Hermiticity are checked here: a label one character short or
    # long used to evaluate a different operator silently, and a non-Hermitian
    # coefficient dict produced three different answers from three engines
    # (E/V-5, E/V-7).
    rust_obs = normalize_for_gradient(observable, circuit.n_qubits)

    dag = circuit.to_ir()
    grad_dict = dag.adjoint_grad(rust_obs, params_dict)

    # No default: a name the engine did not answer for is a bug, and the
    # historical `grad_dict.get(nm, 0.0)` is exactly how all-zero gradients
    # used to ship unnoticed.
    out = np.empty(len(param_names), dtype=float)
    for k, nm in enumerate(param_names):
        try:
            out[k] = float(grad_dict[nm])
        except KeyError:  # pragma: no cover - engine/IR mismatch only
            raise ValueError(
                f"adjoint engine returned no gradient for {nm!r} "
                f"(got {sorted(grad_dict)}); this is an engine bug, not a "
                f"parameter error"
            ) from None
    return out


def adjoint_grad(
    circuit: Circuit,
    observable,
    params: dict[str, float],
) -> dict[str, float]:
    """Dict-form adjoint gradient (analogue of ``parameter_shift_grad``).

    ``params`` must bind every circuit parameter (trainable + embedding);
    the returned dict holds gradients for all of them (embedding entries
    carry the feature-Jacobian and are usually ignored).
    """
    names = list(params.keys())
    vals = np.array([params[nm] for nm in names])
    g = adjoint_grad_vector(circuit, observable, names, vals)
    return {nm: float(v) for nm, v in zip(names, g)}
