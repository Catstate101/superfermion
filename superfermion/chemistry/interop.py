"""Pauli-operator interoperability: superfermion <-> Qiskit.

The point of this module is that a superfermion Hamiltonian should not be a
dead end.  Once a mapping has produced a Pauli sum, everything the Qiskit
ecosystem offers -- ``NumPyEigensolver``, ``NumPyMinimumEigensolver``, VQE and
QAOA with ``SparsePauliOp`` observables, ``PauliList`` arithmetic, circuit
expectation estimation -- takes a ``qiskit.quantum_info.SparsePauliOp``.  This
is the adapter, and it is deliberately the only conversion in the package so
there is one place for the ordering convention to be right.

**Ordering.**  A superfermion Pauli string is indexed by qubit:
``pauli_str[q]`` is the operator on qubit ``q``, and
:func:`~superfermion.chemistry.subspace.pauli_matrix` builds
``kron(str[0], str[1], ...)``.  Qiskit's ``SparsePauliOp`` uses exactly the
same rule -- character ``q`` is qubit ``q`` and ``to_matrix()`` is the
Kronecker product in character order -- so the labels transfer verbatim, with
no reversal.

That is worth stating because a reversal would *not* be caught by an energy
check: relabelling qubits is a permutation similarity, so the spectrum comes
out identical either way while every expectation value against a
superfermion-prepared circuit is wrong.  The tests therefore compare the dense
matrices elementwise, which is the only check that distinguishes the two
conventions.

Coefficients transfer as ``complex``: ``SparsePauliOp`` accepts complex
coefficients and rounds nothing, so the round trip is exact rather than
close.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "QISKIT_AVAILABLE",
    "to_qiskit_op",
    "from_qiskit_op",
    "to_qiskit_labels",
    "operator_matrix",
]


def _probe_qiskit() -> bool:
    try:  # pragma: no cover - import availability, not logic
        import qiskit.quantum_info  # noqa: F401
    except Exception:
        return False
    return True


#: Whether ``qiskit`` is importable.  Conversion raises a ``ModuleNotFoundError``
#: with an install hint rather than an ``ImportError`` from three frames down.
QISKIT_AVAILABLE = _probe_qiskit()


def _require_qiskit() -> None:
    if not QISKIT_AVAILABLE:
        raise ModuleNotFoundError(
            "Qiskit is not installed; to_qiskit_op/from_qiskit_op need "
            "`pip install qiskit` (tested against qiskit 2.6)."
        )


def _pauli_terms(hamiltonian: Any) -> List[Tuple[str, complex]]:
    """``(label, coefficient)`` pairs from any superfermion Pauli container.

    Accepts :class:`~superfermion.observables.core.Hamiltonian` and
    :class:`~superfermion.observables.core.SparsePauliOp`; both expose
    ``terms`` as objects carrying ``pauli_str`` and ``coeffs``, so the
    conversion does not care which one it was handed.
    """
    terms = getattr(hamiltonian, "terms", None)
    if terms is None:
        raise TypeError(
            f"expected a superfermion Hamiltonian or SparsePauliOp, got "
            f"{type(hamiltonian).__name__}"
        )
    out: List[Tuple[str, complex]] = []
    for term in terms:
        label = getattr(term, "pauli_str", None)
        coeff = getattr(term, "coeffs", None)
        if label is None or coeff is None:
            raise TypeError(
                f"term {term!r} does not carry pauli_str/coeffs"
            )
        out.append((str(label).upper(), complex(coeff)))
    return out


def to_qiskit_labels(
    hamiltonian: Any, *, n_qubits: Optional[int] = None
) -> Tuple[List[str], List[complex]]:
    """``(labels, coefficients)`` with every label padded to one width.

    A superfermion operator built by a mapper always emits full-width strings,
    but a hand-assembled one need not, and ``SparsePauliOp`` rejects ragged
    input.  Padding is appended on the right, i.e. identity on the *higher*
    qubits, which is what a shorter string means under the qubit-indexed
    convention above.

    Raises:
        ValueError: if a label is wider than ``n_qubits``, or is empty.
    """
    terms = _pauli_terms(hamiltonian)
    if not terms:
        raise ValueError("cannot convert an operator with no terms")
    width = (
        int(n_qubits)
        if n_qubits is not None
        else max(len(label) for label, _ in terms)
    )
    labels: List[str] = []
    coeffs: List[complex] = []
    for label, coeff in terms:
        if not label:
            raise ValueError("a Pauli label cannot be empty")
        if len(label) > width:
            raise ValueError(
                f"label {label!r} is wider than n_qubits={width}"
            )
        labels.append(label.ljust(width, "I"))
        coeffs.append(coeff)
    return labels, coeffs


def to_qiskit_op(hamiltonian: Any, *, n_qubits: Optional[int] = None):
    """Convert a superfermion Pauli operator to ``qiskit.quantum_info``.

    Args:
        hamiltonian: A superfermion ``Hamiltonian`` or ``SparsePauliOp``.
        n_qubits: Width of the target operator.  Defaults to the widest label,
            which is right when the mapping already used the full register.

    Returns:
        ``qiskit.quantum_info.SparsePauliOp`` with the same terms and complex
        coefficients in the same qubit order.
    """
    _require_qiskit()
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = to_qiskit_labels(hamiltonian, n_qubits=n_qubits)
    return SparsePauliOp(labels, coeffs).simplify(atol=0.0, rtol=0.0)


def from_qiskit_op(op: Any, *, simplify: bool = True):
    """Convert a Qiskit Pauli operator into a superfermion ``Hamiltonian``.

    Args:
        op: ``SparsePauliOp``, ``PauliList``, or anything exposing
            ``paulis``/``coeffs``.
        simplify: Merge repeated labels and drop exact zeros (default).  With
            ``False`` every term is preserved, which is what a round-trip test
            of a deliberately duplicated operator wants.

    Returns:
        ``superfermion.observables.Hamiltonian``.

    Raises:
        ValueError: if ``op`` carries no terms.
    """
    from superfermion.observables.core import Hamiltonian, PauliString

    pairs = _qiskit_pairs(op)
    if not pairs:
        raise ValueError("cannot convert an operator with no terms")
    if simplify:
        merged: Dict[str, complex] = {}
        for label, coeff in pairs:
            merged[label] = merged.get(label, 0.0 + 0.0j) + coeff
        pairs = [
            (label, coeff)
            for label, coeff in merged.items()
            if coeff != 0.0
        ]
    if not pairs:
        raise ValueError("every term of the operator cancelled")
    return Hamiltonian([PauliString(label, coeff=coeff) for label, coeff in pairs])


def _qiskit_pairs(op: Any) -> List[Tuple[str, complex]]:
    """``(label, coefficient)`` pairs from a Qiskit object, without importing
    a concrete class first -- ``SparsePauliOp`` and ``PauliList`` differ in
    exactly which attribute holds the labels."""
    paulis = getattr(op, "paulis", None)
    coeffs = getattr(op, "coeffs", None)
    if paulis is not None and coeffs is not None:
        labels = [_pauli_to_label(pauli) for pauli in paulis]
        values = [complex(value) for value in coeffs]
        return list(zip(labels, values))
    data = getattr(op, "data", None)
    if data is not None:
        return [(_pauli_to_label(pauli), complex(value)) for pauli, value in data]
    raise TypeError(f"cannot read Pauli terms from {type(op).__name__}")


def _pauli_to_label(pauli: Any) -> str:
    """Label of a Qiskit ``Pauli``, character order preserved."""
    to_label = getattr(pauli, "to_label", None)
    if callable(to_label):
        return str(to_label()).upper()
    return str(pauli).upper()


def operator_matrix(hamiltonian: Any) -> Any:
    """Dense matrix of a superfermion Pauli operator, via Qiskit.

    A second implementation of
    :func:`~superfermion.chemistry.subspace.pauli_matrix` that goes through the
    converted operator, so the two can be compared: agreement is the check
    that the conversion preserved the qubit order, which an eigenvalue
    comparison cannot make.
    """
    return to_qiskit_op(hamiltonian).to_matrix()
