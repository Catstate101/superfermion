"""Pauli-operator interoperability: superfermion <-> Qiskit.

The point of this module is that a superfermion Hamiltonian should not be a
dead end.  Once a mapping has produced a Pauli sum, everything the Qiskit
ecosystem offers -- ``NumPyEigensolver``, ``NumPyMinimumEigensolver``, VQE and
QAOA with ``SparsePauliOp`` observables, ``PauliList`` arithmetic, circuit
expectation estimation -- takes a ``qiskit.quantum_info.SparsePauliOp``.  This
is the adapter, and it is deliberately the only conversion in the package so
there is one place for the ordering convention to be right.

**Ordering.**  Superfermion does *not* have a single Pauli ordering, and the
earlier claim that both sides agree verbatim was wrong (INT-1):

* :func:`~superfermion.chemistry.subspace.pauli_matrix` builds
  ``kron(str[0], str[1], ...)``, so character 0 is the **leftmost Kronecker
  factor** -- the same order ``SparsePauliOp.to_matrix()`` uses.
* ``RunResult.expectation`` reads character ``q`` as the operator on **qubit
  ``q``** (``x(0)`` samples ``'01'`` while ``expval('ZI') == -1``).

Qiskit indexes the other way round: character ``i`` of a label is qubit
``n-1-i``, so character 0 is the *highest* wire.  The two conventions are
therefore only interchangeable for symmetric labels.  This module exposes the
choice instead of guessing:

* ``reverse=False`` (default, unchanged) keeps the **matrix** identical:
  ``to_qiskit_op(h).to_matrix()`` reproduces ``pauli_matrix(h)`` elementwise.
* ``reverse=True`` keeps the **qubit assignment** identical: a Qiskit
  expectation value against a superfermion-prepared circuit then measures the
  operator superfermion meant by the label.

Pick by what is being checked.  A reordering is invisible to an energy or
spectrum check -- relabelling qubits is a permutation similarity -- which is
exactly why it has to be asked for explicitly rather than assumed.

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
    hamiltonian: Any,
    *,
    n_qubits: Optional[int] = None,
    reverse: bool = False,
) -> Tuple[List[str], List[complex]]:
    """``(labels, coefficients)`` with every label padded to one width.

    A superfermion operator built by a mapper always emits full-width strings,
    but a hand-assembled one need not, and ``SparsePauliOp`` rejects ragged
    input.  Padding is appended on the right, i.e. identity on the *higher*
    qubits, which is what a shorter string means under the qubit-indexed
    convention above.

    Args:
        hamiltonian: A superfermion ``Hamiltonian`` or ``SparsePauliOp``.
        n_qubits: Width of the emitted labels; defaults to the widest label.
        reverse: Flip each padded label so Qiskit's character ``i`` = qubit
            ``n-1-i`` matches superfermion's character ``q`` = qubit ``q``.
            Leave ``False`` to compare dense matrices; set ``True`` to compare
            expectation values against a superfermion circuit.  See the
            module docstring for why both exist.

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
        padded = label.ljust(width, "I")
        labels.append(padded[::-1] if reverse else padded)
        coeffs.append(coeff)
    return labels, coeffs


def to_qiskit_op(
    hamiltonian: Any, *, n_qubits: Optional[int] = None, reverse: bool = False
):
    """Convert a superfermion Pauli operator to ``qiskit.quantum_info``.

    Args:
        hamiltonian: A superfermion ``Hamiltonian`` or ``SparsePauliOp``.
        n_qubits: Width of the target operator.  Defaults to the widest label,
            which is right when the mapping already used the full register.
        reverse: Emit labels in Qiskit's wire order, i.e. character ``i`` is
            qubit ``n-1-i``.  ``False`` keeps ``to_matrix()`` equal to
            superfermion's own ``pauli_matrix``; ``True`` keeps the *qubit
            assignment* equal (see the module docstring).

    Returns:
        ``qiskit.quantum_info.SparsePauliOp`` with the same terms and complex
        coefficients, in the requested character order.
    """
    _require_qiskit()
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = to_qiskit_labels(
        hamiltonian, n_qubits=n_qubits, reverse=reverse
    )
    return SparsePauliOp(labels, coeffs).simplify(atol=0.0, rtol=0.0)


def from_qiskit_op(op: Any, *, simplify: bool = True, reverse: bool = False):
    """Convert a Qiskit Pauli operator into a superfermion ``Hamiltonian``.

    Args:
        op: ``SparsePauliOp``, ``PauliList``, or anything exposing
            ``paulis``/``coeffs``.
        simplify: Merge repeated labels and drop exact zeros (default).  With
            ``False`` every term is preserved, which is what a round-trip test
            of a deliberately duplicated operator wants.
        reverse: Flip each label, undoing :func:`to_qiskit_op`'s ``reverse``.
            ``False`` (default) reads Qiskit labels in character order, which
            round-trips ``to_qiskit_op(h)`` exactly.

    Returns:
        ``superfermion.observables.Hamiltonian``.

    Raises:
        ValueError: if ``op`` carries no terms.
    """
    from superfermion.observables.core import Hamiltonian, PauliString

    pairs = _qiskit_pairs(op)
    if reverse:
        pairs = [(label[::-1], coeff) for label, coeff in pairs]
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
