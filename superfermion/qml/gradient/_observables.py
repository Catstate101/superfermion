"""One observable language for every gradient entry point (E/V-3,4,5,7).

The four gradient engines and ``RunResult.grad`` each grew their own
observable coercion, and the five of them did not agree:

* ``adjoint_grad`` accepted ``list[PauliString]`` while
  ``parameter_shift_grad`` and ``finite_diff_grad`` raised
  ``Cannot evaluate observable of type list``;
* the engine's own term form ``[([2, 2], 1.0, 0.0)]`` was accepted by
  ``RunResult.grad`` and by nothing else;
* a Pauli label whose width was not the circuit's qubit count was accepted
  and silently meant a *different* operator -- ``'Z'`` on two wires acted as
  ``II`` (every derivative came back ~1e-17) and ``'YYZ'`` dropped the third
  character;
* a non-Hermitian coefficient dict was differentiated three different ways:
  the adjoint returned 0.154897, parameter-shift and finite-difference
  returned the real part 0.621060, and the true derivative is
  0.621060-0.070410j.  Qiskit's LinComb estimator gradient refuses such an
  input outright ("Non-Hermitian input observable"), which is the behaviour
  adopted here: none of these engines defines the gradient of a
  non-observable, so agreeing to differ is worse than refusing.

Everything below is a pure function of its arguments, and the observable
classes are imported lazily so ``superfermion.qml.gradient`` stays importable
while ``superfermion`` itself is still initialising.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

#: Character -> the Rust engine's per-qubit Pauli code.
PAULI_CODES = {"I": 0, "X": 1, "Y": 2, "Z": 3}
#: Code -> character, for rebuilding a label from the engine form.
PAULI_LETTERS = ("I", "X", "Y", "Z")


def pauli_indices(label: Any) -> list[int]:
    """Rust ``u8`` codes for a Pauli label, naming the offending character.

    A bare ``_PAULI_MAP[ch]`` raised ``KeyError: 'N'`` for a typo such as
    ``"not a pauli!"``, which said nothing about where the problem was or
    what a valid string looks like (CH-11).
    """
    indices = []
    for position, ch in enumerate(str(label).upper()):
        code = PAULI_CODES.get(ch)
        if code is None:
            raise ValueError(
                f"invalid Pauli character {ch!r} at position {position} of "
                f"{label!r}: a Pauli string holds only I, X, Y and Z, "
                f"one character per qubit (e.g. 'ZIII' for a 4-qubit system)."
            )
        indices.append(code)
    return indices


def pauli_codes_to_label(codes) -> str:
    """Inverse of :func:`pauli_indices`, for error messages."""
    letters = []
    for code in codes:
        try:
            letters.append(PAULI_LETTERS[int(code)])
        except (IndexError, TypeError, ValueError):
            letters.append(str(code))
    return "".join(letters)


def _as_codes(value) -> Optional[list[int]]:
    """The per-qubit Pauli codes if ``value`` already is that form, else None.

    The engine form carries integers, so a list of four ``0..3`` ints is a
    four-qubit term; a list of anything else is some other spelling and the
    caller must say what it means.
    """
    if isinstance(value, str) or isinstance(value, Mapping):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            return None
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    if not items:
        return None
    codes = []
    for item in items:
        if isinstance(item, (bool, np.bool_)):
            return None
        if not isinstance(item, (int, np.integer)):
            return None
        if not 0 <= int(item) <= 3:
            return None
        codes.append(int(item))
    return codes


def _as_engine_term(item) -> Optional[tuple]:
    """``(codes, re, im)`` when ``item`` is already a single term, else None.

    Accepts the engine spelling ``([2, 2], 1.0, 0.0)`` and the two
    user-sensible short forms a label allows, ``(label, coef)`` and
    ``(label, coef_re, coef_im)``.
    """
    if not isinstance(item, (list, tuple)):
        return None
    if len(item) not in (2, 3):
        return None
    head = item[0]
    codes = _as_codes(head)
    if codes is None and isinstance(head, str):
        try:
            codes = pauli_indices(head)
        except ValueError:
            return None
    if codes is None:
        return None
    if len(item) == 2:
        try:
            coefficient = complex(item[1])
        except (TypeError, ValueError):
            return None
        return (codes, coefficient.real, coefficient.imag)
    try:
        # The engine spelling carries two real numbers, not one complex one.
        return (codes, float(np.real(item[1])), float(np.real(item[2])))
    except (TypeError, ValueError):
        return None


def observable_terms(observable, n_qubits: int = 0) -> list[tuple]:
    """Convert any supported observable spelling to ``[(codes, re, im)]``.

    Supported:

    * ``str`` -- one Pauli character per qubit, character 0 = qubit 0;
    * ``Sequence[str]`` -- several labels summed with unit weight;
    * ``Mapping[str, complex]`` -- labelled terms with coefficients;
    * ``PauliString``, ``SparsePauliOp``, ``Hamiltonian`` (anything with
      ``.terms``), and a ``Sequence`` mixing those with ``str``;
    * the engine's own ``[(codes, re, im)]``, including numpy integer code
      arrays and the ``(label, coef)`` short forms;
    * an integer ``numpy.ndarray`` of codes, meaning one term of unit weight.

    ``n_qubits`` is accepted for signature stability; the width check lives in
    :func:`check_width` so that callers which legitimately do not know the
    register size can skip it.
    """
    from superfermion.observables.core import PauliString, SparsePauliOp

    if isinstance(observable, str):
        return [(pauli_indices(observable), 1.0, 0.0)]

    if isinstance(observable, np.ndarray) and observable.dtype.kind in "iu":
        codes = _as_codes(observable)
        if codes is not None:
            return [(codes, 1.0, 0.0)]

    if isinstance(observable, (list, tuple, np.ndarray)):
        items = list(observable)
        if not items:
            return []
        if all(isinstance(item, str) for item in items):
            return [(pauli_indices(item), 1.0, 0.0) for item in items]
        if all(isinstance(item, (str, PauliString, SparsePauliOp)) for item in items):
            terms: list[tuple] = []
            for item in items:
                terms.extend(observable_terms(item, n_qubits))
            return terms
        engine_terms = [_as_engine_term(item) for item in items]
        if all(term is not None for term in engine_terms):
            return engine_terms
        raise TypeError(
            "Cannot convert this observable sequence: expected Pauli labels, "
            "PauliString/SparsePauliOp terms, or the engine form "
            "[(paulis, coef_re, coef_im)]. Got "
            f"{[type(item).__name__ for item in items]}."
        )

    if isinstance(observable, Mapping):
        terms = []
        for label, coefficient in observable.items():
            try:
                value = complex(coefficient)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Cannot use {coefficient!r} as the coefficient of "
                    f"{label!r}: a Pauli term's weight must be a number."
                ) from None
            terms.append((pauli_indices(label), value.real, value.imag))
        return terms

    if isinstance(observable, PauliString):
        coefficient = complex(observable.coeffs)
        return [(pauli_indices(observable.pauli_str), coefficient.real,
                 coefficient.imag)]

    if isinstance(observable, SparsePauliOp):
        terms = []
        for label, coefficient in observable._terms:
            value = complex(coefficient)
            terms.append((pauli_indices(label), value.real, value.imag))
        return terms

    if hasattr(observable, "terms"):
        terms = []
        for term in observable.terms:
            value = complex(term.coeffs)
            terms.append((pauli_indices(term.pauli_str), value.real, value.imag))
        return terms

    raise TypeError(
        f"Cannot convert observable of type {type(observable).__name__} to "
        f"Rust format. Expected str, Sequence[str], Sequence[PauliString], "
        f"dict, SparsePauliOp, Hamiltonian, or the engine form "
        f"[(paulis, coef_re, coef_im)]."
    )


def check_width(terms, n_qubits: int, what: str = "observable") -> None:
    """Every term must act on exactly the circuit's wires (E/V-7).

    A label one character short used to behave as identity on the missing
    wire and a label one character long used to drop the extra character, so
    both silently evaluated a different operator than the one written.
    """
    if not n_qubits or int(n_qubits) <= 0:
        return
    for codes, _re, _im in terms:
        if len(codes) == int(n_qubits):
            continue
        label = pauli_codes_to_label(codes)
        raise ValueError(
            f"{what} term {label!r} acts on {len(codes)} wire(s) but the "
            f"circuit has {int(n_qubits)} qubit(s). A Pauli label carries one "
            f"character per qubit, character 0 = qubit 0, so a "
            f"{int(n_qubits)}-qubit circuit needs a {int(n_qubits)}-character "
            f"label (e.g. {'Z' * int(n_qubits)!r}). Padding it silently would "
            f"measure a different operator."
        )


def check_hermitian(terms, what: str = "observable") -> None:
    """Refuse a non-Hermitian operator, as Qiskit's gradient estimators do.

    For distinct Pauli labels the sum ``sum_i c_i P_i`` is Hermitian iff every
    ``c_i`` is real: ``(c P)^dagger = conj(c) P`` and distinct Paulis are
    linearly independent.  Repeated labels are summed first, so a pair of
    conjugate terms that cancel their imaginary parts is accepted.
    """
    sums: dict[tuple, complex] = {}
    for codes, re_part, im_part in terms:
        key = tuple(codes)
        sums[key] = sums.get(key, 0.0 + 0.0j) + complex(re_part, im_part)
    offenders = []
    for key, coefficient in sums.items():
        if abs(coefficient.imag) > 1e-12 * max(1.0, abs(coefficient)):
            offenders.append((pauli_codes_to_label(key), coefficient))
    if not offenders:
        return
    detail = ", ".join(f"{label!r}: {coefficient!r}"
                       for label, coefficient in offenders)
    raise ValueError(
        f"{what} is not Hermitian ({detail}): its coefficients have a non-zero "
        f"imaginary part, so <O> is complex and the gradient engines do not "
        f"agree on what to return (the adjoint assumes a Hermitian generator "
        f"and returns a third number). Qiskit's LinComb estimator gradient "
        f"raises 'Non-Hermitian input observable' for the same input, and so "
        f"does this one. Differentiate the real and imaginary coefficient "
        f"terms as two separate observables instead."
    )


def normalize_for_gradient(observable, n_qubits: int,
                           what: str = "observable") -> list[tuple]:
    """The one coercion every gradient entry point applies.

    Converts any spelling, then enforces the two properties the engines
    assume: the label width matches the register, and the operator is
    Hermitian.  Returns the engine's ``[(codes, re, im)]``.
    """
    terms = observable_terms(observable, n_qubits)
    if not terms:
        raise ValueError(
            f"{what} is empty: an observable needs at least one Pauli term "
            f"(got {observable!r})."
        )
    check_width(terms, n_qubits, what)
    check_hermitian(terms, what)
    return terms


def terms_to_dict(terms) -> dict[str, complex]:
    """``{label: coefficient}`` with repeated labels summed, first-seen order."""
    merged: dict[str, complex] = {}
    for codes, re_part, im_part in terms:
        label = pauli_codes_to_label(codes)
        merged[label] = merged.get(label, 0.0 + 0.0j) + complex(re_part, im_part)
    return merged
