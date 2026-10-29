"""Z2 symmetry detection and qubit tapering for Pauli-sum Hamiltonians.

A molecular Hamiltonian in a Jordan-Wigner encoding carries conserved
parities (electron number per spin sector, and point-group symmetries where
the basis respects them).  Written as Pauli sums, those parities are strings
of ``Z`` that commute with every term.  Fixing their eigenvalues lets us
remove one qubit per independent string.

The reduction is exact: it conjugates the whole Hamiltonian by a Clifford
(CNOT) network that maps each symmetry string onto a single qubit, then
substitutes that qubit's eigenvalue and drops the qubit.  The spectrum in the
selected sector is preserved bit-for-bit in energy, so the tapered operator
can be compared against the untransformed one as a correctness check.

Example:
    >>> from superfermion.chemistry.tapering import taper_z2, z2_symmetries
    >>> H = mol.to_hamiltonian(active_space=(6, 6))
    >>> z2_symmetries(H)                      # e.g. ['ZZII...', 'IIZZ...']
    >>> H_t, info = taper_z2(H, reference=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> info["n_qubits_after"]                # 12 -> 10 for a closed-shell singlet

Conventions: superfermion stores Pauli strings big-endian, qubit 0 is the
leftmost character, so a string's character index is the qubit index.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.observables.core import Hamiltonian, PauliString

__all__ = ["z2_symmetries", "taper_z2", "TaperedInfo"]

_PAULI_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
_BITS_PAULI = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def _single_pauli(x: int, z: int) -> np.ndarray:
    """Canonical matrix for one qubit: ``i**(x*z) X**x Z**z``."""
    mat = np.eye(2, dtype=complex)
    if x:
        mat = mat @ np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if z:
        mat = mat @ np.diag([1.0, -1.0]).astype(complex)
    return (1j ** (x * z)) * mat


def _build_cnot_phase_table() -> np.ndarray:
    """Phase (in units of ``i``) picked up by each 2-qubit Pauli under CNOT.

    The symplectic rule ``x_t ^= x_c``, ``z_c ^= z_t`` identifies the image up
    to a power of ``i``: for example ``X_c Z_t`` maps to ``-Y_c Y_t``, not to
    ``+Y_c Y_t``.  Since the canonical representative of ``(x, z)`` is
    ``i**(x*z) X**x Z**z``, that power has to be tracked or the transformed
    Hamiltonian is wrong (it stays Hermitian and commutes with the symmetry,
    but its spectrum is shifted -- the bug is silent until you compare
    energies).
    """
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    cnot_dag = cnot.conj().T
    powers = {1.0: 0, 1j: 1, -1.0: 2, -1j: 3}
    table = np.zeros(16, dtype=np.int8)
    for bits in range(16):
        xc, zc, xt, zt = (bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, bits & 1
        old = np.kron(_single_pauli(xc, zc), _single_pauli(xt, zt))
        new = cnot @ old @ cnot_dag
        predicted = np.kron(
            _single_pauli(xc, zc ^ zt), _single_pauli(xt ^ xc, zt)
        )
        flat = int(np.argmax(np.abs(predicted)))
        ratio = complex(new.ravel()[flat]) / complex(predicted.ravel()[flat])
        key = round(float(ratio.real), 12) + 1j * round(float(ratio.imag), 12)
        table[bits] = powers[key]
    return table


_CNOT_PHASE = _build_cnot_phase_table()


def _term_bits(terms: Sequence[Any], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split Pauli strings into (x, z) bit arrays of shape (n_terms, n_qubits)."""
    x = np.zeros((len(terms), n_qubits), dtype=np.uint8)
    z = np.zeros((len(terms), n_qubits), dtype=np.uint8)
    for i, term in enumerate(terms):
        label = term.pauli_str
        if len(label) != n_qubits:
            raise ValueError(
                f"Pauli string {label!r} has {len(label)} characters, "
                f"expected {n_qubits}"
            )
        for q, ch in enumerate(label):
            try:
                bx, bz = _PAULI_BITS[ch]
            except KeyError:
                raise ValueError(f"invalid Pauli character {ch!r} in {label!r}") from None
            x[i, q] = bx
            z[i, q] = bz
    return x, z


def _null_space_gf2(rows: Sequence[int], n_cols: int) -> List[int]:
    """Basis of the GF(2) null space of ``rows`` (bitmask integers).

    The elimination keeps a pivot -> row map keyed by the highest set bit, so
    every row keeps a *distinct* pivot.  A single pass over an unsorted list
    can otherwise append a row whose pivot is already taken, which silently
    produces vectors that are not in the null space -- the symptom is a
    "symmetry" that does not commute with the Hamiltonian.
    """
    basis: Dict[int, int] = {}
    for row in rows:
        cur = int(row)
        while cur:
            pivot = cur.bit_length() - 1
            if pivot in basis:
                cur ^= basis[pivot]
            else:
                basis[pivot] = cur
                break

    pivots = sorted(basis)
    for pivot in reversed(pivots):
        row = basis[pivot]
        for other in pivots:
            if other != pivot and (row >> other) & 1:
                row ^= basis[other]
        basis[pivot] = row

    free_cols = [c for c in range(n_cols) if c not in basis]
    null: List[int] = []
    for free in free_cols:
        vec = 1 << free
        for pivot in pivots:
            if (basis[pivot] >> free) & 1:
                vec |= 1 << pivot
        null.append(vec)
    return null


def z2_symmetries(hamiltonian: Any) -> List[str]:
    """Return the independent ``{I, Z}`` strings that commute with every term.

    A diagonal Pauli string commutes with a term when its ``Z`` support
    overlaps the term's ``X``/``Y`` support an even number of times.  Each
    independent solution is one removable qubit.

    Args:
        hamiltonian: a ``superfermion`` Hamiltonian / Pauli sum, or any object
            with ``.terms`` whose entries expose ``.pauli_str``.

    Returns:
        Pauli labels (``I``/``Z`` only), one per independent symmetry.
    """
    terms = list(hamiltonian.terms)
    if not terms:
        return []
    n_qubits = len(terms[0].pauli_str)
    x, _ = _term_bits(terms, n_qubits)

    rows = [
        int(sum(1 << q for q in np.flatnonzero(x[i])))
        for i in range(len(terms))
    ]
    rows = [r for r in rows if r]

    labels = []
    for mask in _null_space_gf2(rows, n_qubits):
        labels.append(
            "".join("Z" if (mask >> q) & 1 else "I" for q in range(n_qubits))
        )
    return labels


def _apply_cnot(
    x: np.ndarray, z: np.ndarray, phase: np.ndarray, control: int, target: int
) -> None:
    """Conjugate every term by CNOT(control -> target), in place.

    The adjoint action on Pauli operators is ``X_c -> X_c X_t`` and
    ``Z_t -> Z_c Z_t``, i.e. ``x_t ^= x_c`` and ``z_c ^= z_t``.  ``phase``
    carries the power of ``i`` that this relabelling absorbs for each term.
    """
    index = (
        x[:, control] * 8
        + z[:, control] * 4
        + x[:, target] * 2
        + z[:, target]
    )
    phase += _CNOT_PHASE[index]
    phase %= 4
    x[:, target] ^= x[:, control]
    z[:, control] ^= z[:, target]


def _parse_reference(reference: Any, n_qubits: int) -> np.ndarray:
    if isinstance(reference, str):
        bits = [c for c in reference if c in "01"]
        if len(bits) != n_qubits:
            raise ValueError(
                f"reference bitstring has {len(bits)} bits, expected {n_qubits}"
            )
        return np.array([int(b) for b in bits], dtype=np.uint8)
    occ = np.asarray(list(reference), dtype=np.uint8).ravel()
    if occ.size != n_qubits:
        raise ValueError(f"reference has {occ.size} entries, expected {n_qubits}")
    return occ


class TaperedInfo(dict):
    """``taper_z2``'s metadata, extended with the inverse of the reduction.

    The reduction is exact, but it used to be one-way: ``taper_z2`` handed back
    a smaller Hamiltonian and a description of what it had done, and nothing
    carried a bitstring or an observable *back* across the change of frame.  A
    caller wanting the full-space occupation behind a tapered measurement had
    to re-derive the inversion from ``info["clifford_gates"]``, which is
    exactly the kind of re-derivation that silently transposes a gate order and
    still looks plausible -- a reversed Clifford network produces perfectly
    valid bitstrings, just not the ones that were measured.

    Subclassing ``dict`` keeps every existing access working (item access,
    ``**info``, ``json.dumps``, equality against the old plain literal) while
    adding two methods:

    ``unrotate``
        Map a bitstring read off the reduced register back to the
        ``n_qubits_before`` occupation bitstring, filling in the eliminated
        qubits from the sector and undoing the CNOT network.

    ``expand_observable``
        Lift a Pauli sum on the reduced register onto the original qubits, so
        an expectation value estimated on the tapered circuit can be compared
        against the untransformed operator.

    Both are cheap because a CNOT maps a computational basis state to a
    computational basis state (``|b_c b_t> -> |b_c, b_t ^ b_c>``), so the
    network merely permutes the ``2**n`` basis vectors and its inverse is the
    network replayed backwards.  No matrices, no statevector, no cost
    exponential in the qubit count.

    Args:
        data: The metadata the plain dict used to carry.
        keep: Original indices of the surviving qubits, ascending.
        tapered: Original indices of the eliminated qubits, parallel to
            ``sector``.
        circuit: ``(control, target)`` CNOT pairs in application order.
        sector: Eigenvalue (``+1`` or ``-1``) fixed on each eliminated qubit.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        *,
        keep: Sequence[int] = (),
        tapered: Sequence[int] = (),
        circuit: Sequence[Tuple[int, int]] = (),
        sector: Sequence[int] = (),
    ) -> None:
        super().__init__(data)
        self._keep = [int(q) for q in keep]
        self._tapered = [int(q) for q in tapered]
        self._circuit = [(int(c), int(t)) for c, t in circuit]
        self._sector = [int(s) for s in sector]

    @property
    def keep_qubits(self) -> List[int]:
        """Original indices of the qubits that survived the reduction."""
        return list(self._keep)

    def unrotate(self, bitstring: Any) -> str:
        """Expand a reduced-register bitstring onto the original ``n`` qubits.

        Args:
            bitstring: ``n_qubits_after`` bits, qubit 0 first -- either a
                string of ``0``/``1`` or a 0/1 sequence, the same two forms
                ``taper_z2(reference=...)`` accepts.

        Returns:
            The ``n_qubits_before``-character occupation bitstring, qubit 0
            first, that the reduced outcome corresponds to.

        Note:
            This is a basis relabelling, so it is exact.  It does not certify
            the outcome: the sector fixes the eliminated qubits, but a noisy
            tapered measurement can still land in a block the untransformed
            Hamiltonian reaches with amplitude zero, and the expanded string
            will look perfectly ordinary.
        """
        bits = _parse_reference(bitstring, int(self["n_qubits_after"]))
        full = np.zeros(int(self["n_qubits_before"]), dtype=np.uint8)
        for j, q in enumerate(self._keep):
            full[q] = bits[j]
        for k, q in enumerate(self._tapered):
            full[q] = 0 if self._sector[k] > 0 else 1
        for control, target in reversed(self._circuit):
            full[target] ^= full[control]
        return "".join(str(int(b)) for b in full)

    def expand_observable(self, observable: Any) -> Hamiltonian:
        """Lift a reduced-register Pauli sum back onto the original qubits.

        Given ``O`` acting on the ``n_qubits_after`` register, this returns
        ``U^dag (O kron I) U`` on ``n_qubits_before`` qubits.  For every state
        of the selected sector its matrix elements are the ones measured on
        the tapered circuit, so ``expand_observable(H_t)`` agrees with ``H``
        element for element *inside* that sector -- which is the check that
        says the reduction and its inverse are consistent.

        It deliberately does **not** reproduce ``H`` globally, and comparing
        full dense matrices without projecting both sides will report a large
        difference that is not an error.  ``H_t`` has already been evaluated
        at the sector: ``taper_z2`` replaced ``Z`` on each eliminated qubit by
        its eigenvalue and then dropped the qubit, so the information those
        qubits carried is gone by construction.  Outside the sector this
        operator is not meaningful.

        Nor is there a ``Z -> sector value`` substitution here.  That looks
        right and is wrong: it belongs in the *rotated* frame, on the operator
        being reduced, which is where ``taper_z2`` does it.  After
        back-rotation the eliminated qubit no longer carries the symmetry --
        the surviving original symmetry is ``Z_0 Z_3``, not ``Z_3`` -- so the
        qubit takes both eigenvalues on the sector and replacing it by a
        scalar flips one whole block.  The symptom is specific and easy to
        miss: the two sector energies come out **swapped**, which a sorted
        spectrum and a summed expectation value both accept.

        Args:
            observable: Pauli-sum Hamiltonian (or anything with ``.terms``
                exposing ``.pauli_str``) over the reduced register.

        Returns:
            ``sf.Hamiltonian`` over the original qubit count.

        Raises:
            ValueError: if a term has a qubit count other than
                ``n_qubits_after``, or a Pauli character that is not I/X/Y/Z.
        """
        terms = list(observable.terms)
        if not terms:
            return Hamiltonian([])

        n_after = int(self["n_qubits_after"])
        n_before = int(self["n_qubits_before"])
        x = np.zeros((len(terms), n_before), dtype=np.uint8)
        z = np.zeros((len(terms), n_before), dtype=np.uint8)
        coeffs = np.array([complex(t.coeffs) for t in terms], dtype=complex)

        for i, term in enumerate(terms):
            label = term.pauli_str
            if len(label) != n_after:
                raise ValueError(
                    f"observable term {label!r} has {len(label)} characters, "
                    f"expected {n_after} for a tapered operator"
                )
            for j, ch in enumerate(label):
                try:
                    bx, bz = _PAULI_BITS[ch]
                except KeyError:
                    raise ValueError(
                        f"invalid Pauli character {ch!r} in {label!r}"
                    ) from None
                x[i, self._keep[j]] = bx
                z[i, self._keep[j]] = bz

        # U^dag P U applies the CNOTs in the opposite order to U P U^dag.
        phase = np.zeros(len(terms), dtype=np.int16)
        for control, target in reversed(self._circuit):
            _apply_cnot(x, z, phase, control, target)

        coeffs = coeffs * (1j ** phase)

        combined: Dict[str, complex] = {}
        for i in range(len(terms)):
            label = "".join(
                _BITS_PAULI[(int(x[i, j]), int(z[i, j]))] for j in range(n_before)
            )
            combined[label] = combined.get(label, 0.0) + coeffs[i]

        return Hamiltonian(
            [
                PauliString(label, coeffs=coef)
                for label, coef in combined.items()
                if abs(coef) > 1e-15
            ]
        )


def taper_z2(
    hamiltonian: Any,
    sector: Optional[Sequence[int]] = None,
    reference: Optional[Any] = None,
    *,
    symmetries: Optional[Sequence[str]] = None,
) -> Tuple[Hamiltonian, TaperedInfo]:
    """Remove one qubit per independent Z2 symmetry, exactly.

    Args:
        hamiltonian: Pauli-sum Hamiltonian.
        sector: eigenvalues (±1) to keep, one per symmetry, in the order of
            ``info["symmetries"]``.  Defaults to the values that ``reference``
            has, and otherwise to ``+1``.
        reference: occupation bitstring (``"010110"`` or a 0/1 sequence, qubit
            0 first) whose sector should be kept -- normally the Hartree-Fock
            determinant.  Only used when ``sector`` is None.
        symmetries: Restrict the reduction to these ``{I,Z}`` strings instead
            of every symmetry found.  This is what makes the *symmetry-
            conserving* Bravyi-Kitaev encoding expressible: that encoding
            removes exactly the two particle-number parities, no more, and
            the count is part of the contract rather than a side effect of
            how many symmetries the basis happens to expose.

    Returns:
        ``(tapered_hamiltonian, info)``.  ``info`` is a :class:`TaperedInfo`,
        a ``dict`` subclass, so every key access that worked on the old plain
        dict still works.  It records the original qubit indices that were
        removed, the symmetry strings, the chosen sector, the CNOT gates of
        the Clifford in application order, and the term counts before and
        after.  Qubit order of the remaining qubits is preserved, so original
        qubit ``q`` becomes ``q - #{removed < q}``.

        The same object inverts the reduction:
        ``info.unrotate(bitstring)`` maps a reduced-register measurement back
        to the original occupation bitstring, and
        ``info.expand_observable(O)`` lifts a reduced-register Pauli sum back
        onto the original qubits.  ``info["clifford_gates"]`` remains
        available for inspection, but nothing has to re-derive the inverse
        from it.

    Raises:
        ValueError: on malformed input, or if a term anticommutes with a
            symmetry (which would mean the detection is wrong).
    """
    terms = list(hamiltonian.terms)
    if not terms:
        return Hamiltonian([]), TaperedInfo(
            {
                "n_qubits_before": 0,
                "n_qubits_after": 0,
                "tapered_qubits": [],
                "symmetries": [],
                "sector": [],
                "clifford_gates": [],
                "terms_before": 0,
                "terms_after": 0,
            }
        )

    n_qubits = len(terms[0].pauli_str)
    x, z = _term_bits(terms, n_qubits)
    phase = np.zeros(len(terms), dtype=np.int16)
    coeffs = np.array([complex(t.coeffs) for t in terms], dtype=complex)

    sym_labels = list(symmetries) if symmetries is not None else z2_symmetries(hamiltonian)
    if not sym_labels:
        return Hamiltonian(list(terms)), TaperedInfo(
            {
                "n_qubits_before": n_qubits,
                "n_qubits_after": n_qubits,
                "tapered_qubits": [],
                "symmetries": [],
                "sector": [],
                "clifford_gates": [],
                "terms_before": len(terms),
                "terms_after": len(terms),
            },
            keep=list(range(n_qubits)),
        )
    for label in sym_labels:
        if len(label) != n_qubits or set(label) - set("IZ"):
            raise ValueError(
                f"symmetry {label!r} must be an {{I,Z}} string of length "
                f"{n_qubits}"
            )
    gens = np.zeros((len(sym_labels), n_qubits), dtype=np.uint8)
    for g, label in enumerate(sym_labels):
        gens[g] = [1 if ch == "Z" else 0 for ch in label]
    gens_original = gens.copy()

    # A generator that does not commute with the Hamiltonian would make the
    # whole reduction invalid; catch it here with an honest message rather
    # than letting it surface as a stray X/Y on a tapered qubit.
    for g, label in enumerate(sym_labels):
        overlap = (x.astype(np.int16) @ gens[g].astype(np.int16)) % 2
        if overlap.any():
            raise ValueError(
                f"detected symmetry {label!r} does not commute with "
                f"{int(overlap.sum())} of the {len(terms)} terms; the Pauli "
                "strings are not {I,Z}-only or the detection is wrong"
            )

    # Rotate each symmetry onto a single qubit with CNOT ladders.  Every
    # generator ends as Z on a qubit no other generator touches.
    tapered: List[int] = []
    circuit: List[Tuple[int, int]] = []
    for g in range(len(sym_labels)):
        for q in tapered:
            if gens[g, q]:
                others = [i for i in np.flatnonzero(gens[g]) if i != q]
                if not others:
                    raise ValueError(
                        "two symmetry generators collapse onto the same qubit; "
                        "the generator set is inconsistent"
                    )
                target = int(others[0])
                _apply_cnot(x, z, phase, q, target)
                circuit.append((q, target))
                gens[:, q] = gens[:, q] ^ gens[:, target]
        while gens[g].sum() > 1:
            support = np.flatnonzero(gens[g])
            control, target = int(support[0]), int(support[-1])
            _apply_cnot(x, z, phase, control, target)
            circuit.append((control, target))
            gens[:, control] = gens[:, control] ^ gens[:, target]
        tapered.append(int(np.flatnonzero(gens[g])[0]))

    if len(set(tapered)) != len(tapered):
        raise ValueError("tapered qubit selection is not independent")

    # Sector eigenvalues, evaluated on the original strings.
    if sector is not None:
        sector_values = [int(s) for s in sector]
        if len(sector_values) != len(sym_labels):
            raise ValueError(
                f"sector has {len(sector_values)} entries, expected "
                f"{len(sym_labels)}"
            )
        if any(s not in (-1, 1) for s in sector_values):
            raise ValueError("sector entries must be +1 or -1")
    elif reference is not None:
        occ = _parse_reference(reference, n_qubits)
        sector_values = []
        for g in range(len(sym_labels)):
            overlap = int(np.dot(gens_original[g], occ) % 2)
            sector_values.append(-1 if overlap else 1)
    else:
        warnings.warn(
            "taper_z2 called without sector or reference: every symmetry is "
            "fixed to +1, which is the wrong sector for most molecules. Pass "
            "the Hartree-Fock occupation as reference=.",
            UserWarning,
            stacklevel=2,
        )
        sector_values = [1] * len(sym_labels)

    # Substitute Z_{q_k} = s_k and drop the tapered qubits.
    keep = [q for q in range(n_qubits) if q not in set(tapered)]
    x_keep = x[:, keep]
    z_keep = z[:, keep]

    for k, q in enumerate(tapered):
        if x[:, q].any():
            raise ValueError(
                f"a term carries X/Y on tapered qubit {q}; the Hamiltonian does "
                "not commute with the symmetry, so tapering would be invalid"
            )
        coeffs = np.where(z[:, q] == 1, coeffs * sector_values[k], coeffs)

    coeffs = coeffs * (1j ** phase)

    combined: Dict[str, complex] = {}
    for i in range(len(terms)):
        label = "".join(_BITS_PAULI[(int(x_keep[i, j]), int(z_keep[i, j]))]
                        for j in range(len(keep)))
        combined[label] = combined.get(label, 0.0) + coeffs[i]

    new_terms = [
        PauliString(label, coeffs=coef)
        for label, coef in combined.items()
        if abs(coef) > 1e-15
    ]

    info = TaperedInfo(
        {
            "n_qubits_before": n_qubits,
            "n_qubits_after": len(keep),
            "tapered_qubits": tapered,
            "symmetries": sym_labels,
            "sector": sector_values,
            "clifford_gates": circuit,
            "terms_before": len(terms),
            "terms_after": len(new_terms),
        },
        keep=keep,
        tapered=tapered,
        circuit=circuit,
        sector=sector_values,
    )
    return Hamiltonian(new_terms), info
