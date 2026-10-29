"""
Quantum Chemistry — Fermionic operators and molecular Hamiltonians.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from superfermion.observables.core import Hamiltonian, PauliString


def _term_cut(collected: Dict[str, complex], tol: float, relative: bool) -> float:
    """Magnitude below which a mapped Pauli term is treated as zero."""
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if not relative or not collected:
        return tol
    scale = max(abs(c) for c in collected.values())
    return tol * scale if scale > 0 else tol


class FermionicOperator:
    r"""Represents a sum of fermionic creation and annihilation operators.

    Formula: a^\dagger_i a_j ...
    """

    def __init__(self, terms: Dict[Tuple[Tuple[int, int], ...], complex]):
        r"""
        Args:
            terms: Mapping of ((index, op_type), ...) to coefficient.
                   op_type: 1 for creation (a^\dagger), 0 for annihilation (a).
        """
        self.terms = terms

    @classmethod
    def from_coeffs(
        cls, h1: np.ndarray, h2: Optional[np.ndarray] = None
    ) -> FermionicOperator:
        r"""Build Hamiltonian from one- and two-body integrals.

        H = sum h_ij a^\dagger_i a_j + 0.5 * sum h_ijkl a^\dagger_i a^\dagger_j a_k a_l
        """
        terms: Dict[Tuple[Tuple[int, int], ...], complex] = {}

        # One-body terms -- np.argwhere locates the >1e-10 entries in C speed
        # instead of a full ni*nj Python scan; term construction is unchanged.
        h1 = np.asarray(h1)
        for i, j in np.argwhere(np.abs(h1) > 1e-10):
            terms[((int(i), 1), (int(j), 0))] = complex(h1[i, j])

        # Two-body terms
        if h2 is not None:
            h2 = np.asarray(h2)
            nz = np.argwhere(np.abs(h2) > 1e-10)
            vals = h2[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]]
            for (i, j, k, l), v in zip(nz.tolist(), vals.tolist()):
                terms[((i, 1), (j, 1), (l, 0), (k, 0))] = 0.5 * complex(v)

        return cls(terms)

    # -----------------------------------------------------------------
    #  Jordan-Wigner transformation
    # -----------------------------------------------------------------
    #  a^\dagger_j = 0.5 * (X_j - i Y_j) * Z_{j-1} ... Z_0
    #  a_j         = 0.5 * (X_j + i Y_j) * Z_{j-1} ... Z_0
    #
    #  For a product of n fermionic operators we enumerate the 2^n
    #  X/Y choices, then place Z operators on qubits that see an *odd*
    #  number of fermionic sites to their right.  When a Z chain
    #  overlaps a site that already carries X or Y we multiply the
    #  Paulis using sigma_a sigma_b = delta_{ab} I + i epsilon_{abc} sigma_c.
    # -----------------------------------------------------------------

    # Pauli multiplication: (left, right) → (coeff_factor, result)
    # Encoding: I=0, X=1, Y=2, Z=3  (result 0 = I, 1 = X, 2 = Y, 3 = Z)
    _MUL = {
        (1, 1): (1, 0),    # X*X = I
        (2, 2): (1, 0),    # Y*Y = I
        (3, 3): (1, 0),    # Z*Z = I
        (1, 2): (1j, 3),   # X*Y = iZ
        (1, 3): (-1j, 2),  # X*Z = -iY
        (2, 1): (-1j, 3),  # Y*X = -iZ
        (2, 3): (1j, 1),   # Y*Z = iX
        (3, 1): (1j, 2),   # Z*X = iY
        (3, 2): (-1j, 1),  # Z*Y = -iX
    }
    _PAULI_IDX = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    _IDX_PAULI = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    @staticmethod
    def _mul_pauli(p_left: str, p_right: str):
        """Return (coeff_factor, result_pauli) for left * right."""
        l = FermionicOperator._PAULI_IDX[p_left]
        r = FermionicOperator._PAULI_IDX[p_right]
        if l == 0:
            return 1, p_right
        if r == 0:
            return 1, p_left
        f, res = FermionicOperator._MUL[(l, r)]
        return f, FermionicOperator._IDX_PAULI[res]

    def jordan_wigner(
        self, n_qubits: int, *, tol: float = 1e-12, relative: bool = False
    ) -> Hamiltonian:
        """Perform the Jordan-Wigner transformation.

        Returns a :class:`Hamiltonian` whose terms are the qubit-space
        Pauli-string representation of this fermionic operator.

        Terms whose collected coefficient is below the cut are dropped.  The
        cut is ``tol`` when ``relative`` is false and ``tol * max|coefficient|``
        when it is true.  A *relative* cut is the one that is stable under
        rescaling: on water/6-31G the largest coefficient is ~55 Ha, so an
        absolute 1e-12 cut sits at 2e-14 of the operator norm and a coefficient
        that rounds to 1.0000001e-12 flips in and out between builds.  That is
        the origin of the documented +/-1 term-count wobble, and the reason
        :meth:`MolecularData.to_hamiltonian` defaults to the relative cut.
        """
        if not self.terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        collected: Dict[str, complex] = {}

        for ops, coeff in self.terms.items():
            sites = [idx for idx, _ in ops]
            otypes = [ot for _, ot in ops]
            _n = len(ops)  # kept for clarity — number of fermionic operators

            # Build JW representation iteratively: multiply each fermionic
            # operator's 2-term Pauli expansion with the accumulated product
            # using Pauli algebra (left-to-right ordered by site index).
            acc: Dict[str, complex] = {'I' * n_qubits: coeff}

            for (s, ot) in zip(sites, otypes):
                # 2-term expansion for this fermionic operator
                # a†_s = 0.5 * (X_s * Z_{s-1}...Z_0 - i * Y_s * Z_{s-1}...Z_0)
                # a_s  = 0.5 * (X_s * Z_{s-1}...Z_0 + i * Y_s * Z_{s-1}...Z_0)
                x_factor = 0.5
                if ot == 1:   # creation
                    y_factor = -0.5j
                else:         # annihilation
                    y_factor = 0.5j

                next_acc: Dict[str, complex] = {}

                # Multiply each accumulated Pauli string by both X and Y branches
                for pstr, c in acc.items():
                    pauli = list(pstr)

                    # ---- X branch ----
                    pauli_x = list(pauli)
                    cx = c * x_factor
                    # Apply X at site s (left-multiply in product order)
                    f, p = self._mul_pauli(pauli_x[s], 'X')
                    cx *= f
                    pauli_x[s] = p
                    # Z chain: qubits 0..s-1
                    for k in range(s):
                        f, p = self._mul_pauli(pauli_x[k], 'Z')
                        cx *= f
                        pauli_x[k] = p
                    key_x = ''.join(pauli_x)
                    next_acc[key_x] = next_acc.get(key_x, 0j) + cx

                    # ---- Y branch ----
                    pauli_y = list(pauli)
                    cy = c * y_factor
                    # Apply Y at site s
                    f, p = self._mul_pauli(pauli_y[s], 'Y')
                    cy *= f
                    pauli_y[s] = p
                    # Z chain: qubits 0..s-1
                    for k in range(s):
                        f, p = self._mul_pauli(pauli_y[k], 'Z')
                        cy *= f
                        pauli_y[k] = p
                    key_y = ''.join(pauli_y)
                    next_acc[key_y] = next_acc.get(key_y, 0j) + cy

                acc = next_acc

            # Merge into global collected dict
            for pstr, c in acc.items():
                collected[pstr] = collected.get(pstr, 0j) + c

        # Drop numerically zero terms
        cut = _term_cut(collected, tol, relative)
        terms = []
        for pstr, c in collected.items():
            if abs(c) > cut:
                if abs(c.imag) < cut:
                    terms.append(PauliString(pstr, float(c.real)))
                else:
                    terms.append(PauliString(pstr, c))

        if not terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        return Hamiltonian(terms)

    # -----------------------------------------------------------------
    #  Bravyi-Kitaev transformation
    # -----------------------------------------------------------------
    def bravyi_kitaev(
        self, n_qubits: int, *, tol: float = 1e-12, relative: bool = False
    ) -> Hamiltonian:
        """Perform the Bravyi-Kitaev transformation to qubit space.

        Qubit ``j`` stores the parity of the Fenwick interval
        ``F(j) = {j - lowbit(j+1) + 1, ..., j}`` instead of the occupation of
        a single mode::

            b_j = sum_{i in F(j)} n_i   (mod 2)

        so a ladder operator becomes a Pauli string of weight ``O(log n)``
        rather than the ``O(n)`` Jordan-Wigner chain.

        The derivation used below.  Write ``M[j, i] = [i in F(j)]`` -- unit
        lower triangular, hence invertible over GF(2) -- and let ``S(j)`` be
        row ``j`` of ``M^-1``, so that ``n_j = sum_{k in S(j)} b_k``.  The
        Jordan-Wigner identity ``a_j = (prod_{i<j} Z_i) X_j (1 - Z_j) / 2``
        then reads, in the qubit basis,

            a_j  = 1/2 [ X_{A(j)} Z_{T(j)} + i X_{A(j)-{j}} Y_j Z_{P(j)} ]
            a_j+ = 1/2 [ X_{A(j)} Z_{T(j)} - i X_{A(j)-{j}} Y_j Z_{P(j)} ]

        with the Fenwick column ``A(j) = {k : j in F(k)}`` (the qubits that
        flip when ``n_j`` changes), ``T(j) = XOR_{i<j} S(i)`` (the sign from
        the lower modes) and ``P(j) = (S(j)-{j}) XOR T(j)``.  ``A(j)`` is
        disjoint from ``T(j)`` and ``P(j)``, so neither string needs a
        product-order phase correction.

        The previous revision of this method put ``X``/``Y`` on the mode's own
        site and swapped the update and parity sets.  It produced 38 terms for
        H2 whose spectrum differed from the Jordan-Wigner one by 1.67 Ha and
        which conserved no parity at all, so it was replaced rather than
        relabelled as approximate.

        The term cut behaves exactly as in :meth:`jordan_wigner`.

        Raises:
            ValueError: if a term references a mode outside ``n_qubits``.
        """
        return self._bk_pauli_sum(n_qubits, tol=tol, relative=relative)

    @staticmethod
    def _bk_index_sets(
        n_qubits: int,
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        """Per-mode Bravyi-Kitaev index sets ``(A, T, P)``.

        ``A[j]`` is the Fenwick column (the qubits flipped by a change of
        ``n_j``), ``T[j]`` the sign set of the lower modes and ``P[j]`` the
        set that completes the ``iY`` branch.  ``S(j)`` is recovered by back
        substitution because ``A(i)`` only involves indices ``>= i``.
        """
        if n_qubits < 1:
            return []

        intervals: List[Tuple[int, ...]] = []
        for j in range(n_qubits):
            lowbit = (j + 1) & -(j + 1)
            intervals.append(tuple(range(j - lowbit + 1, j + 1)))

        ancestors: List[List[int]] = [[] for _ in range(n_qubits)]
        for k, interval in enumerate(intervals):
            for i in interval:
                ancestors[i].append(k)

        # S(j) = row j of M^-1: solve (M^T y)_i = sum_{k in A(i)} y_k = d_ij.
        signs: List[set] = []
        for j in range(n_qubits):
            y = [0] * n_qubits
            for i in range(n_qubits - 1, -1, -1):
                acc = 1 if i == j else 0
                for k in ancestors[i]:
                    if k > i:
                        acc ^= y[k]
                y[i] = acc
            signs.append({i for i in range(n_qubits) if y[i]})

        sets: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        lower: set = set()
        for j in range(n_qubits):
            parity = set(lower)
            other = (signs[j] - {j}) ^ parity
            sets.append(
                (tuple(ancestors[j]), tuple(sorted(parity)), tuple(sorted(other)))
            )
            lower ^= signs[j]
        return sets

    @staticmethod
    def _bk_ladder(
        n_qubits: int,
        site: int,
        creation: bool,
        index_sets: Optional[
            List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]
        ] = None,
    ) -> List[Tuple[Dict[int, str], complex]]:
        """The two Pauli strings of ``a_site`` / ``a^dagger_site`` under BK."""
        sets = index_sets if index_sets is not None else (
            FermionicOperator._bk_index_sets(n_qubits)
        )
        ancestors, parity, other = sets[site]
        flipped = tuple(k for k in ancestors if k != site)

        xz: Dict[int, str] = {k: "X" for k in ancestors}
        for k in parity:
            xz[k] = "Z"

        yz: Dict[int, str] = {k: "X" for k in flipped}
        yz[site] = "Y"
        for k in other:
            yz[k] = "Z"

        return [(xz, 0.5 + 0j), (yz, -0.5j if creation else 0.5j)]

    def _bk_pauli_sum(
        self, n_qubits: int, *, tol: float = 1e-12, relative: bool = False
    ) -> Hamiltonian:
        """Expand every term with the :meth:`_bk_ladder` branch structure."""
        if not self.terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        index_sets = self._bk_index_sets(n_qubits)
        collected: Dict[str, complex] = {}

        for ops, coeff in self.terms.items():
            acc: Dict[str, complex] = {'I' * n_qubits: coeff}
            for site, ot in ops:
                site = int(site)
                if not 0 <= site < n_qubits:
                    raise ValueError(
                        f"mode {site} is outside the {n_qubits} qubits this BK "
                        "transform was asked for"
                    )
                branches = self._bk_ladder(n_qubits, site, ot == 1, index_sets)
                next_acc: Dict[str, complex] = {}
                for pstr, c in acc.items():
                    for branch, branch_c in branches:
                        pauli = list(pstr)
                        term_c = c * branch_c
                        for k, p in branch.items():
                            f, res = self._mul_pauli(pauli[k], p)
                            term_c *= f
                            pauli[k] = res
                        key = ''.join(pauli)
                        next_acc[key] = next_acc.get(key, 0j) + term_c
                acc = next_acc
            for pstr, c in acc.items():
                collected[pstr] = collected.get(pstr, 0j) + c

        cut = _term_cut(collected, tol, relative)
        terms = []
        for pstr, c in collected.items():
            if abs(c) > cut:
                if abs(c.imag) < cut:
                    terms.append(PauliString(pstr, float(c.real)))
                else:
                    terms.append(PauliString(pstr, c))

        if not terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        return Hamiltonian(terms)


def get_molecular_hamiltonian(molecule: str, basis: str = "sto-3g") -> Hamiltonian:
    """High-level API to retrieve pre-computed molecular Hamiltonians."""
    # H2 STO-3G at d=0.735 Å, parity-reduced 2-qubit Hamiltonian
    # (O'Malley et al. 2016, Kandala et al. 2017).
    # Nuclear repulsion (0.7199689 Ha) is folded into the II coefficient
    # so VQE eigenvalues are total energies directly.
    # Ground state ≈ -1.1373 Ha in the {|01>, |10>} subspace.
    _CACHED = {
        "h2_sto-3g": [
            PauliString("II", -0.3324043),
            PauliString("ZI",  0.3979374),
            PauliString("IZ", -0.3979374),
            PauliString("ZZ", -0.0112801),
            PauliString("XX",  0.1809312),
        ],
    }
    key = f"{molecule.lower()}_{basis.lower()}"
    if key in _CACHED:
        return Hamiltonian(_CACHED[key])

    raise NotImplementedError(f"Molecule {molecule} not supported yet.")