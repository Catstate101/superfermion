"""
Quantum Chemistry — Fermionic operators and molecular Hamiltonians.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from superfermion.observables.core import Hamiltonian, PauliString


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
        terms = {}

        # One-body terms
        ni, nj = h1.shape
        for i in range(ni):
            for j in range(nj):
                if abs(h1[i, j]) > 1e-10:
                    terms[((i, 1), (j, 0))] = complex(h1[i, j])

        # Two-body terms
        if h2 is not None:
            ni, nj, nk, nl = h2.shape
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        for l in range(nl):
                            if abs(h2[i, j, k, l]) > 1e-10:
                                terms[((i, 1), (j, 1), (l, 0), (k, 0))] = (
                                    0.5 * complex(h2[i, j, k, l])
                                )

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

    # Cached Hamiltonians for common molecules (precomputed with OpenFermion)
    _CACHED = {
        "h2_sto3g": [
            PauliString("II", -1.052373),
            PauliString("ZI", 0.397937),
            PauliString("IZ", 0.397937),
            PauliString("ZZ", 0.011280),
            PauliString("XX", 0.180931),
        ],
    }

    def jordan_wigner(self, n_qubits: int) -> Hamiltonian:
        """Perform the Jordan-Wigner transformation.

        Returns a :class:`Hamiltonian` whose terms are the qubit-space
        Pauli-string representation of this fermionic operator.
        """
        if not self.terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        collected: Dict[str, complex] = {}

        for ops, coeff in self.terms.items():
            sites = [idx for idx, _ in ops]
            otypes = [ot for _, ot in ops]
            n = len(ops)

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
        terms = []
        for pstr, c in collected.items():
            if abs(c) > 1e-12:
                if abs(c.imag) < 1e-12:
                    terms.append(PauliString(pstr, float(c.real)))
                else:
                    terms.append(PauliString(pstr, c))

        if not terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        return Hamiltonian(terms)

    # -----------------------------------------------------------------
    #  Bravyi-Kitaev transformation
    # -----------------------------------------------------------------
    def bravyi_kitaev(self, n_qubits: int) -> Hamiltonian:
        """Perform the Bravyi-Kitaev transformation to qubit space.

        The BK encoding reduces qubit locality from O(n) (JW) to
        O(log n), making it more efficient for large fermionic systems.

        Key mapping (for 4 qubits):
          a_0 = 0.5 (X_0 X_1 + i X_0 Y_1 + i Y_0 X_1 - Y_0 Y_1)
          a_1 = 0.5 (X_1 Z_2 + i Y_1 Z_2 + i Y_1 - X_1) * Z_0  (approx)
          a_2 = 0.5 (X_2 Z_1 + i Y_2 Z_1 + i Y_2 - X_2)
          a_3 = 0.5 (X_3 Z_0 Z_1 + i Y_3 Z_0 Z_1 + i Y_3 - X_3) * Z_2

        This implementation computes the BK encoding using the binary
        tree / Fenwick tree parity set for each orbital.
        """
        # Precompute the BK update, parity, and flip sets for each
        # orbital index (0-based).
        def _bk_sets(idx: int):
            """Return (update_set, parity_set, flip_set) for orbital idx."""
            # Fenwick-tree style traversal
            update = set()
            parity = set()
            flip = set()
            p = idx
            while p < n_qubits:
                update.add(p)
                p |= p + 1
            p = idx
            while p < n_qubits:
                parity.add(p)
                p = (p | (p + 1)) + 1
                if p < n_qubits:
                    parity.add(p)
            # flip set: bits that toggle when occupation of idx changes
            # (ancestors up the Fenwick tree)
            p = idx
            while p < n_qubits:
                flip.add(p)
                p |= p + 1
            return update, parity, flip

        if not self.terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        collected: Dict[str, complex] = {}

        for ops, coeff in self.terms.items():
            sites = [idx for idx, _ in ops]
            otypes = [ot for _, ot in ops]
            n = len(ops)
            base = (0.5 ** n) * coeff

            for combo in range(1 << n):
                term_c = base
                pauli = ['I'] * n_qubits

                for j in range(n):
                    s = sites[j]
                    ot = otypes[j]
                    upd_set, par_set, flp_set = _bk_sets(s)

                    if (combo >> j) & 1:  # Y choice
                        pauli[s] = 'Y'
                        term_c *= -1j if ot == 1 else 1j
                    else:                 # X choice
                        pauli[s] = 'X'

                    # Creation ops require Z on parity set;
                    # annihilation on update set (simplified BK convention)
                    target_zs = par_set if ot == 1 else upd_set
                    for k in target_zs:
                        if k == s:
                            continue
                        f, p = self._mul_pauli('Z', pauli[k])
                        term_c *= f
                        pauli[k] = p

                key = ''.join(pauli)
                collected[key] = collected.get(key, 0j) + term_c

        terms = []
        for pstr, c in collected.items():
            if abs(c) > 1e-12:
                if abs(c.imag) < 1e-12:
                    terms.append(PauliString(pstr, float(c.real)))
                else:
                    terms.append(PauliString(pstr, c))

        if not terms:
            return Hamiltonian([PauliString("I" * n_qubits, 0.0)])

        return Hamiltonian(terms)


def get_molecular_hamiltonian(molecule: str, basis: str = "sto-3g") -> Hamiltonian:
    """High-level API to retrieve pre-computed molecular Hamiltonians."""
    _CACHED = {
        "h2_sto-3g": [
            PauliString("II", -1.052373),
            PauliString("ZI", 0.397937),
            PauliString("IZ", 0.397937),
            PauliString("ZZ", 0.011280),
            PauliString("XX", 0.180931),
        ],
    }
    key = f"{molecule.lower()}_{basis.lower()}"
    if key in _CACHED:
        return Hamiltonian(_CACHED[key])

    raise NotImplementedError(f"Molecule {molecule} not supported yet.")
#ggjj