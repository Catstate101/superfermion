"""
PySCF Bridge — Molecular electronic structure via PySCF integration.

Computes one- and two-electron integrals, active-space transformations,
and builds molecular Hamiltonians for any geometry+basis combination.

If PySCF is not installed, falls back to pre-computed molecule library values.

Usage:
    >>> from superfermion.chemistry.pyscf_bridge import molecule_from_geometry
    >>>
    >>> mol = molecule_from_geometry("H 0 0 0; F 0 0 0.92", basis="sto-3g")
    >>> H = mol.to_hamiltonian(active_space=(2, 2))  # 2 electrons in 2 orbitals
    >>> print(f"{mol.n_qubits}-qubit Hamiltonian ready")

Environment notes
-----------------
* PySCF ships NO Windows wheels on PyPI (sdist-only; a source build needs
  CMake + a C/Fortran toolchain). Run chemistry on Linux/macOS, or inside
  WSL on Windows (``sudo apt install -y build-essential cmake`` then
  ``pip install pyscf``), rather than expecting ``pip install pyscf`` to
  produce a wheel on native Windows.
* When Qiskit Nature / OpenFermion / PennyLane are installed alongside for
  cross-checking, keep the stack coherent. The failures seen when mixing
  pins are ``AttributeError: module 'attrs' has no attribute 'frozen'``
  (OpenFermion/Cirq vs an old or shadowed ``attrs``) and
  ``AttributeError: module 'stevedore' has no attribute 'ExtensionManager'``
  (Qiskit vs a partial ``stevedore``). Fix by creating a clean virtualenv
  and installing a matching set together, e.g.::

      pip install --upgrade pip attrs stevedore
      pip install "pyscf" "qiskit" "qiskit-nature"   # let pip resolve

  Do not hand-pin qiskit below the range qiskit-nature requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Try importing PySCF — graceful degradation if not installed
try:
    import pyscf
    import pyscf.scf
    import pyscf.ao2mo
    import pyscf.symm
    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

#: Accepted values of ``molecule_from_geometry(scf=...)``.
SCF_METHODS = ("auto", "rhf", "rohf", "uhf")


def _mo_eri(
    mol_obj: Any,
    mf: Any,
    coeffs: Union[np.ndarray, Tuple[np.ndarray, ...]],
    *,
    density_fit: bool,
    n: int,
) -> np.ndarray:
    """Full chemist ``(pq|rs)`` MO tensor as ``(n, n, n, n)``.

    ``coeffs`` is one ``(nao, n)`` matrix for a restricted reference, or the
    4-tuple ``(C_alpha, C_beta, C_alpha, C_beta)`` for an unrestricted one.
    The density-fitted ``ao2mo`` indexes its four index groups positionally and
    raises ``IndexError`` on a 1- or 2-tuple, so the four-way form is the only
    spelling that works on both paths; and ``aosym='s1'`` is what keeps the
    exact path from returning a symmetry-packed array that would still reshape
    into a plausible-looking but wrong tensor.
    """
    if isinstance(coeffs, tuple):
        if density_fit:
            flat = mf.with_df.ao2mo(coeffs, compact=False)
        else:
            flat = pyscf.ao2mo.general(mol_obj, coeffs, aosym="s1")
    elif density_fit:
        flat = mf.with_df.ao2mo(coeffs, compact=False)
    else:
        flat = pyscf.ao2mo.full(mol_obj, coeffs, compact=False)
    return np.asarray(flat, dtype=float).reshape(n, n, n, n)


def _spin_orbital_integrals(
    h1: np.ndarray, h2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand spin-free (restricted) MO integrals to spin orbitals.

    Spin-orbital index ordering is interleaved: ``P = 2*p + s`` with
    ``s = 0`` (alpha) / ``1`` (beta), so spatial orbital ``p`` becomes the
    adjacent pair of modes ``(2p, 2p+1)``.

    The physics is the standard restricted expansion::

        h_PQ     = h_pq  delta(s_P, s_Q)
        (PQ|RS)  = (pq|rs) delta(s_P, s_Q) delta(s_R, s_S)

    ``FermionicOperator.from_coeffs`` consumes the two-body tensor in the
    slot order ``T[i,j,k,l] == (i k|j l)``, i.e. it needs
    ``(pq|rs).transpose(0, 2, 1, 3)``, not the plain chemist array. Feeding it
    the plain array silently builds a different operator. Verified against an
    independent occupation-basis matrix and PySCF FCI on H2/STO-3G: with this
    layout ``max|H_sf - H_exact| = 4.4e-16`` and the ground state equals FCI
    (-1.1373060358 Ha); without it the two differ by 7.0e-01.

    Args:
        h1: ``(nao, nao)`` one-body integrals.
        h2: ``(nao, nao, nao, nao)`` chemist-notation ERI ``(pq|rs)``.

    Returns:
        ``(h1_so, h2_so)`` in the ``(2*nao)``-dimensional spin-orbital space,
        ``h2_so`` already in the layout ``from_coeffs`` expects.
    """
    n = h1.shape[0]
    h1_so = np.zeros((2 * n, 2 * n), dtype=float)
    for s in (0, 1):
        h1_so[s::2, s::2] = h1

    h2_so = np.zeros((2 * n, 2 * n, 2 * n, 2 * n), dtype=float)
    # value at slots (2p+a, 2r+b, 2q+a, 2s+b) is (pq|rs), so the source
    # array is traversed with the middle two axes swapped.
    block = np.transpose(h2, (0, 2, 1, 3))
    for a in (0, 1):
        for b in (0, 1):
            h2_so[a::2, b::2, a::2, b::2] = block
    return h1_so, h2_so


def _fold_constant(hamiltonian, constant: float, n_qubits: int):
    """Add ``constant`` to the identity term instead of appending a second one.

    ``FermionicOperator.jordan_wigner`` already collects every contribution to
    the all-identity string and emits it once, so the nuclear repulsion (plus
    any frozen-core energy) must be *added* to that coefficient.  Appending a
    new ``PauliString`` for it was the pre-0.1.21 behaviour and left two entries
    with the same label: H2 came out as 16 terms for 15 distinct strings
    (``'IIII'`` appearing as ``[-0.812170607, +0.715104339]``) and water as 2111
    for 2110.  Energies are unchanged either way -- the sum is the same -- but a
    label-keyed consumer silently keeps only the last coefficient, and a term
    count stops being a term count.
    """
    import math

    from superfermion.observables.core import PauliString

    if not math.isfinite(constant):
        raise ValueError(f"the constant term must be finite, got {constant}")

    identity_key = "I" * n_qubits
    terms = list(hamiltonian.terms)
    is_identity = [set(t.pauli_str) == {"I"} for t in terms]

    if constant == 0.0 and any(is_identity):
        return hamiltonian

    merged = []
    placed = False
    for term, is_id in zip(terms, is_identity):
        if is_id and not placed:
            merged.append(
                PauliString(term.pauli_str, complex(term.coeffs) + constant)
            )
            placed = True
        else:
            merged.append(term)
    if not placed:
        merged.append(PauliString(identity_key, constant))

    return type(hamiltonian)(merged)


def _spin_orbital_integrals_unrestricted(
    h1_alpha: np.ndarray,
    h1_beta: np.ndarray,
    eri_alpha_alpha: np.ndarray,
    eri_alpha_alpha_beta_beta: np.ndarray,
    eri_beta_beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spin-orbital integrals from an unrestricted (UHF) reference.

    Alpha and beta spatial orbitals are different functions, so the spin-free
    ``h1``/``h2`` pair cannot describe the operator at all: there are three
    distinct ERI blocks (``aaaa``, ``aabb`` and ``bbbb``).  Modes stay
    interleaved, ``2p`` alpha and ``2p + 1`` beta, which is what
    :func:`_spin_orbital_integrals`, ``reference_occupation`` and every
    determinant helper in this package assume.

    The stored layout is the one :func:`_spin_orbital_integrals` produces:
    slot ``(2p + a, 2q + b, 2r + a, 2s + b)`` holds
    ``(space_a(p) space_a(r) | space_b(q) space_b(s))``.  With
    ``C_alpha == C_beta`` this function returns exactly what that one returns,
    and that degenerate limit is how the tests pin it down: an assembly that is
    merely *self-consistent* would still be wrong.

    Note which cross block is needed.  The one to pass is
    ``(a_p a_q | b_r b_s)`` from ``ao2mo((Ca, Ca, Cb, Cb))``, **not** the
    ``(a_p b_q | a_r b_s)`` block from ``ao2mo((Ca, Cb, Ca, Cb))``: the stored
    layout puts slots 1 and 3 on one spin and slots 2 and 4 on the other, so the
    mixed block it needs has same-spin partners in the bra *and* in the ket.
    Both mistakes produce a plausible number rather than an error -- OH/STO-3G
    came out at -87.14 Ha and then -81.02 Ha against an exact FCI of
    -74.39 Ha -- which is why the check is FCI basis invariance, not a shape or
    symmetry assertion.

    Args:
        h1_alpha: ``(n, n)`` alpha one-body integrals in the alpha MO basis.
        h1_beta: ``(n, n)`` beta one-body integrals in the beta MO basis.
        eri_alpha_alpha: ``(n, n, n, n)`` chemist ``(pq|rs)``, all-alpha.
        eri_alpha_alpha_beta_beta: ``(n, n, n, n)`` chemist ``(pq|rs)`` with the
            bra pair on alpha and the ket pair on beta, i.e.
            ``(a_p a_q | b_r b_s)`` as produced by ``ao2mo((Ca, Ca, Cb, Cb))``.
        eri_beta_beta: ``(n, n, n, n)`` chemist ``(pq|rs)``, all-beta.

    Returns:
        ``(h1_so, h2_so)`` in the ``2n``-dimensional interleaved space.
    """
    n = int(h1_alpha.shape[0])
    cross = eri_alpha_alpha_beta_beta

    h1_so = np.zeros((2 * n, 2 * n), dtype=float)
    h1_so[0::2, 0::2] = h1_alpha
    h1_so[1::2, 1::2] = h1_beta

    h2_so = np.zeros((2 * n,) * 4, dtype=float)
    h2_so[0::2, 0::2, 0::2, 0::2] = eri_alpha_alpha.transpose(0, 2, 1, 3)
    h2_so[1::2, 1::2, 1::2, 1::2] = eri_beta_beta.transpose(0, 2, 1, 3)
    # slots (a_p, b_q, a_r, b_s) hold (a_p a_r | b_q b_s) = cross[p, r, q, s]
    h2_so[0::2, 1::2, 0::2, 1::2] = cross.transpose(0, 2, 1, 3)
    # slots (b_p, a_q, b_r, a_s) hold (b_p b_r | a_q a_s) = (a_q a_s | b_p b_r)
    # = cross[q, s, p, r].  Note that both surviving index pairs are *repeated*
    # (q with q, p with p), so this is not a transpose of the abab block and
    # neither is it the same permutation applied twice: a permutation that maps
    # output slot 0 onto input slot 0 cannot reach cross[q, s, p, r] at all.
    # ``transpose(2, 0, 3, 1)`` is the one that does, i.e.
    # ``cross.transpose(2, 3, 0, 1).transpose(0, 2, 1, 3)`` written in one step.
    h2_so[1::2, 0::2, 1::2, 0::2] = cross.transpose(2, 0, 3, 1)

    return h1_so, h2_so


def _fold_inactive_spin_orbitals(
    h1_so: np.ndarray,
    h2_so: np.ndarray,
    active_modes: Sequence[int],
    inactive_modes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fold a *doubly occupied* core into a spin-orbital active block.

    The spin-orbital twin of :meth:`MolecularData._slice_active_space`'s
    closed-shell branch, for the unrestricted path where the spin-free form
    does not apply::

        h1_eff[A, B] = h1[A, B] + sum_k [ (A B | k k) - (A k | k B) ]
        e_core       = sum_k h1[k, k]
                       + 0.5 sum_kl [ (k k | l l) - (k l | l k) ]

    ``k`` runs over the inactive modes, each taken as fully occupied.  In the
    stored layout ``h2_so[P, Q, R, S] == (P R | Q S)`` this is
    ``h2_so[A, k, B, k]`` for the Coulomb term and ``h2_so[A, k, k, B]`` for the
    exchange term.

    Returns:
        ``(h1_eff, h2_eff, e_core)``.
    """
    active_modes = [int(i) for i in active_modes]
    inactive_modes = [int(i) for i in inactive_modes]

    h1_eff = np.array(
        h1_so[np.ix_(active_modes, active_modes)], dtype=float, copy=True
    )
    h2_eff = np.array(
        h2_so[np.ix_(active_modes, active_modes, active_modes, active_modes)],
        dtype=float,
        copy=True,
    )
    if not inactive_modes:
        return h1_eff, h2_eff, 0.0

    core = np.asarray(inactive_modes, dtype=int)
    coulomb = h2_so[np.ix_(active_modes, core, active_modes, core)]
    exchange = h2_so[np.ix_(active_modes, core, core, active_modes)]
    h1_eff += coulomb.sum(axis=(1, 3)) - 0.5 * exchange.sum(axis=(1, 3))

    e_core = float(np.trace(h1_so[np.ix_(core, core)]))
    block = h2_so[np.ix_(core, core, core, core)]
    j = block.sum(axis=(1, 3))
    # (k l | l k) is h2_so[k, l, l, k] in the stored layout.
    k_ex = np.array(
        [[block[a, b, b, a] for b in range(core.size)] for a in range(core.size)]
    )
    e_core += 0.5 * float(np.trace(j) - 0.5 * np.trace(k_ex))
    return h1_eff, h2_eff, e_core


@dataclass
class MolecularData:
    """Container for molecular integral data."""

    name: str
    geometry: str  # XYZ format: "H 0 0 0; H 0 0 0.74"
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0  # 2S where S = spin multiplicity − 1

    # One-body integrals (MO basis)
    h1: Optional[np.ndarray] = None  # h_{ij} = ⟨i|h|j⟩
    # Two-body integrals (chemist notation)
    h2: Optional[np.ndarray] = None  # h_{ijkl} = (ij|kl)
    # Nuclear repulsion energy
    nuclear_repulsion: float = 0.0
    # HF energy
    hf_energy: float = 0.0

    n_orbitals: int = 0
    n_electrons: int = 0

    # Active space parameters
    active_orbitals: Optional[Tuple[int, int]] = None  # (n_elec, n_orbs)
    frozen_core: int = 0
    # Explicit active window (spatial orbital indices) and dropped virtuals.
    # ``active_indices`` overrides the contiguous window that ``active_orbitals``
    # implies; ``removed_orbitals`` lists unoccupied orbitals to discard.
    active_indices: Optional[Tuple[int, ...]] = None
    removed_orbitals: Optional[Tuple[int, ...]] = None

    _mo_coeffs: Optional[np.ndarray] = field(default=None, repr=False)
    # AO overlap and per-AO labels ("0 O 2px"), kept so that projection-based
    # selection (AVAS) can run without rebuilding the molecule in PySCF.
    _ao_overlap: Optional[np.ndarray] = field(default=None, repr=False)
    ao_labels: Optional[Tuple[str, ...]] = None
    # Set by chemistry.avas: projector eigenvalues, selected counts and the MO
    # transformation that produced this object.
    avas_info: Optional[Dict[str, Any]] = None

    # ---- mean field provenance (set by _compute_pyscf) ------------------
    # The SCF method that actually ran: "RHF", "ROHF" or "UHF".  Recorded
    # rather than inferred, because PySCF's `scf.RHF` silently returns an
    # `ROHF` object when `mol.spin != 0` and an inferred name would be a guess.
    scf_method: Optional[str] = None
    # <S^2> of the converged reference.  Exact 0.75 for a doublet and 0 for a
    # closed shell; anything larger flags spin contamination, which is the
    # quantity a UHF run is usually asked about.
    spin_squared: Optional[float] = None
    mo_occupations: Optional[np.ndarray] = field(default=None, repr=False)
    mo_energies: Optional[np.ndarray] = field(default=None, repr=False)
    # Fock and exchange matrices in the MO basis, the two inputs APC ranking
    # needs (chemistry/apc.py).  For UHF these are the alpha-spin pair.
    fock_mo: Optional[np.ndarray] = field(default=None, repr=False)
    exchange_mo: Optional[np.ndarray] = field(default=None, repr=False)
    # Point-group information, filled when symmetry=True was requested.
    # ``irrep_ids``/``irrep_labels`` are the group's irreps that carry a basis
    # function (A2 is absent from water/STO-3G in C2v); ``mo_irreps`` is the
    # symmetry label of each MO, which is what a sector decomposition needs.
    point_group: Optional[str] = None
    irrep_ids: Optional[Tuple[int, ...]] = None
    irrep_labels: Optional[Tuple[str, ...]] = None
    mo_irreps: Optional[Tuple[str, ...]] = None
    # Whether the two-electron integrals came from density fitting.
    density_fit: bool = False
    # General (unrestricted) spin-orbital integrals, in the interleaved layout
    # P = 2p + s that everything else in this package uses.  Present only for a
    # UHF reference, where alpha and beta spatial orbitals differ and the
    # spin-free `h1`/`h2` pair cannot describe the operator.
    h1_spin_orbitals: Optional[np.ndarray] = field(default=None, repr=False)
    h2_spin_orbitals: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_qubits(self) -> int:
        """Number of qubits the mapped operator uses (2 per active orbital)."""
        return 2 * self.n_active_orbitals_effective

    @property
    def uses_spin_orbitals(self) -> bool:
        """True when the operator must be built from general spin orbitals."""
        return self.h1_spin_orbitals is not None

    @property
    def n_active_orbitals_effective(self) -> int:
        """Active spatial orbital count after the explicit selections.

        ``active_indices`` wins over everything: it is the exact set that will
        be sliced.  Otherwise ``removed_orbitals`` is subtracted from the
        ``active_orbitals`` window -- or from the whole space when no window is
        set, which is the case the plain ``n_qubits`` used to get wrong.
        """
        if self.active_indices is not None:
            return len(self.active_indices)
        if self.active_orbitals is not None:
            base = int(self.active_orbitals[1])
        else:
            base = int(self.n_orbitals) - int(self.frozen_core or 0)
        if self.removed_orbitals:
            base -= len(set(self.removed_orbitals))
        return max(0, base)

    @property
    def n_active_electrons(self) -> int:
        if self.active_orbitals is not None:
            return self.active_orbitals[0]
        return self.n_electrons

    @property
    def n_active_orbitals(self) -> int:
        if self.active_orbitals is not None:
            return self.active_orbitals[1]
        return self.n_orbitals

    def to_hamiltonian(
        self,
        active_space: Optional[Tuple[int, int]] = None,
        mapping: str = "jordan_wigner",
        *,
        active_indices: Optional[Sequence[int]] = None,
        remove_orbitals: Optional[Sequence[int]] = None,
        frozen_core: Optional[int] = None,
        inactive_density: Optional[np.ndarray] = None,
        core_occupations: Optional[Sequence[float]] = None,
        term_tolerance: float = 1e-12,
    ) -> Any:
        """Build a superfermion Hamiltonian from the molecular integrals.

        Args:
            active_space: Optional (n_electrons, n_orbitals) active space.
            mapping: Fermion-to-qubit mapping (``"jordan_wigner"`` or
                     ``"bravyi_kitaev"``).
            active_indices: Explicit spatial orbital indices to keep active.
                Overrides the contiguous window implied by ``active_space``.
                Enables non-contiguous spaces (π-only, d-only) that the window
                cannot express. Mutually exclusive with ``remove_orbitals``.
            remove_orbitals: Unoccupied orbital indices to drop. Everything
                outside ``frozen_core`` and this list stays active. No energy
                shift is applied, so the indices must really be unoccupied.
            frozen_core: Number of lowest orbitals treated as a closed-shell
                core. Defaults to ``self.frozen_core`` (and to the count
                implied by ``active_space`` when neither is set).
            inactive_density: Optional ``(n_inactive, n_inactive)`` spin-summed
                1-RDM for the frozen core. ``None`` (default) means the core is
                a closed-shell determinant, ``2 * I``. Use this for embedded or
                correlated inactive regions; the two-body part is then the
                Hartree-Fock contraction of that density with itself.
            core_occupations: Optional per-inactive-orbital occupations, i.e.
                the diagonal of the core 1-RDM. This is the user-facing switch
                for an **open-shell** core: ``[2, 1, 2]`` freezes a core with a
                singly occupied orbital, which neither the pair-counting
                ``active_space`` path nor the closed-shell ``2 * I`` default can
                express. Mutually exclusive with ``inactive_density``.
            term_tolerance: Relative cut below which a mapped Pauli term is
                dropped. The cut is ``term_tolerance * max|coefficient|``, so it
                scales with the operator instead of sitting at an absolute
                1e-12 -- which is what made the term count wobble by +/-1
                between identical builds. Pass ``0.0`` to keep every term.

        Returns:
            sf.Hamiltonian.
        """
        if self.h1 is None and not self.uses_spin_orbitals:
            return self._hamiltonian_from_library()

        if self.uses_spin_orbitals:
            # An unrestricted reference has different alpha and beta spatial
            # orbitals, so the spin-free ``h1``/``h2`` pair cannot describe the
            # operator at all; slicing those would silently average the spins.
            return self._to_hamiltonian_spin_orbitals(
                active_space,
                mapping,
                active_indices=active_indices,
                remove_orbitals=remove_orbitals,
                frozen_core=frozen_core,
                core_occupations=core_occupations,
                term_tolerance=term_tolerance,
            )

        # Apply active space transformation
        h1_as, h2_as, n_elec, n_orbs, e_fc = self._apply_active_space(
            active_space,
            active_indices=active_indices,
            remove_orbitals=remove_orbitals,
            frozen_core=frozen_core,
            inactive_density=inactive_density,
            core_occupations=core_occupations,
        )

        # Guard: mapping the FULL orbital space builds a Pauli sum whose size
        # explodes with n_orbitals (benzene/6-31G full = 132 qubits, ~10^6
        # terms). Warn and point at active_space, the same scoping Qiskit
        # Nature / PennyLane rely on. Does not change the returned operator.
        if 2 * n_orbs > 24:
            import warnings
            warnings.warn(
                f"Mapping the full {n_orbs}-orbital space needs {2 * n_orbs} "
                f"qubits and produces a very large Pauli sum. Pass "
                f"active_space=(n_elec, n_orbs) (e.g. (6, 6) for benzene) to "
                f"keep the fermion-to-qubit transform tractable.",
                UserWarning,
                stacklevel=2,
            )

        # Build fermionic operator.  The stored integrals are spin-free
        # (restricted), so they must be expanded to spin orbitals first --
        # the operator spans 2*n_orbs modes.
        from superfermion.chemistry.hamiltonians import FermionicOperator

        h1_so, h2_so = _spin_orbital_integrals(h1_as, h2_as)
        ferm_op = FermionicOperator.from_coeffs(h1_so, h2_so)

        # Map to qubits
        n_qubits = 2 * n_orbs
        if mapping == "jordan_wigner":
            H = ferm_op.jordan_wigner(
                n_qubits, tol=term_tolerance, relative=True
            )
        elif mapping == "bravyi_kitaev":
            H = ferm_op.bravyi_kitaev(
                n_qubits, tol=term_tolerance, relative=True
            )
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        return _fold_constant(H, self.nuclear_repulsion + e_fc, n_qubits)

    def _to_hamiltonian_spin_orbitals(
        self,
        active_space: Optional[Tuple[int, int]],
        mapping: str,
        *,
        active_indices: Optional[Sequence[int]],
        remove_orbitals: Optional[Sequence[int]],
        frozen_core: Optional[int],
        core_occupations: Optional[Sequence[float]],
        term_tolerance: float,
    ) -> Any:
        """``to_hamiltonian`` for an unrestricted reference.

        A UHF reference is not a rotation of an RHF one: alpha and beta MOs are
        different functions, so the stored object is a pair of general
        spin-orbital tensors rather than a spin-free ``h1``/``h2``.  The core is
        therefore folded mode by mode with the same contraction the restricted
        slicer uses,

            h1_eff[A, B] = h1[A, B] + sum_k [ (A B | k k) - (A k | k B) ],

        applied to the spin orbitals that are actually occupied.  With
        ``core_occupations`` a singly occupied core orbital contributes only its
        alpha mode, which the closed-shell ``2 * I`` default cannot express.

        The validation that this layout is right rather than merely
        self-consistent is basis invariance: a rotation of the *full* spin-orbital
        basis cannot change the FCI energy, so a UHF-integral operator must
        reproduce the RHF-integral FCI to numerical precision.
        """
        fc_req = self.frozen_core if frozen_core is None else int(frozen_core)
        core, active, n_elec = self._resolve_explicit_space(
            active_space, active_indices, remove_orbitals, fc_req
        )
        n_orbs = len(active)
        self._warn_if_spin_incompatible(n_elec, n_orbs, stacklevel=3)

        if 2 * n_orbs > 24:
            import warnings
            warnings.warn(
                f"Mapping the full {n_orbs}-orbital space needs {2 * n_orbs} "
                f"qubits and produces a very large Pauli sum. Pass "
                f"active_space=(n_elec, n_orbs) to keep the transform tractable.",
                UserWarning,
                stacklevel=3,
            )

        active_modes = [m for p in active for m in (2 * p, 2 * p + 1)]
        inactive_modes: List[int] = []
        if core_occupations is None:
            inactive_modes = [m for p in core for m in (2 * p, 2 * p + 1)]
        else:
            occ = np.asarray(core_occupations, dtype=float).ravel()
            if occ.shape[0] != len(core):
                raise ValueError(
                    f"core_occupations has {occ.shape[0]} entries but the core "
                    f"holds {len(core)} orbitals"
                )
            if np.any(occ < -1e-10) or np.any(occ > 2.0 + 1e-10):
                raise ValueError(
                    f"core occupations must lie in [0, 2], got {occ.tolist()}"
                )
            for p, value in zip(core, occ):
                if value > 1.5:
                    inactive_modes.extend([2 * p, 2 * p + 1])
                elif value > 0.5:
                    inactive_modes.append(2 * p)

        h1_so, h2_so, e_core = _fold_inactive_spin_orbitals(
            self.h1_spin_orbitals,
            self.h2_spin_orbitals,
            active_modes,
            inactive_modes,
        )

        from superfermion.chemistry.hamiltonians import FermionicOperator

        ferm_op = FermionicOperator.from_coeffs(h1_so, h2_so)
        n_qubits = 2 * n_orbs
        if mapping == "jordan_wigner":
            H = ferm_op.jordan_wigner(n_qubits, tol=term_tolerance, relative=True)
        elif mapping == "bravyi_kitaev":
            H = ferm_op.bravyi_kitaev(n_qubits, tol=term_tolerance, relative=True)
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        return _fold_constant(H, self.nuclear_repulsion + e_core, n_qubits)

    def _apply_active_space(
        self,
        active_space: Optional[Tuple[int, int]],
        *,
        active_indices: Optional[Sequence[int]] = None,
        remove_orbitals: Optional[Sequence[int]] = None,
        frozen_core: Optional[int] = None,
        inactive_density: Optional[np.ndarray] = None,
        core_occupations: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
        """Transform integrals to active space.

        The space can be given three ways, in precedence order: an explicit
        ``active_indices`` list, a ``remove_orbitals`` list (everything outside
        the core and that list stays active), or the contiguous window that
        ``active_space`` implies.  ``frozen_core`` fixes the closed-shell core
        independently of the electron count.

        ``inactive_density`` overrides the closed-shell ``2 * I`` core with a
        general spin-summed inactive 1-RDM; ``core_occupations`` is the same
        thing given only by its diagonal.

        Returns:
            ``(h1_as, h2_as, n_elec, n_orbs, e_fc)`` where ``e_fc`` is the
            frozen-core energy that must be added to the identity term.
        """
        if inactive_density is not None and core_occupations is not None:
            raise ValueError(
                "pass inactive_density (a full core 1-RDM) or "
                "core_occupations (its diagonal), not both"
            )

        fc_req = self.frozen_core if frozen_core is None else int(frozen_core)

        # ---- explicit selection: index list or dropped virtuals --------
        if active_indices is not None or remove_orbitals is not None:
            core, active, n_elec = self._resolve_explicit_space(
                active_space, active_indices, remove_orbitals, fc_req
            )
            return self._slice_active_space(
                core, active, n_elec, inactive_density, core_occupations
            )

        if active_space is None and self.active_orbitals is None:
            if fc_req:
                core, active, n_elec = self._resolve_explicit_space(
                    None, None, None, fc_req
                )
                return self._slice_active_space(
                    core, active, n_elec, inactive_density, core_occupations
                )
            if core_occupations is not None:
                raise ValueError(
                    "core_occupations was given but no orbital is inactive: "
                    "set frozen_core, active_indices or active_space first"
                )
            n_elec = self.n_active_electrons
            n_orbs = self.n_active_orbitals
            return self.h1, self.h2, n_elec, n_orbs, 0.0

        n_elec, n_orbs = active_space or self.active_orbitals or (
            self.n_electrons,
            self.n_orbitals,
        )
        n_total = self.n_orbitals

        # Standard CAS layout: the lowest ``ncore = (n_total_elec - n_elec)/2``
        # orbitals are frozen and doubly occupied, and the active window starts
        # immediately above them.  Slicing the active window from orbital 0 and
        # declaring the *highest* orbitals core (the pre-fix behaviour) is not a
        # variational model: water/STO-3G (4,4) came out at -78.30 Ha, i.e.
        # 3.3 Ha BELOW the exact FCI energy of the same basis.
        import warnings

        ncore = (self.n_electrons - n_elec) // 2
        if ncore < 0:
            warnings.warn(
                f"active_space asks for {n_elec} electrons but the molecule "
                f"has {self.n_electrons}; freezing no core orbitals.",
                UserWarning,
                stacklevel=3,
            )
            ncore = 0
        if (self.n_electrons - n_elec) % 2:
            warnings.warn(
                f"active_space leaves an odd number "
                f"({self.n_electrons - n_elec}) of core electrons; rounding "
                f"the core down to {ncore} doubly occupied orbitals.",
                UserWarning,
                stacklevel=3,
            )

        if n_orbs > n_total - ncore:
            warnings.warn(
                f"active_space=({n_elec}, {n_orbs}) needs {ncore} core + "
                f"{n_orbs} active = {ncore + n_orbs} orbitals but "
                f"{self.name} only has {n_total}; clamping the active space "
                f"to {n_total - ncore} orbitals.",
                UserWarning,
                stacklevel=3,
            )
        n_orbs = max(1, min(n_orbs, n_total - ncore))
        lo, hi = ncore, ncore + n_orbs

        if fc_req and fc_req != ncore:
            warnings.warn(
                f"frozen_core={fc_req} disagrees with the {ncore}-orbital core "
                f"implied by ({n_elec} electrons, {n_orbs} orbitals); the "
                f"active-space window wins. Select the orbitals with "
                f"active_indices= to control the core explicitly.",
                UserWarning,
                stacklevel=3,
            )

        return self._slice_active_space(
            list(range(ncore)),
            list(range(lo, hi)),
            n_elec,
            inactive_density,
            core_occupations,
        )

    def _resolve_explicit_space(
        self,
        active_space: Optional[Tuple[int, int]],
        active_indices: Optional[Sequence[int]],
        remove_orbitals: Optional[Sequence[int]],
        frozen_core: int,
    ) -> Tuple[List[int], List[int], int]:
        """Validate an explicit core / active / dropped split.

        Returns:
            ``(core_indices, active_indices, n_active_electrons)``.
        """
        import warnings

        n_total = self.n_orbitals

        if active_indices is not None and remove_orbitals is not None:
            raise ValueError(
                "active_indices and remove_orbitals are mutually exclusive: "
                "pass the orbitals to keep, or the unoccupied ones to drop."
            )
        if frozen_core < 0:
            raise ValueError(f"frozen_core must be >= 0, got {frozen_core}")
        if frozen_core > n_total:
            raise ValueError(
                f"frozen_core={frozen_core} exceeds the {n_total} orbitals of "
                f"{self.name}"
            )

        if active_space is not None:
            n_elec = int(active_space[0])
        elif frozen_core:
            n_elec = self.n_electrons - 2 * frozen_core
        else:
            n_elec = self.n_electrons

        ncore = frozen_core if frozen_core else max(
            0, (self.n_electrons - n_elec) // 2
        )
        if ncore > n_total:
            raise ValueError(
                f"a {ncore}-orbital core does not fit in {n_total} orbitals"
            )
        core = list(range(ncore))

        if active_indices is not None:
            active = [int(i) for i in active_indices]
            if not active:
                raise ValueError("active_indices is empty")
            if len(set(active)) != len(active):
                raise ValueError(f"active_indices has duplicates: {active}")
            bad = [i for i in active if i < 0 or i >= n_total]
            if bad:
                raise ValueError(
                    f"active_indices outside 0..{n_total - 1}: {bad}"
                )
            clash = sorted(set(active) & set(core))
            if clash:
                raise ValueError(
                    f"active_indices {clash} overlap the {ncore}-orbital "
                    f"frozen core; lower frozen_core or drop them."
                )
        else:
            dropped = [int(i) for i in (remove_orbitals or ())]
            bad = [i for i in dropped if i < 0 or i >= n_total]
            if bad:
                raise ValueError(
                    f"remove_orbitals outside 0..{n_total - 1}: {bad}"
                )
            clash = sorted(set(dropped) & set(core))
            if clash:
                raise ValueError(
                    f"remove_orbitals {clash} are part of the "
                    f"{ncore}-orbital frozen core; they are occupied."
                )
            dropped_set = set(dropped)
            active = [
                i
                for i in range(n_total)
                if i not in core and i not in dropped_set
            ]
            if not active:
                raise ValueError(
                    "remove_orbitals leaves no active orbital"
                )

        if n_elec < 0:
            raise ValueError(
                f"active electron count is negative ({n_elec}); check "
                f"frozen_core and active_space."
            )
        if n_elec > 2 * len(active):
            raise ValueError(
                f"{len(active)} active orbitals hold at most "
                f"{2 * len(active)} electrons, got {n_elec}"
            )
        if (self.n_electrons - n_elec) % 2:
            warnings.warn(
                f"the selection leaves an odd number "
                f"({self.n_electrons - n_elec}) of core electrons; the core "
                f"is still treated as {ncore} doubly occupied orbitals.",
                UserWarning,
                stacklevel=4,
            )

        return core, active, n_elec

    def _spin_split(self, n_elec: int) -> Tuple[int, int]:
        """``(n_alpha, n_beta)`` implied by ``self.spin`` for ``n_elec`` electrons.

        ``self.spin`` is 2S, so the high-spin (ROHF) reference has
        ``n_alpha - n_beta == 2S``.  A split only exists when ``n_elec`` and
        ``2S`` share parity and the molecule has at least ``2S`` electrons.
        """
        delta = int(self.spin)
        if delta < 0:
            raise ValueError(f"spin must be >= 0, got {delta}")
        if delta > n_elec:
            raise ValueError(
                f"spin={delta} (2S) needs at least {delta} electrons but the "
                f"active space holds {n_elec}"
            )
        if (n_elec - delta) % 2:
            raise ValueError(
                f"{n_elec} active electrons and spin={delta} (2S) have "
                f"different parities, so no (n_alpha, n_beta) split exists"
            )
        return (n_elec + delta) // 2, (n_elec - delta) // 2

    def _warn_if_spin_incompatible(
        self, n_elec: int, n_orbs: int, stacklevel: int = 3
    ) -> None:
        """Warn only when the active space cannot host the molecular spin.

        The frozen core is closed-shell by construction, so an open-shell
        molecule whose core electron count is even needs no warning at all.
        The earlier unconditional ``spin != 0`` warning fired on every open
        shell, including cases where the closed-shell core is exact, so it was
        noise rather than a diagnosis.
        """
        import warnings

        try:
            n_alpha, n_beta = self._spin_split(n_elec)
        except ValueError as exc:
            warnings.warn(
                f"{exc}; the mapped operator cannot represent spin="
                f"{self.spin}.",
                UserWarning,
                stacklevel=stacklevel,
            )
            return
        if n_alpha > n_orbs:
            warnings.warn(
                f"spin={self.spin} needs {n_alpha} active alpha orbitals but "
                f"the active space holds {n_orbs}; the operator cannot "
                f"represent this spin.",
                UserWarning,
                stacklevel=stacklevel,
            )

    def reference_occupation(
        self,
        active_space: Optional[Tuple[int, int]] = None,
        *,
        active_indices: Optional[Sequence[int]] = None,
        remove_orbitals: Optional[Sequence[int]] = None,
        frozen_core: Optional[int] = None,
        active_electrons: Optional[Tuple[int, int]] = None,
    ) -> List[int]:
        """Hartree-Fock occupation of the active space, one entry per spin orbital.

        ``to_hamiltonian`` maps ``2 * n_orbs`` modes with the interleaved
        ordering ``P = 2*p + s`` (``s = 0`` alpha, ``1`` beta), so element
        ``P`` of the returned list is the occupation of mode ``P`` -- the same
        position as character ``P`` of the Pauli strings it produces.
        ``sum(result)`` therefore equals the active electron count.

        The default split is the high-spin (ROHF) one implied by ``self.spin``.
        Pass ``active_electrons=(n_alpha, n_beta)`` to pin a different Ms.

        The occupation puts the first ``n_alpha`` alpha modes and the first
        ``n_beta`` beta modes of the active window on, which reproduces the
        ROHF determinant whenever the active window contains every occupied
        orbital.

        Raises:
            ValueError: when the split is inconsistent with the electron
                count, the molecular spin, or the size of the active space.
        """
        _, _, n_elec, n_orbs, _ = self._apply_active_space(
            active_space,
            active_indices=active_indices,
            remove_orbitals=remove_orbitals,
            frozen_core=frozen_core,
        )

        if active_electrons is None:
            n_alpha, n_beta = self._spin_split(n_elec)
        else:
            try:
                n_alpha = int(active_electrons[0])
                n_beta = int(active_electrons[1])
            except (TypeError, IndexError, ValueError):
                raise ValueError(
                    f"active_electrons must be an (n_alpha, n_beta) pair, got "
                    f"{active_electrons!r}"
                ) from None
            if n_alpha < 0 or n_beta < 0:
                raise ValueError(
                    f"active_electrons must be non-negative, got "
                    f"({n_alpha}, {n_beta})"
                )
            if n_alpha < n_beta:
                raise ValueError(
                    f"active_electrons=({n_alpha}, {n_beta}): n_alpha must be "
                    f">= n_beta, because the mode order fixes "
                    f"Ms = (n_alpha - n_beta)/2 >= 0"
                )
            if n_alpha + n_beta != n_elec:
                raise ValueError(
                    f"active_electrons=({n_alpha}, {n_beta}) sums to "
                    f"{n_alpha + n_beta} but the active space holds {n_elec} "
                    f"electrons"
                )
            if n_alpha - n_beta > self.spin:
                raise ValueError(
                    f"active_electrons=({n_alpha}, {n_beta}) implies "
                    f"Ms={(n_alpha - n_beta) / 2} > S={self.spin / 2}"
                )

        if n_alpha > n_orbs or n_beta > n_orbs:
            raise ValueError(
                f"active_electrons=({n_alpha}, {n_beta}) needs at least "
                f"{max(n_alpha, n_beta)} active orbitals but the active space "
                f"holds {n_orbs}"
            )

        occ = [0] * (2 * n_orbs)
        for p in range(n_alpha):
            occ[2 * p] = 1
        for p in range(n_beta):
            occ[2 * p + 1] = 1
        return occ

    def reference_inactive_fock(
        self,
        inactive: Sequence[int],
        active: Sequence[int],
        density: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """General inactive Fock matrix and energy for an arbitrary density.

        The closed-shell path in :meth:`_slice_active_space` assumes every
        inactive orbital is doubly occupied, i.e. an inactive density
        ``rho = 2 * I``.  That covers a frozen core exactly but cannot express
        an embedded or correlated inactive region.  The general form is::

            F_pq  = sum_kl rho_kl [ (pq|kl) - 0.5 (pk|ql) ]
            E_in  = sum_pq rho_pq h_pq
                    + 0.5 sum_pqrs rho_pq rho_rs [ (pq|rs) - 0.5 (pr|qs) ]

        with ``p, q`` on the active orbitals and ``k, l, r, s`` on the
        inactive ones.  An inactive density ``rho = 2 * I`` reproduces the
        closed-shell expressions exactly, which is what the tests assert.

        The two-body part is the Hartree-Fock (Wick) contraction of the
        inactive density against itself.  For a determinant that is exact; for
        a correlated ``rho`` it is the standard mean-field approximation used
        by embedding schemes, not a correlated inactive energy.

        Args:
            inactive: Inactive spatial orbital indices.
            active: Active spatial orbital indices.
            density: ``(len(inactive), len(inactive))`` spin-summed inactive
                1-RDM.  ``None`` means the closed-shell ``2 * I``.

        Returns:
            ``(fock, e_inactive)`` with ``fock`` of shape
            ``(len(active), len(active))``.
        """
        inactive = [int(i) for i in inactive]
        active = [int(i) for i in active]
        n_act = len(active)
        n_inact = len(inactive)

        fock = np.zeros((n_act, n_act), dtype=float)
        if n_inact == 0:
            return fock, 0.0

        if density is None:
            rho = 2.0 * np.eye(n_inact)
        else:
            rho = np.asarray(density, dtype=float)
            if rho.shape != (n_inact, n_inact):
                raise ValueError(
                    f"inactive density must be ({n_inact}, {n_inact}) for the "
                    f"{n_inact} inactive orbitals, got {rho.shape}"
                )
            if not np.allclose(rho, rho.T, atol=1e-10):
                raise ValueError("inactive density must be symmetric")
            if np.any(np.abs(np.diagonal(rho)) > 2.0 + 1e-10):
                raise ValueError(
                    "inactive density occupations exceed 2; a spin-summed "
                    "density has diagonal entries in [0, 2]"
                )

        if self.h2 is not None:
            # J_pq = sum_kl rho_kl (pq|kl),  K_pq = sum_kl rho_kl (pk|ql)
            sub = self.h2[np.ix_(active, active, inactive, inactive)]
            fock += np.einsum("pqkl,kl->pq", sub, rho, optimize=True)
            ksub = self.h2[np.ix_(active, inactive, active, inactive)]
            # (pk|ql): axes (p, k, q, l) -> contract k and l with rho
            fock -= 0.5 * np.einsum("pkql,kl->pq", ksub, rho, optimize=True)

        e_inactive = 0.0
        if self.h1 is not None:
            hsub = self.h1[np.ix_(inactive, inactive)]
            e_inactive += float(np.einsum("pq,pq->", rho, hsub))

        if self.h2 is not None:
            sub2 = self.h2[np.ix_(inactive, inactive, inactive, inactive)]
            jj = np.einsum("pqrs,rs->pq", sub2, rho, optimize=True)
            kk = np.einsum("prqs,rs->pq", sub2, rho, optimize=True)
            e_inactive += 0.5 * float(
                np.einsum("pq,pq->", rho, jj - 0.5 * kk, optimize=True)
            )

        return fock, e_inactive

    def _slice_active_space(
        self,
        core: Sequence[int],
        active: Sequence[int],
        n_elec: int,
        inactive_density: Optional[np.ndarray] = None,
        core_occupations: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
        """Slice the integrals to ``active`` and fold in the ``core`` potential.

        The slices are copies: a view would let the frozen-core corrections
        leak back into ``self.h1`` and a second ``to_hamiltonian`` call would
        apply them twice.

        ``inactive_density`` (or its diagonal ``core_occupations``) replaces the
        closed-shell ``2 * I`` assumption with a general spin-summed inactive
        1-RDM; see :meth:`reference_inactive_fock` for the formula and its
        limits.
        """
        n_orbs = len(active)

        self._warn_if_spin_incompatible(n_elec, n_orbs, stacklevel=4)

        if core_occupations is not None:
            occ = np.asarray(core_occupations, dtype=float).ravel()
            if occ.shape[0] != len(core):
                raise ValueError(
                    f"core_occupations has {occ.shape[0]} entries but the core "
                    f"holds {len(core)} orbitals"
                )
            if np.any(occ < -1e-10) or np.any(occ > 2.0 + 1e-10):
                raise ValueError(
                    f"core occupations must lie in [0, 2], got {occ.tolist()}"
                )
            inactive_density = np.diag(occ)

        if self.h1 is not None:
            h1_as = np.array(self.h1[np.ix_(active, active)], dtype=float, copy=True)
        else:
            h1_as = np.zeros((n_orbs, n_orbs))

        if self.h2 is not None:
            h2_as = np.array(
                self.h2[np.ix_(active, active, active, active)],
                dtype=float,
                copy=True,
            )
        else:
            h2_as = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))

        # Frozen-core / inactive contribution.  The closed-shell special case
        # F_pq = h_pq + sum_i [2(pq|ii) - (pi|qi)] is kept as a scalar loop
        # because it is the hot path for every CAS calculation; anything else
        # goes through the general density-weighted form.
        core = list(core)
        if inactive_density is None:
            if self.h2 is not None and core:
                for p in range(n_orbs):
                    for q in range(n_orbs):
                        for i in core:
                            h1_as[p, q] += (
                                2.0 * self.h2[active[p], active[q], i, i]
                                - self.h2[active[p], i, active[q], i]
                            )

            e_fc = 0.0
            if self.h1 is not None and core:
                for i in core:
                    e_fc += 2.0 * self.h1[i, i]
                    if self.h2 is not None:
                        for j in core:
                            e_fc += (
                                2.0 * self.h2[i, i, j, j]
                                - self.h2[i, j, j, i]
                            )
        else:
            fock, e_fc = self.reference_inactive_fock(
                core, active, inactive_density
            )
            h1_as += fock

        return h1_as, h2_as, n_elec, n_orbs, e_fc

    def _adopt_record(self, record: Dict[str, Any]) -> None:
        """Copy a tabulated record's integrals onto this object."""
        self.h1 = np.array(record["h1"], dtype=float, copy=True)
        self.h2 = np.array(record["h2"], dtype=float, copy=True)
        self.n_electrons = int(record["n_electrons"])
        self.n_orbitals = int(record["n_orbitals"])
        self.nuclear_repulsion = float(record["nuclear_repulsion"])
        self.hf_energy = float(record["hf_energy"])

    def _hamiltonian_from_library(self) -> Any:
        """Build the operator from a tabulated record, or refuse.

        The pre-0.1.21 version matched the molecule name as a substring against
        ``hydrogen`` / ``lithium_hydride`` / ``water``, so ``"h2o"`` resolved to
        hydrogen, and it returned an *empty* operator for water because that
        record's term list was a placeholder.  The lookup is now by geometry,
        basis, charge and spin against records that carry real integrals, and
        anything else raises: once a fabricated operator reaches a solver, it is
        indistinguishable from a computed one.
        """
        from superfermion.chemistry.library import MoleculeLibrary

        record = MoleculeLibrary.get(
            self.geometry, self.basis, self.charge, self.spin
        )
        if record is None:
            raise LookupError(
                f"no integrals for {self.name!r} (geometry {self.geometry!r}, "
                f"basis {self.basis!r}, charge {self.charge}, spin {self.spin}): "
                f"this MolecularData has none, and the tabulated set is "
                f"{MoleculeLibrary.available()}. Build it with "
                f"molecule_from_geometry under PySCF instead."
            )
        self._adopt_record(record)
        return self.to_hamiltonian()

    def set_mean_field(
        self,
        *,
        fock: Optional[np.ndarray] = None,
        exchange: Optional[np.ndarray] = None,
        occupations: Optional[Sequence[float]] = None,
        mo_energies: Optional[Sequence[float]] = None,
    ) -> "MolecularData":
        """Attach an MO-basis mean field for tools that rank orbitals.

        APC ranking (:mod:`superfermion.chemistry.apc`) needs the Fock matrix,
        the exchange matrix and the occupations of *the reference it should
        judge*.  ``molecule_from_geometry`` stores the SCF pair when PySCF is
        present; this method is how a caller supplies a different one -- the
        generalised Fock of an MC-PDFT or CASSCF density, a different
        functional, or a mean field built outside PySCF.  Without it, the only
        available ranking is against the reference determinant, which cannot
        see static correlation that the determinant does not describe.

        Args:
            fock: ``(n, n)`` Fock matrix in the MO basis.  APC reads the
                diagonal only, so a matrix that is not diagonal in this basis
                (ROHF, CASSCF) is used as given.
            exchange: ``(n, n)`` exchange matrix ``K`` in the MO basis.
            occupations: ``(n,)`` occupations, ``2``/``1``/``0`` per spatial
                orbital.
            mo_energies: ``(n,)`` orbital energies, stored for reporting.

        Returns:
            ``self``, so calls chain.

        Raises:
            ValueError: on a shape mismatch with ``self.n_orbitals``, or
                occupations outside ``[0, 2]``.
        """
        n = int(self.n_orbitals) if self.n_orbitals else 0

        def _as_matrix(value, label):
            if value is None:
                return None
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"{label} must be square, got {arr.shape}")
            if n and arr.shape != (n, n):
                raise ValueError(
                    f"{label} is {arr.shape} but this molecule has {n} "
                    f"orbitals, so the mean field is in a different basis"
                )
            if not np.allclose(arr, arr.T, atol=1e-10):
                raise ValueError(f"{label} must be symmetric")
            return arr

        fock = _as_matrix(fock, "fock")
        exchange = _as_matrix(exchange, "exchange")

        if occupations is not None:
            occ = np.asarray(occupations, dtype=float).ravel()
            if n and occ.shape[0] != n:
                raise ValueError(
                    f"occupations has {occ.shape[0]} entries but this molecule "
                    f"has {n} orbitals"
                )
            if np.any(occ < -1e-10) or np.any(occ > 2.0 + 1e-10):
                raise ValueError(
                    f"occupations must lie in [0, 2], got {occ.tolist()}"
                )
            occupations = occ

        if mo_energies is not None:
            energies = np.asarray(mo_energies, dtype=float).ravel()
            if n and energies.shape[0] != n:
                raise ValueError(
                    f"mo_energies has {energies.shape[0]} entries but this "
                    f"molecule has {n} orbitals"
                )
            self.mo_energies = energies

        if fock is not None:
            self.fock_mo = fock
        if exchange is not None:
            self.exchange_mo = exchange
        if occupations is not None:
            self.mo_occupations = occupations
        return self

    def core_description(self) -> Dict[str, Any]:
        """Report which frozen-core convention :meth:`to_hamiltonian` uses.

        Three conventions are reachable and only one of them is electron
        counting, which is why the same molecule can quietly produce two
        different cores depending on which argument was set:

        ``explicit``
            ``active_indices`` or ``remove_orbitals`` selects the active set
            literally; the core is the first ``frozen_core`` orbitals.

        ``active_space``
            A stored ``active_orbitals`` window fixes the core by pair
            counting, ``(n_electrons - n_active_electrons) // 2``.  A
            ``frozen_core`` in this state is *ignored* (with a warning),
            because the window wins.

        ``frozen_core``
            Only ``frozen_core`` is set: that many lowest orbitals form a
            closed-shell core and everything above stays active.

        ``full_space``
            Nothing is frozen; every orbital is active.

        None of these can express an **open-shell** core -- pair counting and
        ``frozen_core`` are both closed-shell by construction.  A singly
        occupied core orbital needs the per-call
        ``core_occupations=[...]`` (or ``inactive_density=...``) override.

        Returns:
            ``{"mode", "n_core_orbitals", "n_active_orbitals", "note"}``.
        """
        n_total = int(self.n_orbitals)
        n_core = int(self.frozen_core or 0)

        if self.active_indices is not None or self.removed_orbitals:
            mode = "explicit"
            if self.active_indices is not None:
                note = (
                    f"active set given as {len(self.active_indices)} explicit "
                    f"orbital indices; the first {n_core} orbital(s) are the core"
                )
            else:
                note = (
                    f"{len(set(self.removed_orbitals))} unoccupied orbital(s) "
                    f"dropped; the first {n_core} orbital(s) are the core"
                )
        elif self.active_orbitals is None:
            if n_core:
                mode = "frozen_core"
                note = (
                    f"first {n_core} orbital(s) frozen, everything above is active"
                )
            else:
                mode = "full_space"
                note = "no core; every orbital is active"
        else:
            n_act_elec = int(self.active_orbitals[0])
            n_act_orbs = int(self.active_orbitals[1])
            n_core = (int(self.n_electrons) - n_act_elec) // 2
            mode = "active_space"
            note = (
                f"({n_act_elec}e, {n_act_orbs}o) window -> "
                f"({int(self.n_electrons)} - {n_act_elec}) // 2 = {n_core} "
                f"core orbital(s) by pair counting"
            )
            if self.frozen_core and int(self.frozen_core) != n_core:
                note += (
                    f"; frozen_core={self.frozen_core} is IGNORED because the "
                    "window wins -- pass active_indices= to control the core"
                )

        return {
            "mode": mode,
            "n_core_orbitals": max(0, min(n_core, n_total)),
            "n_active_orbitals": int(self.n_active_orbitals_effective),
            "note": note,
        }

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of molecular data."""
        core = self.core_description()
        return {
            "name": self.name,
            "geometry": self.geometry,
            "basis": self.basis,
            "n_electrons": self.n_electrons,
            "n_orbitals": self.n_orbitals,
            "n_qubits": self.n_qubits,
            "active_orbitals": self.active_orbitals,
            "active_indices": self.active_indices,
            "removed_orbitals": self.removed_orbitals,
            "frozen_core": self.frozen_core,
            # How the frozen core is actually picked.  `frozen_core` above is
            # the *request*; this is what `to_hamiltonian` will do with it,
            # and the two disagree whenever an `active_orbitals` window is set.
            "core_mode": core["mode"],
            "core_orbitals": core["n_core_orbitals"],
            "core_note": core["note"],
            "hf_energy": self.hf_energy,
            "nuclear_repulsion": self.nuclear_repulsion,
            "pyscf_available": _HAS_PYSCF,
            # Reference provenance.  Recorded, not inferred: PySCF's scf.RHF
            # returns an ROHF object for an open shell, so a caller that read
            # the class name back would report the wrong method.
            "scf_method": self.scf_method,
            "spin_squared": self.spin_squared,
            "density_fit": self.density_fit,
            "point_group": self.point_group,
            "mo_irreps": self.mo_irreps,
            "uses_spin_orbitals": self.uses_spin_orbitals,
        }


# ═════════════════════════════════════════════════════════════════════════
# PySCF bridge
# ═════════════════════════════════════════════════════════════════════════

def molecule_from_geometry(
    geometry: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    name: Optional[str] = None,
    *,
    scf: str = "auto",
    density_fit: bool = False,
    symmetry: bool = False,
) -> MolecularData:
    """Build a MolecularData object from an XYZ geometry string.

    With PySCF available this computes the HF solution and the MO integrals.
    Without it, integrals come from the tabulated set in
    :mod:`superfermion.chemistry.library` and a molecule that is not in that
    set raises ``LookupError`` rather than returning fabricated data.

    Args:
        geometry: XYZ-format geometry string, e.g.
                  ``"H 0 0 0; H 0 0 0.74"``.
        basis: Basis set name (``"sto-3g"``, ``"6-31g"``, ``"cc-pVDZ"``, etc.).
        charge: Total molecular charge.
        spin: 2S where S is spin multiplicity − 1 (0 = singlet, 1 = doublet).
        name: Optional molecule name. Auto-generated if None.
        scf: ``"auto"`` (RHF if ``spin == 0``, ROHF otherwise), ``"rhf"``,
            ``"rohf"`` or ``"uhf"``. ``"rhf"`` with ``spin != 0`` raises,
            because PySCF's ``scf.RHF`` would silently hand back an ROHF
            object and the record would name a method that never ran.
        density_fit: Fit the four-index ERIs with an auxiliary basis. The
            resulting operator is *different* from the exact one (H2O/STO-3G:
            8.7e-05 Ha in the SCF energy), so the flag is recorded on the
            object rather than being an invisible speed switch.
        symmetry: Detect the point group, label the MO irreps and record both
            on the object. Without it PySCF runs in C1.

    Returns:
        MolecularData with populated integrals.

    Raises:
        LookupError: when PySCF is unavailable and the molecule is not
            tabulated.
        ValueError: for an unknown ``scf``, ``scf="rhf"`` with ``spin != 0``,
            or (no PySCF) any of the three PySCF-only keywords.
    """
    if name is None:
        # Auto-name from geometry
        atoms = _parse_geometry(geometry)
        name = "".join(atoms)

    mol = MolecularData(
        name=name,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    if _HAS_PYSCF:
        _compute_pyscf(
            mol, scf=scf, density_fit=density_fit, symmetry=symmetry
        )
    else:
        _compute_fallback(
            mol, scf=scf, density_fit=density_fit, symmetry=symmetry
        )

    return mol


def molecule_from_xyz(
    xyz_path: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    *,
    scf: str = "auto",
    density_fit: bool = False,
    symmetry: bool = False,
) -> MolecularData:
    """Build MolecularData from an XYZ file.

    Args:
        xyz_path: Path to .xyz file.
        basis: Basis set name.
        charge: Total charge.
        spin: 2S.
        scf: Reference choice; see :func:`molecule_from_geometry`.
        density_fit: Use the density-fitted ERIs.
        symmetry: Detect the point group and label the MO irreps.
    """
    with open(xyz_path, "r") as f:
        xyz_content = f.read()

    # Parse standard XYZ format
    lines = xyz_content.strip().split("\n")
    name = lines[1].strip() if len(lines) > 1 else "molecule"
    geometry = "; ".join(line.strip() for line in lines[2:] if line.strip())

    return molecule_from_geometry(
        geometry,
        basis,
        charge,
        spin,
        name,
        scf=scf,
        density_fit=density_fit,
        symmetry=symmetry,
    )


def _parse_geometry(geometry: str) -> List[str]:
    """Extract atom symbols from geometry string."""
    atoms = []
    for part in geometry.split(";"):
        part = part.strip()
        if part:
            atoms.append(part.split()[0])
    return atoms


def _compute_pyscf(
    mol: MolecularData,
    *,
    scf: str = "auto",
    density_fit: bool = False,
    symmetry: bool = False,
) -> None:
    """Compute the mean field and the molecular integrals with PySCF.

    ``scf`` is the reference choice qiskit-nature exposes as ``HFMethodType``
    and this bridge used to hard-code: it always asked for ``RHF``, and PySCF's
    ``scf.RHF`` silently *returns an ROHF object* when ``mol.spin != 0``.  The
    resulting record then said nothing about which method had run or how much
    spin contamination the reference carried.  Here the choice is explicit,
    ``scf="rhf"`` with ``spin != 0`` is an error instead of a silent redirect,
    and ``scf_method`` / ``spin_squared`` are recorded.

    ``density_fit=True`` fits the four-index ERI with a three-index auxiliary
    basis.  That changes the operator and therefore every energy derived from
    it, so ``density_fit`` is recorded on the object rather than being an
    invisible speed switch.

    ``symmetry=True`` lets PySCF detect the point group and label the MOs by
    irrep.  The labels go on the object as ``mo_irreps``; the group itself as
    ``point_group``.

    Args:
        mol: Object to populate in place.
        scf: ``"auto"`` (RHF for a closed shell, ROHF otherwise), ``"rhf"``,
            ``"rohf"`` or ``"uhf"``.
        density_fit: Use the density-fitted ERI instead of the exact one.
        symmetry: Detect the point group and label the MO irreps.

    Raises:
        ValueError: for an unknown ``scf``, or ``scf="rhf"`` with ``spin != 0``.
    """
    mol_obj = pyscf.gto.M(
        atom=mol.geometry,
        basis=mol.basis,
        charge=mol.charge,
        spin=mol.spin,
        verbose=0,
    )
    if symmetry:
        mol_obj.symmetry = True
        mol_obj.build()

    mol.n_electrons = int(mol_obj.nelec[0] + mol_obj.nelec[1])
    mol.n_orbitals = int(mol_obj.nao)
    mol.nuclear_repulsion = float(mol_obj.energy_nuc())

    method = str(scf).lower()
    if method not in SCF_METHODS:
        raise ValueError(
            f"scf must be one of {', '.join(SCF_METHODS)}; got {scf!r}"
        )
    if method == "auto":
        method = "rhf" if mol.spin == 0 else "rohf"
    if method == "rhf" and mol.spin != 0:
        raise ValueError(
            "scf='rhf' needs spin=0: PySCF's scf.RHF silently returns an ROHF "
            "object for an open shell, which hides the reference that was "
            "asked for. Pass scf='rohf' (spin-pure, one set of orbitals) or "
            "scf='uhf' (separate alpha/beta orbitals, spin-contaminated)."
        )

    if method == "rhf":
        mf = pyscf.scf.RHF(mol_obj)
    elif method == "rohf":
        mf = pyscf.scf.ROHF(mol_obj)
    else:
        mf = pyscf.scf.UHF(mol_obj)
    if density_fit:
        mf = mf.density_fit()
    mf.kernel()

    mol.scf_method = method.upper() if method != "auto" else "RHF"
    mol.density_fit = bool(density_fit)
    mol.hf_energy = float(mf.e_tot)
    mol._ao_overlap = np.asarray(mol_obj.intor("int1e_ovlp"), dtype=float)
    mol.ao_labels = tuple(mol_obj.ao_labels())

    # spin_square() is a method, not an attribute, and returns a tuple for the
    # classes whose <S^2> needs an S^+S^- correction.
    _s2 = mf.spin_square()
    mol.spin_squared = float(
        _s2[0] if isinstance(_s2, (tuple, list)) else _s2
    )

    if not mf.converged:
        import warnings
        warnings.warn(
            f"the {mol.scf_method} calculation for {mol.name!r} did not "
            f"converge; the integrals below are from the last iteration, not "
            f"from a stationary point.",
            UserWarning,
            stacklevel=3,
        )

    if symmetry:
        mol.point_group = str(mol_obj.groupname)
        mol.irrep_ids = tuple(int(i) for i in mol_obj.irrep_id)
        mol.irrep_labels = tuple(str(s) for s in mol_obj.irrep_name)

    hcore = np.asarray(mf.get_hcore(), dtype=float)
    nao = mol_obj.nao

    if method == "uhf":
        C_alpha, C_beta = (np.asarray(c, dtype=float) for c in mf.mo_coeff)
        occupations = np.asarray(mf.mo_occ, dtype=float)
        # A UHF reference is not a rotation of an RHF one: the alpha and beta
        # orbitals are different functions, so the spin-free h1/h2 pair cannot
        # describe the operator.  Both are stored: h1/h2 from the *alpha* block
        # so that legacy consumers still see a consistent set, and the general
        # spin-orbital tensors -- which to_hamiltonian prefers -- for the
        # operator itself.
        mol._mo_coeffs = C_alpha
        mol.mo_occupations = occupations.sum(axis=0)
        mol.mo_energies = np.asarray(mf.mo_energy, dtype=float).mean(axis=0)

        h1_alpha = C_alpha.T @ hcore @ C_alpha
        h1_beta = C_beta.T @ hcore @ C_beta
        mol.h1 = np.asarray(h1_alpha, dtype=float)

        eri_aa = _mo_eri(
            mol_obj, mf, (C_alpha, C_alpha, C_alpha, C_alpha),
            density_fit=density_fit, n=nao,
        )
        eri_bb = _mo_eri(
            mol_obj, mf, (C_beta, C_beta, C_beta, C_beta),
            density_fit=density_fit, n=nao,
        )
        # The cross block the interleaved layout needs is the same-spin bra /
        # same-spin ket one, (a_p a_q | b_r b_s), not (a_p b_q | a_r b_s).
        eri_aabb = _mo_eri(
            mol_obj, mf, (C_alpha, C_alpha, C_beta, C_beta),
            density_fit=density_fit, n=nao,
        )
        mol.h2 = np.array(eri_aa, dtype=float, copy=True)
        mol.h1_spin_orbitals, mol.h2_spin_orbitals = (
            _spin_orbital_integrals_unrestricted(
                h1_alpha, h1_beta, eri_aa, eri_aabb, eri_bb
            )
        )

        fock = np.asarray(mf.get_fock(), dtype=float)
        exchange = np.asarray(mf.get_k(), dtype=float)
        mol.fock_mo = C_alpha.T @ fock[0] @ C_alpha
        mol.exchange_mo = C_alpha.T @ exchange[0] @ C_alpha
        if symmetry:
            mol.mo_irreps = tuple(
                str(s)
                for s in pyscf.symm.label_orb_symm(
                    mol_obj, mol_obj.irrep_name, mol_obj.symm_orb, C_alpha
                )
            )
        return

    # ---- restricted path (RHF or ROHF) --------------------------------
    C = np.asarray(mf.mo_coeff, dtype=float)
    mol._mo_coeffs = C
    mol.mo_occupations = np.asarray(mf.mo_occ, dtype=float)
    mol.mo_energies = np.asarray(mf.mo_energy, dtype=float)
    mol.h1 = np.asarray(C.T @ hcore @ C, dtype=float)
    mol.h2 = _mo_eri(
        mol_obj, mf, C, density_fit=density_fit, n=nao
    )

    # ROHF's get_k() is one matrix per spin, and the spin-free exchange is the
    # sum.  Its get_fock() is the *effective* Fock, which is diagonal in the
    # canonical ROHF basis (verified: max off-diagonal 1.2e-09) and therefore
    # the correct thing for a criterion that reads F_pp.
    exchange = np.asarray(mf.get_k(), dtype=float)
    if exchange.ndim == 3:
        exchange = exchange.sum(axis=0)
    mol.fock_mo = np.asarray(
        C.T @ np.asarray(mf.get_fock(), dtype=float) @ C, dtype=float
    )
    mol.exchange_mo = np.asarray(C.T @ exchange @ C, dtype=float)
    if symmetry:
        mol.mo_irreps = tuple(
            str(s)
            for s in pyscf.symm.label_orb_symm(
                mol_obj, mol_obj.irrep_name, mol_obj.symm_orb, C
            )
        )


def _compute_fallback(
    mol: MolecularData,
    *,
    scf: str = "auto",
    density_fit: bool = False,
    symmetry: bool = False,
) -> None:
    """Populate from a tabulated record, and refuse to invent one.

    This is the no-PySCF path.  It used to match ``"h2"`` before ``"h2o"``, so
    ``name="h2o"`` silently received hydrogen's 2 electrons and 2 orbitals, the
    unmatched case received zeros, and ``hf_energy`` was ``0.0`` everywhere
    because no record carried the key.  Matching is now geometric, and a miss is
    an error rather than a plausible-looking molecule.

    The tabulated record is an exact restricted-HF/STO-3G calculation, so the
    three PySCF-only keywords cannot be honoured: silently ignoring them would
    make ``density_fit=True`` return an operator that was never
    density-fitted.
    """
    from superfermion.chemistry.library import MoleculeLibrary

    requested = []
    if str(scf).lower() not in ("auto", "rhf"):
        requested.append(f"scf={scf!r}")
    if density_fit:
        requested.append("density_fit=True")
    if symmetry:
        requested.append("symmetry=True")
    if requested:
        raise ValueError(
            f"{', '.join(requested)} needs PySCF; the tabulated records are "
            f"exact restricted-HF/STO-3G calculations and carry no "
            f"alternative reference, auxiliary basis or symmetry labels. "
            f"Install PySCF, or drop the keyword."
        )

    record = MoleculeLibrary.get(
        mol.geometry, mol.basis, mol.charge, mol.spin
    )
    if record is None:
        raise LookupError(
            f"PySCF is not installed, so integrals must come from the tabulated "
            f"set, and {mol.name!r} ({mol.geometry!r}, basis {mol.basis!r}, "
            f"charge {mol.charge}, spin {mol.spin}) is not in it. Tabulated: "
            + "; ".join(MoleculeLibrary.available())
            + ". Install PySCF for any other molecule."
        )
    mol._adopt_record(record)
    mol.scf_method = "RHF"


# ═════════════════════════════════════════════════════════════════════════
# Active space selection
# ═════════════════════════════════════════════════════════════════════════

def active_space_from_homo_lumo(
    mol: MolecularData,
    n_active_electrons: int,
    n_active_orbitals: int,
    frozen_core: Optional[int] = None,
) -> MolecularData:
    """Select active space around HOMO-LUMO gap.

    Args:
        mol: MolecularData with MO integrals.
        n_active_electrons: Number of electrons in active space.
        n_active_orbitals: Number of spatial orbitals in active space.
        frozen_core: Number of frozen core orbitals. Auto-detected if None.

    Returns:
        mol with active space applied (mutated in place).
    """
    if frozen_core is None:
        # Auto-detect: freeze inner orbitals
        # For first/second row: 1s core = 1 orbital per heavy atom
        atoms = _parse_geometry(mol.geometry)
        heavy_atoms = [a for a in atoms if a not in ("H", "He")]
        frozen_core = len(heavy_atoms)

    mol.active_orbitals = (n_active_electrons, n_active_orbitals)
    mol.frozen_core = frozen_core

    return mol
