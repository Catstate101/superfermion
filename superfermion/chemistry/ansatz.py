"""
UCCSD-inspired variational ansatz for quantum chemistry.

For the minimal 2-qubit H2 case, uses the standard single-parameter
ansatz from O'Malley et al. (2016). For larger systems, builds a
number-conserving unitary-coupled-cluster expansion:

* every **single** excitation ``i -> a`` is an exact fermionic hopping
  rotation ``exp(-i·theta/2 (X_i Y_a - Y_i X_a))``, which moves an electron
  instead of copying it;
* every **double** excitation ``(i,j) -> (a,b)`` is a genuine rotation: the
  many-body qubit generator ``exp(-i·theta/2 · X_i Y_j Y_a Y_b)`` when its
  four spin-orbitals are the whole register (there the operator is exactly
  number conserving), and otherwise two number-conserving hopping rotations
  on the spin-matched pairs. ``double_mode`` selects either form explicitly.

Both operators are real rotations, so the ansatz can lower the energy below
the Hartree-Fock reference (CH-1..CH-5, CH-8). The pre-fix construction is
kept as :func:`hardware_efficient_ansatz` for reproducibility: it applied
``ry`` to the virtual qubit and then a ``cx`` *from* the occupied one, which
copies the electron rather than moving it, and every ``d_i`` turned out to be
a global phase.

Why ``double_mode="auto"`` picks the form per excitation
---------------------------------------------------------
No single Pauli string conserves particle number except an all-``Z``/``I``
one, so ``exp(-i·theta/2 · XYYY)`` is closed only on the four-qubit sector
that happens to hold exactly two electrons. When the quartet is the *whole*
register that sector is the entire state space of the reference and the
operator is exact. When the quartet is a subset, a hopping rotation can
leave one or three electrons on it, the same ``XYYY`` rotation then moves the
state to ``N ± 2`` or ``N ± 4``, and the reported energy is no longer the
electron energy of the molecule -- measured on (6,4): 16 determinants spanned
instead of the 9 of the reference ``Sz`` sector, reaching ``N = 0`` and
``N = 2``. ``"auto"`` therefore uses the many-body generator exactly where it
is provably closed and falls back to the hopping pair everywhere else.

Where each double form loses expressiveness
-------------------------------------------
The two forms fail in different ways, and neither failure is visible from the
circuit alone. Measured with differential evolution over the whole parameter
box ``[-pi, pi]``, against exact diagonalisation in the matching ``N`` and
``S_z`` sector:

* H2 / STO-3G, register ``(4, 2)`` -- ``E_HF = -1.1169989968``, sector FCI
  ``-1.1373060358``, 4 determinants. ``auto`` and ``pauli`` reach the FCI
  energy (-1.1373060358, 100 % of the 0.0203 Ha correlation energy);
  ``number_conserving`` stops at ``E_HF`` exactly (-1.1169989968, 0 %). Keeping
  only the two singles and dropping the double gives that same pinned minimum,
  which is what "the double adds no direction" means here: its two hopping
  rotations act in the singles' own rotation planes with one shared angle. The
  overlap with the FCI state is still 0.9876 in the pinned case -- the HF
  determinant already overlaps the true ground state that much -- so at this
  size the energy, not the overlap, is the number that shows the failure.
* a synthetic three-orbital / four-electron molecule, register ``(6, 4)`` --
  ``E_HF = -4.13563821``, sector FCI ``-4.52058324``, 9 determinants. ``auto``
  and ``number_conserving`` are the same circuit (no quartet is the whole
  register) and reach -4.37505461: 62.2 % of the 0.3849 Ha correlation
  energy, with a best overlap of 0.5306 against the sector ground state. A
  plateau that far below 1 is a limit of the parameter manifold, not of the
  optimiser. ``pauli`` reaches -4.46399215 with overlap 0.9363, but it is not
  in the ``N = 4`` sector at all (``max |<N> - 4| = 2.244``), so that number is
  not an energy of this molecule and is not a variational bound either. It is
  also *lower* than the -4.37505461 the sector-conserving circuit reaches, so
  the leak looks like the better answer of the two; it is not, it is simply
  the wrong particle number.

So use ``"auto"``. Prefer ``"number_conserving"`` when staying inside the
reference electron count matters more than reaching the exact ground state,
and ``"pauli"`` only on a register whose every double spans all four qubits.
"""

from __future__ import annotations

import superfermion as sf
from superfermion.parameters import param

__all__ = [
    "uccsd_ansatz",
    "hardware_efficient_ansatz",
    "single_excitation_gate",
    "double_excitation_gate",
    "excitation_pairs",
    "pauli_rotation",
]


def uccsd_ansatz(
    n_qubits: int,
    n_electrons: int,
    conserve_spin: bool = True,
    double_mode: str = "auto",
) -> sf.Circuit:
    """Create a unitary-coupled-cluster (UCCSD) variational circuit.

    The reference state is the Hartree-Fock determinant (qubits
    ``0 .. n_electrons-1`` occupied). Every excitation is a genuine rotation
    in the corresponding subspace, so the variational minimum is no longer
    pinned to ``E_HF``.

    Args:
        n_qubits: Total qubits (spin-orbitals).
        n_electrons: Number of electrons in the system.
        conserve_spin: When the spin-orbital ordering is interleaved
            (``2k`` = alpha, ``2k+1`` = beta, the convention used by
            :mod:`superfermion.chemistry.pyscf_bridge`), only excitations that
            keep the spin of every electron are generated. Set to ``False``
            for a spin-orbital ordering that is not interleaved.
        double_mode: How the double excitations are built.

            ``"auto"`` (default) uses the many-body generator
            ``exp(-i·theta/2 · X_i Y_j Y_a Y_b)`` when that quartet is the
            whole register -- the one case where the operator is provably
            number conserving -- and the hopping pair everywhere else, so the
            circuit never leaves the reference electron count.

            ``"number_conserving"`` always composes two hopping rotations on
            the spin-matched pairs. It stays inside the reference electron
            count, but those are the singles' own rotation planes with one
            shared angle, so it may not lower the energy at all: on H2 at
            ``(4, 2)`` it is pinned at ``E_HF`` (see the module docstring).

            ``"pauli"`` always emits the many-body generator. It is what the
            qubit-UCCSD literature writes and it is exact on the four-qubit
            sector holding exactly two electrons, but outside that sector it
            moves the state to ``N ± 2``. Only choose it when you know the
            quartet stays at two electrons: outside the sector the energy it
            reports is not a variational bound for the molecule.

    Examples:
        >>> circuit = uccsd_ansatz(4, 2)
        >>> list(circuit.parameters)
        ['s_0_2', 's_1_3', 'd_0_1_2_3']
    """
    if n_qubits == 2 and n_electrons == 2:
        return _h2_minimal_ansatz()
    if not 0 <= n_electrons <= n_qubits:
        raise ValueError(
            f"n_electrons={n_electrons} must be between 0 and n_qubits={n_qubits}"
        )
    if double_mode not in ("auto", "number_conserving", "pauli"):
        raise ValueError(
            "double_mode must be 'auto', 'number_conserving' or 'pauli', got "
            f"{double_mode!r}"
        )

    c = sf.Circuit(n_qubits)

    # Hartree-Fock reference determinant.
    for i in range(n_electrons):
        c.x(i)

    singles, doubles = excitation_pairs(n_qubits, n_electrons, conserve_spin)
    for i, a in singles:
        single_excitation_gate(c, i, a, param(f"s_{i}_{a}"))
    for i, j, a, b in doubles:
        if double_mode == "auto":
            # ``X_i Y_j Y_a Y_b`` is closed on the two-electron sector of its
            # quartet and leaks outside it, so it is only exact when the
            # quartet *is* the register (then every state in the space has the
            # reference electron count).
            conserving = {i, j, a, b} != set(range(n_qubits))
        else:
            conserving = double_mode == "number_conserving"
        double_excitation_gate(
            c, i, j, a, b, param(f"d_{i}_{j}_{a}_{b}"),
            number_conserving=conserving,
        )

    return c


def excitation_pairs(
    n_qubits: int, n_electrons: int, conserve_spin: bool = True
):
    """``(singles, doubles)`` index lists for the UCCSD expansion.

    ``singles`` holds ``(occupied, virtual)`` pairs and ``doubles`` holds
    ``(i, j, a, b)`` with ``i, j`` occupied and ``a, b`` virtual. With
    ``conserve_spin`` and an interleaved spin-orbital ordering, a single keeps
    the parity of its index and a double needs ``(i%2, j%2) == (a%2, b%2)``
    as multisets.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    interleaved = conserve_spin and n_qubits % 2 == 0

    singles = []
    for i in occupied:
        for a in virtual:
            if interleaved and i % 2 != a % 2:
                continue
            singles.append((i, a))

    doubles = []
    for pos, i in enumerate(occupied):
        for j in occupied[pos + 1:]:
            for pos_a, a in enumerate(virtual):
                for b in virtual[pos_a + 1:]:
                    if interleaved and sorted((i % 2, j % 2)) != sorted((a % 2, b % 2)):
                        continue
                    doubles.append((i, j, a, b))
    return singles, doubles


def pauli_rotation(c: sf.Circuit, theta, qubits, letters: str) -> sf.Circuit:
    """Append ``exp(-i·theta/2 · P)`` for the Pauli string ``P``.

    ``letters`` is aligned with ``qubits``: ``"XYYY"`` on four qubits means
    ``X_{qubits[0]} Y_{qubits[1]} Y_{qubits[2]} Y_{qubits[3]}``. The basis
    change maps each letter onto ``Z``, a CX ladder reduces the string to a
    single ``Z``, and ``rz(theta)`` supplies the rotation.
    """
    if len(qubits) != len(letters):
        raise ValueError(
            f"pauli_rotation needs one letter per qubit, got {len(qubits)} "
            f"and {len(letters)}"
        )
    chain = list(qubits)

    for q, letter in zip(chain, letters):
        if letter == "X":
            c.h(q)
        elif letter == "Y":
            # H·S† maps Y onto Z: S† Y S = X and H X H = Z.
            c.sdg(q)
            c.h(q)
        elif letter != "Z":
            raise ValueError(f"unknown Pauli letter {letter!r}")

    ladder = list(zip(chain, chain[1:]))
    for a, b in ladder:
        c.cx(a, b)
    c.rz(theta, chain[-1])
    # Undo the ladder in reverse order but with the *same* (control, target)
    # pairs: reversing the endpoints would compute the inverse of a different
    # circuit and leave a non-Pauli operator behind.
    for a, b in reversed(ladder):
        c.cx(a, b)

    for q, letter in reversed(list(zip(chain, letters))):
        if letter == "X":
            c.h(q)
        elif letter == "Y":
            c.h(q)
            c.s(q)
    return c


def single_excitation_gate(c: sf.Circuit, i: int, a: int, theta) -> sf.Circuit:
    """Append the fermionic single excitation ``i -> a``.

    Implements ``exp(-i·theta/2 (X_i Y_a - Y_i X_a))``, the real rotation that
    moves one electron between the two spin-orbitals and acts as the identity
    on ``|00>`` and ``|11>``. The old circuit applied ``ry`` on the virtual
    qubit and then ``cx(occupied, virtual)``, which leaves the occupied qubit
    occupied (a copy, not a move).
    """
    c.sdg(a)
    c.rxx(theta, i, a)
    c.ryy(theta, i, a)
    c.s(a)
    return c


def double_excitation_gate(
    c: sf.Circuit,
    i: int,
    j: int,
    a: int,
    b: int,
    theta,
    number_conserving: bool = True,
) -> sf.Circuit:
    """Append the double excitation ``(i, j) -> (a, b)``.

    Args:
        c: circuit to append to.
        i, j: occupied spin-orbitals.
        a, b: virtual spin-orbitals.
        theta: variational angle.
        number_conserving: build the excitation out of two hopping rotations
            on the spin-matched pairs (default). ``False`` emits the
            many-body qubit generator ``exp(-i·theta/2 · X_i Y_j Y_a Y_b)``
            instead, which is only closed on the two-electron sector of the
            quartet -- see the module docstring.

    The number-conserving form applies ``i -> alpha`` and ``j -> beta`` where
    ``alpha``/``beta`` are whichever of ``a``, ``b`` has the same spin as the
    electron being moved. A double whose two electrons cannot be paired that
    way would have to change the spin of an electron, so it is rejected.
    """
    if len({i, j, a, b}) != 4:
        raise ValueError(
            "double_excitation_gate needs four distinct spin-orbitals, got "
            f"{(i, j, a, b)}"
        )
    if not number_conserving:
        return pauli_rotation(c, theta, [i, j, a, b], "XYYY")

    def _pairs(ia: bool, ib: bool, ja: bool, jb: bool):
        """Choose the (occupied, virtual) pairing that keeps every spin."""
        if ia and jb and not ib and not ja:
            return (i, a), (j, b)
        if ib and ja and not ia and not jb:
            return (i, b), (j, a)
        # Same-spin electrons (i and j both alpha, or both beta): either
        # assignment is spin-conserving, so keep the listed order.
        if (ia or ib) and (ja or jb):
            return (i, a), (j, b)
        raise ValueError(
            f"the double ({(i, j)} -> {(a, b)}) changes the spin of an "
            f"electron, so it cannot be built from number-conserving hopping "
            f"rotations. Use number_conserving=False or conserve_spin=False."
        )

    first, second = _pairs(i % 2 == a % 2, i % 2 == b % 2,
                           j % 2 == a % 2, j % 2 == b % 2)
    single_excitation_gate(c, first[0], first[1], theta)
    single_excitation_gate(c, second[0], second[1], theta)
    return c


def hardware_efficient_ansatz(n_qubits: int, n_electrons: int) -> sf.Circuit:
    """The pre-fix construction, kept for reproducibility.

    Do not use it for energies: the singles copy instead of move and the
    doubles are global phases.

    The two H2 instances behave differently, and the CH-4 result is about the
    4-qubit one only:

    * ``(4, 2)`` on ``molecule_from_geometry(...).to_hamiltonian()``: qubits 0
      and 1 are only ever used as controls, so they stay at ``|1>`` and only
      ``2**(4-2) = 4`` states are reachable. The lowest eigenvalue of that
      subspace is exactly ``E_HF``, and Nelder-Mead over the five parameters
      returns ``E_HF`` (+1e-12). The whole-parameter-space minimum therefore
      equals the Hartree-Fock energy -- the CH-4 finding, a gap of +2.031e-2
      to FCI.
    * ``(2, 2)`` on ``get_molecular_hamiltonian("H2")``: the reference is
      ``|11>``, not the reduced model's HF state ``|10>``, and the only
      parameter ``d_0`` is a global phase. The energy is the constant
      ``-0.3436844`` at every ``d_0`` -- 0.7733 Ha *above* ``E_HF``. So this
      instance is not pinned to ``E_HF``; it is simply wrong, which is the
      other reason the construction is retired.
    """
    c = sf.Circuit(n_qubits)

    for i in range(n_electrons):
        c.x(i)

    for i in range(n_electrons):
        for j in range(n_electrons, n_qubits):
            c.ry(param(f"s_{i}_{j}"), j)
            c.cx(i, j)

    for i in range(n_electrons - 1):
        c.cx(i, i + 1)
        c.rz(param(f"d_{i}"), i + 1)
        c.cx(i, i + 1)

    return c


def _h2_minimal_ansatz() -> sf.Circuit:
    """Standard 2-qubit H2 ansatz (O'Malley et al. 2016).

    The ground state lives in the {|01>, |10>} subspace.  This circuit
    prepares the HF reference |10> (one occupied spin-orbital on q0)
    then applies a Y-rotation on q1 followed by CNOT(q1, q0) to span
    cos(t/2)|10> + sin(t/2)|01>.
    """
    c = sf.Circuit(2)
    c.x(0)
    c.ry(param("theta"), 1)
    c.cx(1, 0)
    return c
