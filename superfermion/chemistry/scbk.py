"""SCBK: symmetry-conserving Bravyi-Kitaev, exactly two qubits removed.

Setia, Chen, Rice, Mezzacapo, Pistoia and Whitfield, *PRA* 102, 032414 (2020).

Bravyi-Kitaev already keeps Pauli weights logarithmic in the mode count, but
it spends qubits on the particle-number parity.  A molecular Hamiltonian
conserves ``N_alpha`` and ``N_beta`` separately, so those two parities are
exact Z2 symmetries and their qubits are redundant: fix the sector and two
qubits disappear *for every molecule*, not just for the ones whose basis
happens to expose extra symmetries.

Implementation note, stated plainly: this module does not re-derive the SCBK
circuit.  It maps with the package's verified ``bravyi_kitaev`` and then
removes exactly two symmetry qubits with the same Clifford substitution used
by :mod:`superfermion.chemistry.tapering`, restricted to two generators.  That
restriction is the whole point of the encoder: the qubit saving is a fixed
promise of the encoding, so it is asserted rather than measured after the
fact.  What is measured is the spectrum.

Because the reduction is exact, the four sector Hamiltonians tile the full
spectrum of the untapered operator.  That identity is the correctness test
(see ``tests/domain/test_scbk.py``): no sector may be missing and none may
overlap, or an energy somewhere is wrong.

Not implemented: BKSF (the *superfast* encoder), which spends qubits on the
edges of the interaction graph and enforces the resulting constraints by
projection.  It shares the ``Bravyi-Kitaev`` name but none of the algebra
below, and a half-transcribed edge-operator construction would produce a
Hermitian operator with plausible energies and no relationship to the
molecule.  It is listed as unimplemented rather than approximated.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.chemistry.hamiltonians import FermionicOperator
from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals
from superfermion.chemistry.tapering import taper_z2, z2_symmetries

__all__ = ["scbk_reduce", "scbk_from_integrals"]

# Refuse to build a dense 2**n matrix past this; the caller has its own solver.
_DENSE_QUBIT_LIMIT = 12


def _lowest_eigenvalue(hamiltonian, n_qubits: int) -> Optional[float]:
    if n_qubits > _DENSE_QUBIT_LIMIT:
        # Silence would be the wrong answer here: the caller gets ``None`` and
        # has to know why, or the dict looks like a solver that failed.
        warnings.warn(
            f"SCBK sector on {n_qubits} qubits is past the dense-spectrum "
            f"limit of {_DENSE_QUBIT_LIMIT}; its energy is reported as None. "
            f"Diagonalise the sector Hamiltonian with a solver, or pass "
            f"compute_spectrum=False to skip this deliberately.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    from superfermion.chemistry.subspace import pauli_matrix

    return float(np.linalg.eigvalsh(pauli_matrix(hamiltonian))[0])


def scbk_reduce(
    h1_so: np.ndarray,
    h2_so: Optional[np.ndarray],
    n_modes: int,
    *,
    n_symmetries: int = 2,
    compute_spectrum: bool = True,
    energy_offset: float = 0.0,
) -> Dict[str, Any]:
    """Bravyi-Kitaev map, then remove ``n_symmetries`` parity qubits.

    Args:
        h1_so: ``(n_modes, n_modes)`` spin-orbital one-body integrals.
        h2_so: Two-body integrals in the layout ``FermionicOperator.from_coeffs``
            expects, or ``None``.
        n_modes: Spin-orbital count (``2 * n_spatial``).
        n_symmetries: Generators to remove. ``2`` is the SCBK contract.
        compute_spectrum: Evaluate the lowest eigenvalue of each sector
            Hamiltonian.  This is the expensive part of the call: it builds a
            dense ``2**n`` matrix and diagonalises it, so it is free on the
            handful of qubits the tests use and takes minutes once a sector is
            ten or twelve qubits wide.  Turn it off when the point of the call
            is the reduction rather than a debugging spectrum; ``energies``
            then holds ``None`` per sector.
        energy_offset: Constant added to every reported sector energy.  The
            operator is built from integrals, so it is *electronic* only:
            pass the nuclear repulsion here and the numbers are total
            energies.  Without it water/STO-3G reports -84.2735 where the
            molecule's ground state is -74.9861.

    Returns:
        Dict with

        ``hamiltonians``
            ``{sector_tuple: Hamiltonian}`` for every combination of sector
            eigenvalues, each on ``n_modes - n_symmetries`` qubits.
        ``energies``
            ``{sector_tuple: float}`` lowest eigenvalue per sector plus
            ``energy_offset``, or ``None`` where the dense guard refused or
            ``compute_spectrum`` was false.
        ``symmetries``
            The ``{I,Z}`` strings that were removed.
        ``n_qubits_before``, ``n_qubits_after``, ``n_terms_before``
            Sizes before and after.

    Raises:
        ValueError: if fewer than ``n_symmetries`` independent symmetries exist
            -- in that case the requested saving is not available and saying
            so is better than silently removing fewer qubits.
    """
    op = FermionicOperator.from_coeffs(h1_so, None if h2_so is None else h2_so)
    full = op.bravyi_kitaev(n_modes)

    found = z2_symmetries(full)
    if len(found) < n_symmetries:
        raise ValueError(
            f"the Bravyi-Kitaev Hamiltonian exposes {len(found)} Z2 "
            f"symmetries but SCBK needs {n_symmetries}; the encoding saving "
            f"is not available for this operator"
        )
    chosen = found[:n_symmetries]

    hamiltonians: Dict[Tuple[int, ...], Any] = {}
    energies: Dict[Tuple[int, ...], Optional[float]] = {}
    for sector in product((-1, 1), repeat=n_symmetries):
        reduced, info = taper_z2(full, sector=list(sector), symmetries=chosen)
        if info["n_qubits_after"] != n_modes - n_symmetries:
            raise ValueError(
                f"expected {n_modes - n_symmetries} qubits after SCBK, got "
                f"{info['n_qubits_after']}"
            )
        hamiltonians[sector] = reduced
        if compute_spectrum:
            lowest = _lowest_eigenvalue(reduced, info["n_qubits_after"])
            energies[sector] = (
                None if lowest is None else float(lowest) + float(energy_offset)
            )
        else:
            energies[sector] = None

    return {
        "hamiltonians": hamiltonians,
        "energies": energies,
        "symmetries": chosen,
        "n_qubits_before": n_modes,
        "n_qubits_after": n_modes - n_symmetries,
        "n_terms_before": len(list(full.terms)),
        "n_terms_after": {
            sector: len(list(h.terms)) for sector, h in hamiltonians.items()
        },
    }


def scbk_from_integrals(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    *,
    n_symmetries: int = 2,
    compute_spectrum: bool = True,
    energy_offset: float = 0.0,
) -> Dict[str, Any]:
    """Same as :func:`scbk_reduce`, from spin-free (restricted) MO integrals.

    ``h1`` is ``(n, n)`` and ``h2`` is ``(n, n, n, n)`` in chemist order, as
    stored on :class:`MolecularData`; the spin-orbital expansion is the one
    used by every other solver here.  Pass ``energy_offset=molecule
    .nuclear_repulsion`` to get total rather than electronic energies.
    """
    h1 = np.asarray(h1, dtype=float)
    h1_so, h2_so = _spin_orbital_integrals(
        h1, None if h2 is None else np.asarray(h2, dtype=float)
    )
    return scbk_reduce(
        h1_so, h2_so, 2 * h1.shape[0],
        n_symmetries=n_symmetries,
        compute_spectrum=compute_spectrum,
        energy_offset=energy_offset,
    )
