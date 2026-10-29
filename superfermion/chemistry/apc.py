"""APC ranked-orbital active-space selection.

Implements the APC ("approximate pair correlation") orbital entropy of King et
al., *JCTC* 2021, doi:10.1021/acs.jctc.1c00037, the APC-N variant of *JCTC* 2022,
doi:10.1021/acs.jctc.2c00630, and the ranked-orbital chooser that turns a set of
entropies into an active space.

Why this is not a literature entry any more
-------------------------------------------
``_v021_CHEM_ACTIVE_SPACE.md`` §8.4 called this row "λAPC" and declined it on the
grounds that it needs an MC-PDFT density.  That conflates two papers by the same
author.  The ranked-orbital criterion is built from three things only — the Fock
matrix, the exchange matrix and a set of orbital occupations — and all three are
*arguments* here, so the criterion is well defined without any functional.  The
density-dependent variant is then a caller choice: pass the Fock/exchange pair
of an RHF, of an ROHF/UHF, or of a correlated density.

The pair amplitude
------------------
For a doubly occupied orbital ``d`` and a virtual ``v`` the 2x2 CIS problem has
the amplitude

    k12   = (1/2) K_vv
    delta = F_vv - F_dd
    c     = -k12 / (delta + sqrt(k12**2 + delta**2))

so ``c -> 0`` when the gap dominates the coupling and ``|c| -> 1/sqrt(2)`` when
the coupling dominates -- which is exactly "how much this pair wants to be
correlated".  The entropy of the vector ``(c_1, ..., c_m, 1) / norm`` is the
orbital's score.  A singly occupied orbital has no partner in this construction,
so it is given ``max + eps`` and therefore always selected.

Two entropy variants, and why the default is not the reference's
The reference implementation (``pyscf.mcscf.apc``) collapses the amplitudes to
one outcome before the logarithm::

    S_ref = -p log p - q log q,   p = sum_i c_i^2 / norm^2,  q = 1 / norm^2

whereas the paper's entropy -- and this module's default -- is the sum over
outcomes::

    S_paper = -sum_i w_i log w_i - q log q,   w_i = c_i^2 / norm^2

Merging outcomes cannot add entropy, so ``S_paper >= S_ref`` with equality when
the orbital has a single partner.  Measured on water: STO-3G (two virtuals)
max |S_paper - S_ref| = 7.5e-02, 6-31G (eight virtuals, two rounds) 4.5e-02,
and the *ranking* is unchanged -- both promote the same virtuals and the chooser
returns the same index list.  ``variant="pyscf"`` reproduces the reference bit
for bit, for anyone porting a published selection.

Relationship to the rest of the package
---------------------------------------
:mod:`superfermion.chemistry.entropy` ranks orbitals by the entanglement entropy
of a correlated wavefunction and :mod:`superfermion.chemistry.avas` ranks them by
an atomic projection; APC ranks them by a *mean-field* correlation estimate, so it
is the only one of the three that needs no wavefunction at all.  That is its
selling point and also its limit: it cannot see static correlation that the
reference determinant does not already describe.

Nothing here imports PySCF or scipy at module level (``n_csfs`` uses
``math.comb``), so the module is importable on a machine without either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ActiveSpaceChoice",
    "apc_entropies",
    "apc_active_space",
    "chooser",
    "n_csfs",
]


def _xlogx(x: np.ndarray) -> np.ndarray:
    """``x log x`` with ``0 log 0 = 0``.

    The guard changes nothing where the argument is positive; it only keeps an
    orbital with no coupling to any partner (every amplitude zero) from
    returning ``nan``.
    """
    out = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    out[mask] = x[mask] * np.log(x[mask])
    return out


def apc_entropies(
    fock: np.ndarray,
    exchange: np.ndarray,
    occupations: Sequence[float],
    *,
    rounds: int = 2,
    eps: float = 1e-3,
    variant: str = "paper",
) -> np.ndarray:
    """APC-N orbital entropies for a set of MOs.

    Args:
        fock: ``(n, n)`` Fock matrix in the MO basis.  Only the diagonal is used.
        exchange: ``(n, n)`` exchange matrix ``K`` in the MO basis.
        occupations: ``(n,)`` occupations, ``2``/``1``/``0`` per spatial orbital.
        rounds: ``n`` in APC-N -- how many high-entropy virtuals are promoted to
            singly occupied before the reported entropies are computed.  ``1``
            reproduces plain APC.
        eps: Offset applied to the promoted and singly occupied orbitals so that
            they sort above everything else.  Must be positive.
        variant: ``"paper"`` (default) is the published entropy,
            ``-sum_i p_i log p_i`` over the pair amplitudes and the vacuum term.
            ``"pyscf"`` reproduces ``pyscf.mcscf.apc.APC._apc``, which collapses
            the pair amplitudes into one outcome first and costs
            ``-(sum_i p_i) log(sum_i p_i)``.  The two agree exactly when an
            orbital has a single partner and differ otherwise; see the module
            docstring.

    Returns:
        ``(n,)`` entropies, one per MO.  Higher means "wants to be active".

    Raises:
        ValueError: on a shape mismatch, a non-positive ``eps``, ``rounds < 1``,
            an unknown ``variant``, or occupations not in ``{0, 1, 2}``.
    """
    if variant not in (
        "paper",
        "pyscf",
    ):
        raise ValueError(f"variant must be 'paper' or 'pyscf', got {variant!r}")
    fock = np.asarray(fock, dtype=float)
    exchange = np.asarray(exchange, dtype=float)
    occ = np.asarray(occupations, dtype=float).ravel()
    n = occ.shape[0]

    if fock.shape != (n, n) or exchange.shape != (n, n):
        raise ValueError(
            f"fock {fock.shape} and exchange {exchange.shape} must both be "
            f"({n}, {n}) to match the {n} occupations"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not np.allclose(occ, np.round(occ)):
        raise ValueError(
            f"occupations must be integers, got {occ.tolist()}"
        )
    if np.any(occ < 0) or np.any(occ > 2):
        raise ValueError(
            f"occupations must lie in [0, 2], got {occ.tolist()}"
        )

    occ = np.round(occ).astype(int)
    original_singly = np.where(occ == 1)[0]
    removed: List[int] = []

    entropies = _apc(fock, exchange, occ, eps, variant)
    for round_index in range(rounds):
        if round_index > 0:
            entropies = _apc(fock, exchange, occ, eps, variant)
        virtual = np.where(occ == 0)[0]
        if virtual.size == 0:
            break
        promote = int(virtual[np.argmax(entropies[virtual])])
        removed.append(promote)
        occ[removed] = 1

    entropies = _apc(fock, exchange, occ, eps, variant)

    # The promoted virtuals and the original open shells have no APC score of
    # their own; place them above every computed value, in a stable order.
    top = float(np.max(entropies)) if entropies.size else 0.0
    for i, orbital in enumerate(original_singly):
        entropies[orbital] = top + 2.0 * eps - i * eps * 1e-2
    for i, orbital in enumerate(removed):
        entropies[orbital] = top + eps - i * eps * 1e-2

    return entropies


def _apc(
    fock: np.ndarray,
    exchange: np.ndarray,
    occ: np.ndarray,
    eps: float,
    variant: str = "paper",
) -> np.ndarray:
    """One pass of the APC entropy (``pyscf.mcscf.apc.APC._apc``)."""
    n = occ.shape[0]
    doubly = np.where(occ == 2)[0]
    singly = np.where(occ == 1)[0]
    virtual = np.where(occ == 0)[0]

    amplitudes = np.zeros((doubly.size, virtual.size), dtype=float)
    for i, d in enumerate(doubly):
        for j, v in enumerate(virtual):
            k12 = 0.5 * exchange[v, v]
            delta = fock[v, v] - fock[d, d]
            amplitudes[i, j] = -k12 / (delta + math.hypot(k12, delta))

    entropies = np.zeros(n, dtype=float)
    for orbital in range(n):
        if orbital in singly:
            continue  # filled in by the caller
        if orbital in doubly:
            row = int(np.where(doubly == orbital)[0][0])
            vector = amplitudes[row, :]
        else:
            col = int(np.where(virtual == orbital)[0][0])
            vector = amplitudes[:, col]

        squared = vector ** 2
        norm = math.sqrt(float(np.sum(squared)) + 1.0)
        if variant == "pyscf":
            # The reference sums the pair weights into one outcome before taking
            # the logarithm, so its "entropy" is a two-outcome one.
            pair = float(np.sum(squared)) / (norm ** 2)
            vacuum = 1.0 / (norm ** 2)
            entropies[orbital] = -float(np.sum(_xlogx(np.array([pair, vacuum]))))
        else:
            weights = np.append(squared, 1.0) / (norm ** 2)
            entropies[orbital] = -float(np.sum(_xlogx(weights)))

    if entropies.size:
        entropies[singly] = float(np.max(entropies)) + eps
    return entropies


def n_csfs(nactel: int, norbs: int) -> int:
    """Number of configuration state functions of a ``(nactel, norbs)`` space.

    ``C(n, a) C(n, b) - C(n, a+1) C(n, b-1)`` with ``a = ceil(nactel/2)``,
    ``b = floor(nactel/2)`` -- the ``M_S = 0`` (or ``M_S = 1/2``) count.
    """
    nactel = int(nactel)
    norbs = int(norbs)
    if norbs < 0 or nactel < 0:
        raise ValueError(
            f"nactel={nactel} and norbs={norbs} must be non-negative"
        )
    alpha = nactel // 2 + nactel % 2
    beta = nactel // 2
    if alpha > norbs or beta > norbs:
        return 0
    term1 = math.comb(norbs, alpha) * math.comb(norbs, beta)
    term2 = (
        math.comb(norbs, alpha + 1) * math.comb(norbs, beta - 1)
        if beta >= 1 and alpha + 1 <= norbs
        else 0
    )
    return term1 - term2


@dataclass
class ActiveSpaceChoice:
    """The ranked-orbital result: which orbitals, how many electrons, how big."""

    active_indices: Tuple[int, ...]
    n_alpha: int
    n_beta: int
    inactive_indices: Tuple[int, ...]
    secondary_indices: Tuple[int, ...]
    n_csfs: int
    entropies: Tuple[float, ...] = field(repr=False)

    @property
    def n_active_orbitals(self) -> int:
        return len(self.active_indices)

    @property
    def n_active_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def active_orbitals(self) -> Tuple[int, int]:
        """``(n_electrons, n_orbitals)``, the form ``MolecularData`` takes."""
        return (self.n_active_electrons, self.n_active_orbitals)

    def __repr__(self) -> str:  # keep the tuple dump readable
        return (
            f"ActiveSpaceChoice(active={list(self.active_indices)}, "
            f"n_alpha={self.n_alpha}, n_beta={self.n_beta}, "
            f"n_csfs={self.n_csfs})"
        )


def chooser(
    occupations: Sequence[float],
    entropies: Sequence[float],
    *,
    max_size: Any = (8, 8),
    fixed: bool = False,
) -> ActiveSpaceChoice:
    """Ranked-orbital selection (``pyscf.mcscf.apc.Chooser``).

    Args:
        occupations: ``(n,)`` MO occupations, ``2``/``1``/``0``.
        entropies: ``(n,)`` scores; higher is more important.
        max_size: ``(nactel, norbs)`` to bound the **CSF count**, or an ``int``
            to bound the **orbital count**.
        fixed: Select exactly ``max_size`` from the highest-entropy orbitals
            instead of dropping the cheapest orbitals until the bound holds.
            Used when orbital consistency matters, e.g. along a reaction path.

    Returns:
        :class:`ActiveSpaceChoice`.

    Raises:
        ValueError: if ``fixed`` is used with an integer ``max_size``, if the
            requested size cannot be reached, or if the resulting space is not
            reasonable (no electrons, a full space, or fewer orbitals than open
            shells).
    """
    occ = np.asarray(occupations, dtype=float).ravel()
    ent = np.asarray(entropies, dtype=float).ravel()
    if occ.shape != ent.shape:
        raise ValueError(
            f"got {occ.size} occupations and {ent.size} entropies; they must "
            f"match, one score per orbital"
        )
    if not np.allclose(occ, np.round(occ)) or np.any(occ < 0) or np.any(occ > 2):
        raise ValueError(
            f"occupations must be 0/1/2, got {occ.tolist()}"
        )
    occ = np.round(occ).astype(int)
    n = occ.size

    # Open shells have no APC partner; lift them above the computed scores so
    # the ranking never drops one.  The reference does this here, not in _apc,
    # because the *ranking* is what needs them first.
    scores = ent.copy()
    open_shell = np.where(occ == 1)[0]
    if open_shell.size and scores.size:
        scores[open_shell] = float(np.max(scores)) + 0.01

    if fixed:
        if isinstance(max_size, (int, np.integer)):
            raise ValueError(
                "fixed=True needs max_size=(nactel, norbs); an orbital count "
                "does not say how many of them are doubly occupied"
            )
        return _choose_fixed(occ, scores, tuple(int(v) for v in max_size))
    return _choose_dynamic(occ, scores, max_size, n)


def _choose_fixed(occ, scores, max_size) -> ActiveSpaceChoice:
    nactel, norbs = max_size
    doubly = np.where(occ == 2)[0]
    open_shell = np.where(occ == 1)[0]
    virtual = np.where(occ == 0)[0]

    n_open = open_shell.size
    n_core = int((nactel - n_open) / 2)
    n_virt = int(norbs - n_core - n_open)
    if n_core < 0 or n_virt < 0:
        raise ValueError(
            f"max_size={max_size} cannot host {n_open} open-shell orbitals with "
            f"{nactel} electrons"
        )
    if n_core > doubly.size or n_virt > virtual.size:
        raise ValueError(
            f"max_size={max_size} needs {n_core} doubly occupied and {n_virt} "
            f"virtual orbitals but only {doubly.size} and {virtual.size} exist"
        )

    keep_core = doubly[np.argsort(scores[doubly])[-n_core:]] if n_core else []
    keep_virt = virtual[np.argsort(scores[virtual])[-n_virt:]] if n_virt else []
    active = np.hstack([keep_core, open_shell, keep_virt]).astype(int)
    inactive = np.setdiff1d(doubly, keep_core)
    secondary = np.setdiff1d(virtual, keep_virt)
    return _build_choice(active, inactive, secondary, occ, scores)


def _choose_dynamic(occ, scores, max_size, n) -> ActiveSpaceChoice:
    active = list(range(n))
    inactive: List[int] = []
    secondary: List[int] = []

    if isinstance(max_size, (tuple, list, np.ndarray)):
        bound_csfs = _csf_budget(occ, max_size)
        size = _count(occ, active, as_csfs=True)
        as_csfs = True
    else:
        bound_csfs = int(max_size)
        size = len(active)
        as_csfs = False

    while size > bound_csfs:
        ranking = sorted(active, key=lambda i: scores[i])
        dropped = None
        for candidate in ranking:
            trial = [i for i in active if i != candidate]
            if _is_reasonable(occ, trial):
                dropped = candidate
                break
        if dropped is None:
            raise ValueError(
                "no orbital can be dropped without making the active space "
                "unreasonable (no electrons, a full space, or fewer orbitals "
                "than open shells); raise max_size"
            )
        active.remove(dropped)
        if occ[dropped] > 0:
            inactive.append(dropped)
        else:
            secondary.append(dropped)
        size = (
            _count(occ, active, as_csfs=True) if as_csfs else len(active)
        )

    return _build_choice(
        np.array(sorted(active), dtype=int),
        np.array(sorted(inactive), dtype=int),
        np.array(sorted(secondary), dtype=int),
        occ,
        scores,
    )


def _csf_budget(occ, max_size) -> int:
    nactel, norbs = (int(v) for v in max_size)
    return n_csfs(nactel, norbs)


def _count(occ, active, *, as_csfs: bool) -> int:
    if not as_csfs:
        return len(active)
    nactel = int(np.sum(occ[active])) if active else 0
    return n_csfs(nactel, len(active))


def _is_reasonable(occ, active) -> bool:
    """``pyscf.mcscf.apc.Chooser._as_is_reasonable``: ``0 < nactel < 2 n``."""
    if not active:
        return False
    nactel = int(np.sum(occ[active]))
    n_open = int(np.sum(occ == 1))
    return nactel > 0 and nactel < 2 * len(active) and len(active) >= n_open


def _build_choice(active, inactive, secondary, occ, scores) -> ActiveSpaceChoice:
    active = np.asarray(active, dtype=int)
    n_open = int(np.sum(occ[active] == 1)) if active.size else 0
    # ``np.sum(occ[active] == 2)`` counts *orbitals*, so it is already the
    # number of closed-shell pairs -- and each pair contributes one alpha and
    # one beta electron.  It used to be divided by two as if it were counting
    # electrons, which halved the closed-shell contribution: a window holding a
    # single doubly occupied orbital reported n_alpha = n_beta = 0, and a (2,2)
    # window with three CSFs reported n_csfs = 1.  The invariant that pins the
    # arithmetic is ``n_alpha + n_beta == nactel``.
    n_both = int(np.sum(occ[active] == 2)) if active.size else 0
    n_alpha = n_both + n_open
    n_beta = n_both
    total = n_alpha + n_beta
    return ActiveSpaceChoice(
        active_indices=tuple(int(i) for i in active),
        n_alpha=n_alpha,
        n_beta=n_beta,
        inactive_indices=tuple(int(i) for i in inactive),
        secondary_indices=tuple(int(i) for i in secondary),
        n_csfs=n_csfs(total, active.size),
        entropies=tuple(float(s) for s in scores),
    )


def apc_active_space(
    mol,
    *,
    max_size: Any = (8, 8),
    rounds: int = 2,
    eps: float = 1e-3,
    fixed: bool = False,
    variant: str = "paper",
) -> ActiveSpaceChoice:
    """APC selection for a :class:`~superfermion.chemistry.pyscf_bridge.MolecularData`.

    Needs the Fock and exchange matrices in the MO basis, which
    ``molecule_from_geometry`` stores when PySCF is present (``fock_mo``,
    ``exchange_mo``) and which :meth:`MolecularData.set_mean_field` sets
    explicitly for a correlated or a different reference.

    Args:
        mol: ``MolecularData`` with ``fock_mo`` and ``exchange_mo`` populated.
        max_size: ``(nactel, norbs)`` or an int orbital count; see :func:`chooser`.
        rounds: Number of APC-N promotion rounds.
        eps: Offset for open shells and promoted virtuals.
        fixed: Select exactly ``max_size``; see :func:`chooser`.
        variant: Entropy definition, ``"paper"`` or ``"pyscf"``; see
            :func:`apc_entropies`.

    Returns:
        :class:`ActiveSpaceChoice`.

    Raises:
        ValueError: if the molecule has no mean field to rank against.
    """
    fock = getattr(mol, "fock_mo", None)
    exchange = getattr(mol, "exchange_mo", None)
    occupations = getattr(mol, "mo_occupations", None)
    if fock is None or exchange is None or occupations is None:
        raise ValueError(
            "APC ranks orbitals against a mean field, and this MolecularData "
            "has no fock_mo / exchange_mo / mo_occupations. Build it with "
            "molecule_from_geometry under PySCF, or call set_mean_field()."
        )
    entropies = apc_entropies(
        fock, exchange, occupations, rounds=rounds, eps=eps, variant=variant
    )
    return chooser(occupations, entropies, max_size=max_size, fixed=fixed)
