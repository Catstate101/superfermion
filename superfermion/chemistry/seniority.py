"""Seniority truncation of a determinant space.

Seniority ``Omega`` counts the *singly occupied spatial orbitals* of a
determinant.  A closed-shell determinant has ``Omega = 0``; a determinant with
one electron promoted out of a pair has ``Omega = 2``, because both the orbital
it left and the orbital it entered are singly occupied.  ``Omega`` is therefore
always even for an even electron count, and it is a property of a single
determinant, not of a state.

Why truncate on it: for a weakly correlated closed-shell molecule the exact
ground state is dominated by ``Omega = 0`` and ``Omega = 2`` determinants --
"pair" excitations that keep the closed-shell pairing pattern -- while
``Omega >= 4`` carries little weight.  Keeping only the low-seniority sector
shrinks the determinant space by orders of magnitude for a many-orbital active
space.  For a stretched bond the pairing is broken, high-``Omega`` determinants
carry real weight, and the truncated energy is *above* the exact one by an
amount that is a direct measure of how bad the truncation is.  That direction is
what the tests assert, rather than a tolerance that would hide it.

The functions here take and return plain determinant bitmasks in the interleaved
spin-orbital layout used everywhere else in this package (``2p`` alpha, ``2p+1``
beta), so they compose with :mod:`superfermion.chemistry.subspace` and
:mod:`superfermion.chemistry.sqd` without translation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "seniority",
    "seniority_of",
    "seniority_of_state",
    "seniority_distribution",
    "filter_by_seniority",
    "seniority_pool",
]


def seniority(det: int) -> int:
    """Number of singly occupied spatial orbitals in ``det``.

    Args:
        det: Determinant bitmask, ``2p`` alpha and ``2p + 1`` beta.

    Returns:
        The seniority ``Omega``; ``0`` for a closed-shell occupation.
    """
    det = int(det)
    if det < 0:
        raise ValueError("a determinant bitmask must be non-negative")
    omega = 0
    p = 0
    while (det >> (2 * p)) > 0:
        alpha = (det >> (2 * p)) & 1
        beta = (det >> (2 * p + 1)) & 1
        omega += alpha ^ beta
        p += 1
    return omega


def seniority_of(det: int, *, n_orbitals: Optional[int] = None) -> int:
    """Alias of :func:`seniority` that optionally validates the width.

    ``n_orbitals`` only bounds-checks the bitmask: a determinant that sets a
    bit at or above ``2 * n_orbitals`` is a caller error, and silently returning
    a seniority for it would hide a mode-indexing mistake.
    """
    det = int(det)
    if n_orbitals is not None and det >> (2 * int(n_orbitals)):
        raise ValueError(
            f"determinant {det} sets a spin orbital at or above "
            f"2*n_orbitals={2 * int(n_orbitals)}"
        )
    return seniority(det)


def seniority_distribution(determinants: Iterable[int]) -> Dict[int, int]:
    """Count determinants per seniority sector.

    Returns:
        ``{Omega: count}`` with the keys sorted ascending.
    """
    counts: Dict[int, int] = {}
    for det in determinants:
        omega = seniority(det)
        counts[omega] = counts.get(omega, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def filter_by_seniority(
    determinants: Iterable[int], max_seniority: int
) -> List[int]:
    """Keep the determinants whose ``Omega`` is at most ``max_seniority``.

    Duplicates are removed and the result is sorted, so the output can be fed
    straight to :func:`superfermion.chemistry.subspace.ground_state`.
    """
    limit = int(max_seniority)
    if limit < 0:
        raise ValueError("max_seniority must be non-negative")
    return sorted({int(d) for d in determinants if seniority(d) <= limit})


def seniority_pool(
    n_orbitals: int,
    n_electrons: int,
    *,
    max_seniority: Optional[int] = 2,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> List[int]:
    """Every determinant of the requested sector with ``Omega <= max_seniority``.

    Args:
        n_orbitals: Spatial orbitals; the spin-orbital count is ``2 * n``.
        n_electrons: Total electrons.
        max_seniority: Seniority cap. ``None`` keeps the whole sector, which is
            only useful as a self-check that the filter is inactive.
        n_alpha, n_beta: Fix the ``S_z`` sector, as elsewhere in this package.

    Returns:
        Ascending list of determinant bitmasks.

    Raises:
        ValueError: If the cap excludes the whole sector, which would make a
            downstream diagonalisation fail with a less useful message.
    """
    from superfermion.chemistry.subspace import all_determinants

    pool = all_determinants(
        n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
    )
    if max_seniority is None:
        return pool
    kept = filter_by_seniority(pool, max_seniority)
    if not kept:
        raise ValueError(
            f"max_seniority={max_seniority} leaves no determinant of "
            f"({n_orbitals} orbitals, {n_electrons} electrons); with "
            f"n_alpha={n_alpha} and n_beta={n_beta} the minimum seniority is "
            f"{min(seniority(d) for d in pool)}"
        )
    return kept


def seniority_of_state(
    amplitudes: Sequence[complex],
    determinants: Sequence[int],
    *,
    threshold: float = 1e-8,
) -> Dict[str, float]:
    """Seniority content of a state given its determinant amplitudes.

    Reports the probability weight in each seniority sector, which is what
    decides whether a truncation is safe for *this* state rather than for a
    qualitative argument about correlation strength.

    Args:
        amplitudes: Coefficient per entry of ``determinants``.
        determinants: Determinant bitmasks, same length as ``amplitudes``.
        threshold: Sectors below this weight are omitted.

    Returns:
        Dict with ``weights`` (``{Omega: weight}``, summing to ~1),
        ``mean_seniority`` (``sum Omega * weight``) and ``weight_above``
        (``{Omega: weight}`` for sectors above ``max(Omega)`` -- see
        ``max_seniority`` below) is not included; callers get ``weights`` and
        pick their own cut.

    Raises:
        ValueError: If the lengths differ or every amplitude is below
            ``threshold`` (a state that cannot be normalised).
    """
    amps = list(amplitudes)
    dets = list(determinants)
    if len(amps) != len(dets):
        raise ValueError(
            f"{len(amps)} amplitudes for {len(dets)} determinants"
        )
    weights: Dict[int, float] = {}
    total = 0.0
    for amp, det in zip(amps, dets):
        p = float(abs(complex(amp)) ** 2)
        if p <= threshold:
            continue
        omega = seniority(det)
        weights[omega] = weights.get(omega, 0.0) + p
        total += p
    if total <= 0.0:
        raise ValueError("all amplitudes are below the threshold")
    weights = {k: weights[k] / total for k in sorted(weights)}
    mean = sum(omega * w for omega, w in weights.items())
    return {"weights": weights, "mean_seniority": mean, "norm": total}


#: Seniority of the Hartree-Fock reference of a closed-shell molecule, kept
#: as a named constant so a caller can pass ``max_seniority=HF_SENIORITY + 2``
#: and mean "one pair broken" without restating the convention.
HF_SENIORITY = 0
