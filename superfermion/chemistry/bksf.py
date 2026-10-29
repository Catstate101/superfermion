"""Bravyi-Kitaev superfast (original superfast encoding, OSE): cost, not a build.

Status: this module measures whether the encoding is worth having for a given
coupling graph and provides the two operator families whose definitions are
unambiguous.  It deliberately does **not** build a qubit Hamiltonian.  The
reason is measured below and is not a matter of taste.

The encoding
------------
A graph is placed on the fermionic modes: one vertex per mode, one edge per
non-negligible coupling.  Each *directed* edge carries one qubit, so a graph
with ``E`` undirected edges costs ``2E`` qubits.  Vertex and edge operators
(Chien, Xue, Hardikar, Setia and Whitfield, arXiv:1907.02976, Table 1) are

    B_i  = prod_{j in n(i)} Z_(i,j)
    A_pq = eps_pq X_(p,q) prod_{l<q, l in n(p)} Z_(l,p)
                          prod_{s<p, s in n(q)} Z_(s,q)

with ``n(i)`` the neighbours of ``i`` and ``eps_pq = +1`` for ``p < q``,
``-1`` otherwise.  The payoff is that ``A_pq`` has weight at most
``1 + deg(p) + deg(q)``, so on a bounded-degree graph every term of the
Hamiltonian is O(1): the Jordan-Wigner strings of length O(n) are gone.

Why there is no Hamiltonian builder
-----------------------------------
The paper's Table 1 lists representative second-quantised terms, not a complete
decomposition.  The general two-electron operator needs the full
double-excitation formula, which is long enough that the only accessible copy
of it is visibly corrupted in the HTML rendering we could obtain, and the
qubit-level relations hold *on the code space* -- the stabiliser (gauge)
constraints that pick it out are not given there at all.  Writing either from
memory and shipping it would break the rule every other module here follows:
nothing returns an energy until it has been checked against an exact
diagonalisation.  So the honest deliverable for a molecular toolkit is the cost
verdict, which is exactly computable, plus the operators that are not in doubt.

The verdict, from the primary source
------------------------------------
Section V of arXiv:1907.02976, measured on the AE6 molecules:

    "OSE performs much worse than JW on all metrics of comparison ... We
    conclude from this, that OSE is not suited to molecular simulations which
    use standard atomic orbital basis sets."

with Table 2 giving, for STO-3G,

    molecule   Q_OSE   Q_JW
    Silane       156     26
    SiO          182     28
    Sulfur       306     36
    Propyne      342     38

The cause is the interaction graph: in an atom-centred basis almost every pair
of orbitals has a non-zero integral, so the graph is near-complete and
``2E`` is quadratic in the mode count.  ``cost`` reproduces that verdict from
the integrals themselves instead of taking it on trust, and the reproduction is
blunter than the paper: measuring STO-3G H2, LiH, H2O, NH3, CH4, N2, CO and
C2H4 with PySCF, the interaction graph comes out **complete for every one of
them even at a 1e-4 cutoff**, so the spin-orbital count is ``Q_OSE >= 4 * E``
with ``E = C(n, 2)`` -- a ratio of ``n_spatial - 1`` against Jordan-Wigner,
from 5x on LiH to 13x on C2H4.  A complete graph also destroys the term-weight
advantage, because ``deg = n - 1``: measured ``max_weight_ose`` is ``n_so - 3``
for the same set.  For molecules this encoding loses on both metrics.

Where it wins is a bounded-degree coupling graph, and the metric it wins on is
*term weight*, not qubit count: ``A_pq`` has weight at most
``1 + deg(p) + deg(q)``, so on a lattice every term is O(1) instead of an
O(n) Jordan-Wigner string.

On qubit count it does not win, and this module will not pretend otherwise.
The cost is ``2 |E|`` against ``n`` for Jordan-Wigner, so even the sparsest
connected graph -- a tree with ``|E| = n - 1``, the least the encoding can get
away with while every mode is still coupled -- costs ``2 (n - 1)``, which is
more than ``n`` for every ``n >= 3``.  The paper's Table 3 reports
``Q_OSE = 2 (N - 1)`` against ``Q_JW = 2 N`` for a 1-D chain, which is a lower
count than that and uses a different convention for what ``N`` counts; rather
than guess at the convention, ``cost`` computes the number from the graph it
is handed and reports which side it lands on.

References:
    S. B. Bravyi and A. Yu. Kitaev, Ann. Phys. 298, 210 (2002).
    R. W. Chien et al., arXiv:1907.02976.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from superfermion.observables.core import PauliString

__all__ = [
    "chain_edges",
    "directed_edge_register",
    "qubit_count",
    "vertex_operator",
    "edge_operator",
    "interaction_graph",
    "cost",
]


def chain_edges(n_modes: int) -> List[Tuple[int, int]]:
    """Edges of the path graph on ``n_modes`` vertices, i.e. a 1-D chain.

    A path is the sparsest graph on which every mode is still coupled, so
    ``2 (n - 1)`` qubits is the *best* this encoding can do for ``n`` modes.
    That is already worse than Jordan-Wigner's ``n`` for any ``n >= 3``, which
    is why the chain is here as the favourable bound rather than as a
    recommendation.  It is also the graph that makes the term weights smallest.
    """
    n = int(n_modes)
    if n < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    return [(i, i + 1) for i in range(n - 1)]


def _normalise_edges(
    n_modes: int, edges: Iterable[Sequence[int]]
) -> List[Tuple[int, int]]:
    """Validate an undirected edge list: no loops, in range, one entry per pair.

    ``(u, v)`` and ``(v, u)`` are the same edge and a repeated pair is not an
    error: both collapse to one entry, because the encoding spends a qubit pair
    per *undirected* edge and counting an edge twice would overstate it.
    """
    n = int(n_modes)
    if n < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    out = set()
    for edge in edges:
        try:
            u, v = (int(x) for x in edge)
        except (TypeError, ValueError):
            raise ValueError(f"edge {edge!r} is not a pair of integers") from None
        if u == v:
            raise ValueError(f"edge ({u}, {v}) is a self-loop")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(
                f"edge ({u}, {v}) is outside the {n} vertices 0..{n - 1}"
            )
        out.add((min(u, v), max(u, v)))
    return sorted(out)


def _neighbours(n_modes: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    neighbours: List[List[int]] = [[] for _ in range(int(n_modes))]
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)
    return [sorted(nbrs) for nbrs in neighbours]


def directed_edge_register(
    n_modes: int, edges: Iterable[Sequence[int]]
) -> Dict[Tuple[int, int], int]:
    """Qubit index of every *directed* edge.

    One qubit per direction, so an undirected edge gets two.  The register is
    ordered by (tail, head), which makes the indexing reproducible across runs
    -- a labellings-dependent encoding with a nondeterministic qubit order would
    make every later operator string impossible to compare.
    """
    normalised = _normalise_edges(n_modes, edges)
    register: Dict[Tuple[int, int], int] = {}
    for u, v in normalised:
        register[(u, v)] = len(register)
        register[(v, u)] = len(register)
    return register


def qubit_count(n_modes: int, edges: Iterable[Sequence[int]]) -> int:
    """Qubits this encoding needs: two per undirected edge."""
    return 2 * len(_normalise_edges(n_modes, edges))


def vertex_operator(
    n_modes: int, edges: Iterable[Sequence[int]], i: int
) -> PauliString:
    """``B_i = prod_{j in n(i)} Z_(i,j)``.

    ``B_i = 1 - 2 n_i``: it is the occupation parity of mode ``i``, so
    ``(1 - B_i) / 2`` is the number operator.

    Raises:
        ValueError: if vertex ``i`` has no incident edge.  The empty product is
            the identity, and ``(1 - B_i) / 2`` would then assert that the mode
            is *always empty* -- not a harmless default but a wrong operator.
            A mode the graph does not connect cannot be represented at all
            (arXiv:1907.02976, section III.3), so this refuses rather than
            returning something plausible.
    """
    n = int(n_modes)
    normalised = _normalise_edges(n, edges)
    neighbours = _neighbours(n, normalised)
    i = int(i)
    if not 0 <= i < n:
        raise ValueError(f"vertex {i} is outside 0..{n - 1}")
    if not neighbours[i]:
        raise ValueError(
            f"vertex {i} has no incident edge, so mode {i} cannot be "
            "represented by this encoding; add an edge to it or drop it"
        )
    register = directed_edge_register(n, normalised)
    label = ["I"] * len(register)
    for j in neighbours[i]:
        label[register[(i, j)]] = "Z"
    return PauliString("".join(label), coeffs=1.0)


def edge_operator(
    n_modes: int, edges: Iterable[Sequence[int]], p: int, q: int
) -> PauliString:
    """``A_pq`` from arXiv:1907.02976 Table 1, as a single Pauli string.

    ``eps_pq X_(p,q)`` times a ``Z`` on ``(l, p)`` for each neighbour ``l`` of
    ``p`` with ``l < q``, and a ``Z`` on ``(s, q)`` for each neighbour ``s`` of
    ``q`` with ``s < p``.  The Z strings are what make ``A_pq`` commute up to
    the signs the fermionic algebra needs; their length is bounded by the
    degrees, which is the whole point of the encoding.

    Raises:
        ValueError: if ``(p, q)`` is not an edge of the graph.  There is no
            fermionic current between uncoupled modes, and silently returning
            an operator for one would hide a bad graph.
    """
    n = int(n_modes)
    normalised = _normalise_edges(n, edges)
    p, q = int(p), int(q)
    if p == q:
        raise ValueError(f"A_pq needs two distinct vertices, got ({p}, {q})")
    edges_set = set(normalised)
    if (min(p, q), max(p, q)) not in edges_set:
        raise ValueError(
            f"({p}, {q}) is not an edge of this graph, so A_pq does not exist"
        )
    neighbours = _neighbours(n, normalised)
    register = directed_edge_register(n, normalised)

    label = ["I"] * len(register)
    label[register[(p, q)]] = "X"
    for l in neighbours[p]:
        if l < q:
            label[register[(l, p)]] = "Z"
    for s in neighbours[q]:
        if s < p:
            label[register[(s, q)]] = "Z"
    coefficient = 1.0 if p < q else -1.0
    return PauliString("".join(label), coeffs=coefficient)


def interaction_graph(
    h1: Any,
    h2: Optional[Any] = None,
    *,
    threshold: float = 1e-8,
) -> List[Tuple[int, int]]:
    """Undirected edges whose one- or two-electron coupling exceeds ``threshold``.

    An edge exists when ``|h1[i, j]| > threshold`` (a one-body hop) or when any
    ``|h2[i, j, k, l]| > threshold`` for that pair (a two-body coupling needs
    both ``i`` and ``j`` to be present, since the operator annihilates a pair).
    The two-body test is the max over the last two indices, which is why the
    graph becomes near-complete in an atom-centred basis: the number of
    non-negligible integrals grows as ``n^4`` while the number of possible edges
    only grows as ``n^2``.

    Args:
        h1: ``(n, n)`` one-body integrals in the MO basis.
        h2: ``(n, n, n, n)`` chemist-ordered ERIs, or ``None`` for a
            one-body-only model.
        threshold: Integral magnitude below which a coupling is treated as
            absent.  This is the only free parameter, and it is what makes the
            qubit count a property of the *basis*, not of the encoding.

    Returns:
        Sorted ``(i, j)`` pairs with ``i < j``.
    """
    one = np.asarray(h1, dtype=float)
    if one.ndim != 2 or one.shape[0] != one.shape[1]:
        raise ValueError(f"h1 must be square, got {one.shape}")
    n = one.shape[0]
    if float(threshold) < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    # Symmetrise: a graph is undirected, so a coupling reported in one triangle
    # only is still an edge.  Taking abs first keeps the sign out of it.
    coupling = np.maximum(np.abs(one), np.abs(one).T)

    if h2 is not None:
        two = np.asarray(h2, dtype=float)
        if two.shape != (n, n, n, n):
            raise ValueError(f"h2 must be {(n, n, n, n)}, got {two.shape}")
        pair_coupling = np.max(np.abs(two), axis=(2, 3))
        coupling = np.maximum(coupling, np.maximum(pair_coupling, pair_coupling.T))

    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if coupling[i, j] > float(threshold)
    ]
    return edges


def cost(n_modes: int, edges: Iterable[Sequence[int]], *, threshold: float = 1e-8) -> Dict[str, Any]:
    """What this graph costs, against Jordan-Wigner on the same operator.

    Args:
        n_modes: Fermionic modes (spin orbitals).
        edges: Coupling graph, normally from :func:`interaction_graph`.
        threshold: Only recorded in the result, so the number can be reported
            next to the cut-off that produced it.

    Returns:
        Dict with ``n_modes``, ``n_edges``, ``n_isolated``, ``ose_qubits``,
        ``jw_qubits``, ``qubit_ratio`` (``ose / jw``), ``max_weight_ose``
        measured from the operators this module builds, ``max_weight_jw`` the
        worst-case Jordan-Wigner term weight ``n_modes`` (a bound, not a
        measurement), and ``ose_cheaper_on_qubits``.

    ``max_weight_ose`` is measured, not bounded: it is the largest number of
    non-identity characters over every ``A_pq`` and ``B_i`` of the graph.  That
    is the quantity that decides gate depth, and it is the one metric on which
    the encoding can win.

    ``n_isolated`` counts modes with no incident edge.  Those modes cannot be
    represented (see :func:`vertex_operator`), so a non-zero value means the
    graph -- and ``threshold`` -- need revisiting before the counts above mean
    anything.
    """
    n = int(n_modes)
    normalised = _normalise_edges(n, edges)
    register = directed_edge_register(n, normalised)
    neighbours = _neighbours(n, normalised)
    ose_qubits = len(register)
    heaviest = 0
    if normalised:
        for p, q in normalised:
            for operator in (edge_operator(n, normalised, p, q),
                             edge_operator(n, normalised, q, p)):
                heaviest = max(
                    heaviest, sum(1 for ch in operator.pauli_str if ch != "I")
                )
        for i in range(n):
            if not neighbours[i]:
                continue  # unrepresentable; counted in n_isolated instead
            heaviest = max(
                heaviest,
                sum(1 for ch in vertex_operator(n, normalised, i).pauli_str
                    if ch != "I"),
            )
    n_isolated = sum(1 for nbrs in neighbours if not nbrs)
    return {
        "n_modes": n,
        "n_edges": len(normalised),
        "n_isolated": n_isolated,
        "ose_qubits": ose_qubits,
        "jw_qubits": n,
        "qubit_ratio": (ose_qubits / n) if n else 0.0,
        "max_weight_ose": heaviest,
        "max_weight_jw": n,
        "threshold": float(threshold),
        "ose_cheaper_on_qubits": ose_qubits < n,
    }
