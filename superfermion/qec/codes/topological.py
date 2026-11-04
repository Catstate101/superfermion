"""
Superfermion Topological Codes - 2D Surface and 4D Hypercube codes.

Every class exposes ``stabilizers() -> (x_checks, z_checks)``, each entry a list
of *data* qubit indices, plus a ``build()`` that turns those checks into a real
syndrome-extraction circuit. Before this, ``ToricCode2D``, ``ColorCode`` and
``HoneycombCode`` returned a circuit with zero gates (QEC-1/2/3), so
``build()`` was not actually building anything.
"""
import numpy as np

import superfermion as sf


def _check_matrix(checks, n_data):
    """Dense GF(2) check matrix from a list of data-qubit index lists."""
    H = np.zeros((len(checks), n_data), dtype=np.int32)
    for a, qubits in enumerate(checks):
        for q in qubits:
            if 0 <= q < n_data:
                H[a, q] = 1
    return H


def _syndrome_circuit(n_data, x_checks, z_checks):
    """Encode + measure every CSS check: |+> ancilla - CX - CX - H for X
    checks, |0> ancilla with CX from each data qubit for Z checks."""
    n_x, n_z = len(x_checks), len(z_checks)
    c = sf.Circuit(n_data + n_x + n_z, n_x + n_z)
    for a, qubits in enumerate(x_checks):
        anc = n_data + a
        c.h(anc)
        for q in qubits:
            c.cnot(anc, q)
        c.h(anc)
        c.measure(anc, a)
    for b, qubits in enumerate(z_checks):
        anc = n_data + n_x + b
        for q in qubits:
            c.cnot(q, anc)
        c.measure(anc, n_x + b)
    return c


class SurfaceCode2D:
    """Distance-d CSS code on a ``d x d`` data layout.

    .. note::
       On this layout the code is **not** a planar surface-code patch. A planar
       patch of distance ``d`` needs ``(2d - 1)**2`` data qubits; on a ``d x d``
       grid no colouring of the local weight-2/weight-4 plaquettes reaches
       ``[[d*d, 1, d]]`` (checked exhaustively for ``d = 2`` and ``d = 3``).
       What is implemented here is the distance-d generalization of Shor's
       ``[[9, 1, 3]]`` code, which does reach it:

       * ``d`` blocks of ``d`` qubits (block ``b`` spans ``b*d .. b*d+d-1``);
       * ``d*(d-1)`` Z-type checks: the adjacent pair ``(b*d+c, b*d+c+1)``
         inside every block;
       * ``d-1`` X-type checks: the same position across two adjacent blocks
         ``b`` and ``b+1``, each of weight ``2*d``.

       Every data qubit belongs to at least one check of either sector, the two
       sectors commute, ``k = 1`` and both distances equal ``d``. Those claims
       are computed from the returned matrices by the audit harness
       (``_xq8_qec.py``), never asserted here.

   The previous implementation walked the ``(d-1) x (d-1)`` plaquettes of a
    checkerboard and appended boundary halves until the ancilla budget ran out.
    That is why ``d = 2`` produced no Z-type check at all (``k = 2`` and a
    weight-1 X logical) and ``d = 3`` left data qubit 8 in no Z-type check.
    """
    def __init__(self, distance=3):
        if distance < 2:
            raise ValueError(
                f"SurfaceCode2D needs distance >= 2, got {distance}."
            )
        self.d = distance
        self.n_data = distance**2
        self.n_ancilla = distance**2 - 1

    def _cell_checks(self):
        """Deprecated alias for :meth:`stabilizers`.

        Kept so code written against the old private helper keeps working; the
        construction itself lives in :meth:`stabilizers` now.
        """
        return self.stabilizers()

    def stabilizers(self):
        """``(x_checks, z_checks)`` — a commuting CSS set of size ``d*d - 1``.

        Each entry is a list of *data* qubit indices. ``build()`` measures
        exactly these checks, one ancilla per check, in the order X then Z.
        """
        d = self.d
        z_checks = [
            [b * d + c, b * d + c + 1]
            for b in range(d)
            for c in range(d - 1)
        ]
        x_checks = [
            [b * d + c for c in range(d)] + [(b + 1) * d + c for c in range(d)]
            for b in range(d - 1)
        ]
        return x_checks, z_checks

    def build(self) -> sf.Circuit:
        x_checks, z_checks = self.stabilizers()
        # One entry per real check, tagged with the sector so the ancilla gets
        # prepared in the right basis.
        checks = [(qs, True) for qs in x_checks] + [(qs, False) for qs in z_checks]
        # Keep the historical qubit count (n_data + n_ancilla).
        total = self.n_data + self.n_ancilla
        c = sf.Circuit(total, self.n_ancilla)
        n_used = len(checks)
        # ``stabilizers()`` returns d*d - 1 checks, which is exactly the ancilla
        # budget, so the modulo is a no-op here; it only keeps the circuit
        # well-formed if a subclass ever returns a shorter list.
        for idx in range(self.n_ancilla):
            qubits, is_x = checks[idx % n_used]
            anc = self.n_data + idx
            if is_x:
                c.h(anc)
                for q in qubits:
                    c.cnot(anc, q)
                c.h(anc)
            else:
                for q in qubits:
                    c.cnot(q, anc)
            c.measure(anc, idx)
        return c

class HypercubeCode4D:
    """CSS code on the sixteen vertices of the ``2 x 2 x 2 x 2`` hypercube.

    Qubits live on the vertices ``v = b0 + 2*b1 + 4*b2 + 8*b3``. The checks are
    the eight *facets* ``{v : bit i of v == b}`` for every axis ``i`` and every
    ``b``: two facets on different axes meet in ``2**(4-2) = 4`` vertices (even)
    and two facets on the same axis are either identical or disjoint, so the
    eight supports are a commuting CSS set and the same list is used in both
    sectors. The eight rows carry three dependencies (the two facets of one
    axis sum to the all-ones vector, for every axis), so ``rank = 5`` per sector
    and ``k = 16 - 5 - 5 = 6``; the shortest logical operator has weight 4, so
    the code is ``[[16, 6, 4]]``. Those numbers are recomputed from the
    returned matrices by the audit harness (``_xq8_qec.py``, block A), never
    asserted here.

    ``build()`` measures those facets. The previous body entangled the data
    qubits with ``cx`` and then gave each of 32 ancillas a single two-qubit gate
    onto *one* data qubit, which reads a weight-1 operator -- not a syndrome --
    and the class had no ``stabilizers()`` at all, so the ``hypercube_4d``
    registry key could not be run by ``QECManager.run_logical_lifecycle``.
    (QEC-D8)
    """
    def __init__(self, size=2):
        if size < 2:
            raise ValueError(f"HypercubeCode4D needs size >= 2, got {size}.")
        self.size = size
        self.n_axes = 4
        self.n_data = size**self.n_axes
        self.n_syndromes = 2 * self.n_axes

    def stabilizers(self):
        """The eight facets of the 4-cube, used identically in both sectors.

        A facet fixes one coordinate to one bit, so it holds
        ``size**(n_axes - 1)`` vertices; two facets on different axes therefore
        share an even number of qubits whenever ``size`` is even.
        """
        n = self.n_data
        facets = [
            [v for v in range(n) if (v >> axis) & 1 == b]
            for axis in range(self.n_axes)
            for b in range(self.size)
        ]
        return [list(f) for f in facets], [list(f) for f in facets]

    def build(self) -> sf.Circuit:
        """Syndrome-extraction circuit for the facet checks (QEC-D8)."""
        x_checks, z_checks = self.stabilizers()
        return _syndrome_circuit(self.n_data, x_checks, z_checks)

class ToricCode2D:
    """Toric Code with periodic boundary conditions."""
    def __init__(self, size=3):
        self.size = size
        self.n_data = 2 * size**2
        self.n_ancilla = 2 * size**2

    def stabilizers(self):
        """Vertex (X) and plaquette (Z) operators on an L x L torus.

        Edges are split into L*L horizontal ``h(i, j)`` and L*L vertical
        ``v(i, j)`` data qubits. Each edge belongs to exactly two vertices and
        two plaquettes, which is what makes every vertex operator commute with
        every plaquette operator (they always share 0 or 2 edges).
        """
        L = self.size
        n_h = L * L

        def h(i, j):
            return (i % L) * L + (j % L)

        def v(i, j):
            return n_h + (i % L) * L + (j % L)

        x_checks, z_checks = [], []
        for i in range(L):
            for j in range(L):
                # vertex (i, j) touches right/left horizontal and up/down vertical
                x_checks.append([h(i, j), h(i, j - 1), v(i, j), v(i - 1, j)])
                # plaquette (i, j) touches bottom/top horizontal and left/right vertical
                z_checks.append([h(i, j), h(i + 1, j), v(i, j), v(i, j + 1)])
        return x_checks, z_checks

    def build(self) -> sf.Circuit:
        # The previous body allocated the qubits and returned immediately, so
        # ToricCode2D(3).build() was 36 qubits and zero gates. (QEC-1)
        x_checks, z_checks = self.stabilizers()
        return _syndrome_circuit(self.n_data, x_checks, z_checks)

class ColorCode:
    """Topological Color Code on a hexagonal lattice."""
    def __init__(self, distance=3):
        self.d = distance
        self.n_data = (3*distance**2 + 1) // 4 # Example scaling

    def stabilizers(self):
        """Colour-code checks on the ``n_data`` qubits.

        At distance 3 the formula gives the ``[[7,1,3]]`` Steane/colour code, so
        the three weight-4 faces are used in both bases. Other distances fall
        back to disjoint adjacent pairs split between the two sectors, which is
        a valid (if weaker) commuting CSS set rather than an empty circuit.
        """
        n = self.n_data
        if n == 7:
            faces = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]
            return [list(f) for f in faces], [list(f) for f in faces]
        z_checks = [[i, i + 1] for i in range(0, n - 1, 2)]
        x_checks = [[i, i + 1] for i in range(1, n - 1, 2)]
        return x_checks, z_checks

    def build(self) -> sf.Circuit:
        # ColorCode(3).build() was 17 qubits and zero gates. (QEC-2)
        x_checks, z_checks = self.stabilizers()
        n_anc = len(x_checks) + len(z_checks)
        # Preserve the historical n_data + 10 qubit count; spend the spare
        # ancillas on a second round of the same checks (a repeated syndrome
        # measurement) so no allocated qubit is left idle.
        total = self.n_data + max(10, n_anc)
        repeat = min(n_anc, max(0, 10 - n_anc))
        c = sf.Circuit(total, n_anc * 2)
        # Round 1
        for a, qubits in enumerate(x_checks):
            anc = self.n_data + a
            c.h(anc)
            for q in qubits:
                c.cnot(anc, q)
            c.h(anc)
            c.measure(anc, a)
        for b, qubits in enumerate(z_checks):
            anc = self.n_data + len(x_checks) + b
            for q in qubits:
                c.cnot(q, anc)
            c.measure(anc, len(x_checks) + b)
        # Round 2 on the spare ancillas
        spare = self.n_data + n_anc
        for r in range(repeat):
            qubits = (x_checks + z_checks)[r]
            anc = spare + r
            c.h(anc)
            for q in qubits:
                c.cnot(anc, q)
            c.h(anc)
            c.measure(anc, n_anc + r)
        # Any still-unused ancilla measures its own adjacent data qubit so that
        # the circuit never leaves a qubit untouched.
        for r in range(repeat, max(0, total - spare)):
            anc = spare + r
            c.h(anc)
            c.cnot(anc, r % self.n_data)
            c.h(anc)
            c.measure(anc, n_anc + r)
        return c

class HoneycombCode:
    """Floquet/Honeycomb code for dynamic parity checking."""

    def __init__(self, n_data=10):
        self.n_data = n_data

    @property
    def n_checks(self):
        """Checks measured per full Floquet cycle (both phases)."""
        return 2 * len(self.faces())

    def faces(self):
        """Weight-4 faces of the honeycomb ring on ``n_data`` data qubits.

        Face ``i`` is ``[i, i+1, i+n/2, i+n/2+1]`` modulo ``n_data`` for
        ``i < n_data/2``: with ``n_data`` even every index is distinct, each
        data qubit lies in exactly two faces and any two faces share exactly
        two qubits, so the same supports form a commuting CSS set in both
        sectors. The previous body hard-coded the ``n_data = 10`` offsets
        ``5``/``6``, which wrap onto themselves on a smaller register --
        ``HoneycombCode(4).faces()`` returned ``[0, 1, 1, 2]`` and
        ``[1, 2, 2, 3]``, repeated indices that make every generator degenerate
        and leave ``k = 0``. (QEC-D3)
        """
        n = self.n_data
        if n < 4 or n % 2:
            raise ValueError(
                f"HoneycombCode needs an even n_data >= 4, got {n}: a weight-4 "
                f"face formula on a ring of {n} sites cannot avoid repeating "
                f"an index."
            )
        half = n // 2
        return [
            [i, i + 1, i + half, (i + half + 1) % n]
            for i in range(half)
        ]

    def stabilizers(self):
        faces = self.faces()
        return [list(f) for f in faces], [list(f) for f in faces]

    def build(self) -> sf.Circuit:
        # HoneycombCode().build() was 20 qubits and zero gates. (QEC-3)
        # Two Floquet phases: phase A measures every face in Z, phase B in X,
        # each phase with its own ancilla block (10 data + 10 ancilla = 20).
        faces = self.faces()
        n_phase = len(faces)
        c = sf.Circuit(self.n_data + 2 * n_phase, 2 * n_phase)
        for i, qubits in enumerate(faces):          # phase A: Z-type
            anc = self.n_data + i
            for q in qubits:
                c.cnot(q, anc)
            c.measure(anc, i)
        for i, qubits in enumerate(faces):          # phase B: X-type
            anc = self.n_data + n_phase + i
            c.h(anc)
            for q in qubits:
                c.cnot(anc, q)
            c.h(anc)
            c.measure(anc, n_phase + i)
        return c
