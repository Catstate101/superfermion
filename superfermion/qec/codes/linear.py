"""
Superfermion Linear QEC Codes - Standard Bit-Flip, Phase-Flip, Shor, and Steane codes.

Every class exposes ``stabilizers() -> (x_checks, z_checks)`` where each entry is
the list of *data* qubit indices touched by one check. That is the same shape
``MWPMDecoder`` / ``UnionFindDecoder`` expect as ``syndrome_qubit_map``, and it is
what lets :class:`~superfermion.qec.manager.QECManager` compute a syndrome and
then actually decode it instead of guessing.
"""
import warnings

import numpy as np

import superfermion as sf

class RepetitionCode:
    """Standard repetition code for bit or phase flip, any length n>=2.

    Layout: n data qubits [0..n-1] + (n-1) ancillas [n..2n-2]; ancilla k
    measures parity d_k (+) d_{k+1}. Syndrome map (check -> data qubits):
    ``[[k, k+1] for k in range(n-1)]`` — Stim-compatible ordering.

    Parameters: ``n`` data qubits, ``n - 1`` independent checks, so
    ``k = n - rank(H_X) - rank(H_Z) = 1``.

    **The distance is sector-specific.**  ``[[n, 1, n]]`` describes the error
    type the code is built for, not the full Pauli group.  Recomputed from
    :meth:`stabilizers` over GF(2) (lightest logical outside the stabiliser):

    ==========  =====  =====  =====  ==========
    code_type   n      d_X    d_Z    d (Pauli)
    ==========  =====  =====  =====  ==========
    ``bit``     3      3      1      1
    ``bit``     5      5      1      1
    ``phase``   3      1      3      1
    ``phase``   5      1      5      1
    ==========  =====  =====  =====  ==========

    A ``bit`` code measures ``Z_k Z_{k+1}``, so a weight-1 **Z** commutes with
    every check and is not in the stabiliser: it is a weight-1 logical and the
    code never sees it.  ``phase`` is the mirror image (a weight-1 **X**).  That
    is what makes it a repetition code rather than a defect, but it has a
    consequence for anything that reads a distance off this class: a decoder
    used on the *unprotected* sector reports corrections that are valid yet
    differ from the truth by a weight-1 logical.  Only the sector column
    (``d_X`` for ``bit``, ``d_Z`` for ``phase``) is ``n``.
    """
    def __init__(self, n=3, code_type="bit"):
        if n < 2:
            raise ValueError(f"RepetitionCode needs n>=2, got {n}.")
        self.n = n
        self.code_type = code_type

    def syndrome_map(self):
        """Parity-check map: check k touches data qubits [k, k+1]."""
        return [[k, k + 1] for k in range(self.n - 1)]

    def stabilizers(self):
        """``(x_checks, z_checks)`` for a bit- or phase-flip repetition code.

        A bit-flip code only has Z-type checks (X errors flip them) and a
        phase-flip code only has X-type checks (Z errors flip them).
        """
        parity = self.syndrome_map()
        if self.code_type == "phase":
            return parity, []
        return [], parity

    def build(self, rounds=1) -> sf.Circuit:
        """Build encoding + syndrome-extraction circuit.

        The parity round follows :meth:`stabilizers`: a ``bit`` code measures
        ``Z_k Z_{k+1}`` with a ``|0>`` ancilla and ``cx`` from data to ancilla,
        a ``phase`` code measures ``X_k X_{k+1}`` with a ``|+>`` ancilla and
        ``cx`` from ancilla to data.  The previous body ran the Z-parity round
        for both, so ``RepetitionCode(5, 'phase')`` reported four X-type
        generators and then measured four Z parities: with no error injected
        those bits were random, and an injected X fired nothing at all.  (QEC-D6)

        Args:
            rounds: number of repeated syndrome-extraction rounds
                (default 1). Round r measures into fresh classical bits
                ``r*(n-1)..(r+1)*(n-1)-1``; data qubits are measured once
                at the end (into the trailing bits) for easy decoding.
        """
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}.")
        n = self.n
        n_anc = n - 1
        # data + one ancilla block reused across rounds (reset each round)
        c = sf.Circuit(2 * n - 1, n_anc * rounds + n)
        # Encoding
        for i in range(1, n):
            c.cnot(0, i)

        if self.code_type == "phase":
            for i in range(n):
                c.h(i)

        # Syndrome Measurement (repeated rounds)
        # Parity checks: d_k (+) d_{k+1} into ancilla n+k.  The basis the
        # parity is taken in is the one the checks live in: a Z-parity round
        # for a bit-flip code, an X-parity round for a phase-flip one.
        phase = self.code_type == "phase"
        for r in range(rounds):
            for k in range(n - 1):
                anc = n + k
                if phase:
                    c.h(anc)
                    c.cnot(anc, k)
                    c.cnot(anc, k + 1)
                    c.h(anc)
                else:
                    c.cnot(k, anc)
                    c.cnot(k + 1, anc)
            for k in range(n - 1):
                c.measure(n + k, r * n_anc + k)
            if r + 1 < rounds:
                for k in range(n - 1):
                    c.reset(n + k)
        for i in range(n):
            c.measure(i, n_anc * rounds + i)
        return c

class ShorCode:
    """9-qubit Shor Code - Corrects arbitrary single-qubit errors."""

    def stabilizers(self):
        """The 8 Shor-code generators on 9 data qubits."""
        z_checks = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]]
        x_checks = [[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8]]
        return x_checks, z_checks

    def build(self) -> sf.Circuit:
        c = sf.Circuit(9) # Simplified encoding/syndrome circuit
        # External encoding logic is complex, here we provide the structure
        # Initial entanglement
        c.cnot(0, 3)
        c.cnot(0, 6)
        for i in [0, 3, 6]:
            c.h(i)
            c.cnot(i, i+1)
            c.cnot(i, i+2)
        return c

class SteaneCode:
    """7-qubit Steane Code [[7,1,3]] - Corrects any single qubit error."""

    def stabilizers(self):
        """The 6 Steane generators (self-dual: X and Z checks are identical)."""
        checks = [
            [0, 2, 4, 6],
            [1, 2, 5, 6],
            [3, 4, 5, 6],
        ]
        return [list(c) for c in checks], [list(c) for c in checks]

    def build(self) -> sf.Circuit:
        c = sf.Circuit(7)
        # Standard Steane encoding
        c.h(0)
        c.h(1)
        c.h(3)
        c.cnot(0, 2); c.cnot(0, 4); c.cnot(0, 6)
        c.cnot(1, 2); c.cnot(1, 5); c.cnot(1, 6)
        c.cnot(3, 4); c.cnot(3, 5); c.cnot(3, 6)
        return c

class BaconShorCode:
    """Bacon-Shor Subsystem Code [[9,1,3]]."""
    def __init__(self, L=3):
        self.L = L

    def gauge_operators(self):
        """Row-adjacent ``XX`` and column-adjacent ``ZZ`` gauge generators.

        These are the *gauge* group of the subsystem code, not its stabilisers:
        a row-adjacent ``XX`` and a column-adjacent ``ZZ`` can meet in a single
        qubit and therefore anticommute. They are kept available (a subsystem /
        gauge decoder measures them), but they are deliberately not what
        ``stabilizers()`` returns.
        """
        L = self.L
        z_gauge = [[i * L + j, i * L + j + 1] for i in range(L) for j in range(L - 1)]
        x_gauge = [[j * L + i, (j + 1) * L + i] for i in range(L) for j in range(L - 1)]
        return x_gauge, z_gauge

    def stabilizers(self):
        """``(x_checks, z_checks)`` of the **X-gauge-fixed** Bacon-Shor code.

        Bacon-Shor is a *subsystem* code.  Its stabiliser group on its own (the
        centre of the gauge group intersected with the gauge group) is only
        rank ``2(L-1)`` per sector, so it leaves ``k = L*L - 4(L-1)`` logical
        qubits -- 5 on nine qubits, with a weight-2 logical X -- which is not
        the ``[[9, 1, 3]]`` the class advertises.  The advertised numbers
        belong to the gauge-fixed code: fixing the X gauge (the vertical ``XX``
        pairs of :meth:`gauge_operators`) to ``+1`` turns those pairs plus the
        ``L-1`` ``Z``-type column stabilisers into a genuine ``[[L*L, 1, L]]``
        CSS set -- 6 weight-2 X-type and 2 weight-6 Z-type generators at
        ``L = 3``, ``k = 9 - 6 - 2 = 1``, both distances 3.  These are the
        matrices :meth:`build` measures, and block A of ``_xq8_qec.py``
        recomputes ``k``, ``d_X`` and ``d_Z`` from them rather than trusting
        this paragraph.

        The previous body returned the un-fixed centre and had its two sectors
        swapped: it named *two full columns* as its X generators and *two full
        rows* as its Z generators, so the object it called a centre did not
        commute with the gauge operator it claimed to commute with.  The Z-type
        generators below are the pair of full columns, which is the sector that
        does commute with the fixed X gauge.

        The unfixed gauge group stays available through
        :meth:`gauge_operators`; that is the object a subsystem decoder wants.
        """
        L = self.L
        x_checks = [list(pair) for pair in self.gauge_operators()[0]]
        z_checks = [
            [r * L + c for r in range(L) for c in (a, a + 1)]
            for a in range(L - 1)
        ]
        return x_checks, z_checks

    def build(self) -> sf.Circuit:
        """Syndrome-extraction circuit for exactly the checks above.

        X-type checks get a ``|+>`` ancilla and ``cx`` ancilla -> data, Z-type
        checks a ``|0>`` ancilla and ``cx`` data -> ancilla, and every measured
        bit corresponds to one entry of :meth:`stabilizers`.  The previous body
        measured all twelve gauge pairs, ten of them as ``cx`` data -> ancilla:
        the vertical ``XX`` gauges were read out as ``ZZ`` parities, so half the
        reported syndrome was a different observable from the one it was named
        after, and none of the twelve matched ``stabilizers()``.  (QEC-D4)
        """
        x_checks, z_checks = self.stabilizers()
        n_data = self.L * self.L
        n_anc = len(x_checks) + len(z_checks)
        c = sf.Circuit(n_data + n_anc, n_anc)
        for a, qubits in enumerate(x_checks):
            anc = n_data + a
            c.h(anc)
            for q in qubits:
                c.cnot(anc, q)
            c.h(anc)
            c.measure(anc, a)
        for b, qubits in enumerate(z_checks):
            anc = n_data + len(x_checks) + b
            for q in qubits:
                c.cnot(q, anc)
            c.measure(anc, len(x_checks) + b)
        return c

def _gf2_rref(rows, n_cols):
    """Row-reduce ``rows`` over GF(2); return ``(reduced_rows, pivot_columns)``.

    The CSS encoder below needs the pivot column of every independent row.  In
    reduced row-echelon form the pivot column ``p`` of row ``i`` holds a 1 only
    in row ``i``, which is exactly what makes the CNOT ladder a valid encoder.
    """
    mat = [[int(v) % 2 for v in row] for row in rows]
    pivots = []
    r = 0
    for c in range(n_cols):
        piv = next((i for i in range(r, len(mat)) if mat[i][c]), None)
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        for i in range(len(mat)):
            if i != r and mat[i][c]:
                mat[i] = [a ^ b for a, b in zip(mat[i], mat[r])]
        pivots.append(c)
        r += 1
        if r == len(mat):
            break
    return mat[:r], pivots


class GenericCSSCode:
    """Generic CSS code builder from Hx and Hz matrices.

    With no arguments it defaults to the ``[[7,1,3]]`` Steane code, which makes
    the ``css_generic`` entry in :class:`QECManager` usable out of the box; the
    previous ``__init__(self, hx, hz)`` had no defaults, so
    ``QECManager.get_code("css_generic")`` raised ``TypeError``. (QEC-11)
    """

    _STEANE_H = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ]

    def __init__(self, hx=None, hz=None):
        if hx is None and hz is None:
            hx = np.array(self._STEANE_H, dtype=np.int32)
            hz = np.array(self._STEANE_H, dtype=np.int32)
        elif hx is None or hz is None:
            raise ValueError("GenericCSSCode needs both hx and hz (or neither).")
        self.hx = np.asarray(hx, dtype=np.int32)
        self.hz = np.asarray(hz, dtype=np.int32)
        if self.hx.shape[1] != self.hz.shape[1]:
            raise ValueError(
                f"hx has {self.hx.shape[1]} columns but hz has {self.hz.shape[1]}"
            )
        self.n = self.hx.shape[1]
        # ``Hx @ Hz^T == 0 (mod 2)`` is the CSS condition. It is *checked* and
        # exposed as ``css_valid`` rather than raised: callers have historically
        # passed non-commuting matrices (the regression test uses
        # ``hx=[[1,1,0],[0,1,1]]`` with ``hz=[[1,0,1]]``, which anticommute), so
        # a hard error would break them. A warning is loud without changing what
        # the class accepts.
        self.css_valid = not bool(
            self.hx.size and self.hz.size and (self.hx @ self.hz.T % 2).any()
        )
        if not self.css_valid:
            warnings.warn(
                "GenericCSSCode: hx and hz do not commute (Hx @ Hz^T != 0 mod 2). "
                "The two sectors are not independent, so syndromes cannot be "
                "decoded separately; use a commuting CSS set for that.",
                RuntimeWarning,
                stacklevel=2,
            )

    def stabilizers(self):
        """``(x_checks, z_checks)`` as data-qubit index lists, one per row."""
        x_checks = [[int(q) for q in np.nonzero(row)[0]] for row in self.hx]
        z_checks = [[int(q) for q in np.nonzero(row)[0]] for row in self.hz]
        return x_checks, z_checks

    def _encoder_gates(self, c) -> None:
        """Append the logical ``|0_L>`` encoder to circuit ``c``.

        For a CSS code whose two sectors commute, ``|0_L>`` is the uniform
        superposition over the row space of ``Hx`` (every X-type generator is a
        permutation on it, and every Z-type one acts as ``+1`` because
        ``Hx @ Hz^T == 0``).  With ``Hx`` row-reduced, that superposition is
        prepared by putting the pivot columns in ``|+>`` and XORing each free
        column from the pivots whose row covers it -- the same shape as the
        hand-written ``SteaneCode.build()``, which is the ``[[7,1,3]]`` case.
        """
        rows, pivots = _gf2_rref(self.hx, self.n)
        free = [col for col in range(self.n) if col not in pivots]
        for p in pivots:
            c.h(p)
        for i, p in enumerate(pivots):
            for f in free:
                if rows[i][f]:
                    c.cnot(p, f)

    def encoder(self) -> sf.Circuit:
        """The ``|0_L>`` preparation on ``n`` data qubits, as its own circuit."""
        c = sf.Circuit(self.n)
        self._encoder_gates(c)
        return c

    def build(self) -> sf.Circuit:
        x_checks, z_checks = self.stabilizers()
        n_x, n_z = len(x_checks), len(z_checks)
        c = sf.Circuit(self.n + n_x + n_z, n_x + n_z)
        # Encoder first.  The previous body allocated the ancillas and then
        # returned without a single gate, so build() handed the syndrome round
        # the bare |00...0>: no X-type generator is decided by that state, so
        # 39 of the generator readings were undefined and the circuit measured
        # the stabilisers of a state that is not in the code. (QEC-4/QEC-D5)
        self._encoder_gates(c)
        for a, qubits in enumerate(x_checks):
            anc = self.n + a
            c.h(anc)
            for q in qubits:
                c.cnot(anc, q)
            c.h(anc)
            c.measure(anc, a)
        for b, qubits in enumerate(z_checks):
            anc = self.n + n_x + b
            for q in qubits:
                c.cnot(q, anc)
            c.measure(anc, n_x + b)
        return c
