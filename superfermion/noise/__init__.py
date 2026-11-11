"""
Noise Models — Quantum channel simulation for realistic circuit execution.

Unified noise model supporting both statevector (JAX) and density matrix
(Kraus operator) simulation paths.

Usage:
    noise = sf.NoiseModel()
    noise.add_depolarizing(0.01)
    noise.add_readout_error(0.02)
    result = sf.run(circuit, method="density_matrix", noise_model=noise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np



_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kraus_depolarizing_1q(p: float) -> List[np.ndarray]:
    """Single-qubit depolarizing: rho -> (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ)"""
    return [
        math.sqrt(1 - p) * _I2,
        math.sqrt(p / 3) * _X,
        math.sqrt(p / 3) * _Y,
        math.sqrt(p / 3) * _Z,
    ]


def kraus_depolarizing_2q(p: float) -> List[np.ndarray]:
    """Two-qubit depolarizing channel with 15 non-identity Pauli terms."""
    paulis_1q = [_I2, _X, _Y, _Z]
    kraus = [math.sqrt(1 - p) * np.kron(_I2, _I2)]
    for a in paulis_1q:
        for b in paulis_1q:
            if not (np.allclose(a, _I2) and np.allclose(b, _I2)):
                kraus.append(math.sqrt(p / 15) * np.kron(a, b))
    return kraus


def kraus_amplitude_damping(gamma: float) -> List[np.ndarray]:
    """Amplitude damping (T1 decay): K0 = diag(1, sqrt(1-g)), K1 = [[0,sqrt(g)],[0,0]]"""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128),
    ]


def kraus_phase_damping(gamma: float) -> List[np.ndarray]:
    """Phase damping (T2 dephasing): K0 = diag(1, sqrt(1-g)), K1 = diag(0, sqrt(g))"""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, 0], [0, math.sqrt(gamma)]], dtype=np.complex128),
    ]


def kraus_bit_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _X]


def kraus_phase_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _Z]


def kraus_bit_phase_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _Y]


def kraus_reset(p0: float, p1: float = 0.0) -> List[np.ndarray]:
    """Reset channel: ``rho -> (1-p0-p1) rho + p0 |0><0| Tr(rho) + p1 |1><1| Tr(rho)``.

    The two reset terms are non-unital: they cannot be written as a single
    ``K rho K^dagger``, so each expands into one Kraus operator per input
    basis state (``|0><i|`` for the p0 term, ``|1><i|`` for the p1 term).
    The five-term set below is trace preserving by construction and matches
    ``qiskit_aer.noise.reset_error(p0, p1)`` (Aer builds it from the
    circuits ``id`` / ``reset`` / ``reset;x`` with probabilities
    ``1-p0-p1`` / ``p0`` / ``p1``; measured: identical superoperator).

    Register it with :meth:`NoiseModel.add_reset_error`, which attaches it to
    the RESET instruction.
    """
    if p0 < 0 or p1 < 0 or p0 + p1 > 1:
        raise ValueError(
            f"invalid reset probabilities: p0={p0!r}, p1={p1!r} "
            "(both >= 0 and p0 + p1 <= 1)"
        )
    keep = math.sqrt(max(0.0, 1.0 - p0 - p1))
    k00 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    k01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    k10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    k11 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    return [
        keep * _I2,
        math.sqrt(p0) * k00,
        math.sqrt(p0) * k01,
        math.sqrt(p1) * k10,
        math.sqrt(p1) * k11,
    ]


def kraus_leakage(p: float) -> List[np.ndarray]:
    """Leakage channel: move population from |0> and |1> into the |2> level.

    ``K0 = diag(sqrt(1-p), sqrt(1-p), 1)``, ``K1 = sqrt(p) |2><0|`` and
    ``K2 = sqrt(p) |2><1|`` -- trace preserving, with |2> absorbing (no decay
    back).  This is the first-order transmon leakage model used in Qiskit's
    leakage examples.  Register it with :meth:`NoiseModel.add_leakage`, which
    embeds the three levels on a ``(data, flag)`` qubit pair.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"leakage probability must be in [0, 1]; got {p!r}")
    keep = math.sqrt(1.0 - p)
    to_2_from_0 = np.zeros((3, 3), dtype=np.complex128)
    to_2_from_0[2, 0] = 1.0
    to_2_from_1 = np.zeros((3, 3), dtype=np.complex128)
    to_2_from_1[2, 1] = 1.0
    return [
        np.diag([keep, keep, 1.0]).astype(np.complex128),
        math.sqrt(p) * to_2_from_0,
        math.sqrt(p) * to_2_from_1,
    ]


def _compose_kraus(second: List[np.ndarray], first: List[np.ndarray]) -> List[np.ndarray]:
    """Kraus set of ``first`` followed by ``second`` (physical application order).

    Same element order as the historical composition inside
    :meth:`NoiseModel.to_rust_kraus_ops` (``L @ K`` for each ``K`` of
    ``first`` and each ``L`` of ``second``), so composing one pair
    reproduces that loop bit-for-bit.
    """
    return [L @ K for K in first for L in second]


def kraus_thermal_relaxation(
    t1: float, t2: float, gate_time: float
) -> List[np.ndarray]:
    """Thermal relaxation (T1/T2) channel for a gate of duration ``gate_time``.

    Zero-temperature bath: populations decay as ``exp(-t/T1)`` and
    coherences as ``exp(-t/T2)``, built exactly as amplitude damping
    (``g1 = 1 - exp(-t/T1)``) followed by pure dephasing
    (``g_phi = 1 - exp(-2t/T2 + t/T1)``).  The composite Kraus set equals
    ``qiskit_aer.noise.thermal_relaxation_error(t1, t2, t)`` at machine
    precision (measured max |dChoi| <= 3.4e-16 on three parameter sets).
    ``T2 > 2*T1`` is rejected, matching Aer's ``NoiseError``.
    """
    if t1 <= 0 or t2 <= 0:
        raise ValueError(f"t1 and t2 must be positive; got t1={t1!r}, t2={t2!r}")
    if gate_time < 0:
        raise ValueError(f"gate_time must be non-negative; got {gate_time!r}")
    if t2 > 2 * t1:
        raise ValueError(
            "invalid T2 relaxation time parameter: T2 greater than 2 * T1 "
            f"(got t1={t1!r}, t2={t2!r})"
        )
    g1 = 1.0 - math.exp(-gate_time / t1)
    g_phi = 1.0 - math.exp(-2.0 * gate_time / t2 + gate_time / t1)
    return _compose_kraus(kraus_phase_damping(g_phi), kraus_amplitude_damping(g1))


def _flat_kraus(kraus_set: List[np.ndarray], dim: int) -> List[float]:
    """Row-major re/im flattened Kraus set (the Rust binding's layout)."""
    flat: List[float] = []
    for K in kraus_set:
        for r in range(dim):
            for c in range(dim):
                flat.append(float(K[r, c].real))
                flat.append(float(K[r, c].imag))
    return flat


#: Wire swap of a 4x4 operator's basis: |00>,|01>,|10>,|11> -> |00>,|10>,|01>,|11>.
_TWO_QUBIT_SWAP = np.array([0, 2, 1, 3])


def _qiskit_order_2q(matrix: np.ndarray) -> np.ndarray:
    """Re-order a 4x4 channel matrix from Qiskit's wire order to SF's kernel.

    Qiskit treats the *first* qubit of a gate/error argument list as the least
    significant (``Operator(circuit.x(0)) == kron(I, X)``), while the SF
    density-matrix kernels treat the first wire of a local 2-qubit block as the
    most significant.  A matrix written Qiskit-side as ``kron(A, B)`` on
    ``[a, b]`` therefore acts with ``A`` on ``b``; the SF kernel would put ``A``
    on ``a``.  Conjugating by the wire-swap permutation maps one to the other
    (``M' = P M P``, i.e. ``M'[i, j] = M[p(i), p(j)]``), so a channel registered
    as ``qubits=(a, b)`` realises exactly the Qiskit error measured against Aer.

    Channels that are symmetric under the swap (the 15-Pauli depolarizing set,
    ``Z (x) Z``) are returned unchanged up to floating point, which is why the
    pre-existing 2-qubit adders are unaffected by this conversion.
    """
    a = np.asarray(matrix)
    if a.shape != (4, 4):
        return a
    return a[np.ix_(_TWO_QUBIT_SWAP, _TWO_QUBIT_SWAP)].copy()


def _normalize_targets(qubits: Any, n_qubits: int) -> Optional[Tuple[int, ...]]:
    """Validate/normalize the ``qubits=`` target of an ``add_*`` call.

    ``None`` (all qubits / all pairs) passes through.  A 1-qubit channel
    takes an int or a sequence of ints (deduplicated, first-occurrence
    order); a 2-qubit channel takes exactly one ordered pair.
    """
    if qubits is None:
        return None
    if isinstance(qubits, (int, np.integer)):
        seq = [int(qubits)]
    else:
        try:
            seq = [int(q) for q in qubits]
        except TypeError as exc:
            raise TypeError(
                f"qubits must be None, an int, or a sequence of ints; got {qubits!r}"
            ) from exc
    if any(q < 0 for q in seq):
        raise ValueError(f"qubits must be non-negative; got {qubits!r}")
    if n_qubits == 1:
        if not seq:
            raise ValueError("qubits must not be empty (pass None for all qubits)")
        unique: List[int] = []
        for q in seq:
            if q not in unique:
                unique.append(q)
        return tuple(unique)
    if len(seq) != 2 or seq[0] == seq[1]:
        raise ValueError(
            "a two-qubit channel targets exactly one ordered pair; pass "
            f"qubits=(a, b) and call the method once per pair; got {qubits!r}"
        )
    return (seq[0], seq[1])


#: Canonical instruction names accepted by ``instructions=``.  These are the
#: exact strings ``OpType::name()`` returns in the Rust core; the comparison at
#: simulation time is case-insensitive, and gates that SF lowers before
#: execution (e.g. ``u1``/``u2``/``u3``) must be named by what actually runs.
_INSTRUCTION_ALIASES: Dict[str, str] = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "Sdg",
    "t": "T",
    "tdg": "Tdg",
    "sx": "SX",
    "sxdg": "SXdg",
    "id": "Id",
    "i": "Id",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
    "r1": "R1",
    "p": "P",
    "u": "U",
    "cu": "Cu",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "cy": "CY",
    "swap": "SWAP",
    "iswap": "iSWAP",
    "ecr": "ECR",
    "rzz": "Rzz",
    "rxx": "Rxx",
    "ryy": "Ryy",
    "crx": "CRx",
    "crz": "CRz",
    "cp": "CP",
    "cphase": "CP",
    "ccx": "CCX",
    "toffoli": "CCX",
    "cswap": "CSWAP",
    "fredkin": "CSWAP",
    "measure": "Measure",
    "reset": "Reset",
    "barrier": "Barrier",
    "unitary": "UNITARY",
}

#: Instruction filter a qutrit / leakage channel fires on by default: every
#: canonical gate name, but not ``Measure``, ``Reset`` or ``Barrier`` -- they
#: do not drive the transmon, and a kick after a terminal measurement would
#: perturb the state that is returned to the caller.
_QUTRIT_DEFAULT_INSTRUCTIONS: Tuple[str, ...] = tuple(
    sorted(set(_INSTRUCTION_ALIASES.values()) - {"Measure", "Reset", "Barrier"})
)


def _normalize_instructions(instructions: Any) -> Optional[Tuple[str, ...]]:
    """``None`` (every instruction) or a tuple of canonical instruction names.

    Accepts a single name or a sequence of names, in any letter case.  Gates
    SF decomposes before execution (``u1``, ``u2``, ``u3``, ``cry``, ``ch``)
    are rejected by name because the simulated instruction stream no longer
    contains them: the channel would never fire.
    """
    if instructions is None:
        return None
    if isinstance(instructions, str):
        instructions = (instructions,)
    names: List[str] = []
    for raw in instructions:
        if not isinstance(raw, str):
            raise TypeError(
                f"instruction names must be strings; got {type(raw).__name__}"
            )
        key = raw.strip().lower()
        if key in ("u1", "u2", "u3", "cry", "ch"):
            raise ValueError(
                f"instruction {raw!r} is decomposed before simulation, so a "
                "channel filtered on it would never fire; target the lowered "
                "gates instead (e.g. 'p', 'u', 'rz', 'ry', 'cx')"
            )
        canon = _INSTRUCTION_ALIASES.get(key)
        if canon is None:
            supported = ", ".join(sorted(set(_INSTRUCTION_ALIASES.values())))
            raise ValueError(
                f"unknown instruction {raw!r}; supported names: {supported}"
            )
        if canon not in names:
            names.append(canon)
    if not names:
        raise ValueError("instructions must not be empty (pass None for every gate)")
    return tuple(names)


class ErrorChannel:
    """A reusable, composable quantum error: a 1- or 2-qubit Kraus set.

    Mirrors the part of Qiskit's ``QuantumError`` the density-matrix engine can
    express -- ``compose`` (self after other), ``tensor`` (two wires) and
    ``power`` (n-fold repetition).  Build one with a class method and register
    it on a model with :meth:`NoiseModel.add_quantum_error`, which is what the
    ``add_kraus`` / ``add_unitary`` / ``add_mixed_unitary`` / ``add_reset_error``
    helpers do internally.

    Kraus operators are validated at construction: wrong shape or a
    non-trace-preserving set raises ``ValueError`` (as Aer's ``kraus_error``
    does), so an invalid channel can never be registered silently.
    """

    def __init__(
        self,
        kraus: Any,
        n_qubits: int,
        name: str = "custom",
        error_rate: float = 0.0,
        _validated: bool = False,
    ) -> None:
        if n_qubits not in (1, 2):
            raise ValueError(
                f"only 1- and 2-qubit error channels are supported; got {n_qubits}"
            )
        dim = 2 ** n_qubits
        ops = [np.asarray(k, dtype=np.complex128) for k in kraus]
        if not ops:
            raise ValueError("an error channel needs at least one Kraus operator")
        for i, op in enumerate(ops):
            if op.shape != (dim, dim):
                raise ValueError(
                    f"Kraus operator {i} has shape {op.shape}; expected "
                    f"{(dim, dim)} for {n_qubits} qubit(s)"
                )
        self._kraus = ops
        self.n_qubits = n_qubits
        self.name = name
        self.error_rate = float(error_rate)
        if not _validated and not self.is_trace_preserving:
            raise ValueError(
                "Kraus operators do not define a channel (sum K^dagger K != I "
                f"within 1e-8, max deviation {self.tp_deviation:.3e})"
            )

    @property
    def kraus(self) -> List[np.ndarray]:
        """Copies of the Kraus operators (mutating them cannot alter the error)."""
        return [k.copy() for k in self._kraus]

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def tp_deviation(self) -> float:
        """``max |sum K^dagger K - I|`` -- 0 for a trace-preserving channel."""
        total = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for k in self._kraus:
            total += k.conj().T @ k
        return float(np.abs(total - np.eye(self.dim)).max())

    @property
    def is_trace_preserving(self) -> bool:
        return self.tp_deviation <= 1e-8

    @classmethod
    def from_kraus(cls, kraus: Any, name: str = "kraus") -> "ErrorChannel":
        """Error from an explicit Kraus set (1 or 2 qubits, inferred)."""
        ops = list(kraus)
        if not ops:
            raise ValueError("an error channel needs at least one Kraus operator")
        dim = np.asarray(ops[0]).shape[0]
        if dim == 2:
            n = 1
        elif dim == 4:
            n = 2
        else:
            raise ValueError(
                f"Kraus operators must be 2x2 (1 qubit) or 4x4 (2 qubits); "
                f"got {dim}x{dim}"
            )
        return cls(ops, n, name=name)

    @classmethod
    def from_unitary(cls, unitary: Any) -> "ErrorChannel":
        """Coherent (unitary) error -- a one-term Kraus set."""
        u = np.asarray(unitary, dtype=np.complex128)
        if u.shape not in ((2, 2), (4, 4)):
            raise ValueError(
                f"unitary must be 2x2 or 4x4; got {u.shape[0]}x{u.shape[1]}"
            )
        gram = u.conj().T @ u
        if np.abs(gram - np.eye(u.shape[0])).max() > 1e-8:
            raise ValueError("the given matrix is not unitary")
        return cls([u], 1 if u.shape == (2, 2) else 2, name="unitary", _validated=True)

    @classmethod
    def from_mixed_unitary(cls, pairs: Any) -> "ErrorChannel":
        """Mixture of unitaries -- ``pairs`` is ``[(U, p), ...]`` with ``sum p = 1``.

        The Kraus set is ``sqrt(p_i) U_i``, i.e. the same channel Aer builds for
        ``mixed_unitary_error``.
        """
        items = list(pairs)
        if not items:
            raise ValueError("a mixed unitary needs at least one (unitary, prob)")
        first = np.asarray(items[0][0], dtype=np.complex128)
        n = 1 if first.shape == (2, 2) else 2 if first.shape == (4, 4) else 0
        if n == 0:
            raise ValueError("each unitary must be 2x2 or 4x4")
        dim = 2 ** n
        probs: List[float] = []
        ops: List[np.ndarray] = []
        gram = np.eye(dim)
        for i, (u, p) in enumerate(items):
            u = np.asarray(u, dtype=np.complex128)
            p = float(p)
            if u.shape != (dim, dim):
                raise ValueError(
                    f"unitary {i} has shape {u.shape}; expected all {(dim, dim)}"
                )
            if p < 0:
                raise ValueError(f"probability {i} is negative ({p!r})")
            if np.abs(u.conj().T @ u - gram).max() > 1e-8:
                raise ValueError(f"matrix {i} is not unitary")
            probs.append(p)
            ops.append(math.sqrt(p) * u)
        total = sum(probs)
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                f"mixed unitary probabilities must sum to 1; got {total!r}"
            )
        return cls(ops, n, name="mixed_unitary", _validated=True)

    @classmethod
    def from_reset(cls, p0: float, p1: float = 0.0) -> "ErrorChannel":
        """Reset error -- see :func:`kraus_reset`."""
        return cls(kraus_reset(p0, p1), 1, name="reset", _validated=True)

    @classmethod
    def from_qutrit_kraus(cls, kraus: Any, name: str = "qutrit") -> "ErrorChannel":
        """A 3-level channel embedded on a ``(data, flag)`` qubit pair.

        The third level does not fit on one wire, so each 3x3 Kraus operator
        is embedded into the 4x4 space of the pair (Qiskit wire order: the
        first wire is the least significant bit)

            |0> -> |data=0, flag=0>
            |1> -> |data=1, flag=0>
            |2> -> |data=0, flag=1>

        and the unused fourth basis state ``|11>`` is mapped to ``|00>``, so
        the 4x4 set is trace preserving.  The leaked amplitude always lands
        on ``flag=1``, which is what makes the pair readable as
        "computational subspace + leak flag".  Register the result with
        :meth:`NoiseModel.add_qutrit_error`, which supplies the matching pair
        and firing rule.
        """
        if isinstance(kraus, np.ndarray) and kraus.ndim == 2:
            # A bare 3x3 matrix was passed: iterating it would yield its rows,
            # so treat the whole matrix as a one-term set.
            kraus = [kraus]
        ops = [np.asarray(k, dtype=np.complex128) for k in kraus]
        if not ops:
            raise ValueError("an error channel needs at least one Kraus operator")
        for i, op in enumerate(ops):
            if op.shape != (3, 3):
                raise ValueError(
                    f"qutrit Kraus operator {i} has shape {op.shape}; "
                    "expected (3, 3)"
                )
        total = np.zeros((3, 3), dtype=np.complex128)
        for op in ops:
            total += op.conj().T @ op
        deviation = float(np.abs(total - np.eye(3)).max())
        if deviation > 1e-8:
            raise ValueError(
                "qutrit Kraus operators do not define a channel "
                f"(sum K^dagger K != I within 1e-8, max deviation {deviation:.3e})"
            )
        embedded: List[np.ndarray] = []
        for op in ops:
            emb = np.zeros((4, 4), dtype=np.complex128)
            emb[:3, :3] = op
            embedded.append(emb)
        # Completion on the orthogonal complement: |11> has no qutrit
        # preimage, so it is reset to |00>.  Together with the isometry this
        # makes sum K^dagger K = I on the whole 4-dimensional space.
        complement = np.zeros((4, 4), dtype=np.complex128)
        complement[0, 3] = 1.0
        embedded.append(complement)
        return cls(embedded, 2, name=name)

    def compose(self, other: "ErrorChannel") -> "ErrorChannel":
        """``self`` applied *after* ``other`` (Qiskit ``compose`` order)."""
        other = _as_error(other)
        if other.n_qubits != self.n_qubits:
            raise ValueError(
                "compose needs two errors on the same number of qubits "
                f"({self.n_qubits} vs {other.n_qubits}); use tensor for distinct wires"
            )
        return ErrorChannel(
            _compose_kraus(self._kraus, other._kraus),
            self.n_qubits,
            name=f"{other.name}->{self.name}",
            _validated=True,
        )

    def tensor(self, other: "ErrorChannel") -> "ErrorChannel":
        """Kronecker product; ``self`` is the LEFT factor, i.e. its wire is the
        higher-index one — Qiskit's ``A.tensor(B) == A (x) B``.

        With Qiskit's least-significant-first wire order, ``a.tensor(b)``
        registered on ``qubits=(0, 1)`` acts with ``b`` on qubit 0 and ``a`` on
        qubit 1.
        """
        other = _as_error(other)
        if self.n_qubits + other.n_qubits > 2:
            raise ValueError("tensor products beyond 2 qubits are not supported")
        return ErrorChannel(
            [np.kron(a, b) for a in self._kraus for b in other._kraus],
            self.n_qubits + other.n_qubits,
            name=f"{self.name}(x){other.name}",
            _validated=True,
        )

    def power(self, n: int) -> "ErrorChannel":
        """``self`` applied ``n`` times in a row (``n=0`` is the identity)."""
        n = int(n)
        if n < 0:
            raise ValueError(f"power must be non-negative; got {n}")
        result = ErrorChannel(
            [np.eye(self.dim, dtype=np.complex128)],
            self.n_qubits,
            name="identity",
            _validated=True,
        )
        for _ in range(n):
            result = self.compose(result)
        return result

    def __repr__(self) -> str:
        return (
            f"ErrorChannel(name={self.name!r}, n_qubits={self.n_qubits}, "
            f"kraus={len(self._kraus)})"
        )


def _as_error(obj: Any) -> "ErrorChannel":
    if isinstance(obj, ErrorChannel):
        return obj
    raise TypeError(
        f"expected an ErrorChannel; got {type(obj).__name__} "
        "(build one with ErrorChannel.from_kraus / from_unitary / from_reset)"
    )


@dataclass
class NoiseChannel:
    """A single noise channel applied after a gate."""
    name: str
    error_rate: float
    apply: Optional[Callable] = None
    #: Construction-time normalization of ``error_rate`` for depolarizing
    #: channels: "total" (nominal = total non-identity-Pauli probability)
    #: or "qiskit" (nominal = Qiskit lambda, rescaled when the Kraus set is
    #: built).  Always "total" for non-depolarizing channels.
    convention: str = "total"
    #: Target qubits.  ``None`` (default) = every qubit for a 1-qubit
    #: channel / every two-qubit gate for a 2-qubit channel.  A tuple of
    #: qubit indices for a 1-qubit channel; exactly one ``(a, b)`` pair for
    #: a 2-qubit channel, matched in instruction order (Qiskit
    #: ``add_quantum_error`` parity - measured: an error on ``(0, 1)`` does
    #: not fire on ``cx(1, 0)``).
    qubits: Optional[Tuple[int, ...]] = None
    #: Extra construction parameters a channel needs to be rebuilt but that
    #: are not recoverable from ``error_rate``.  Only thermal relaxation sets
    #: it today: ``error_rate`` carries ``gate_time`` (the quantity the
    #: backend consumes), while ``t1``/``t2`` live here so
    #: :meth:`NoiseModel.to_dict` can serialize a rebuildable channel.
    params: Optional[Dict[str, float]] = None
    #: Instruction filter: ``None`` (default) fires after every instruction
    #: that touches the channel's qubits; a tuple of canonical gate names
    #: fires only after instructions of those types (Qiskit
    #: ``add_quantum_error(error, instructions, qubits)`` parity).  Matching
    #: happens on the simulated IR, after SF's own lowering.
    instructions: Optional[Tuple[str, ...]] = None
    #: Explicit Kraus set for channels that are not one of the built-in
    #: families (custom Kraus / unitary / mixed unitary / reset).  ``None``
    #: for every built-in channel, whose set is rebuilt from ``name`` and
    #: ``error_rate``.
    kraus: Optional[List[np.ndarray]] = None
    #: When the channel fires (2-qubit channels only).
    #:
    #: ``"gate"`` (default) = after a matching two-qubit gate on the target
    #: pair.  ``"spectator"`` = cycle-aware crosstalk: at the end of every
    #: parallel cycle it accumulates one unit per *driven endpoint* of the
    #: pair and fires that many times, so an idle partner is kicked while
    #: its neighbour is busy, a gate on the edge itself counts twice, and
    #: two simultaneous neighbours double the strength (see
    #: :meth:`NoiseModel.add_crosstalk`).
    mode: str = "gate"

    @property
    def gate(self):
        return self.name

    @property
    def rate(self):
        return self.error_rate


def _channel_targets(
    channels: List[NoiseChannel], idx: int
) -> Optional[Tuple[int, ...]]:
    """Targets of the channel backing ``_kraus[idx]`` (None when out of range).

    The ``*_channels`` and ``*_kraus`` lists are appended in lockstep by
    the ``add_*`` methods; the length guard keeps hand-manipulated model
    internals behaving like the historical untargeted lists.
    """
    if idx < len(channels):
        return channels[idx].qubits
    return None


def _channel_instructions(
    channels: List[NoiseChannel], idx: int
) -> Optional[Tuple[str, ...]]:
    """Instruction filter of the channel backing ``_kraus[idx]`` (None = any)."""
    if idx < len(channels):
        return channels[idx].instructions
    return None


def _flat_kraus_list(kraus_set: List[np.ndarray]) -> List[List[float]]:
    """Kraus set as the flat re/im float lists the Rust bridge consumes.

    Used by the serializer so a custom channel round-trips through
    :meth:`NoiseModel.to_dict` without rebuilding it from a formula.
    """
    dim = kraus_set[0].shape[0]
    out: List[List[float]] = []
    for k in kraus_set:
        flat: List[float] = []
        for r in range(dim):
            for c in range(dim):
                flat.append(float(k[r, c].real))
                flat.append(float(k[r, c].imag))
        out.append(flat)
    return out


def _kraus_from_flat(flat_sets: Any) -> List[np.ndarray]:
    """Inverse of :func:`_flat_kraus_list` (used by ``from_dict``)."""
    sets = list(flat_sets or [])
    if not sets:
        raise ValueError("serialized channel has an empty 'kraus' list")
    size = len(sets[0])
    dim = int(round(math.sqrt(size // 2)))
    if dim not in (2, 4) or dim * dim * 2 != size:
        raise ValueError(
            f"serialized Kraus matrix has {size} floats; expected 8 (2x2) or 32 (4x4)"
        )
    ops: List[np.ndarray] = []
    for flat in sets:
        if len(flat) != size:
            raise ValueError("serialized Kraus matrices differ in size")
        m = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim * dim):
            m[i // dim, i % dim] = complex(flat[2 * i], flat[2 * i + 1])
        ops.append(m)
    return ops


def _channel_dict(ch: NoiseChannel) -> Dict[str, Any]:
    """Serialized form of one channel.

    ``qubits``, ``convention``, ``params``, ``instructions`` and ``mode``
    appear only when they carry information (explicit target, non-default
    depolarizing normalization, extra rebuild parameters, an instruction
    filter, a non-gate firing mode), so a default untargeted model keeps the
    historical exact dict.
    """
    d: Dict[str, Any] = {"name": ch.name, "error_rate": ch.error_rate}
    if ch.qubits is not None:
        d["qubits"] = list(ch.qubits)
    if ch.convention != "total":
        d["convention"] = ch.convention
    if ch.params is not None:
        d["params"] = dict(ch.params)
    if ch.instructions is not None:
        d["instructions"] = list(ch.instructions)
    if ch.kraus is not None:
        d["kraus"] = _flat_kraus_list(ch.kraus)
    if ch.mode != "gate":
        d["mode"] = ch.mode
    return d


def _restore_mode(
    model: "NoiseModel", serialized: Dict[str, Any], n_qubits: int
) -> None:
    """Re-apply a channel's firing mode after :meth:`NoiseModel.from_dict`
    rebuilt it.

    The ``add_*`` / ``add_quantum_error`` path always appends exactly one
    channel, so the freshly appended entry is the one being restored.
    """
    mode = serialized.get("mode", "gate")
    if mode != "gate" and n_qubits == 2 and model.two_qubit_channels:
        model.two_qubit_channels[-1].mode = mode


class NoiseModel:
    """Unified noise model for all simulation methods.

    Stores both JAX callables (statevector path) and Kraus operators
    (density matrix path). The ``has_noise`` property indicates whether
    any noise channels are registered.

    Two construction knobs control how gate noise is placed and scaled
    relative to other simulators (both default to the historical
    SuperFermion semantics — existing numbers do not change):

    - ``NoiseModel(placement=...)`` (native density-matrix path):
      ``"touch"`` (default) applies a qubit's 1-qubit channels after
      every gate touching that qubit, so a two-qubit gate gets two
      1-qubit channels (one per qubit); ``"gate"`` applies 1-qubit
      channels only after 1-qubit gates — the Qiskit ``NoiseModel`` /
      Aer placement.
    - ``add_depolarizing(..., convention=...)``:
      ``"total"`` (default) treats ``p`` as the total non-identity-Pauli
      probability (per Pauli: p/3 for 1 qubit, p/15 for 2);
      ``"qiskit"`` treats ``p`` as Qiskit's lambda and rescales
      internally (p_kraus = 3*lam/4 for 1 qubit, 15*lam/16 for 2), so the
      channel is identical to ``qiskit_aer.noise.depolarizing_error`` at
      the same nominal number.
    - ``add_*(..., qubits=...)``: binds a channel to specific wires
      instead of every qubit.  A 1-qubit channel takes an int or int
      sequence; a 2-qubit channel takes exactly one ordered pair matched
      in instruction order, like Qiskit's ``add_quantum_error``
      (``qubits=(0, 1)`` does not fire on ``cx(1, 0)``).

    ``add_thermal_relaxation(t1, t2, gate_time)`` builds the standard
    T1/T2 channel (amplitude damping followed by pure dephasing) equal to
    ``qiskit_aer.noise.thermal_relaxation_error`` at the same parameters;
    ``t2 > 2*t1`` is rejected.

    Reproducing an Aer model that applies ``depolarizing_error(lam, 1)``
    after every 1-qubit gate and ``depolarizing_error(lam, 2)`` after
    every 2-qubit gate::

        noise = NoiseModel(placement="gate")
        noise.add_depolarizing(lam, convention="qiskit")
        noise.add_depolarizing(lam, n_qubits=2, convention="qiskit")

    Usage::

        noise = NoiseModel()
        noise.add_depolarizing(0.01)
        noise.add_amplitude_damping(0.005)
        noise.add_readout_error(0.02)

        # Use with sf.run()
        result = sf.run(circuit, method="density_matrix", noise_model=noise)
    """

    def __init__(self, placement: str = "touch") -> None:
        if placement not in ("touch", "gate"):
            raise ValueError(
                "placement must be 'touch' (channels after every gate "
                "touching the qubit) or 'gate' (1q channels only after 1q "
                f"gates, like Qiskit); got {placement!r}"
            )
        self.placement = placement
        self.single_qubit_channels: List[NoiseChannel] = []
        self.two_qubit_channels: List[NoiseChannel] = []
        self.readout_error: float = 0.0
        self._1q_kraus: List[List[np.ndarray]] = []
        self._2q_kraus: List[List[np.ndarray]] = []
        self._readout_p: float = 0.0

    @property
    def has_noise(self) -> bool:
        return bool(
            self.single_qubit_channels
            or self.two_qubit_channels
            or self._1q_kraus
            or self._2q_kraus
            or self.readout_error > 0
            or self._readout_p > 0
        )

    def add_depolarizing(
        self,
        p: float,
        n_qubits: int = 1,
        convention: str = "total",
        qubits: Any = None,
        instructions: Any = None,
    ) -> "NoiseModel":
        """Add depolarizing noise with error probability p.

        Placement: the channel is applied after every gate that touches a
        qubit — a two-qubit gate therefore gets two 1-qubit channels (one
        per qubit).  ``n_qubits=2`` selects the 15-Pauli 2-qubit channel,
        applied after each two-qubit gate instead.  With
        ``NoiseModel(placement="gate")`` the 1-qubit channels apply only
        after 1-qubit gates.

        ``qubits`` binds the channel to specific wires instead of every
        qubit: an int or int sequence for ``n_qubits=1`` (e.g.
        ``qubits=[0, 2]``), exactly one ordered pair for ``n_qubits=2``
        (e.g. ``qubits=(0, 1)``, instruction order).  Explicit targets are
        validated against the simulated circuit in
        :meth:`to_rust_kraus_ops`.

        Normalization: ``convention="total"`` (default) interprets ``p``
        as the total non-identity-Pauli probability (per Pauli: p/3 for
        1 qubit, p/15 for 2 qubits).  Qiskit's ``depolarizing_error`` uses
        the lambda convention
        ``E(rho) = (1 - lam) rho + lam * Tr(rho) * I / 2**n`` (per Pauli:
        lam/4, lam/16), so at equal nominal numbers this channel is heavier
        by exactly 4/3 (1 qubit) and 16/15 (2 qubits); with
        ``convention="qiskit"`` the stored Kraus set is rescaled internally
        (p_kraus = 3*lam/4 for 1 qubit, 15*lam/16 for 2) so the channel is
        identical to Qiskit's at the same nominal number.  ``error_rate``
        (and ``to_dict``) keeps the nominal value as passed.
        """
        if convention not in ("total", "qiskit"):
            raise ValueError(
                f"convention must be 'total' or 'qiskit'; got {convention!r}"
            )
        targets = _normalize_targets(qubits, n_qubits)
        p_kraus = p
        if convention == "qiskit":
            p_kraus = p * 3.0 / 4.0 if n_qubits == 1 else p * 15.0 / 16.0
        channel = NoiseChannel(
            "depolarizing", p, None, convention, targets, None,
            _normalize_instructions(instructions),
        )
        if n_qubits == 1:
            self.single_qubit_channels.append(channel)
            self._1q_kraus.append(kraus_depolarizing_1q(p_kraus))
        else:
            self.two_qubit_channels.append(channel)
            self._2q_kraus.append(kraus_depolarizing_2q(p_kraus))
        return self

    def add_amplitude_damping(
        self, gamma: float, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Add amplitude damping (T1 decay) with rate gamma.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "amplitude_damping", gamma, None, "total",
            _normalize_targets(qubits, 1), None,
            _normalize_instructions(instructions),
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_amplitude_damping(gamma))
        return self

    def add_phase_damping(
        self, gamma: float, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Add phase damping (T2 dephasing) with rate gamma.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "phase_damping", gamma, None, "total",
            _normalize_targets(qubits, 1), None,
            _normalize_instructions(instructions),
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_phase_damping(gamma))
        return self

    def add_thermal_relaxation(
        self,
        t1: float,
        t2: float,
        gate_time: float,
        qubits: Any = None,
        instructions: Any = None,
    ) -> "NoiseModel":
        """Add an Aer-compatible T1/T2 thermal-relaxation channel.

        The channel is applied after each gate touching a targeted qubit
        (placement applies), with the gate's noise strength set by the
        ``gate_time`` passed here.  Same units for ``t1``, ``t2`` and
        ``gate_time`` (e.g. ns).  Requires ``t2 <= 2*t1``; the composite
        Kraus set equals ``qiskit_aer.noise.thermal_relaxation_error``
        (see :func:`kraus_thermal_relaxation`).  The same ``gate_time``
        is used after every touching gate — per-instruction durations are
        not available at this layer, so pass the gate time this channel
        should model.
        """
        kraus = kraus_thermal_relaxation(t1, t2, gate_time)
        channel = NoiseChannel(
            "thermal_relaxation",
            float(gate_time),
            None,
            "total",
            _normalize_targets(qubits, 1),
            # Without t1/t2 a serialized thermal channel cannot be rebuilt;
            # gate_time alone does not determine the Kraus set.
            {"t1": float(t1), "t2": float(t2)},
            _normalize_instructions(instructions),
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus)
        return self

    def add_bit_flip(
        self, p: float, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Add bit-flip noise with probability p.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "bit_flip", p, None, "total", _normalize_targets(qubits, 1), None,
            _normalize_instructions(instructions),
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_bit_flip(p))
        return self

    def add_phase_flip(
        self, p: float, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Add phase-flip noise with probability p.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "phase_flip", p, None, "total", _normalize_targets(qubits, 1), None,
            _normalize_instructions(instructions),
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_phase_flip(p))
        return self

    def add_readout_error(self, p: float) -> "NoiseModel":
        """Add readout bit-flip error with probability p."""
        self.readout_error = p
        self._readout_p = p
        return self

    def add_two_qubit_depolarizing(
        self, p: float, convention: str = "total", qubits: Any = None,
        instructions: Any = None,
    ) -> "NoiseModel":
        """Convenience wrapper for add_depolarizing with n_qubits=2.

        ``qubits`` binds the channel to exactly one ordered pair (e.g.
        ``qubits=(0, 1)``); ``None`` applies it after every two-qubit
        gate.
        """
        return self.add_depolarizing(
            p, n_qubits=2, convention=convention, qubits=qubits,
            instructions=instructions,
        )

    # ─── Custom errors (Qiskit ``QuantumError`` parity) ──────────────────
    def add_quantum_error(
        self,
        error: "ErrorChannel",
        instructions: Any = None,
        qubits: Any = None,
    ) -> "NoiseModel":
        """Register a composable :class:`ErrorChannel`.

        This is the direct counterpart of Aer's
        ``NoiseModel.add_quantum_error(error, instructions, qubits)``:

        * ``error`` — an :class:`ErrorChannel` (custom Kraus set, coherent
          unitary, mixed unitary, reset, or any ``compose``/``tensor``/
          ``power`` product of those).
        * ``instructions`` — restrict the channel to those gate types
          (``add_quantum_error(err, ["cx"], [0, 1])`` fires only after ``cx``
          on that ordered pair).  ``None`` keeps the historical behaviour:
          the channel fires after every instruction touching its qubits.
          Matching is on the simulated IR, after SF's own lowering.
        * ``qubits`` — same target semantics as every other adder.

        A two-qubit matrix is read in Qiskit's wire order: the first listed
        qubit is the least significant, so ``kron(A, B)`` registered on
        ``(a, b)`` acts with ``A`` on ``b`` and ``B`` on ``a`` — the same
        channel Aer builds for ``coherent_unitary_error(kron(A, B))`` on
        ``[a, b]``.  The conversion to the kernel's block order happens in
        :meth:`to_rust_kraus_ops_2q` / :meth:`apply_2q`.

        The channel is applied on the density-matrix path, the only engine
        that carries gate noise.
        """
        error = _as_error(error)
        targets = _normalize_targets(qubits, error.n_qubits)
        channel = NoiseChannel(
            error.name,
            error.error_rate,
            None,
            "total",
            targets,
            None,
            _normalize_instructions(instructions),
            error.kraus,
        )
        if error.n_qubits == 1:
            self.single_qubit_channels.append(channel)
            self._1q_kraus.append(error.kraus)
        else:
            self.two_qubit_channels.append(channel)
            self._2q_kraus.append(error.kraus)
        return self

    def add_kraus(
        self, kraus_ops: Any, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Custom Kraus channel; 1 or 2 qubits, inferred from the shape.

        The set must be trace preserving (``sum K^dagger K = I`` within
        1e-8), otherwise ``ValueError`` is raised — the same contract as
        Aer's ``kraus_error``.
        """
        return self.add_quantum_error(
            ErrorChannel.from_kraus(kraus_ops),
            instructions=instructions,
            qubits=qubits,
        )

    def add_unitary(
        self, unitary: Any, qubits: Any = None, instructions: Any = None
    ) -> "NoiseModel":
        """Coherent error: one unitary, applied as a one-term Kraus set.

        The direct counterpart of Aer's ``coherent_unitary_error`` for 1- and
        2-qubit unitaries; ``instructions`` narrows it to specific gates.
        """
        return self.add_quantum_error(
            ErrorChannel.from_unitary(unitary),
            instructions=instructions,
            qubits=qubits,
        )

    def add_mixed_unitary(
        self,
        unitaries: Any,
        probabilities: Any,
        qubits: Any = None,
        instructions: Any = None,
    ) -> "NoiseModel":
        """Mixture of unitaries — Aer's ``mixed_unitary_error`` parity.

        ``probabilities`` must sum to 1; the Kraus set is ``sqrt(p_i) U_i``.
        """
        pairs = list(zip(list(unitaries), list(probabilities)))
        return self.add_quantum_error(
            ErrorChannel.from_mixed_unitary(pairs),
            instructions=instructions,
            qubits=qubits,
        )

    def add_reset_error(
        self,
        p0: float,
        p1: float = 0.0,
        qubits: Any = None,
        instructions: Any = ("reset",),
    ) -> "NoiseModel":
        """Reset error — Aer's ``reset_error`` parity.

        ``p0`` is the probability of ending in ``|0>`` after the reset, ``p1``
        the probability of ending in ``|1>`` (``p0 + p1 <= 1``); the channel is
        :func:`kraus_reset`, whose five-term Kraus set reproduces Aer's
        ``id`` / ``reset`` / ``reset;x`` mixture exactly.

        It is attached to the RESET instruction by default.  Note that under
        the model-level ``placement="touch"`` an *untargeted* channel fires
        after every gate; this channel always carries an instruction filter,
        so it fires on resets only.

        Reach: on ``method="density_matrix"`` a ``reset`` is realised exactly
        (the engine applies the deterministic reset channel, then this
        error), so the default binding fires end to end.  Circuits whose
        reset is followed by a re-use of the same qubit still go through the
        trajectory engine when they also contain a non-terminal measurement
        or feed-forward, and the pure-state methods (statevector / mps /
        stabilizer) keep their per-shot trajectory path, where gate noise is
        not applied.
        """
        return self.add_quantum_error(
            ErrorChannel.from_reset(p0, p1),
            instructions=instructions,
            qubits=qubits,
        )

    def add_crosstalk(
        self,
        edges: Any,
        zz_angle: float,
        error_rate: float = 0.0,
        instructions: Any = None,
        mode: str = "static",
    ) -> "NoiseModel":
        """ZZ coupling on each edge, either static or cycle-aware.

        For every pair ``(a, b)`` the coherent error
        ``exp(-i * zz_angle/2 * Z_a Z_b)`` is registered on that pair, plus an
        optional two-qubit depolarizing channel of strength ``error_rate``.
        ``instructions`` narrows both to specific gate types.

        ``mode="static"`` (default) is the measured residual coupling folded
        onto the edge: the channel fires after every matching gate on the
        pair, with a strength that does not depend on what else runs.

        ``mode="spectator"`` is cycle-aware.  A cycle is one parallel step —
        a maximal run of instructions on pairwise-disjoint qubits — and the
        edge accumulates one unit of coupling per *driven endpoint* in that
        cycle: ``+1`` when an instruction touches ``a``, ``+1`` when one
        touches ``b``.  The channel is then applied that many times, so

        * a neighbour gate on ``a`` while ``b`` idles → one kick (spectator
          ZZ: the idle partner does not escape the residual coupling),
        * a gate on the edge itself → two kicks,
        * two disjoint neighbours, one per endpoint → two kicks
          (simultaneous-gate inflation, ``θ → 2θ``),
        * an idle edge → nothing.

        The one case static mode cannot express is the first: the edge
        responds to *which* gates run concurrently, not just to its own.

        This is a first-order model, not a calibration: every driven endpoint
        contributes exactly one unit per cycle regardless of pulse shape, a
        gate's own error rate is not inflated (that needs a parametric
        channel), and the kicks land at the end of the cycle.
        """
        if mode not in ("static", "spectator"):
            raise ValueError(
                f"crosstalk mode must be 'static' or 'spectator'; got {mode!r}"
            )
        if mode == "spectator" and error_rate < 0:
            raise ValueError(f"error_rate must be >= 0; got {error_rate!r}")
        z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        zz = np.kron(z, z)
        theta = float(zz_angle)
        coherent = (
            math.cos(theta / 2.0) * np.eye(4, dtype=np.complex128)
            - 1j * math.sin(theta / 2.0) * zz
        )
        pairs = list(edges)
        if not pairs:
            raise ValueError("add_crosstalk needs at least one (a, b) edge")
        channel_mode = "spectator" if mode == "spectator" else "gate"
        for edge in pairs:
            try:
                a, b = edge
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"each edge must be a pair (a, b); got {edge!r}"
                ) from exc
            target = (int(a), int(b))
            added = len(self.two_qubit_channels)
            self.add_unitary(coherent, qubits=target, instructions=instructions)
            if error_rate > 0:
                self.add_depolarizing(
                    error_rate, n_qubits=2, qubits=target, instructions=instructions
                )
            if channel_mode != "gate":
                # Mark only the channels this call added; the flag selects the
                # per-cycle firing rule in the native density-matrix core.
                for ch in self.two_qubit_channels[added:]:
                    ch.mode = channel_mode
        return self

    def add_qutrit_error(
        self,
        kraus_ops: Any,
        qubits: Any,
        instructions: Any = None,
    ) -> "NoiseModel":
        """A 3-level (qutrit) channel on an explicit ``(data, flag)`` pair.

        A transmon's second excited level does not fit on one wire, so the
        qutrit is simulated on two physical qubits -- ``data`` carries the
        computational subspace, ``flag`` marks the leaked level -- with the
        embedding of :meth:`ErrorChannel.from_qutrit_kraus`::

            |0> -> |data=0, flag=0>
            |1> -> |data=1, flag=0>
            |2> -> |data=0, flag=1>

        ``qubits`` is the ordered pair ``(data, flag)`` and is required: the
        embedding needs a wire for the leaked amplitude.  The flag wire is
        read-only in this model -- do not put gates on it, or the leaked
        level will be driven.

        The channel fires once at the end of every parallel cycle in which the
        data qubit is driven (the two-qubit ``mode="spectator"`` rule; the
        flag is never driven, so one gate on the data qubit is exactly one
        hit).  ``instructions=None`` (default) fires on every gate and skips
        ``measure`` / ``reset`` / ``barrier``; pass an explicit list to narrow
        it, e.g. ``instructions=["cx"]``.

        Scope: this is exact 3-level dynamics for the *embedded* model, not a
        native d=3 kernel.  It costs one extra qubit per leaky transmon, so a
        circuit on ``n`` qubits with ``k`` leaky transmons runs on ``n + k``
        wires.  The channel is applied on the density-matrix path.
        """
        if instructions is None:
            instructions = _QUTRIT_DEFAULT_INSTRUCTIONS
        targets = _normalize_targets(qubits, 2)
        if targets is None:
            raise ValueError(
                "add_qutrit_error needs an explicit (data, flag) qubit pair; "
                "the embedding has nowhere to put the leaked amplitude otherwise"
            )
        added = len(self.two_qubit_channels)
        self.add_quantum_error(
            ErrorChannel.from_qutrit_kraus(kraus_ops),
            instructions=instructions,
            qubits=targets,
        )
        # Per-cycle firing: one hit per driven endpoint, which for a flag that
        # is never gated means one hit per gate on the data qubit.
        self.two_qubit_channels[added].mode = "spectator"
        return self

    def add_leakage(
        self,
        p: float,
        qubit: int,
        flag: int,
        instructions: Any = None,
    ) -> "NoiseModel":
        """First-order leakage of ``qubit`` into the flagged |2> level.

        After every gate on ``qubit`` the channel of :func:`kraus_leakage`
        moves population out of the computational subspace with probability
        ``p``: both |0> and |1> leak into |2>, and |2> is absorbing (no decay
        back; compose :meth:`add_qutrit_error` with a return channel for
        that).  ``flag`` is the physical wire that records the leaked level --
        see :meth:`add_qutrit_error` for the embedding and the one-extra-qubit
        cost per leaky transmon.
        """
        return self.add_qutrit_error(
            kraus_leakage(p), qubits=(qubit, flag), instructions=instructions
        )

    def apply_1q(self, rho: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply the 1-qubit Kraus channels targeting ``qubit``."""
        from superfermion.backends.density_matrix import _apply_kraus_1q
        for idx, kraus_set in enumerate(self._1q_kraus):
            targets = _channel_targets(self.single_qubit_channels, idx)
            if targets is not None and qubit not in targets:
                continue
            rho = _apply_kraus_1q(rho, kraus_set, qubit, n)
        return rho

    def apply_2q(self, rho: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
        """Apply the 2-qubit Kraus channels matching the ordered pair ``(q0, q1)``.

        Matrices are stored in Qiskit's wire order (first listed qubit = least
        significant) and converted to the kernel's order here, so this path and
        the native engine realise the same channel.
        """
        from superfermion.backends.density_matrix import _apply_kraus_2q
        for idx, kraus_set in enumerate(self._2q_kraus):
            targets = _channel_targets(self.two_qubit_channels, idx)
            if targets is not None and targets != (q0, q1):
                continue
            ordered = [_qiskit_order_2q(k) for k in kraus_set]
            rho = _apply_kraus_2q(rho, ordered, q0, q1, n)
        return rho

    def apply_to_counts(self, counts: Dict[str, int], rng=None) -> Dict[str, int]:
        """Apply readout error to measurement counts."""
        p = self.readout_error or self._readout_p
        if p <= 0:
            return counts
        if rng is None:
            rng = np.random.default_rng(42)
        elif not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(int(np.asarray(rng).flat[0]))
        noisy: Dict[str, int] = {}
        for bs, count in counts.items():
            for _ in range(count):
                bits = list(bs)
                for i in range(len(bits)):
                    if rng.random() < p:
                        bits[i] = '1' if bits[i] == '0' else '0'
                new_bs = ''.join(bits)
                noisy[new_bs] = noisy.get(new_bs, 0) + 1
        return noisy

    def apply_to_counts_fast(
        self, counts: Dict[str, int], rng=None
    ) -> Dict[str, int]:
        """Vectorized readout-error sampling for binary count keys.

        Mathematically identical to :meth:`apply_to_counts` (each measured
        bit flips independently with probability ``p``), but samples all
        shots of a key at once with NumPy.  Falls back to the scalar loop
        for keys wider than 63 bits, where the packed-integer mapping is not
        available.
        """
        p = self.readout_error or self._readout_p
        if p <= 0:
            return counts
        if rng is None:
            rng = np.random.default_rng(42)
        elif not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(int(np.asarray(rng).flat[0]))

        noisy: Dict[str, int] = {}
        for bs, count in counts.items():
            count = int(count)
            n_bits = len(bs)
            if count <= 0:
                continue
            if n_bits == 0 or n_bits > 63:
                for _ in range(count):
                    bits = list(bs)
                    for i in range(len(bits)):
                        if rng.random() < p:
                            bits[i] = '1' if bits[i] == '0' else '0'
                    new_bs = ''.join(bits)
                    noisy[new_bs] = noisy.get(new_bs, 0) + 1
                continue

            bits = np.frombuffer(bs.encode("ascii"), dtype=np.uint8) - 48
            if bits.size != n_bits:
                # Non-binary key: preserve the scalar behavior instead of
                # silently packing an unexpected representation.
                for _ in range(count):
                    chars = list(bs)
                    for i in range(len(chars)):
                        if rng.random() < p:
                            chars[i] = '1' if chars[i] == '0' else '0'
                    new_bs = ''.join(chars)
                    noisy[new_bs] = noisy.get(new_bs, 0) + 1
                continue

            weights = np.left_shift(
                np.uint64(1), np.arange(n_bits - 1, -1, -1, dtype=np.uint64)
            )
            chunk = max(1, 1_000_000 // n_bits)
            remaining = count
            while remaining > 0:
                block = min(chunk, remaining)
                flips = rng.random((block, n_bits)) < p
                packed = (bits ^ flips).astype(np.uint64) @ weights
                values, occurrences = np.unique(packed, return_counts=True)
                for value, occurrences_i in zip(values, occurrences):
                    key = format(int(value), f"0{n_bits}b")
                    noisy[key] = noisy.get(key, 0) + int(occurrences_i)
                remaining -= block
        return noisy

    @property
    def has_targeted_2q(self) -> bool:
        """True when any 2-qubit channel carries an explicit ``qubits=`` pair.

        The native DM path then needs the aligned pair list from
        :meth:`to_rust_kraus_ops_2q_targets` so the Rust core can match
        channels to gates in instruction order; older builds without the
        pair argument must not receive it (TypeError on the binding).
        """
        return any(ch.qubits is not None for ch in self.two_qubit_channels)

    def _validate_targets(self, n_qubits: int) -> None:
        """Check explicit ``qubits=`` targets against the simulated circuit.

        Called from :meth:`to_rust_kraus_ops` — the first place the wire
        count is known (the ``add_*`` methods cannot see ``n``); the
        historical all-qubit call skips all checks.
        """
        for channel in self.single_qubit_channels:
            if channel.qubits is None:
                continue
            for q in channel.qubits:
                if q >= n_qubits:
                    raise ValueError(
                        f"noise channel '{channel.name}' targets qubit {q}, "
                        f"but the circuit only has {n_qubits} qubits"
                    )
        for channel in self.two_qubit_channels:
            if channel.qubits is None:
                continue
            for q in channel.qubits:
                if q >= n_qubits:
                    raise ValueError(
                        f"noise channel '{channel.name}' targets qubits "
                        f"{channel.qubits}, but the circuit only has "
                        f"{n_qubits} qubits"
                    )

    def to_rust_kraus_ops(self, n_qubits: int) -> List[Tuple[int, List[float]]]:
        """Convert the 1-qubit noise channels to the flat format expected by Rust.

        Returns list of (qubit, flat_kraus) tuples where flat_kraus encodes the
        **composed** channel's Kraus matrices for that qubit as
        [re00, im00, re01, im01, ...] per matrix.  The channel is applied
        after every gate touching the qubit.

        Multiple 1q channels on the same qubit are composed in the order they
        were added (C2 ∘ C1 after the gate), exactly matching the sequential
        application of the pure-Python path.  Passing the raw concatenation of
        every channel's Kraus set instead would make the Rust core sum the
        per-channel superoperators (non-TP: tr(rho) = #channels per gate
        touch) — the composite set below is exact by construction.

        Channels constructed with ``qubits=`` contribute to their target
        wires only; a wire with no applicable channel is omitted (no
        noise).  Targets past ``n_qubits`` raise ``ValueError``.  With no
        targets every wire receives the same composite list,
        byte-identical to the historical format.
        """
        self._validate_targets(n_qubits)
        ops: List[Tuple[int, List[float]]] = []
        if not self._1q_kraus:
            return ops

        for q, _key, composite in self._iter_1q_groups(n_qubits):
            ops.append((q, _flat_kraus(composite, 2)))
        return ops

    def _iter_1q_groups(
        self, n_qubits: int
    ) -> "List[Tuple[int, Optional[Tuple[str, ...]], List[np.ndarray]]]":
        """``(qubit, instruction_filter, composed_kraus)`` for every 1q group.

        Channels are composed per qubit in add order, but only *within* one
        instruction filter: two channels that fire on different gate types
        cannot be merged into a single set.  With no filters (the historical
        case) every qubit yields exactly one group, so the emitted Kraus set
        is byte-identical to the pre-filter format.
        """
        out: List[Tuple[int, Optional[Tuple[str, ...]], List[np.ndarray]]] = []
        for q in range(n_qubits):
            groups: List[List[Any]] = []  # [filter, composite] in add order
            for idx, kraus_set in enumerate(self._1q_kraus):
                targets = _channel_targets(self.single_qubit_channels, idx)
                if targets is not None and q not in targets:
                    continue
                key = _channel_instructions(self.single_qubit_channels, idx)
                slot = next((g for g in groups if g[0] == key), None)
                if slot is None:
                    groups.append([key, list(kraus_set)])
                else:
                    slot[1] = _compose_kraus(kraus_set, slot[1])
            for key, composite in groups:
                out.append((q, key, composite))
        return out

    @property
    def has_targeted_1q(self) -> bool:
        """True when a 1-qubit channel carries an instruction filter."""
        return any(
            ch.instructions is not None for ch in self.single_qubit_channels
        )

    def to_rust_kraus_ops_1q_instructions(
        self, n_qubits: int
    ) -> Optional[List[Optional[Tuple[str, ...]]]]:
        """Per-entry instruction filters aligned with :meth:`to_rust_kraus_ops`.

        ``None`` when no 1q channel is filtered — the adapters then call the
        historical binding without the extra argument, so unfiltered runs keep
        their exact code path.  A ``None`` entry means "every instruction".
        """
        if not self.has_targeted_1q:
            return None
        return [
            key for _q, key, _composite in self._iter_1q_groups(n_qubits)
        ]

    def to_rust_kraus_ops_2q(self) -> List[List[float]]:
        """Flat Kraus sets for the 2-qubit channels (native DM path).

        Returns one flat list per 2q channel, in add-order.  Each 4x4 Kraus
        matrix is flattened as 32 floats (re00, im00, re01, im01, ... re33,
        im33), **after** converting it from Qiskit's wire order to the
        kernel's (:func:`_qiskit_order_2q`).  The Rust core applies the
        channels sequentially after every 2q gate (after the fused gate +
        1q-channel sweep), which matches the per-channel sequential
        semantics of :meth:`apply_2q`.
        """
        return [
            _flat_kraus([_qiskit_order_2q(k) for k in kraus_set], 4)
            for kraus_set in self._2q_kraus
        ]

    def to_rust_kraus_ops_2q_targets(
        self,
    ) -> List[Optional[Tuple[int, int]]]:
        """Per-channel ordered pair aligned with :meth:`to_rust_kraus_ops_2q`.

        ``None`` = every two-qubit gate.  A channel registered for
        ``(a, b)`` fires only on gates emitted in that exact order — the
        Rust core checks ``inst.qubits == (a, b)``, matching Qiskit's
        ``add_quantum_error`` (measured: an error on ``[0, 1]`` is silent
        on ``cx(1, 0)``).
        """
        return [ch.qubits for ch in self.two_qubit_channels]

    @property
    def has_targeted_2q_instructions(self) -> bool:
        """True when a 2-qubit channel carries an instruction filter."""
        return any(
            ch.instructions is not None for ch in self.two_qubit_channels
        )

    def to_rust_kraus_ops_2q_instructions(
        self,
    ) -> Optional[List[Optional[Tuple[str, ...]]]]:
        """Per-channel instruction filters aligned with
        :meth:`to_rust_kraus_ops_2q` (``None`` when nothing is filtered)."""
        if not self.has_targeted_2q_instructions:
            return None
        return [ch.instructions for ch in self.two_qubit_channels]

    def to_rust_kraus_2q_modes(self) -> Optional[List[int]]:
        """Per-channel firing mode aligned with :meth:`to_rust_kraus_ops_2q`.

        ``0`` = after matching gates (historical), ``1`` = spectator: fired
        once per driven endpoint per parallel cycle, as described in
        :meth:`add_crosstalk`.  ``None`` when every 2-qubit channel is
        gate-placed, so the native core skips its cycle bookkeeping entirely
        for ordinary models.
        """
        modes = [
            1 if ch.mode == "spectator" else 0 for ch in self.two_qubit_channels
        ]
        return modes if any(modes) else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize noise model to a dictionary.

        Includes the model-level ``placement``.  ``error_rate`` is the nominal
        value as passed to the ``add_*`` methods.  A channel's ``qubits``
        target appears only when one was given, its ``convention`` only when
        it is not the default ``"total"``, and ``params`` only for channels
        that need more than ``error_rate`` to be rebuilt (thermal relaxation
        stores ``t1``/``t2`` there).  Untargeted default models therefore
        keep the historical exact dict, and :meth:`from_dict` restores every
        model exactly.
        """
        return {
            "single_qubit_channels": [
                _channel_dict(ch) for ch in self.single_qubit_channels
            ],
            "two_qubit_channels": [
                _channel_dict(ch) for ch in self.two_qubit_channels
            ],
            "readout_error": self.readout_error,
            "placement": self.placement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseModel":
        """Rebuild a model from :meth:`to_dict` output.

        Accepts dicts written before ``convention`` and ``params`` existed:
        a missing ``convention`` means the historical ``"total"``
        normalization, and a missing ``qubits`` means untargeted.  A thermal
        channel with no ``params`` cannot be rebuilt (``error_rate`` is only
        the gate time), so it raises ``ValueError`` naming the missing
        fields instead of inventing ``t1``/``t2``.
        """
        model = cls(placement=data.get("placement", "touch"))
        for key, n_qubits in (
            ("single_qubit_channels", 1),
            ("two_qubit_channels", 2),
        ):
            for ch in data.get(key, []) or []:
                name = ch.get("name")
                rate = ch.get("error_rate")
                targets = ch.get("qubits")
                convention = ch.get("convention", "total")
                instrs = ch.get("instructions")
                if "kraus" in ch:
                    # Custom channels serialize their Kraus set (row-major
                    # re/im pairs, the same layout the Rust bridge takes), so
                    # they rebuild exactly instead of being re-derived from a
                    # formula that does not exist for them.
                    ops = _kraus_from_flat(ch["kraus"])
                    model.add_quantum_error(
                        ErrorChannel.from_kraus(ops, name=name or "kraus"),
                        instructions=instrs,
                        qubits=targets,
                    )
                    _restore_mode(model, ch, n_qubits)
                    continue
                if name == "depolarizing":
                    model.add_depolarizing(
                        rate, n_qubits=n_qubits, convention=convention,
                        qubits=targets, instructions=instrs,
                    )
                elif name == "amplitude_damping":
                    model.add_amplitude_damping(
                        rate, qubits=targets, instructions=instrs
                    )
                elif name == "phase_damping":
                    model.add_phase_damping(
                        rate, qubits=targets, instructions=instrs
                    )
                elif name == "bit_flip":
                    model.add_bit_flip(rate, qubits=targets, instructions=instrs)
                elif name == "phase_flip":
                    model.add_phase_flip(
                        rate, qubits=targets, instructions=instrs
                    )
                elif name == "thermal_relaxation":
                    params = ch.get("params") or {}
                    if "t1" not in params or "t2" not in params:
                        raise ValueError(
                            "thermal_relaxation channel has no t1/t2: this "
                            "dict predates param serialization and gate_time="
                            f"{rate!r} does not determine the channel"
                        )
                    model.add_thermal_relaxation(
                        float(params["t1"]), float(params["t2"]), rate,
                        qubits=targets, instructions=instrs,
                    )
                else:
                    raise ValueError(f"unknown noise channel {name!r}")
                _restore_mode(model, ch, n_qubits)
        readout = data.get("readout_error", 0.0) or 0.0
        if readout:
            model.add_readout_error(readout)
        return model

    def __repr__(self) -> str:
        channels = len(self.single_qubit_channels) + len(self.two_qubit_channels)
        return (
            f"NoiseModel(channels={channels}, "
            f"readout_error={self.readout_error}, placement='{self.placement}')"
        )


def ibm_eagle_noise() -> NoiseModel:
    """Approximate noise model for IBM Eagle (127-qubit) processor."""
    return (NoiseModel()
        .add_depolarizing(0.001)
        .add_depolarizing(0.01, n_qubits=2)
        .add_amplitude_damping(0.0005)
        .add_phase_damping(0.001)
        .add_readout_error(0.01))


def ideal_noise() -> NoiseModel:
    """No noise (ideal simulator)."""
    return NoiseModel()
