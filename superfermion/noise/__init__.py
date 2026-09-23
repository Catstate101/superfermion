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


def _channel_dict(ch: NoiseChannel) -> Dict[str, Any]:
    """Serialized form of one channel; ``qubits`` only when explicitly set."""
    d: Dict[str, Any] = {"name": ch.name, "error_rate": ch.error_rate}
    if ch.qubits is not None:
        d["qubits"] = list(ch.qubits)
    return d


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
        channel = NoiseChannel("depolarizing", p, None, convention, targets)
        if n_qubits == 1:
            self.single_qubit_channels.append(channel)
            self._1q_kraus.append(kraus_depolarizing_1q(p_kraus))
        else:
            self.two_qubit_channels.append(channel)
            self._2q_kraus.append(kraus_depolarizing_2q(p_kraus))
        return self

    def add_amplitude_damping(
        self, gamma: float, qubits: Any = None
    ) -> "NoiseModel":
        """Add amplitude damping (T1 decay) with rate gamma.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "amplitude_damping", gamma, None, "total", _normalize_targets(qubits, 1)
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_amplitude_damping(gamma))
        return self

    def add_phase_damping(
        self, gamma: float, qubits: Any = None
    ) -> "NoiseModel":
        """Add phase damping (T2 dephasing) with rate gamma.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "phase_damping", gamma, None, "total", _normalize_targets(qubits, 1)
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_phase_damping(gamma))
        return self

    def add_thermal_relaxation(
        self, t1: float, t2: float, gate_time: float, qubits: Any = None
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
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus)
        return self

    def add_bit_flip(self, p: float, qubits: Any = None) -> "NoiseModel":
        """Add bit-flip noise with probability p.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "bit_flip", p, None, "total", _normalize_targets(qubits, 1)
        )
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_bit_flip(p))
        return self

    def add_phase_flip(self, p: float, qubits: Any = None) -> "NoiseModel":
        """Add phase-flip noise with probability p.

        ``qubits`` binds the channel to specific wires instead of every
        qubit (int or int sequence).
        """
        channel = NoiseChannel(
            "phase_flip", p, None, "total", _normalize_targets(qubits, 1)
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
        self, p: float, convention: str = "total", qubits: Any = None
    ) -> "NoiseModel":
        """Convenience wrapper for add_depolarizing with n_qubits=2.

        ``qubits`` binds the channel to exactly one ordered pair (e.g.
        ``qubits=(0, 1)``); ``None`` applies it after every two-qubit
        gate.
        """
        return self.add_depolarizing(
            p, n_qubits=2, convention=convention, qubits=qubits
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
        """Apply the 2-qubit Kraus channels matching the ordered pair ``(q0, q1)``."""
        from superfermion.backends.density_matrix import _apply_kraus_2q
        for idx, kraus_set in enumerate(self._2q_kraus):
            targets = _channel_targets(self.two_qubit_channels, idx)
            if targets is not None and targets != (q0, q1):
                continue
            rho = _apply_kraus_2q(rho, kraus_set, q0, q1, n)
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

        for q in range(n_qubits):
            composite: Optional[List[np.ndarray]] = None
            for idx, kraus_set in enumerate(self._1q_kraus):
                targets = _channel_targets(self.single_qubit_channels, idx)
                if targets is not None and q not in targets:
                    continue
                if composite is None:
                    composite = list(kraus_set)
                else:
                    composite = _compose_kraus(kraus_set, composite)
            if composite is not None:
                ops.append((q, _flat_kraus(composite, 2)))
        return ops

    def to_rust_kraus_ops_2q(self) -> List[List[float]]:
        """Flat Kraus sets for the 2-qubit channels (native DM path).

        Returns one flat list per 2q channel, in add-order.  Each 4x4 Kraus
        matrix is flattened as 32 floats (re00, im00, re01, im01, ... re33,
        im33).  The Rust core applies the channels sequentially after every
        2q gate (after the fused gate + 1q-channel sweep), which matches the
        per-channel sequential semantics of :meth:`apply_2q`.
        """
        return [_flat_kraus(kraus_set, 4) for kraus_set in self._2q_kraus]

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize noise model to a dictionary.

        Includes the model-level ``placement``.  Per-channel construction
        conventions (the depolarizing normalization) are not serialized;
        ``error_rate`` is the nominal value as passed to the ``add_*``
        methods.  A channel's ``qubits`` target appears only when one was
        given, so untargeted models keep the historical exact dict.
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
