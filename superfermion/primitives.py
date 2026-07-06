"""
Qiskit-compatible Primitives API for SuperFermion.

Provides ``SFEstimator`` and ``SFSampler`` with the same interface as
``qiskit.primitives.StatevectorEstimator`` and ``qiskit.primitives.StatevectorSampler``,
so code written for Qiskit Primitives v2 runs unchanged with SF backends.

Cross-validated against:
  qiskit.primitives.StatevectorEstimator
  qiskit.primitives.StatevectorSampler

Usage:
    # Exact replicas of Qiskit API
    >>> from superfermion.primitives import SFEstimator, SFSampler
    >>> from superfermion.observables.core import SparsePauliOp

    # Estimator: ⟨ψ|H|ψ⟩
    >>> est = SFEstimator(backend='statevector')
    >>> job = est.run([(circuit, observable)])
    >>> result = job.result()
    >>> print(result[0].data.evs)   # expectation value

    # Sampler: counts/quasi-probability distribution
    >>> samp = SFSampler(backend='statevector')
    >>> job = samp.run([circuit], shots=1000)
    >>> result = job.result()
    >>> print(result[0].data.meas)  # ShotResult with counts/quasi_probs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import superfermion as sf
from superfermion.observables.core import Observable, SparsePauliOp, PauliString, Hamiltonian


# ── Result types (mirror Qiskit Primitives v2) ──────────────────────────────────

@dataclass
class EstimatorData:
    evs: float           # expectation value
    stds: float = 0.0    # standard deviation (0 for exact)


@dataclass
class EstimatorPubResult:
    """Result of one Estimator PUB (Primitive Unified Bloc)."""
    data: EstimatorData
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotResult:
    """Measurement result: counts + quasi-probability distribution."""
    counts: Dict[str, int]
    quasi_probs: Dict[str, float]
    n_qubits: int

    @property
    def get_counts(self) -> Dict[str, int]:
        return self.counts


@dataclass
class SamplerData:
    meas: ShotResult


@dataclass
class SamplerPubResult:
    """Result of one Sampler PUB."""
    data: SamplerData
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Primitive Job (synchronous) ────────────────────────────────────────────────

class PrimitiveJob:
    """Synchronous job that holds a list of PubResults."""

    def __init__(self, pub_results: List):
        self._results = pub_results

    def result(self) -> List:
        return self._results

    def __getitem__(self, idx: int):
        return self._results[idx]


# ── SFEstimator ────────────────────────────────────────────────────────────────

class SFEstimator:
    """Compute expectation values ⟨ψ|O|ψ⟩ using any SF backend.

    Matches the Qiskit ``StatevectorEstimator`` interface (Primitives v2).

    Args:
        backend: SF backend name.
        shots:   0 = exact statevector (default); > 0 = shot-based.
        seed:    RNG seed for shot-based runs.

    Examples:
        Single PUB:
            job = estimator.run([(circuit, observable)])
            evs = job.result()[0].data.evs

        Multiple PUBs with parameter bindings:
            pubs = [(circ, H, param_values)]
            job = estimator.run(pubs)

        Qiskit-compatible (circuit is a Qiskit QuantumCircuit):
            from superfermion.bridge import from_qiskit
            sf_circ = from_qiskit(qk_circuit)
            sf_obs  = SparsePauliOp.from_qiskit(qk_observable)
            job = estimator.run([(sf_circ, sf_obs)])
    """

    def __init__(
        self,
        backend: str = "statevector",
        shots: int = 0,
        seed: int = 42,
    ):
        self.backend = backend
        self.shots = shots
        self.seed = seed

        from superfermion.backends.factory import get_backend
        self._sim = get_backend(backend)

    def run(
        self,
        pubs: Sequence[Tuple],
        shots: Optional[int] = None,
        precision: float = 0.0,
    ) -> PrimitiveJob:
        """Execute a sequence of PUBs.

        Each PUB is a tuple: ``(circuit, observable)`` or
        ``(circuit, observable, parameter_values)``.

        Args:
            pubs:      List of (circuit, observable [, param_values]).
            shots:     Override shots (0 = exact).
            precision: Ignored (Qiskit API compatibility).

        Returns:
            PrimitiveJob whose ``.result()`` returns a list of EstimatorPubResult.
        """
        _shots = shots if shots is not None else self.shots
        results: List[EstimatorPubResult] = []

        for pub in pubs:
            circuit, observable = pub[0], pub[1]
            param_values = pub[2] if len(pub) > 2 else None

            # Bind parameters if provided
            if param_values is not None:
                circuit = _bind_circuit(circuit, param_values)

            ev, std = self._estimate(circuit, observable, _shots)
            results.append(EstimatorPubResult(
                data=EstimatorData(evs=ev, stds=std),
                metadata={"backend": self.backend, "shots": _shots},
            ))

        return PrimitiveJob(results)

    def _estimate(self, circuit: sf.Circuit, observable: Observable, shots: int) -> Tuple[float, float]:
        result = self._sim.run(circuit, shots=shots, seed=self.seed)

        if result.statevector is not None:
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            ev = float(np.real(observable._fast_expval(sv)))
            return ev, 0.0

        # Shot-based: estimate from counts
        from superfermion.qml.gradient.parameter_shift import _expval_from_counts
        ev = _expval_from_counts(result.counts, observable, circuit.n_qubits)
        # Shot noise std: σ ≈ 1/√shots for bounded observables
        std = 1.0 / math.sqrt(max(shots, 1)) if shots > 0 else 0.0
        return ev, std


import math


# ── SFSampler ─────────────────────────────────────────────────────────────────

class SFSampler:
    """Sample quasi-probability distributions using any SF backend.

    Matches the Qiskit ``StatevectorSampler`` interface (Primitives v2).

    Args:
        backend: SF backend name.
        default_shots: Default shots per run.
        seed: RNG seed.

    Examples:
        job = sampler.run([circuit], shots=1000)
        result = job.result()
        counts = result[0].data.meas.counts
        qprobs = result[0].data.meas.quasi_probs
    """

    def __init__(
        self,
        backend: str = "statevector",
        default_shots: int = 1024,
        seed: int = 42,
    ):
        self.backend = backend
        self.default_shots = default_shots
        self.seed = seed

        from superfermion.backends.factory import get_backend
        self._sim = get_backend(backend)

    def run(
        self,
        pubs: Sequence,
        shots: Optional[int] = None,
    ) -> PrimitiveJob:
        """Execute a sequence of PUBs.

        Each PUB is a circuit or a tuple ``(circuit,)`` or
        ``(circuit, parameter_values)``.

        Args:
            pubs:  List of circuits or (circuit [, param_values]).
            shots: Override default_shots.

        Returns:
            PrimitiveJob whose ``.result()`` returns a list of SamplerPubResult.
        """
        _shots = shots if shots is not None else self.default_shots
        results: List[SamplerPubResult] = []

        for pub in pubs:
            if isinstance(pub, (list, tuple)):
                circuit = pub[0]
                param_values = pub[1] if len(pub) > 1 else None
            else:
                circuit = pub
                param_values = None

            if param_values is not None:
                circuit = _bind_circuit(circuit, param_values)

            counts, quasi_probs = self._sample(circuit, _shots)
            n = circuit.n_qubits
            shot_result = ShotResult(counts=counts, quasi_probs=quasi_probs, n_qubits=n)
            results.append(SamplerPubResult(
                data=SamplerData(meas=shot_result),
                metadata={"backend": self.backend, "shots": _shots},
            ))

        return PrimitiveJob(results)

    def _sample(
        self,
        circuit: sf.Circuit,
        shots: int,
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        result = self._sim.run(circuit, shots=shots, seed=self.seed)

        if result.counts:
            counts = dict(result.counts)
        elif result.statevector is not None:
            # Exact statevector → sample
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            probs = np.abs(sv) ** 2
            probs /= probs.sum()
            n = circuit.n_qubits
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(probs), size=shots, p=probs)
            counts = {}
            for idx in indices:
                bs = format(idx, f'0{n}b')
                counts[bs] = counts.get(bs, 0) + 1
        else:
            counts = {}

        total = sum(counts.values()) or 1
        quasi_probs = {bs: cnt / total for bs, cnt in counts.items()}
        return counts, quasi_probs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bind_circuit(circuit: sf.Circuit, param_values) -> sf.Circuit:
    """Bind parameter values to a parametric circuit.

    param_values can be:
      - dict: {name: value}
      - sequence: aligned with circuit.parameters
      - numpy array: same
    """
    if isinstance(param_values, dict):
        return circuit.bind(param_values)

    names = list(circuit.parameters)
    values = list(param_values)
    if len(values) < len(names):
        raise ValueError(
            f"Parameter mismatch: circuit has {len(names)} params "
            f"but {len(values)} values were provided."
        )
    return circuit.bind(dict(zip(names, [float(v) for v in values])))
