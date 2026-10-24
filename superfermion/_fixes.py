"""Additive compatibility shims for SuperFermion 0.1.15 core-output defects.

Nothing in the compiled core is replaced.  This module adds:

* ``LargeUnitaryDAG`` — a thin proxy returned by ``Circuit.to_ir()`` when a
  circuit contains an opaque unitary on more than three qubits.  The 0.1.15
  statevector kernel only applies opaque matrices for arities 1-3 and
  silently drops larger ones (D1); the proxy evaluates the same circuit from
  the full circuit unitary instead, and raises for methods that cannot
  represent it.
* `unpermute_statevector` / `remap_counts` — helpers that undo the MPS
  routing permutation (D10).
* A wrapper on ``MPSState.to_statevector`` that applies that inverse
  permutation, so direct MPS reads use the documented qubit order.
* Matrix JSON helpers shared by the two circuit serializers (D2).

All additions are read-only with respect to the Rust extension and the
installed wheel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

import superfermion._sf_core as _core

# The D1 fallback materializes a 2^n x 2^n unitary.  Above this cap the
# allocation is larger than the benefit of a silent fallback; raise instead.
_MAX_DENSE_UNITARY_QUBITS = 12

def _mps_perm_error() -> str:
    """Build the arity error with the running package version, not a literal.

    The message is user-facing, so it carries the version of the package that
    actually raised it; a hardcoded one goes stale on every release.
    """
    from superfermion import __version__

    return (
        f"SuperFermion {__version__} applies opaque unitary matrices on at "
        "most three qubits for this execution path. The circuit contains a "
        "larger unitary; use method='statevector' (which has a compatibility "
        "fallback) or decompose the matrix into <=3-qubit gates."
    )


def matrix_to_json(matrix: np.ndarray) -> List[List[List[float]]]:
    """Encode a complex matrix as JSON-safe ``[[[re, im], ...], ...]``."""
    arr = np.asarray(matrix, dtype=complex)
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in arr
    ]


def matrix_from_json(raw: Any) -> np.ndarray:
    """Decode the ``matrix_to_json`` representation back to complex128."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(
            "Serialized unitary matrix must be a list of rows of "
            "[real, imag] pairs; got shape "
            f"{arr.shape}"
        )
    return arr[..., 0] + 1j * arr[..., 1]


def mps_routing_permutation(
    gate_records: Sequence[Sequence[Any]],
    n_qubits: int,
) -> List[int]:
    """Replicate the MPS lazy-SWAP routing map in pure Python.

    Mirrors ``MPSState::apply_gate_lazy_routed`` from the 0.1.15 Rust core
    (``crates/sf-ir/src/mps.rs``): a non-adjacent two-qubit gate moves the
    higher-positioned virtual qubit leftward until it is adjacent to the
    lower one, and the qubit stays at its new site (no swap-back).  The
    returned list uses ``perm[virtual] = site``, matching
    ``MPSState.perm()``, so it can be obtained without a second MPS
    evolution just to read the map.
    """
    perm = list(range(int(n_qubits)))
    perm_inv = list(range(int(n_qubits)))
    for record in gate_records:
        qubits = record[1] if len(record) > 1 else ()
        if len(qubits) != 2:
            continue
        v1, v2 = int(qubits[0]), int(qubits[1])
        if v1 == v2:
            continue
        p1, p2 = perm[v1], perm[v2]
        if abs(p1 - p2) <= 1:
            continue
        plo = min(p1, p2)
        phi = v1 if p1 > p2 else v2
        while perm[phi] > plo + 1:
            lo = perm[phi] - 1
            vi, vj = perm_inv[lo], perm_inv[lo + 1]
            perm[vi], perm[vj] = lo + 1, lo
            perm_inv[lo], perm_inv[lo + 1] = vj, vi
    return perm


def unpermute_statevector(
    statevector: Sequence[complex] | np.ndarray,
    perm: Sequence[int],
) -> np.ndarray:
    """Return amplitudes in qubit order from MPS physical-site order.

    ``perm[virtual] = site`` is the convention exposed by
    ``MPSState.perm()``.  A basis state at physical sites is relabeled by
    ``out[v] = sv[s]`` with ``s[perm[q]] = v[q]``; in tensor form that is a
    transpose by the *inverse* permutation (verified against the 0.1.15
    wheel: a long-range CNOT gives ``perm=[0, 2, 3, 1]`` and this transpose
    reproduces the statevector method amplitude for amplitude).
    """
    perm_list = [int(p) for p in perm]
    n = len(perm_list)
    sv = np.asarray(statevector)
    if n <= 1 or perm_list == list(range(n)):
        return sv
    if sv.ndim != 1 or sv.size != (1 << n):
        return sv
    # ``reshape([2] * n)`` makes axis 0 the most-significant bit, while the
    # permutation is expressed per qubit (least-significant bit q); the
    # axis order therefore reverses ``perm`` in both position and value.
    axes = [n - 1 - perm_list[n - 1 - k] for k in range(n)]
    return sv.reshape([2] * n).transpose(axes).reshape(-1)


def remap_counts(
    counts: Dict[str, int],
    perm: Sequence[int],
) -> Dict[str, int]:
    """Relabel MPS counts from physical-site order to qubit order.

    Character ``i`` of an SF counts key is qubit ``n-1-i``; the mapping is
    applied per observed key, so sampling cost stays proportional to the
    number of distinct outcomes rather than ``2^n``.
    """
    perm_list = [int(p) for p in perm]
    n = len(perm_list)
    if n == 0 or perm_list == list(range(n)):
        return counts
    out: Dict[str, int] = {}
    for key, count in counts.items():
        site_bits = int(key, 2)
        virtual_bits = 0
        for qubit, site in enumerate(perm_list):
            if (site_bits >> site) & 1:
                virtual_bits |= 1 << qubit
        new_key = format(virtual_bits, f"0{n}b")
        out[new_key] = out.get(new_key, 0) + int(count)
    return out


class PermutedMPSState:
    """Expose an MPS ``State`` handle in documented qubit order (D10).

    ``State.numpy()`` / ``State.sample()`` / ``State.probabilities()`` keep
    the physical-site layout of the lazy-SWAP routing.  Every other method
    is delegated unchanged to the Rust handle.
    """

    def __init__(self, state: Any, perm: Sequence[int]) -> None:
        self._state = state
        self._perm = [int(p) for p in perm]

    def numpy(self) -> np.ndarray:
        return unpermute_statevector(self._state.numpy(), self._perm)

    def sample(self, shots: int, seed: int = 42) -> Dict[str, int]:
        return remap_counts(self._state.sample(shots, seed), self._perm)

    def probabilities(self) -> np.ndarray:
        statevector = self.numpy()
        return (np.abs(statevector) ** 2).astype(float)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)

    def __repr__(self) -> str:
        return f"PermutedMPSState(perm={self._perm}, state={self._state!r})"


class LargeUnitaryDAG:
    """Duck-typed ``QuantumDAG`` for circuits with a >3-qubit unitary.

    ``Circuit.to_ir()`` returns this proxy only when such a gate is present.
    Normal circuits keep the raw Rust DAG and pay no Python dispatch cost.
    """

    def __init__(self, dag: Any, n_qubits: int) -> None:
        self._dag = dag
        self._n_qubits = int(n_qubits)

    # -- introspection -------------------------------------------------
    def n_qubits(self) -> int:
        return self._n_qubits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dag, name)

    def __repr__(self) -> str:
        return f"LargeUnitaryDAG(n_qubits={self._n_qubits}, dag={self._dag!r})"

    # -- statevector fallback -----------------------------------------
    def _statevector(self) -> np.ndarray:
        if self._n_qubits > _MAX_DENSE_UNITARY_QUBITS:
            raise ValueError(
                "Circuit contains a >3-qubit opaque unitary and "
                f"{self._n_qubits} qubits. The compatibility path builds the "
                f"full 2^n x 2^n unitary, capped at "
                f"{_MAX_DENSE_UNITARY_QUBITS} qubits. Decompose the matrix "
                "into <=3-qubit gates, or use a smaller register."
            )
        unitary = np.asarray(self._dag.to_unitary())
        return unitary[:, 0].copy()

    def simulate_to_state(self, method: str = "statevector", *args: Any, **kwargs: Any) -> Any:
        method_name = str(method).lower()
        if method_name == "statevector":
            return _core.State.from_numpy(self._statevector(), self._n_qubits)
        if method_name in ("mps", "density_matrix", "stabilizer"):
            raise ValueError(_mps_perm_error())
        return self._dag.simulate_to_state(method, *args, **kwargs)

    def simulate_and_sample(self, shots: int, seed: int = 42) -> Dict[str, int]:
        return self.simulate_to_state("statevector").sample(shots, seed)

    def simulate(self) -> np.ndarray:
        return np.asarray(self.simulate_to_state("statevector").numpy())

    def _unsupported(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError(_mps_perm_error())

    # Methods that would otherwise reach the arity-limited Rust kernels.
    simulate_dm = _unsupported
    simulate_dm_noisy = _unsupported
    simulate_dm_noisy_state = _unsupported
    simulate_dm_noisy_state_view = _unsupported
    simulate_dynamic = _unsupported
    simulate_dynamic_mps = _unsupported
    simulate_f32 = _unsupported
    simulate_msb = _unsupported
    simulate_on = _unsupported
    evolve_mps = _unsupported
    mps_norm_sq = _unsupported
    sample_mps = _unsupported
    simulate_mps_pauli_expval = _unsupported
    simulate_mps_pauli_expval_batch = _unsupported
    adjoint_grad = _unsupported


def _install_mps_output_fix() -> None:
    """Apply the inverse routing permutation in ``MPSState.to_statevector``."""
    mps_state = getattr(_core, "MPSState", None)
    if mps_state is None:
        return
    original = getattr(mps_state, "to_statevector", None)
    if original is None or getattr(original, "_sf_qubit_order", False):
        return

    def to_statevector(self: Any) -> np.ndarray:
        raw = np.asarray(original(self))
        return unpermute_statevector(raw, self.perm())

    to_statevector._sf_qubit_order = True  # type: ignore[attr-defined]
    mps_state.to_statevector = to_statevector

    # Preserve the pre-fix physical-site view for diagnostics/tests.
    try:
        mps_state.to_statevector_sites = original
    except (AttributeError, TypeError):
        pass


_install_mps_output_fix()


def reject_compat_dag(dag: Any, api: str) -> None:
    """Refuse a :class:`LargeUnitaryDAG` where the core needs a QuantumDAG.

    ``sf.cat`` and ``sf.prepare`` hand their argument to a PyO3 constructor
    typed ``QuantumDAG``, so the D1 compatibility proxy would surface as a
    bare ``TypeError`` from the binding layer.  This raises the directed
    error instead, at the API the caller actually used.
    """
    if isinstance(dag, LargeUnitaryDAG):
        raise ValueError(
            f"{api} requires a native QuantumDAG and cannot take the "
            "compatibility proxy used for opaque unitaries above three "
            "qubits. Decompose the matrix into <=3-qubit gates, or execute "
            "the circuit with sf.run(method='statevector')."
        )


__all__ = [
    "LargeUnitaryDAG",
    "PermutedMPSState",
    "matrix_from_json",
    "matrix_to_json",
    "mps_routing_permutation",
    "reject_compat_dag",
    "remap_counts",
    "unpermute_statevector",
]
