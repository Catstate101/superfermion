"""Noisy density-matrix state handle (additive fix for noisy ``density_matrix`` runs).

``RustDevice._run_density_matrix`` computes the *noisy* density matrix and
stores it in ``RunResult.metadata["density_matrix"]`` (used by
``probabilities``/``counts``/``metadata["purity"]``), but builds
``RunResult.state`` via ``dag.simulate_to_state("density_matrix")`` — a Rust
binding with no noise path — so the state handle (and therefore
``RunResult.expectation()``/``variance()``, ``state.purity()``, etc.)
silently reflected the *noiseless* state under a gate-noise model.

This module provides ``NoisyDensityMatrixState``: a drop-in Python state
handle backed by the noisy rho, mirroring the interface and semantics of
the Rust ``density_matrix`` state handle (``sf.State`` with
``method='density_matrix'``) so noisy runs behave exactly as the Rust
handle would if it had been built from the noisy rho.

Convention notes (verified empirically against the Rust handle):
- Rust stores the density matrix in the little-endian computational basis
  (index bit q = qubit q); the public ``metadata["density_matrix"]`` is
  bit-reversed (qubit-0-first) relative to it.
- ``expectation()`` maps ``paulis[q]`` to qubit q (bit position q), same as
  the Rust ``DensityMatrixStateWrapper``.
- ``numpy()`` returns the row-major flatten of the little-endian matrix,
  matching the Rust handle's ``numpy()`` layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from superfermion.utils.exceptions import MethodError


def _to_little_endian(rho_public: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert public (qubit-0-first) rho to the Rust little-endian basis.

    The permutation is its own inverse, so the same helper converts back.
    The public rho is already the Hermitian density matrix (the Rust path
    applies ``.conj()`` before the reversal), so no conjugation is needed.
    """
    from superfermion.backends.density_matrix import _reverse_qubits_dm
    return _reverse_qubits_dm(rho_public, n_qubits)


class NoisyDensityMatrixState:
    """Drop-in ``sf.State`` handle for the noisy density-matrix method.

    Implements the surface of the Rust density-matrix state handle backed
    by the noisy rho (little-endian basis). ``grad()``/``qfim()`` raise
    ``MethodError`` with the same messages as the Rust wrapper.

    Args:
        rho_le: Density matrix in the little-endian basis (2^n x 2^n).
        n_qubits: Number of qubits.
        device: Device label, mirroring the Rust handle's ``device``.
    """

    def __init__(
        self,
        rho_le: np.ndarray,
        n_qubits: int,
        device: str = "cpu",
    ) -> None:
        rho = np.asarray(rho_le, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(f"rho must be a square matrix, got shape {rho.shape}")
        if rho.shape[0] != (1 << n_qubits):
            raise ValueError(
                f"rho shape {rho.shape} does not match 2^{n_qubits} = {1 << n_qubits}"
            )
        self._rho_le = rho
        self._n_qubits = n_qubits
        self._device = device

    @classmethod
    def from_public_rho(
        cls,
        rho_public: np.ndarray,
        n_qubits: int,
        device: str = "cpu",
    ) -> NoisyDensityMatrixState:
        """Build from the public metadata rho (qubit-0-first ordering)."""
        return cls(_to_little_endian(np.asarray(rho_public), n_qubits), n_qubits, device)

    # ── attributes (mirror the Rust handle getters) ──────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def method(self) -> str:
        return "density_matrix"

    @property
    def device(self) -> str:
        return self._device

    @property
    def shape(self) -> list[int]:
        dim = 1 << self._n_qubits
        return [dim, dim]

    def __repr__(self) -> str:
        return (
            f"State(n_qubits={self._n_qubits}, method='density_matrix', "
            f"device='{self._device}')"
        )

    # ── measurement / information API (mirror DensityMatrixStateWrapper) ─

    def expectation(self, observable) -> float:
        """⟨O⟩ = Tr(ρ O) for Pauli observable terms.

        Same term format and qubit mapping as the Rust handle:
        terms are ``(paulis, coef_re, coef_im)`` with ``paulis[q]`` the
        Pauli (0=I, 1=X, 2=Y, 3=Z) on qubit q (bit position q).
        """
        return float(np.real(self._expval_complex(observable)))

    def variance(self, observable) -> float:
        """Var(O) = ⟨O²⟩ − ⟨O⟩² for a Pauli observable."""
        obs = self._observable_matrix(observable)
        rho = self._rho_le
        mean = float(np.real(np.trace(rho @ obs)))
        mean_sq = float(np.real(np.trace(rho @ obs @ obs)))
        return mean_sq - mean * mean

    def sample(self, shots: int, seed: int = 42) -> dict[str, int]:
        """Sample computational-basis outcomes from the noisy rho.

        Bitstring display matches the Rust handle: qubit q is bit q of the
        sampled index, rendered qubit-0-last.
        """
        if shots <= 0:
            return {}
        n = self._n_qubits
        dim = 1 << n
        probs = np.maximum(np.real(np.diag(self._rho_le)), 0.0)
        total = probs.sum()
        if total <= 0.0:
            return {}
        probs = probs / total
        rng = np.random.default_rng(seed)
        idx = rng.choice(dim, size=shots, p=probs)
        counts: dict[str, int] = {}
        for i in idx:
            bs = "".join(str((i >> q) & 1) for q in range(n - 1, -1, -1))
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def probabilities(self) -> np.ndarray:
        """Diagonal of the noisy rho (little-endian index order)."""
        return np.maximum(np.real(np.diag(self._rho_le)), 0.0)

    def numpy(self) -> np.ndarray:
        """Row-major flatten of the little-endian noisy rho (complex128)."""
        return self._rho_le.ravel()

    def purity(self) -> float:
        return float(np.real(np.trace(self._rho_le @ self._rho_le)))

    def entropy(self) -> float:
        """Von Neumann entropy S = −Σ λ ln λ (natural log)."""
        eig = np.linalg.eigvalsh((self._rho_le + self._rho_le.conj().T) / 2)
        s = 0.0
        for lam in eig:
            if lam > 1e-15:
                s -= lam * np.log(lam)
        return float(s)

    def fidelity(self, other: Any) -> float:
        """|Tr(ρ σ)| for another density-matrix state (same convention as Rust)."""
        n = self._n_qubits
        if isinstance(other, NoisyDensityMatrixState):
            if other._n_qubits != n:
                raise MethodError("fidelity() requires same number of qubits")
            sigma = other._rho_le
        elif type(other).__name__ == "State" and getattr(other, "method", None) == "density_matrix":
            dim = 1 << n
            flat = np.asarray(other.numpy(), dtype=np.complex128)
            if flat.size != dim * dim:
                raise MethodError("fidelity() requires same number of qubits")
            sigma = flat.reshape(dim, dim)
        else:
            raise MethodError(
                "fidelity() between density_matrix states requires both to be density_matrix"
            )
        return abs(float(np.real(np.trace(self._rho_le @ sigma))))

    def partial_trace(self, keep_qubits: list[int]) -> NoisyDensityMatrixState:
        """Trace out qubits not in ``keep_qubits`` (mirrors the Rust method).

        In the returned state, qubit k corresponds to the original qubit
        ``keep_qubits[k]`` (same indexing as the Rust handle).
        """
        n = self._n_qubits
        n_keep = len(keep_qubits)
        dim_keep = 1 << n_keep
        traced = sorted(q for q in range(n) if q not in keep_qubits)
        rho = self._rho_le
        reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        for i_keep in range(dim_keep):
            for j_keep in range(dim_keep):
                val = 0.0 + 0.0j
                for t in range(1 << len(traced)):
                    idx_i = 0
                    idx_j = 0
                    for k, q in enumerate(keep_qubits):
                        if (i_keep >> k) & 1:
                            idx_i |= 1 << q
                        if (j_keep >> k) & 1:
                            idx_j |= 1 << q
                    for k, q in enumerate(traced):
                        if (t >> k) & 1:
                            idx_i |= 1 << q
                            idx_j |= 1 << q
                    val += rho[idx_i, idx_j]
                reduced[i_keep, j_keep] = val
        return NoisyDensityMatrixState(reduced, n_keep, self._device)

    def mutual_info(self, wires_a: list[int], wires_b: list[int]) -> float:
        """I(A;B) = S(A) + S(B) − S(AB) in nats (mirrors the Rust default)."""
        ab = sorted(set(wires_a) | set(wires_b))
        s_a = self.partial_trace(sorted(wires_a)).entropy()
        s_b = self.partial_trace(sorted(wires_b)).entropy()
        s_ab = self.partial_trace(ab).entropy()
        return s_a + s_b - s_ab

    # ── unsupported operations (mirror the Rust wrapper errors) ──────────

    def grad(self, observable, dag=None, param_values=None):
        raise MethodError("grad() not supported for density_matrix method")

    def qfim(self, dag=None, param_values=None):
        raise MethodError("qfim() not supported for density_matrix method")

    # ── internal helpers ─────────────────────────────────────────────────

    def _pauli_matrix(self, pauli: int) -> np.ndarray:
        if pauli == 0:
            return np.eye(2, dtype=np.complex128)
        if pauli == 1:
            return np.array([[0, 1], [1, 0]], dtype=np.complex128)
        if pauli == 2:
            return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        if pauli == 3:
            return np.array([[1, 0], [0, -1]], dtype=np.complex128)
        raise ValueError(f"unknown Pauli code {pauli} (expected 0..3)")

    def _expval_complex(self, observable) -> complex:
        """Complex Tr(ρ O) mirroring the Rust wrapper's bit-phase loop."""
        n = self._n_qubits
        dim = 1 << n
        rho = self._rho_le
        total = 0.0 + 0.0j
        for paulis, coef_re, coef_im in observable:
            coef = complex(coef_re, coef_im)
            if len(paulis) > n:
                raise ValueError(
                    f"observable paulis length {len(paulis)} exceeds n_qubits={n}"
                )
            for i in range(dim):
                phase = 1.0 + 0.0j
                target = i
                for q, op in enumerate(paulis):
                    bit = 1 << q
                    if op == 1:  # X flips bit q
                        target ^= bit
                    elif op == 2:  # Y flips bit q with phase
                        target ^= bit
                        if (i >> q) & 1 == 0:
                            phase *= -1j
                        else:
                            phase *= 1j
                    elif op == 3 and ((i >> q) & 1) == 1:  # Z phase
                        phase *= -1.0
                total += coef * phase * rho[target, i]
        return total

    def _observable_matrix(self, observable) -> np.ndarray:
        """Full little-endian observable matrix O (for variance)."""
        n = self._n_qubits
        dim = 1 << n
        obs = np.zeros((dim, dim), dtype=np.complex128)
        for paulis, coef_re, coef_im in observable:
            if len(paulis) > n:
                raise ValueError(
                    f"observable paulis length {len(paulis)} exceeds n_qubits={n}"
                )
            op = np.array([[1.0]], dtype=np.complex128)
            # little-endian kron: qubit q is factor (n-1-q), q0 rightmost
            for q in range(n - 1, -1, -1):
                p = paulis[q] if q < len(paulis) else 0
                op = np.kron(op, self._pauli_matrix(p))
            obs += complex(coef_re, coef_im) * op
        return obs
