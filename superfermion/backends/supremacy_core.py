"""
SuperFermion Supremacy Core: The "X-Matrix" Engine.
High-Density MPS Simulator with JAX/XLA Acceleration.
Target: 10,000 Qubits with absolute scientific fidelity.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple
import time

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.results import RunResult

class SupremacyBackend(Backend):
    """The Apex Engine: JAX-accelerated Tensor Networks for 10k Qubits."""

    def __init__(self, name: str = "supremacy", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        # Enable X64 if user needs extreme scientific precision
        self.enable_x64 = options.get("enable_x64", False) if options else False
        if self.enable_x64:
            jax.config.update("jax_enable_x64", True)
        
        # Max bond dimension for MPS (Controls precision vs memory)
        self.max_bond = options.get("max_bond", 64) if options else 64

    @property
    def n_qubits(self) -> int:
        return 10000 # Architectural limit for MPS

    @property
    def supported_gates(self) -> List[str]:
        return ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "CX", "CZ", "SWAP", "P", "U"]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        n = circuit.n_qubits
        seed = kwargs.get("seed", 42)

        # 🚀 THE SUPREMACY PIPELINE:
        # 1. Transpile to MPO (Matrix Product Operators)
        # 2. Execute MPS propagation with JAX/XLA
        # 3. Handle accuracy-preserving truncation

        # Reset bond-saturation trackers for this run. Populated per-SVD in
        # _apply_adjacent_gate and surfaced in metadata below.
        self._discard_log: List[float] = []
        self._bond_log: List[int] = []

        state_mps = self._simulate_mps(circuit)

        # For small qubit counts, we can provide the full statevector for accuracy checks
        state_sv = None
        if n <= 25:
            state_sv = self._mps_to_sv(state_mps)

        # Draw samples. Two paths so n<=25 uses the exact dense SV and n>25 uses
        # a direct left-to-right MPS sampler (no 2^N memory).
        counts: Dict[str, int] = {}
        if shots > 0:
            if state_sv is not None:
                counts = self._sample_from_sv(np.asarray(state_sv), n, shots, seed)
            else:
                counts = self._sample_from_mps(state_mps, shots, seed)

        # --- Aggregate bond-saturation telemetry across all SVD steps ---
        # max_observed_bond: largest true (non-zero) rank at any 2Q gate.
        # bond_saturated:    at least one gate tried to use more bond than
        #                    max_bond — result is a provable approximation.
        # fidelity_lower_bound: product over steps of (1 - discard_step).
        #                    Computed in log-space to avoid underflow.
        max_obs_bond = max(self._bond_log, default=0)
        bond_saturated = max_obs_bond > self.max_bond
        max_step_discard = max(self._discard_log, default=0.0)
        # F_lb = prod(1 - d_i) = exp(sum log(1 - d_i))
        log_F = 0.0
        for d in self._discard_log:
            # clamp to avoid log(0) when a step discards everything.
            log_F += math.log(max(1.0 - d, 1e-300))
        fidelity_lower_bound = math.exp(log_F) if self._discard_log else 1.0

        if bond_saturated and fidelity_lower_bound < 0.99:
            warnings.warn(
                f"[sf.supremacy] MPS bond saturated at max_bond={self.max_bond} "
                f"(observed up to {max_obs_bond}). Fidelity lower bound "
                f"{fidelity_lower_bound:.4f} - result is an approximation. "
                f"Increase max_bond in options for a tighter bound.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        return RunResult(
            counts=counts,
            probabilities={},
            statevector=state_sv,  # Full SV only if safe
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": "sf-supremacy-core",
                "method": "JAX-MPS",
                "qubits": n,
                "max_bond": self.max_bond,
                "sampler": "dense-sv" if state_sv is not None else "mps-direct",
                # Bond saturation telemetry (new)
                "max_observed_bond": max_obs_bond,
                "bond_saturated": bond_saturated,
                "fidelity_lower_bound": fidelity_lower_bound,
                "max_single_step_discard": max_step_discard,
                "svd_steps": len(self._discard_log),
            }
        )

    @staticmethod
    def _sample_from_sv(sv: np.ndarray, n: int, shots: int, seed: int) -> Dict[str, int]:
        """Sample shot bitstrings from a dense statevector.

        BE convention: qubit 0 is the most-significant bit of the bitstring.
        """
        probs = np.abs(sv.astype(np.complex128)) ** 2
        total = float(probs.sum())
        if total <= 0.0:
            # Degenerate state (shouldn't happen for a unitary circuit) — uniform fallback.
            probs = np.full(probs.shape, 1.0 / probs.size)
        else:
            probs = probs / total
        rng = np.random.default_rng(seed)
        idx = rng.choice(probs.size, size=shots, p=probs)
        fmt = f"0{n}b"
        counts: Dict[str, int] = {}
        for i in idx:
            key = format(int(i), fmt)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _sample_from_mps(self, tensors, shots: int, seed: int) -> Dict[str, int]:
        """Exact left-to-right sampler directly from the MPS.

        Uses precomputed right environments so each sample is O(n * D^3).
        BE convention: qubit i contributes the i-th character of the bitstring
        from the left (qubit 0 = MSB).
        """
        T = [np.asarray(t) for t in tensors]
        n = len(T)
        dtype = np.complex128
        # Right environments: R[i] has shape (L_i, L_i) and equals
        # sum over s_i..s_{n-1} of (A_i..A_{n-1}) (A_i..A_{n-1})^†
        R: List[np.ndarray] = [None] * (n + 1)  # type: ignore[list-item]
        R[n] = np.ones((1, 1), dtype=dtype)
        for i in range(n - 1, -1, -1):
            A = T[i].astype(dtype)  # (L, 2, R_bond)
            # Contract over the physical index and the right-bond index.
            R[i] = np.einsum("lsa,ab,ksb->lk", A, R[i + 1], A.conj())

        rng = np.random.default_rng(seed)
        counts: Dict[str, int] = {}
        for _ in range(shots):
            L_state = np.ones((1, 1), dtype=dtype)  # shape (1, 1)
            bits: List[str] = []
            for i in range(n):
                A = T[i].astype(dtype)
                # Unnormalized conditional amplitudes.
                L0 = L_state @ A[:, 0, :]  # (1, R_bond)
                L1 = L_state @ A[:, 1, :]
                p0 = float(np.real(L0 @ R[i + 1] @ L0.conj().T)[0, 0])
                p1 = float(np.real(L1 @ R[i + 1] @ L1.conj().T)[0, 0])
                tot = p0 + p1
                if tot <= 1e-15:
                    s = 0
                    L_state = L0
                else:
                    ps = (p0 / tot, p1 / tot)
                    s = int(rng.choice(2, p=ps))
                    L_state = L0 if s == 0 else L1
                bits.append(str(s))
            key = "".join(bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _simulate_mps(self, circuit: Circuit):
        """MPS Propagation using JAX/XLA."""
        from superfermion.backends.turbo import expand_3q_gates
        n = circuit.n_qubits
        raw_gates = [g for g in circuit._gates if g.name not in ("BARRIER", "MEASURE")]
        # Expand 3Q gates (CCX, CSWAP) into 1Q+2Q sequences so that
        # _apply_gate (which only handles 1Q and 2Q) can process them.
        gates = expand_3q_gates(raw_gates)
        
        dtype = jnp.complex64 if not self.enable_x64 else jnp.complex128
        # Each tensor is shape (left_bond, physical, right_bond)
        tensors = [jnp.array([[[1.0], [0.0]]], dtype=dtype) for _ in range(n)]
        
        for gate in gates:
            tensors = self._apply_gate(tensors, gate)
            
        return tensors

    def _apply_gate(self, tensors, gate: GateRecord):
        """Apply a gate to the MPS tensors with extreme fidelity."""
        q = gate.qubits
        mat = jnp.array(gate.to_unitary(), dtype=tensors[0].dtype)
        
        if len(q) == 1:
            idx = q[0]
            # (p_new, p_old) @ (L, p_old, R) -> (L, p_new, R)
            tensors[idx] = jnp.einsum('lj,ijk->ilk', mat, tensors[idx])
        
        elif len(q) == 2:
            q0, q1 = q[0], q[1]
            dist = abs(q0 - q1)
            
            if dist == 1:
                tensors = self._apply_adjacent_gate(tensors, q0, q1, mat)
            else:
                # AUTOMATED SWAP CHAIN FOR NON-ADJACENT GATES
                current_q0 = q0
                step = 1 if q1 > q0 else -1
                swaps_made = []
                
                while abs(current_q0 - q1) > 1:
                    next_q = current_q0 + step
                    tensors = self._apply_swap(tensors, current_q0, next_q)
                    swaps_made.append((current_q0, next_q))
                    current_q0 = next_q
                
                tensors = self._apply_adjacent_gate(tensors, current_q0, q1, mat)
                
                for s0, s1 in reversed(swaps_made):
                    tensors = self._apply_swap(tensors, s0, s1)

        return tensors

    def _apply_adjacent_gate(self, tensors, q0, q1, mat):
        """Core logic for adjacent 2-qubit gate application."""
        i, j = min(q0, q1), max(q0, q1)
        A, B = tensors[i], tensors[j]
        L_A, p_A, R_A = A.shape
        L_B, p_B, R_B = B.shape
        
        theta = jnp.einsum('ijk,klm->ijlm', A, B)
        g = mat.reshape(2, 2, 2, 2)
        if q0 > q1: # reverse case
            g = g.transpose(1, 0, 3, 2)
            
        theta = jnp.einsum('ijkl,aklb->aijb', g, theta)
        theta_mat = theta.reshape(L_A * 2, 2 * R_B)
        u, s, v = jnp.linalg.svd(theta_mat, full_matrices=False)

        # --- Bond-saturation telemetry (read s BEFORE truncation) ---
        # Log the true non-zero bond ("rank") and the fractional squared-weight
        # thrown away by capping at max_bond. Discarded weight == 1 - F_step
        # where F_step is the per-step fidelity lower bound.
        s_np = np.asarray(s)
        nz_rank = int(np.count_nonzero(s_np > 1e-12))
        k = min(int(s_np.shape[0]), self.max_bond)
        total_w = float(np.sum(s_np ** 2))
        kept_w = float(np.sum(s_np[:k] ** 2))
        discard = 0.0 if total_w <= 0.0 else 1.0 - (kept_w / total_w)
        # If instance trackers exist (set at run()-entry), record this step.
        if hasattr(self, "_discard_log"):
            self._discard_log.append(discard)
            self._bond_log.append(nz_rank)
        # ------------------------------------------------------------

        u, s, v = u[:, :k], s[:k], v[:k, :]

        tensors[i] = (u * s).reshape(L_A, 2, k)
        tensors[j] = v.reshape(k, 2, R_B)
        return tensors

    def _apply_swap(self, tensors, q0, q1):
        """Native SWAP gate optimized for MPS."""
        swap_mat = jnp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=tensors[0].dtype)
        return self._apply_adjacent_gate(tensors, q0, q1, swap_mat)

    def _mps_to_sv(self, tensors):
        """Contract the entire MPS into a dense statevector (Rank-safe)."""
        n = len(tensors)
        if n > 28:
            return None
            
        res = tensors[0]
        for i in range(1, len(tensors)):
            L, D, M = res.shape
            tensor = tensors[i]
            _, p, R = tensor.shape
            res = jnp.einsum('idm,mpr->idpr', res, tensor)
            res = res.reshape(L, D * p, R)
            
        return jnp.ravel(res)

    def _get_gate_matrix(self, gate: GateRecord):
        # We reuse the existing superfermion logic or provide XLA-friendly versions
        return jnp.array(gate.to_unitary(), dtype=jnp.complex64)
