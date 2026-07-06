
"""
Industrial JAX-MPS Engine with Federated Lazy Routing.
Directly maps arbitrary topologies to physical sites via XLA-static kernels.
"""
from __future__ import annotations
import math
import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from typing import Any, Dict, List, Optional

import superfermion as sf
from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.results import RunResult

class JAXMPSBackend(Backend):
    """The High-Fidelity XLA-Accelerated Industrial Core."""

    def __init__(self, name: str = "jax_mps", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        self.bond_dim = options.get("max_bond_dim", 32) if options else 32
        self._n_max_qubits = 200

    @property
    def n_qubits(self) -> int: return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "RX", "RY", "RZ", "U3", "CU3", "P", "CP"]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        n = circuit.n_qubits
        max_bond = kwargs.get("max_bond_dim", self.bond_dim)
        accurate = kwargs.get("accurate", True)

        # NOTE: no Clifford auto-dispatch — see SingularityBackend for the
        # auto-routed Clifford fast path; this backend always runs the
        # MPS path so callers see MPS-style metadata.

        # 1. State Init |00...0>
        tensors = jnp.zeros((n, max_bond, 2, max_bond), dtype=jnp.complex64)
        tensors = tensors.at[:, 0, 0, 0].set(1.0)
        
        # 2. Gate Preparation (Lazy Routing on Python side for compilation simplicity)
        v2p = list(range(n))
        p2v = list(range(n))

        # Expand CCX/CSWAP into 1Q+2Q before routing; jax_mps's scan is
        # designed for gates with at most 2 qubits.
        from superfermion.backends.turbo import expand_3q_gates
        expanded_gates = expand_3q_gates(
            [g for g in circuit._gates if g.name != "BARRIER"]
        )

        # We collect 'Physical' gates
        # (p1, p2, matrix, type)
        # type 0: 1Q, type 1: SWAP, type 2: 2Q
        phys_gates = []

        for g in expanded_gates:
            if g.name == "BARRIER": continue
            u_mat = g.to_unitary()
            u_flat = jnp.array(u_mat.flatten(), dtype=jnp.complex64)
            u_pad = jnp.pad(u_flat, (0, 16 - u_flat.shape[0]))
            
            q_v = g.qubits
            if len(q_v) == 1:
                p = v2p[q_v[0]]
                phys_gates.append((p, -1, u_pad, 0))
            elif len(q_v) == 2:
                p1_orig, p2_orig = v2p[q_v[0]], v2p[q_v[1]]
                p1, p2 = p1_orig, p2_orig
                # Routing
                while abs(p1 - p2) > 1:
                    step = 1 if p2 > p1 else -1
                    neigh = p2 - step
                    # Swap physical sites
                    sw_mat = jnp.array([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1], dtype=jnp.complex64)
                    phys_gates.append((min(p2, neigh), max(p2, neigh), sw_mat, 1))
                    # Update maps
                    v_p2, v_neigh = p2v[p2], p2v[neigh]
                    p2v[p2], p2v[neigh] = v_neigh, v_p2
                    v2p[v_p2], v2p[v_neigh] = neigh, p2
                    p2 = neigh
                
                # Now they are adjacent at (min(p1, p2), max(p1, p2))
                # CRITICAL: If p1 < p2, then site min(p1,p2) is q_v[0] and site max(p1,p2) is q_v[1].
                # If p1 > p2, then site min(p1,p2) is q_v[1] and site max(p1,p2) is q_v[0].
                # Our 4x4 matrix assumes (q_v[0], q_v[1]) ordering. 
                # If site order is (q_v[1], q_v[0]), we must transpose the operator to match sites.
                if p1 > p2:
                    # Permute matrix: (a,b,p,q) -> (b,a,q,p)
                    u_adj = u_mat.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).flatten()
                    u_pad = jnp.array(jnp.pad(u_adj, (0, 16 - u_adj.shape[0])), dtype=jnp.complex64)
                
                phys_gates.append((min(p1, p2), max(p1, p2), u_pad, 2))

        if not phys_gates:
            # Identity propagator: return |00...0> counts
            if shots > 0:
                bs = "0" * n
                counts = {bs: shots}
            else:
                counts = {}
            return RunResult(counts=counts, shots=shots, circuit=circuit, metadata={"max_bond": max_bond, "status": "empty_circuit"})

        # Prepare JAX scan data
        p1s = jnp.array([g[0] for g in phys_gates], dtype=jnp.int32)
        p2s = jnp.array([g[1] for g in phys_gates], dtype=jnp.int32)
        mats = jnp.stack([g[2] for g in phys_gates])
        types = jnp.array([g[3] for g in phys_gates], dtype=jnp.int32)
        
        # 3. Execution (Propagator)
        final_tensors, log_fid_acc, max_bond_seen = self._propagate_jax(
            tensors, p1s, p2s, mats, types, max_bond
        )

        # --- Bond-saturation telemetry ---
        # fidelity_lower_bound is the authoritative signal — it's a product of
        # per-step kept-weight ratios which float32 computes reliably.
        # max_observed_bond is a best-effort rank estimate (in-scan cumsum can
        # accumulate float32 rounding on deep circuits, so treat as advisory).
        fidelity_lower_bound = float(math.exp(float(np.asarray(log_fid_acc))))
        max_obs_bond = int(np.asarray(max_bond_seen))
        # Bond is "saturated" iff meaningful information was discarded.
        bond_saturated = fidelity_lower_bound < 0.99
        if bond_saturated:
            warnings.warn(
                f"[sf.jax_mps] MPS truncation reduced fidelity lower bound to "
                f"{fidelity_lower_bound:.4f} at max_bond={max_bond}. "
                f"Result is an approximation. "
                f"Increase max_bond_dim in options for a tighter bound.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        # 4. Canonicalization and Sampling
        counts = {}
        if shots > 0:
            norm_tensors = self._right_normalize_jax(final_tensors, n)
            key = jax.random.PRNGKey(kwargs.get("seed", 42))
            bits_raw = self._sample_batched_jax(norm_tensors, n, shots, key, max_bond)

            # Remap bits back to virtual order
            bits_np = np.array(bits_raw)
            for sh in range(shots):
                b_sh = bits_np[sh]
                v_bits = [0]*n
                for v in range(n): v_bits[v] = int(b_sh[v2p[v]])
                bs = "".join(map(str, v_bits))
                counts[bs] = counts.get(bs, 0) + 1

        # Build dense statevector (shots=0 path, or always populated for
        # downstream introspection). Skipped at large n where 2^n is huge.
        sv = None
        if n <= 20:
            sv = self._mps_to_statevector(final_tensors, n, v2p)
            # Also populate counts from the SV if shots>0 was not run
            if shots > 0 and not counts:
                pass  # already filled from sampling above

        return RunResult(
            counts=counts, shots=shots, circuit=circuit,
            statevector=sv,
            metadata={
                "max_bond": max_bond,
                "max_observed_bond": max_obs_bond,
                "bond_saturated": bool(bond_saturated),
                "fidelity_lower_bound": fidelity_lower_bound,
            },
        )

    def _mps_to_statevector(self, tensors, n, v2p):
        """Contract MPS to dense statevector in SF MSB (virtual-qubit) order.

        The MPS stores tensors of shape (max_bond, 2, max_bond) per site, with
        the chain ends pinned: the first site's left-bond is effectively 1
        (active region = [0, :, :]), the last site's right-bond is effectively
        1 (active region = [:, :, 0]).
        """
        # Pull tensors out of JAX into NumPy (also forces materialisation)
        t = np.asarray(tensors)  # shape (n, max_bond, 2, max_bond), complex64
        # Contract the chain. Start with the first site's leftmost slice.
        state = t[0][0]   # shape (2, max_bond)
        for k in range(1, n):
            # state shape so far: (2^k, max_bond); next tensor: (max_bond, 2, max_bond)
            state = np.einsum('ab,bpc->apc', state, t[k])
            # Flatten the new physical dim into the existing physical axes
            old_shape = state.shape  # (2^k, 2, max_bond)
            state = state.reshape(-1, old_shape[-1])  # (2^(k+1), max_bond)
        # Final right-bond is pinned at index 0
        sv_physical = state[:, 0].astype(np.complex128)
        # Permute bits from physical -> virtual qubit order (only if reordered).
        if list(v2p) == list(range(n)):
            return sv_physical
        # Bit permutation:  index_virtual encodes (v_0, v_1, ..., v_{n-1}) in MSB,
        # and v_k == phys_bit_at_position_{v2p[k]}.
        # Equivalently: phys_index = sum_{v} v_bits[v] * 2^{n-1-v2p[v]}.
        # We build virtual by iterating over phys_index and decoding bits.
        sv_virtual = np.zeros_like(sv_physical)
        for p_idx in range(1 << n):
            # phys_bits[p] = bit at MSB-position p in p_idx
            phys_bits = [(p_idx >> (n - 1 - p)) & 1 for p in range(n)]
            v_idx = 0
            for v in range(n):
                v_idx = (v_idx << 1) | phys_bits[v2p[v]]
            sv_virtual[v_idx] = sv_physical[p_idx]
        return sv_virtual

    @partial(jit, static_argnums=(0, 6))
    def _propagate_jax(self, tensors, p1s, p2s, mats, types, max_bond):
        # Scan carry = (tensors, log_fidelity_acc, max_bond_seen).
        # - log_fidelity_acc accumulates sum_i log(kept_w_i / total_w_i) so the
        #   final fidelity lower bound is exp(log_fidelity_acc).
        # - max_bond_seen is the running max of true (non-negligible) rank at
        #   any 2Q SVD — > max_bond means the run truncated lossily somewhere.
        def apply(carry, x):
            sv, log_fid, max_b = carry
            p1, p2, mat, g_t = x

            def h1(state):
                sv_, log_fid_, max_b_ = state
                t = sv_[p1]
                m22 = mat[:4].reshape(2, 2)
                new_t = jnp.einsum('lpr,sp->lsr', t, m22)
                # 1Q gates don't touch entanglement; pass telemetry through.
                return sv_.at[p1].set(new_t), log_fid_, max_b_

            def h2(state):
                sv_, log_fid_, max_b_ = state
                t1, t2 = sv_[p1], sv_[p2]
                m44 = mat.reshape(2, 2, 2, 2)
                cb = jnp.einsum('lpr,rqs,abpq->labs', t1, t2, m44)
                res = cb.reshape(t1.shape[0]*2, -1)
                u, s_vals, vh = jnp.linalg.svd(res, full_matrices=False)

                # --- Bond-saturation telemetry (use full s_vals BEFORE trunc) ---
                s_sq = s_vals.real ** 2
                total_w = jnp.sum(s_sq)
                k = max_bond
                kept_w = jnp.sum(s_sq[:k])
                # kept_ratio in (0, 1]; log is <= 0. Clamp to avoid -inf when
                # truncation kills everything (shouldn't happen in a unitary
                # circuit but guard anyway).
                kept_ratio = kept_w / jnp.maximum(total_w, 1e-30)
                log_fid_new = log_fid_ + jnp.log(jnp.maximum(kept_ratio, 1e-30))
                # Effective rank = smallest k that captures (1 - 1e-8) of
                # the squared-singular-value mass. Robust to the float32
                # noise tail that accumulates in the zero-padded regions
                # of jax_mps tensors after many 2Q gates — only "real"
                # singular values contribute non-trivially to the cumsum.
                # (Note: s_vals is descending from jnp.linalg.svd, so the
                # cumsum is monotone.)
                cum = jnp.cumsum(s_sq) / jnp.maximum(total_w, 1e-30)
                # 1e-5 tolerance: float32 accumulation error grows with gate
                # count; 1e-8 is below the noise floor on deep circuits.
                # Cast everything to int32 explicitly so the cond branches
                # match (h1 passes max_b_ through as int32). With x64 enabled,
                # jnp.sum over int32 promotes to int64 — force back to int32.
                rank = jnp.sum((cum < 1.0 - 1e-5).astype(jnp.int32)).astype(jnp.int32) + jnp.int32(1)
                max_b_new = jnp.maximum(max_b_, rank).astype(jnp.int32)
                # ----------------------------------------------------------------

                u = u[:, :k]; s_trunc = s_vals[:k]; vh = vh[:k, :]
                ss = jnp.diag(jnp.sqrt(jnp.maximum(s_trunc.real, 1e-12)))
                sv_new = sv_.at[p1].set((u @ ss).reshape(-1, 2, k)) \
                            .at[p2].set((ss @ vh).reshape(k, 2, -1))
                return sv_new, log_fid_new, max_b_new

            return lax.cond(g_t == 0, h1, h2, (sv, log_fid, max_b)), None

        init_carry = (
            tensors,
            jnp.float32(0.0),      # log_fid_acc
            jnp.int32(0),          # max_bond_seen
        )
        (final_tensors, log_fid_acc, max_bond_seen), _ = lax.scan(
            apply, init_carry, (p1s, p2s, mats, types)
        )
        return final_tensors, log_fid_acc, max_bond_seen

    @partial(jit, static_argnums=(0, 2))
    def _right_normalize_jax(self, tensors, n):
        max_bond = tensors.shape[1]
        def body(carry, idx_tuple):
            t, next_r = carry
            i = idx_tuple
            curr_t = t[i]
            tc = jnp.einsum('lpr,rk->lpk', curr_t, next_r)
            res = tc.reshape(tc.shape[0], -1)
            u, s, vh = jnp.linalg.svd(res, full_matrices=False)
            
            k = max_bond
            u = u[:, :k]
            s = s[:k]
            vh = vh[:k, :]
            
            new_t = vh.reshape(-1, 2, curr_t.shape[2])
            new_r = u @ jnp.diag(s)
            return (t.at[i].set(new_t), new_r), None
        
        init_r = jnp.eye(tensors[-1].shape[2], dtype=jnp.complex64)
        (final_t, _), _ = lax.scan(body, (tensors, init_r), jnp.arange(n-1, -1, -1))
        return final_t

    @partial(jit, static_argnums=(0, 2, 3, 5))
    def _sample_batched_jax(self, tensors, n, shots, key, max_bond):
        # BUGFIX: previously `keys = jax.random.split(key, n)` was computed but
        # never threaded into the scan — the carry reused the *same* `key` at
        # every site, so all qubits drew identical uniforms and the resulting
        # bits were fully correlated. GHZ coincidentally looked correct because
        # its conditionals are degenerate (0 or 1) after the first measurement.
        # Fix: pass per-site subkeys as scan inputs.
        keys = jax.random.split(key, n)

        def step(vecs, x):
            i, k_i = x
            t = tensors[i]

            # vecs: (shots, L)
            psi0 = jnp.einsum('sl,lr->sr', vecs, t[:, 0, :])
            psi1 = jnp.einsum('sl,lr->sr', vecs, t[:, 1, :])

            p0 = jnp.sum(jnp.abs(psi0)**2, axis=1)
            p1 = jnp.sum(jnp.abs(psi1)**2, axis=1)
            prob0 = p0 / (p0 + p1 + 1e-12)

            # Per-shot uniforms drawn from *this site's* key.
            bits = (jax.random.uniform(k_i, (shots,)) > prob0).astype(jnp.int8)

            # pick branch
            sel = bits[:, None]
            new_vecs = jnp.where(sel == 0, psi0, psi1)
            norms = jnp.linalg.norm(new_vecs, axis=1, keepdims=True)
            new_vecs = jnp.where(norms > 1e-12, new_vecs / norms, new_vecs)

            return new_vecs, bits

        init_v = jnp.zeros((shots, max_bond), dtype=jnp.complex64).at[:, 0].set(1.0)
        _, all_bits = lax.scan(step, init_v, (jnp.arange(n), keys))
        return all_bits.T
