
"""
CUDA-Accelerated Matrix Product State (MPS) core with Batched Sampling.
Implements a GPU-native tensor-network propagator with Batch-Sized shots.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Any, Dict, List, Optional

try:
    import cupy as cp
except ImportError:
    cp = None

import superfermion as sf
from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.results import RunResult

class CupyMPSBackend(Backend):
    """Zero-Latency NVIDIA GPU MPS Engine with Batched Sampling."""
    
    def __init__(self, name: str = "cuda_mps", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        if cp is None:
             raise ImportError("CuPy not installed. Required for cuda_mps.")
        self.bond_dim = options.get("max_bond_dim", 32) if options else 32
        self._n_max_qubits = 1000

    @property
    def n_qubits(self) -> int: return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "RX", "RY", "RZ", "U3", "CU3", "P", "CP"]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        n = circuit.n_qubits
        max_bond = kwargs.get("max_bond_dim", self.bond_dim)

        # NOTE: no Clifford auto-dispatch — sf.singularity routes
        # Clifford circuits explicitly.

        # 1. GPU Initial State (Product State) |00...0>
        tensors = [cp.zeros((1, 2, 1), dtype=cp.complex64) for _ in range(n)]
        for i in range(n): tensors[i][0, 0, 0] = 1.0
        
        # Mapping: virtual_qubit_id -> physical_site_index
        v2p = list(range(n))
        p2v = list(range(n))
        
        t0 = time.time()

        # Expand CCX/CSWAP into 1Q+2Q sequences before routing;
        # the MPS architecture only supports gates with at most 2 qubits.
        from superfermion.backends.turbo import expand_3q_gates
        expanded_gates = expand_3q_gates(
            [g for g in circuit._gates if g.name not in ("BARRIER", "MEASURE")]
        )

        # 2. Lazy Routed Gate Application
        for gate in expanded_gates:
            if gate.name == "BARRIER": continue
            q_virtual = gate.qubits
            u_mat = cp.array(gate.to_unitary(), dtype=cp.complex64)
            
            if len(q_virtual) == 1:
                p_site = v2p[q_virtual[0]]
                tensors[p_site] = cp.einsum('lpr,sp->lsr', tensors[p_site], u_mat)
            elif len(q_virtual) == 2:
                p1 = v2p[q_virtual[0]]
                p2 = v2p[q_virtual[1]]
                
                # Make adjacent in physical space
                while abs(p1 - p2) > 1:
                    step = 1 if p2 > p1 else -1
                    neigh = p2 - step
                    # Apply SWAP at physical sites
                    self._apply_2q_adjacent_gpu(tensors, min(p2, neigh), max(p2, neigh), 
                                              cp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=cp.complex64), 
                                              max_bond)
                    # Update Map
                    v_p2 = p2v[p2]
                    v_neigh = p2v[neigh]
                    p2v[p2], p2v[neigh] = v_neigh, v_p2
                    v2p[v_p2], v2p[v_neigh] = neigh, p2
                    p2 = neigh
                
                # Polarity fix: handle control/target inversion
                if p1 > p2:
                    u_mat = u_mat.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
                
                self._apply_2q_adjacent_gpu(tensors, min(p1, p2), max(p1, p2), u_mat, max_bond)

        # 3. Final State and Canonicalization
        norm_tensors = self._right_normalize_gpu(tensors, max_bond)
        
        counts = {}
        if shots > 0:
            # 🚀 BATHTED SAMPLING: Sample all shots in parallel across sites
            final_bits_raw = self._sample_batched_gpu(norm_tensors, shots)
            
            # Map physical bits back to virtual bitstrings
            for sh_idx in range(shots):
                bits_raw = final_bits_raw[sh_idx]
                mapped_bits = [0] * n
                for v in range(n):
                    mapped_bits[v] = int(bits_raw[v2p[v]])
                bs = "".join(map(str, mapped_bits))
                counts[bs] = counts.get(bs, 0) + 1

        t_total = (time.time() - t0) * 1000
        return RunResult(counts=counts, shots=shots, circuit=circuit, metadata={"max_bond": max_bond, "latency_ms": t_total})

    def _apply_2q_adjacent_gpu(self, tensors, i, j, m44, max_bond):
        m44 = m44.reshape(2, 2, 2, 2)
        cb = cp.einsum('lpr,rqs,abpq->labs', tensors[i], tensors[j], m44)
        shape = cb.shape
        res = cb.reshape(shape[0]*2, -1)
        u, s, vh = cp.linalg.svd(res, full_matrices=False)
        k = min(max_bond, s.shape[0])
        u, s, vh = u[:, :k], s[:k], vh[:k, :]
        ss = cp.diag(cp.sqrt(cp.maximum(s.real, 0.0)).astype(cp.complex64))
        tensors[i] = (u @ ss).reshape(-1, 2, k)
        tensors[j] = (ss @ vh).reshape(k, 2, -1)

    def _right_normalize_gpu(self, tensors, max_bond):
        n = len(tensors)
        next_r = cp.eye(tensors[-1].shape[2], dtype=cp.complex64)
        for i in range(n-1, -1, -1):
            t = tensors[i]
            tc = cp.einsum('lpr,rk->lpk', t, next_r)
            shape = tc.shape
            res = tc.reshape(shape[0], -1)
            u, s, vh = cp.linalg.svd(res, full_matrices=False)
            k = min(max_bond, s.shape[0])
            u, s, vh = u[:, :k], s[:k], vh[:k, :]
            tensors[i] = vh.reshape(-1, 2, shape[2])
            next_r = u @ cp.diag(s)
        return tensors

    def _sample_batched_gpu(self, tensors, shots):
        """Sequential site sampling with batched vector updates."""
        n = len(tensors)
        # current_vecs: (shots, L)
        current_vecs = cp.zeros((shots, 1), dtype=cp.complex64)
        current_vecs[:, 0] = 1.0
        
        all_bits = np.zeros((shots, n), dtype=np.int8)
        
        for i, t in enumerate(tensors):
            # t: (L, 2, R)
            # project to physical=0: (shots, L) @ (L, R) -> (shots, R)
            psi0 = current_vecs @ t[:, 0, :]
            psi1 = current_vecs @ t[:, 1, :]
            
            p0 = cp.sum(cp.abs(psi0)**2, axis=1) # (shots,)
            p1 = cp.sum(cp.abs(psi1)**2, axis=1)
            norm = p0 + p1
            prob0 = cp.asnumpy(p0 / norm) # (shots,)
            
            # CPU draw for all shots
            r = np.random.random(shots)
            bits = (r > prob0).astype(np.int8)
            all_bits[:, i] = bits
            
            # Update current_vecs for next site
            # This is the difficult part in batching: select psi0 or psi1 per shot
            # psi0 is (shots, R), psi1 is (shots, R)
            selector = cp.array(bits)[:, None] # (shots, 1)
            new_vecs = cp.where(selector == 0, psi0, psi1)
            
            # Normalize each vector in the batch
            vec_norms = cp.linalg.norm(new_vecs, axis=1, keepdims=True)
            current_vecs = cp.where(vec_norms > 1e-12, new_vecs / vec_norms, new_vecs)
            
        return all_bits
