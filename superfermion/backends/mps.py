
"""
Optimized Matrix Product State (MPS) simulator with Lazy SWAP Routing.
Maintains a virtual-to-physical mapping to minimize redundant swaps.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import superfermion as sf
from superfermion.backends.base import Backend
from superfermion.circuit import Circuit
from superfermion.results import RunResult

class MPSSimulatorBackend(Backend):
    """Industrial-Grade MPS Engine with Lazy Routing and Matrix Caching."""

    # Keep this in sync with _RUST_MPS_GATES inside run().
    # Single source of truth: anything the Rust kernel parses cleanly is listed
    # here. Anything not listed must be decomposed at the Python layer first.
    _SUPPORTED_GATES = (
        "H", "X", "Y", "Z", "CX", "CNOT", "CZ", "RX", "RY", "RZ",
        "SWAP", "U3", "CU3", "P", "CP", "BARRIER", "MEASURE",
    )

    def __init__(self, name: str = "mps", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        self.bond_dim = options.get("max_bond_dim", 32) if options else 32
        self._n_max_qubits = 200
        self._gate_cache = {}
        # Cache for evolved MPS state to speed up repeated expval calls on the same circuit
        self._last_circuit_id = None
        self._last_circuit_len = None
        self._last_bond_dim = None
        self._last_mps_state = None

    @property
    def n_qubits(self) -> int: return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return list(self._SUPPORTED_GATES)

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        n = circuit.n_qubits
        max_bond = kwargs.get("max_bond_dim", self.bond_dim)
        seed = kwargs.get("seed", 42)

        # ── Clifford fast path (shots > 0 only) ──
        # The Rust `sample_mps` doesn't canonicalise tensors before
        # measurement so its sampled distribution is biased on Clifford
        # circuits.  StabilizerBackend gives the exact distribution in
        # poly time.  When shots == 0 (caller wants the SV), fall through
        # to the regular MPS path.
        if shots > 0:
            from superfermion.backends.stabilizer import maybe_clifford_dispatch
            stab_res = maybe_clifford_dispatch(
                circuit, shots, seed=seed, require_statevector=False,
            )
            if stab_res is not None:
                stab_res.metadata = {**(stab_res.metadata or {}),
                                     "dispatched_from": "mps",
                                     "max_bond_dim": max_bond,
                                     "bond_saturated": False,
                                     "fidelity_lower_bound": 1.0}
                return stab_res

        # ── Rust MPS fast path ─────────────────────────────────────────
        # The Rust kernel only supports a limited gate set. If the circuit
        # contains unknown gates it silently treats them as identity, giving
        # wrong results. Gate-check before using this path.
        _RUST_MPS_GATES = frozenset(self._SUPPORTED_GATES)
        _circuit_gates = {g.name.upper() for g in circuit._gates}
        _rust_safe = _circuit_gates.issubset(_RUST_MPS_GATES)

        if _rust_safe and n <= 22 and shots > 0:
            # Route via the Rust dense simulate + numpy sampling: this is
            # both fast and CORRECT.  The Rust `sample_mps` has a known
            # canonicalisation bug that biases probabilities on
            # non-Clifford circuits — using `simulate()` to get a dense
            # statevector and then sampling with numpy avoids that
            # entirely and is still ~ms at n<=22.
            try:
                dag = circuit.to_ir()
                sv = np.asarray(dag.simulate(), dtype=np.complex128)
                # Rust core uses LSB; SF uses MSB. Reverse axes for SF.
                sv = sv.reshape([2]*n).transpose(list(range(n))[::-1]).reshape(-1)
                probs = np.abs(sv) ** 2
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                rng = np.random.default_rng(seed)
                idxs = rng.choice(len(probs), size=shots, p=probs)
                uniq, freq = np.unique(idxs, return_counts=True)
                counts = {format(int(i), f"0{n}b"): int(c)
                          for i, c in zip(uniq, freq)}
                return RunResult(
                    counts=counts,
                    statevector=sv,
                    shots=shots,
                    circuit=circuit,
                    metadata={
                        "max_bond": max_bond,
                        "engine": "rust-dense-then-sample",
                        "bond_saturated": False,
                        "fidelity_lower_bound": 1.0,
                    },
                )
            except (ImportError, AttributeError, ValueError, RuntimeError):
                pass  # fall through to Python sweep

        # ── Python MPS fallback ─────────────────────────────────────────
        norm_tensors, v2p = self._evolve_and_canonicalize(circuit, max_bond)

        counts: Dict[str, int] = {}
        if shots > 0:
            rng = np.random.default_rng(seed)
            for _ in range(shots):
                phys_bits = self._sample_one_rng(norm_tensors, rng)
                # Remap physical bits back to virtual qubit order
                v_bits = [0] * n
                for v in range(n):
                    v_bits[v] = phys_bits[v2p[v]]
                bs = "".join(map(str, v_bits))
                counts[bs] = counts.get(bs, 0) + 1

        return RunResult(
            counts=counts,
            statevector=None,
            shots=shots,
            circuit=circuit,
            metadata={
                "max_bond": max_bond,
                "engine": "python-mps-fallback",
                "bond_saturated": None,
                "fidelity_lower_bound": None,
                "bond_saturation_tracking": "unavailable (python fallback)",
                "_v2p": v2p,
            },
        )

    # ------------------------------------------------------------------
    # Internal helper: evolve a circuit into an MPS, then right-canonicalize.
    # Returns the (canonicalized) list of MPS tensors and the v2p map.
    # ------------------------------------------------------------------
    def _evolve_and_canonicalize(self, circuit: Circuit, max_bond: int):
        n = circuit.n_qubits
        from superfermion.backends.turbo import expand_3q_gates
        raw_gates = [g for g in circuit._gates if g.name not in ("BARRIER", "MEASURE")]
        gates = expand_3q_gates(raw_gates)

        # Initial state |00...0>
        tensors = [np.zeros((1, 2, 1), dtype=np.complex64) for _ in range(n)]
        for i in range(n):
            tensors[i][0, 0, 0] = 1.0

        v2p = list(range(n))
        p2v = list(range(n))
        swap_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                             [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64)

        for gate in gates:
            q_v = gate.qubits
            u_mat = gate.to_unitary().astype(np.complex64)

            if len(q_v) == 1:
                p = v2p[q_v[0]]
                tensors[p] = np.einsum('sp,lpr->lsr', u_mat, tensors[p])
            elif len(q_v) == 2:
                p1, p2 = v2p[q_v[0]], v2p[q_v[1]]
                while abs(p1 - p2) > 1:
                    step = 1 if p2 > p1 else -1
                    neigh = p2 - step
                    self._apply_2q_adjacent(tensors, min(p2, neigh), max(p2, neigh),
                                            swap_mat, max_bond)
                    v_p2, v_neigh = p2v[p2], p2v[neigh]
                    p2v[p2], p2v[neigh] = v_neigh, v_p2
                    v2p[v_p2], v2p[v_neigh] = neigh, p2
                    p2 = neigh
                if p1 > p2:
                    u_mat = u_mat.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
                self._apply_2q_adjacent(tensors, min(p1, p2), max(p1, p2), u_mat, max_bond)

        norm_tensors = self._canonicalize(list(tensors), max_bond)
        return norm_tensors, v2p

    # ------------------------------------------------------------------
    # Exact expectation values via direct MPS contraction (no sampling).
    # ------------------------------------------------------------------
    def expval(self, circuit: Circuit, observable, max_bond: Optional[int] = None) -> float:
        """Fast MPS Pauli expectation. Hands off to the Rust QR-based
        boundary-contraction kernel when the gate set is supported; falls
        back to the pure-Python sweep otherwise."""
        bd = max_bond if max_bond is not None else self.bond_dim
        terms = self._normalize_observable(observable, circuit.n_qubits)
        _RUST_MPS_GATES = frozenset(self._SUPPORTED_GATES)
        circuit_gates = {g.name.upper() for g in circuit._gates}
        if circuit_gates.issubset(_RUST_MPS_GATES):
            try:
                _PB = {"I": 0, "X": 1, "Y": 2, "Z": 3}
                # Pauli string is in SF MSB convention: position 0 = qubit 0.
                # The Rust MPS uses the same indexing — no reversal needed.
                # (A previous edit had [::-1] here; that flipped <X⊗I⊗I⊗I⊗I>
                # into <I⊗I⊗I⊗I⊗X> on every multi-Pauli observable.)
                pauli_bytes_list = [
                    [_PB[c.upper()] for c in pauli_str] for pauli_str, _ in terms
                ]

                # Cache the evolved MPS state keyed by circuit content hash.
                # id(circuit) is NOT safe — Python reuses memory addresses after
                # GC, causing stale states to be returned for different circuits.
                try:
                    circuit_hash = hash(tuple(
                        (g.name, tuple(g.qubits),
                         tuple(float(p) if hasattr(p, '__float__') else str(p)
                               for p in g.params))
                        for g in circuit._gates
                    ))
                except Exception:
                    circuit_hash = None  # unhashable params: skip cache

                if (circuit_hash is not None and
                        circuit_hash == self._last_circuit_id and
                        bd == self._last_bond_dim):
                    mps_state = self._last_mps_state
                else:
                    dag = circuit.to_ir()
                    mps_state = dag.evolve_mps(bd)
                    self._last_circuit_id = circuit_hash
                    self._last_circuit_len = len(circuit._gates)
                    self._last_bond_dim = bd
                    self._last_mps_state = mps_state

                # Remap Pauli bytes from virtual qubit order to physical site order.
                # Lazy-SWAP routing may have moved virtual qubit v to physical
                # site perm[v] != v. pauli_expval contracts in physical order.
                perm = mps_state.perm()  # perm[virtual] = physical site
                n_q = circuit.n_qubits
                remapped = []
                for virt_pb in pauli_bytes_list:
                    phys_pb = [0] * n_q
                    for v in range(n_q):
                        phys_pb[perm[v]] = virt_pb[v]
                    remapped.append(phys_pb)
                pairs = mps_state.pauli_expval_batch(remapped)
                
                total = 0.0 + 0.0j
                for (pauli_str, coeff), (re, im) in zip(terms, pairs):
                    total += complex(coeff) * complex(re, im)
                if abs(total.imag) > 1e-7:
                    import warnings
                    warnings.warn(
                        f"expval has imag part {total.imag:.2e} (Hermitian "
                        f"observable expected); returning real part only.",
                        RuntimeWarning, stacklevel=2,
                    )
                return float(total.real)
            except Exception:
                pass  # fall through to Python sweep
        return self._expval_python(circuit, observable, max_bond)

    def _expval_python(self, circuit: Circuit, observable, max_bond: Optional[int] = None) -> float:
        """Exact expectation <psi|O|psi> on the MPS state of `circuit`.

        Parameters
        ----------
        circuit : sf.Circuit
        observable : str | dict | sf.SparsePauliOp
            * Pauli string of length ``n`` over {I, X, Y, Z} (e.g. ``"IZII"``)
            * dict mapping pauli-string -> coefficient
            * ``sf.SparsePauliOp`` instance
        max_bond : int, optional
            Override the backend's default bond dimension.

        Returns
        -------
        float
            The real expectation value. (For Hermitian observables the
            imaginary part is dropped after a sanity-check < 1e-9.)

        Notes
        -----
        Cost: ``O(n * chi^3)`` for a single Pauli string of weight 1, plus
        ``O(n * chi^2 * d^2)`` for boundary contractions; far cheaper than
        the 10k-shot sampling fallback that previously was the only option,
        and crucially **noise-free** (no shot-noise floor at sigma~1e-2).
        """
        n = circuit.n_qubits
        bd = max_bond if max_bond is not None else self.bond_dim

        # Normalize the observable to a list of (pauli_string, coeff) terms.
        terms = self._normalize_observable(observable, n)

        # Evolve the MPS once; reuse for every term.
        norm_tensors, v2p = self._evolve_and_canonicalize(circuit, bd)

        # Single-qubit Pauli matrices
        I2 = np.eye(2, dtype=np.complex64)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

        total = 0.0 + 0.0j
        for pauli_str, coeff in terms:
            # Map virtual qubit index -> physical site (after lazy SWAPs)
            # The pauli_str is indexed in MSB-first virtual order:
            # pauli_str[k] is the operator on virtual qubit k.
            # Apply at physical site v2p[k].
            ops_at_phys = [I2] * n
            for k_virt, p in enumerate(pauli_str):
                ops_at_phys[v2p[k_virt]] = PAULI[p.upper()]
            total += complex(coeff) * self._contract_with_ops(norm_tensors, ops_at_phys)

        if abs(total.imag) > 1e-7:
            # Numerical leakage — surface it for the user.
            import warnings
            warnings.warn(f"expval has imag part {total.imag:.2e} (Hermitian "
                          f"observable expected); returning real part only.",
                          RuntimeWarning, stacklevel=2)
        return float(total.real)

    @staticmethod
    def _normalize_observable(obs, n: int):
        """Return a list of (pauli_string, coeff) tuples."""
        if isinstance(obs, str):
            if len(obs) != n:
                raise ValueError(f"Pauli string length {len(obs)} != n_qubits {n}")
            return [(obs, 1.0)]
        if isinstance(obs, dict):
            for k in obs:
                if len(k) != n:
                    raise ValueError(f"Pauli key '{k}' length != n_qubits {n}")
            return list(obs.items())
        # sf.SparsePauliOp duck-typed: has `.terms` or iter of (str, coeff)
        if hasattr(obs, "to_dict"):
            return list(obs.to_dict().items())
        if hasattr(obs, "terms"):
            return list(obs.terms)
        raise TypeError(f"Unsupported observable type: {type(obs).__name__}")

    @staticmethod
    def _contract_with_ops(tensors, ops_at_phys):
        """Contract <psi|O|psi> where O = otimes_i ops_at_phys[i].

        Uses a left-to-right boundary contraction with bond_dim^2 working set.
        Numerically safe in single precision because the canonicalization
        already normalized the MPS (norm ~ 1).
        """
        # Boundary tensor: shape (l_ket, l_bra)
        left = np.array([[1.0]], dtype=np.complex64)
        for t, op in zip(tensors, ops_at_phys):
            # Apply op on the ket side: t_op[l, p', r] = sum_p op[p',p] t[l,p,r]
            t_op = np.einsum('qp,lpr->lqr', op.astype(np.complex64), t)
            # Absorb tensor pair into the boundary
            #   left[l_k, l_b] * t_op[l_k, p, r_k] * t.conj[l_b, p, r_b]
            #   = left[r_k, r_b]
            left = np.einsum('ab,aps,bpt->st', left, t_op, t.conj())
        # Final boundary should be a (1, 1) scalar tensor
        return complex(left.reshape(-1)[0])

    def _apply_2q_adjacent(self, tensors, i, j, mat, max_bond):
        m44 = mat.reshape(2, 2, 2, 2)
        cb = np.einsum('lpr,rqs,abpq->labs', tensors[i], tensors[j], m44)
        shape = cb.shape
        res = cb.reshape(shape[0]*2, -1)
        u, s, vh = np.linalg.svd(res, full_matrices=False)
        # Relative SVD threshold: keep singular values above 1e-12 * max(s)
        # More robust than absolute threshold for varying circuit norms.
        k = min(max_bond, int(np.sum(s > 1e-12 * s[0])))
        k = max(k, 1)
        u, s, vh = u[:, :k], s[:k], vh[:k, :]
        ss = np.diag(np.sqrt(s))
        tensors[i] = (u @ ss).reshape(-1, 2, k)
        tensors[j] = (ss @ vh).reshape(k, 2, -1)

    def _canonicalize(self, tensors, max_bond):
        # Right to left normalization
        n = len(tensors)
        next_r = np.eye(tensors[-1].shape[2], dtype=np.complex64)
        for i in range(n-1, -1, -1):
            t = tensors[i]
            tc = np.einsum('lpr,rk->lpk', t, next_r)
            shape = tc.shape
            u, s, vh = np.linalg.svd(tc.reshape(shape[0], -1), full_matrices=False)
            # Relative SVD threshold: keep singular values above 1e-12 * max(s)
            k = min(max_bond, int(np.sum(s > 1e-12 * s[0]))); k = max(k, 1)
            u, s, vh = u[:, :k], s[:k], vh[:k, :]
            tensors[i] = vh.reshape(-1, 2, shape[2])
            next_r = u @ np.diag(s)
        return tensors

    def _sample_one_rng(self, tensors, rng):
        """Sample one bitstring using a seeded numpy RNG."""
        bits = []
        curr_vec = np.array([1.0], dtype=np.complex64).reshape(1, 1)
        for t in tensors:
            psi0 = curr_vec[0] @ t[:, 0, :]
            psi1 = curr_vec[0] @ t[:, 1, :]
            p0 = float(np.sum(np.abs(psi0)**2))
            p1 = float(np.sum(np.abs(psi1)**2))
            norm = p0 + p1
            prob0 = p0 / norm if norm > 1e-12 else 0.5
            bit = 0 if rng.random() < prob0 else 1
            bits.append(bit)
            vec = psi0 if bit == 0 else psi1
            v_norm = float(np.linalg.norm(vec))
            curr_vec = (vec / v_norm if v_norm > 1e-12 else vec).reshape(1, -1)
        return bits

    def _sample_one(self, tensors):
        bits = []
        curr_vec = np.array([1.0], dtype=np.complex64).reshape(1, 1)
        for t in tensors:
            psi0 = curr_vec[0] @ t[:, 0, :]
            psi1 = curr_vec[0] @ t[:, 1, :]
            p0 = np.sum(np.abs(psi0)**2)
            p1 = np.sum(np.abs(psi1)**2)
            norm = p0 + p1
            prob0 = p0 / norm if norm > 1e-12 else 0.5
            bit = 0 if np.random.random() < prob0 else 1
            bits.append(bit)
            vec = psi0 if bit == 0 else psi1
            v_norm = np.linalg.norm(vec)
            curr_vec = (vec / v_norm if v_norm > 1e-12 else vec).reshape(1, -1)
        return bits
