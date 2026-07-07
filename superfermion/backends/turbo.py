"""
SuperFermion Turbo Engine — Zero-overhead simulation pipeline.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from superfermion.circuit import Circuit, GateRecord


# ════════════════════════════════════════════════════════════
#  GATE FUSION
# ════════════════════════════════════════════════════════════

def _gate_to_u2(gate: GateRecord) -> np.ndarray:
    name = gate.name.upper()
    p = [float(x) if not hasattr(x, 'value') else x.value for x in gate.params]
    if name == "H": return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    if name == "X": return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "Y": return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if name == "Z": return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if name == "S": return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if name == "SDG": return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if name == "T": return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    if name == "TDG": return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)
    if name == "SX": return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex128)
    if name == "RX":
        t = p[0] if p else 0.0
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if name == "RY":
        t = p[0] if p else 0.0
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        t = p[0] if p else 0.0
        return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=np.complex128)
    if name in ("P", "U1", "R1"):
        phi = p[0] if p else 0.0
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
    if name in ("U", "U3"):
        t, ph, lm = (p[0], p[1], p[2]) if len(p) >= 3 else (p[0], 0, 0)
        ct, st = math.cos(t / 2), math.sin(t / 2)
        return np.array([[ct, -np.exp(1j * lm) * st], [np.exp(1j * ph) * st, np.exp(1j * (ph + lm)) * ct]], dtype=np.complex128)
    return gate.to_unitary().astype(np.complex128)

def _get_2q_matrix(gate: GateRecord) -> np.ndarray:
    name = gate.name.upper()
    p = [float(x) if not hasattr(x, 'value') else x.value for x in gate.params]
    if name in ("CNOT", "CX"): return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    if name == "CZ": return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=np.complex128)
    if name in ("CP", "CR1"):
        phi = p[0] if p else 0.0
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(1j*phi)]], dtype=np.complex128)
    if name == "RZZ":
        t = p[0] if p else 0.0
        em, ep = np.exp(-1j*t/2), np.exp(1j*t/2)
        return np.diag([em, ep, ep, em]).astype(np.complex128)
    return gate.to_unitary().astype(np.complex128)

def _u2_to_params(u: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 2x2 unitary into U(θ, φ, λ) parameters.
    
    U(θ, φ, λ) = [[cos(θ/2), -exp(iλ)sin(θ/2)],
                  [exp(iφ)sin(θ/2), exp(i(φ+λ))cos(θ/2)]]
    
    The decomposition preserves the matrix up to a global phase.
    """
    # Get matrix elements
    a, b = u[0, 0], u[0, 1]
    c, d = u[1, 0], u[1, 1]
    
    # Check if matrix is diagonal (off-diagonal elements are zero)
    # This handles RZ gates correctly
    is_diagonal = abs(b) < 1e-10 and abs(c) < 1e-10
    
    if is_diagonal:
        # Diagonal case: θ = 0
        # U = [[a, 0], [0, d]] where |a| = |d| = 1
        # This equals exp(i*phase(a)) * [[1, 0], [0, exp(i*phase(d/a))]]
        # So: global_phase = phase(a), λ = phase(d/a) = phase(d*conj(a))
        theta = 0.0
        phi = 0.0
        if abs(a) > 1e-10 and abs(d) > 1e-10:
            lam = float(np.angle(d * np.conj(a)))
        else:
            lam = 0.0
    else:
        # General case: compute θ from |a|
        cos_half = abs(a)
        theta = 2 * math.acos(np.clip(cos_half, -1.0, 1.0))
        
        # Extract φ and λ from the phases
        # From U definition (accounting for global phase):
        # c/a has phase φ
        # -b/a has phase λ
        
        if abs(a) > 1e-10:
            phi = float(np.angle(c * np.conj(a))) if abs(c) > 1e-10 else 0.0
            lam = float(np.angle(-b * np.conj(a))) if abs(b) > 1e-10 else 0.0
        else:
            # a ≈ 0, so θ ≈ π, use different approach
            # U = [[0, -exp(iλ)], [exp(iφ), 0]]
            phi = float(np.angle(c)) if abs(c) > 1e-10 else 0.0
            lam = float(np.angle(-b)) if abs(b) > 1e-10 else 0.0
    
    return (theta, phi, lam)

def _is_diagonal_rz(u: np.ndarray) -> Optional[float]:
    """Check if matrix is a pure RZ rotation (diagonal with symmetric phases).
    
    RZ(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
    
    Returns the theta angle if it's an RZ, or None if not.
    """
    # Check if off-diagonal elements are zero
    if abs(u[0, 1]) > 1e-10 or abs(u[1, 0]) > 1e-10:
        return None
    
    # Check if diagonal elements have opposite phases (RZ signature)
    # RZ: a = exp(-i*theta/2), d = exp(i*theta/2)
    # So a * d = 1 (unit magnitude product)
    a, d = u[0, 0], u[1, 1]
    
    # The phases should be negatives of each other
    phase_a = np.angle(a)
    phase_d = np.angle(d)
    
    # For RZ: phase_a = -theta/2, phase_d = theta/2
    # So phase_d = -phase_a
    if abs(phase_a + phase_d) < 1e-10:
        # This is an RZ with theta = 2 * phase_d = -2 * phase_a
        theta = -2 * phase_a  # or 2 * phase_d
        return theta
    
    return None

def fuse_two_qubit_gates(circuit: Circuit) -> Circuit:
    gates = circuit._gates
    new_gates: List[GateRecord] = []
    i = 0
    while i < len(gates):
        if (i+2 < len(gates) and gates[i].name.upper() in ("CX", "CNOT") and 
            gates[i+1].name.upper() == "RZ" and gates[i+2].name.upper() in ("CX", "CNOT") and
            gates[i].qubits == gates[i+2].qubits and gates[i+1].qubits == [gates[i].qubits[1]]):
            c, t = gates[i].qubits
            theta = gates[i+1].params[0]
            new_gates.append(GateRecord("RZZ", [c, t], [theta]))
            i += 3
        else:
            new_gates.append(gates[i])
            i += 1
    res = Circuit(circuit.n_qubits)
    res._gates = new_gates
    return res

_KNOWN_1Q_GATES = frozenset({
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "ID",
    "RX", "RY", "RZ", "P", "U1", "R1", "U", "U3", "U2",
})


def fuse_single_qubit_gates(circuit: Circuit) -> Circuit:
    """Combine consecutive single-qubit gates on the same qubit into a
    single U(theta, phi, lambda).

    Unrecognised gate names raise ``ValueError`` rather than silently
    being dropped (which used to make typos like ``FOO_BAR`` evaluate to
    the identity — a subtle and dangerous bug).
    """
    gates = circuit._gates
    new_gates: List[GateRecord] = []
    pending: Dict[int, np.ndarray] = {}
    def _flush_qubit(q: int):
        if q in pending:
            u = pending.pop(q)
            if np.allclose(u, np.eye(2), atol=1e-10): return
            t, p, l = _u2_to_params(u)
            if abs(t) < 1e-10 and abs(p) < 1e-10 and abs(l) < 1e-10: return
            new_gates.append(GateRecord("U", [q], [t, p, l]))
    def _flush_all():
        for q in sorted(pending.keys()): _flush_qubit(q)
    for gate in gates:
        name = gate.name.upper()
        if name in ("BARRIER", "MEASURE", "RESET"):
            _flush_all()
            new_gates.append(gate)
            continue
        qs = gate.qubits
        if len(qs) == 1:
            if name not in _KNOWN_1Q_GATES:
                raise ValueError(
                    f"fuse_single_qubit_gates: unknown 1-qubit gate "
                    f"'{gate.name}'. Known gates: "
                    f"{sorted(_KNOWN_1Q_GATES)}.  Add it to "
                    f"`_KNOWN_1Q_GATES` and `_gate_to_u2` in turbo.py "
                    f"if it should be supported."
                )
            q = qs[0]
            mat = _gate_to_u2(gate)
            pending[q] = mat @ pending.get(q, np.eye(2))
        else:
            for q in qs: _flush_qubit(q)
            new_gates.append(gate)
    _flush_all()
    res = Circuit(circuit.n_qubits)
    res._gates = new_gates
    return res

def fuse_all_gates(circuit: Circuit) -> Circuit:
    return fuse_single_qubit_gates(fuse_two_qubit_gates(circuit))

# ════════════════════════════════════════════════════════════
#  DECOMPOSITION
# ════════════════════════════════════════════════════════════

def decompose_for_rust(gates: List[GateRecord]) -> List[GateRecord]:
    result: List[GateRecord] = []
    for g in gates:
        name = g.name.upper()
        nq = len(g.qubits)
        if name in ("CP", "CR1", "CPHASE") and nq == 2:
            phi = g.params[0] if g.params else 0.0
            p_val = float(phi) if not hasattr(phi, "value") else float(phi.value)
            result.extend([GateRecord("P", [g.qubits[0]], [p_val/2]), GateRecord("CX", [g.qubits[0], g.qubits[1]]),
                           GateRecord("P", [g.qubits[1]], [-p_val/2]), GateRecord("CX", [g.qubits[0], g.qubits[1]]),
                           GateRecord("P", [g.qubits[1]], [p_val/2])])
        elif name == "CRY" and nq == 2:
            th = g.params[0] if g.params else 0.0
            th_v = float(th) if not hasattr(th, "value") else float(th.value)
            result.extend([GateRecord("RY", [g.qubits[1]], [th_v/2]), GateRecord("CX", [g.qubits[0], g.qubits[1]]), 
                           GateRecord("RY", [g.qubits[1]], [-th_v/2]), GateRecord("CX", [g.qubits[0], g.qubits[1]])])
        elif name == "CH" and nq == 2:
            c, t = g.qubits
            result.extend([GateRecord("S", [t]), GateRecord("H", [t]), GateRecord("T", [t]), GateRecord("CX", [c, t]), 
                           GateRecord("TDG", [t]), GateRecord("H", [t]), GateRecord("SDG", [t])])
        elif name == "U1" and nq == 1:
            result.append(GateRecord("P", [g.qubits[0]], [float(g.params[0])]))
        elif name == "U2" and nq == 1:
            result.append(GateRecord("U", [g.qubits[0]], [math.pi/2, float(g.params[0]), float(g.params[1])]))
        elif name == "U3" and nq == 1:
            ps = [float(p) for p in g.params]
            while len(ps) < 3: ps.append(0.0)
            result.append(GateRecord("U", [g.qubits[0]], ps[:3]))
        else:
            result.append(g)
    return result

def expand_3q_gates(gates: List[GateRecord]) -> List[GateRecord]:
    result: List[GateRecord] = []
    for g in gates:
        if g.name.upper() in ("CCX", "TOFFOLI") and len(g.qubits) == 3:
            c0, c1, t = g.qubits
            result.extend([GateRecord("H", [t]), GateRecord("CX", [c1, t]), GateRecord("TDG", [t]), GateRecord("CX", [c0, t]), 
                           GateRecord("T", [t]), GateRecord("CX", [c1, t]), GateRecord("TDG", [t]), GateRecord("CX", [c0, t]), 
                           GateRecord("T", [t]), GateRecord("T", [c1]), GateRecord("H", [t]), GateRecord("CX", [c0, c1]), 
                           GateRecord("T", [c0]), GateRecord("TDG", [c1]), GateRecord("CX", [c0, c1])])
        else:
            result.append(g)
    return result

# ════════════════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════════════════

def simulate_statevector_turbo(circuit: Circuit, seed: int = 42) -> np.ndarray:
    # Expand 3q gates (CCX/Toffoli, CSWAP) into 1Q+2Q sequences before fusion.
    # Without this, CCX gates silently fail because the tensordot loop
    # only handles 1Q and 2Q gates.
    expanded = expand_3q_gates(circuit._gates)
    if expanded is not circuit._gates:
        circuit = Circuit(circuit.n_qubits)
        circuit._gates = expanded
    fused = fuse_all_gates(circuit)
    n = fused.n_qubits
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    st_t = state.reshape([2] * n) if n > 0 else state
    for gate in fused._gates:
        if gate.name.upper() in ("BARRIER", "MEASURE"): continue
        qs = gate.qubits
        mat = GateMatrixCache.get(gate).astype(np.complex128)
        if len(qs) == 1:
            res = np.tensordot(mat, st_t, axes=([1], [qs[0]]))
            st_t = np.moveaxis(res, 0, qs[0])
        elif len(qs) == 2:
            res = np.tensordot(mat.reshape(2,2,2,2), st_t, axes=([2,3], [qs[0], qs[1]]))
            st_t = np.moveaxis(res, [0,1], [qs[0], qs[1]])
    return np.ascontiguousarray(st_t).reshape(dim)

class GateMatrixCache:
    _cache: Dict[Tuple, np.ndarray] = {}
    @classmethod
    def get(cls, gate: GateRecord):
        ps = tuple(round(float(p if not hasattr(p, 'value') else p.value), 12) for p in gate.params)
        key = (gate.name, ps)
        if key not in cls._cache:
            if len(gate.qubits) == 1: cls._cache[key] = _gate_to_u2(gate)
            elif len(gate.qubits) == 2: cls._cache[key] = _get_2q_matrix(gate)
            else: cls._cache[key] = gate.to_unitary()
        return cls._cache[key]
    @classmethod
    def clear(cls): cls._cache.clear()

def encode_circuit_flat(circuit: Circuit):
    ns, qs, ps = [], [], []
    for g in circuit._gates:
        if g.name.upper() in ("MEASURE", "BARRIER"): continue
        ns.append(g.name); qs.append(g.qubits)
        ps.append([float(p) if not hasattr(p, 'value') else p.value for p in g.params])
    return ns, qs, ps

def direct_mps_sample(tensors, v2p, n, shots, seed=None):
    rng = np.random.RandomState(seed)
    counts = {}
    for _ in range(shots):
        raw = _sample_one_mps(tensors, rng)
        mapped = [raw[v2p[v]] for v in range(n)]
        bs = "".join(map(str, mapped))
        counts[bs] = counts.get(bs, 0) + 1
    return counts

def _sample_one_mps(tensors, rng):
    curr = np.array([1.0], dtype=np.complex64).reshape(1)
    bits = []
    for t in tensors:
        if t.ndim == 3: res = np.einsum('l,lpk->pk', curr, t)
        else: 
            d_l = t.shape[0]//2
            res = np.einsum('l,lpk->pk', curr, t.reshape(d_l, 2, -1))
        p0, p1 = float(np.sum(np.abs(res[0])**2)), float(np.sum(np.abs(res[1])**2))
        total = p0 + p1
        bit = 0 if (total < 1e-30 or rng.random() < p0/total) else 1
        bits.append(bit)
        vec = res[bit]
        norm = np.linalg.norm(vec)
        curr = vec / norm if norm > 1e-12 else vec
    return bits
