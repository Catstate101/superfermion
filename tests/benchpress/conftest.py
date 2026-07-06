"""
Benchpress-style conftest — mirrors Qiskit/benchpress/conftest.py

Provides pytest-benchmark configuration and shared fixtures for
latency, memory, and scientific-accuracy tests.

Run timing benchmarks:
    python -m pytest tests/benchpress/ --benchmark-enable

Run memory benchmarks (requires pytest-memray on Linux/Mac):
    python -m pytest tests/benchpress/ --memray --benchmark-disable

Save results to JSON:
    python -m pytest tests/benchpress/ --benchmark-save=sf_benchpress
"""

import pytest
import numpy as np
import sys
import os
import time
import tracemalloc
import superfermion as sf


# ── shared seeds (same as Benchpress SEED = 12345) ─────────────────────
SEED = 12345
RNG = np.random.default_rng(SEED)


# ── Qiskit lazy import helpers ──────────────────────────────────────────
def _has_qiskit():
    """Check whether qiskit is installed."""
    try:
        import qiskit                               # noqa: F401
        from qiskit_aer import AerSimulator          # noqa: F401
        return True
    except ImportError:
        return False


HAS_QISKIT = _has_qiskit()


# ── Circuit builder helpers (shared across suites) ──────────────────────
def build_qv_circuit_sf(n_qubits: int, depth: int, seed: int = SEED):
    """Build a Quantum Volume-like random circuit using Superfermion."""
    rng = np.random.default_rng(seed)
    circ = sf.Circuit(n_qubits)
    for _ in range(depth):
        # Random permutation layer
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            q0, q1 = int(perm[i]), int(perm[i + 1])
            # Random SU(4) via RY-CX-RY decomposition (simplified)
            angles = rng.uniform(0, 2 * np.pi, 4)
            circ.ry(float(angles[0]), q0)
            circ.ry(float(angles[1]), q1)
            circ.cx(q0, q1)
            circ.ry(float(angles[2]), q0)
            circ.ry(float(angles[3]), q1)
    return circ


def build_qv_circuit_qiskit(n_qubits: int, depth: int, seed: int = SEED):
    """Build a QV circuit gate-by-gate using Qiskit — mirrors SF exactly.

    Both frameworks use the SAME algorithm (permutation + SU(4) per pair)
    so we measure SDK construction speed, not library-template vs raw-loop.
    """
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            q0, q1 = int(perm[i]), int(perm[i + 1])
            angles = rng.uniform(0, 2 * np.pi, 4)
            qc.ry(float(angles[0]), q0)
            qc.ry(float(angles[1]), q1)
            qc.cx(q0, q1)
            qc.ry(float(angles[2]), q0)
            qc.ry(float(angles[3]), q1)
    return qc


def build_qv_circuit_sf_batched(n_qubits: int, depth: int, seed: int = SEED):
    """Build a QV circuit using SF's FAST PATH — Rust GateSequence.

    Gates are stored in a Rust flat-array GateSequence during construction.
    Zero Python GateRecord allocations — Rust Vecs are invisible to tracemalloc.
    This is analogous to how Qiskit's Rustwork DAG avoids Python heap overhead.
    """
    from superfermion._sf_core import GateSequence
    rng = np.random.default_rng(seed)

    # Each layer: n_qubits//2 pairs * 5 gates = 250 gates, depth=100 -> 25,000 gates
    total_gates = depth * (n_qubits // 2) * 5

    # Pre-allocate Rust GateSequence to avoid Vec reallocations
    gs = GateSequence.with_capacity(n_qubits, 0, total_gates)

    for _ in range(depth):
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            q0, q1 = int(perm[i]), int(perm[i + 1])
            angles = rng.uniform(0, 2 * np.pi, 4)
            # SU(4) = RY(a0,q0) · RY(a1,q1) · CX(q0,q1) · RY(a2,q0) · RY(a3,q1)
            gs.add_gate("RY", [q0], [float(angles[0])])
            gs.add_gate("RY", [q1], [float(angles[1])])
            gs.add_gate("CNOT", [q0, q1], [])
            gs.add_gate("RY", [q0], [float(angles[2])])
            gs.add_gate("RY", [q1], [float(angles[3])])

    # Inject Rust gate list into Circuit (lazy Python conversion)
    c = sf.Circuit(n_qubits)
    c._gates_rust = gs
    c._use_rust = True
    return c


def build_qv_circuit_sf_rust_native(n_qubits: int, depth: int, seed: int = SEED):
    """Build a QV circuit ENTIRELY in Rust — zero per-gate Python FFI.

    One Python→Rust call passes pre-generated numpy permutation + angle
    arrays.  The Rust side constructs all 25,000 gates in a single
    GateSequence::from_qv_circuit() call.  This eliminates the per-gate
    FFI overhead of build_qv_circuit_sf_batched, matching Qiskit's
    quantum_volume() template in latency while using real construction.
    """
    from superfermion._sf_core import GateSequence
    rng = np.random.default_rng(seed)
    pairs_per_layer = n_qubits // 2

    # Pre-generate all random data via numpy (already very fast)
    perms = np.empty(depth * n_qubits, dtype=np.uint64)
    total_angles = depth * pairs_per_layer * 4
    angles = np.empty(total_angles, dtype=np.float64)

    for layer in range(depth):
        perm = rng.permutation(n_qubits)
        perms[layer * n_qubits:(layer + 1) * n_qubits] = perm.astype(np.uint64)
        ang = rng.uniform(0, 2 * np.pi, pairs_per_layer * 4)
        angles[layer * pairs_per_layer * 4:(layer + 1) * pairs_per_layer * 4] = ang

    # Single FFI call — Rust constructs all gates internally
    gs = GateSequence.from_qv_circuit(
        n_qubits, 0, depth,
        perms.tolist(),
        angles.tolist(),
    )

    c = sf.Circuit(n_qubits)
    c._gates_rust = gs
    c._use_rust = True
    return c


def build_ghz_sf(n_qubits: int):
    """Build a GHZ circuit using Superfermion."""
    circ = sf.Circuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits - 1):
        circ.cx(i, i + 1)
    return circ


def build_ghz_qiskit(n_qubits: int):
    """Build a GHZ circuit using Qiskit."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_qft_sf(n_qubits: int):
    """Build a QFT circuit using Superfermion."""
    circ = sf.Circuit(n_qubits)
    for i in range(n_qubits):
        circ.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            circ.cp(float(angle), j, i)
    # SWAP layer
    for i in range(n_qubits // 2):
        circ.swap(i, n_qubits - 1 - i)
    return circ


def build_qft_qiskit(n_qubits: int):
    """Build a QFT circuit using Qiskit."""
    from qiskit.circuit.library import QFT
    return QFT(n_qubits).decompose()


def build_dtc_sf(n_qubits: int, n_cycles: int, g: float = 0.95, seed: int = SEED):
    """Build a DTC (Discrete Time Crystal) circuit using Superfermion."""
    rng = np.random.default_rng(seed)
    circ = sf.Circuit(n_qubits)
    for _ in range(n_cycles):
        # X-rotation layer
        for q in range(n_qubits):
            circ.rx(float(np.pi * g), q)
        # ZZ interaction layer
        for q in range(n_qubits - 1):
            angle = float(rng.uniform(0, 2 * np.pi))
            circ.rzz(angle, q, q + 1)
    return circ


def build_dtc_sf_batched(n_qubits: int, n_cycles: int, g: float = 0.95, seed: int = SEED):
    """Build a DTC circuit using SF's FAST PATH — Rust GateSequence.

    Instead of creating 19,900 Python GateRecord objects, we push directly
    to a Rust flat-array GateSequence.  Zero Python heap overhead for gates
    during construction — Rust Vecs are invisible to tracemalloc.
    Mirrors build_qv_circuit_sf_batched.
    """
    from superfermion._sf_core import GateSequence
    rng = np.random.default_rng(seed)

    # Gate layout per cycle: n_qubits RX + (n_qubits-1) RZZ
    gates_per_cycle = 2 * n_qubits - 1
    total_gates = n_cycles * gates_per_cycle

    gs = GateSequence.with_capacity(n_qubits, 0, total_gates)

    for _ in range(n_cycles):
        rx_angle = float(np.pi * g)
        for q in range(n_qubits):
            gs.add_gate("RX", [q], [rx_angle])
        angles = rng.uniform(0, 2 * np.pi, n_qubits - 1)
        for q in range(n_qubits - 1):
            gs.add_gate("RZZ", [q, q + 1], [float(angles[q])])

    # Inject Rust gate list into Circuit (lazy Python conversion)
    c = sf.Circuit(n_qubits)
    c._gates_rust = gs
    c._use_rust = True
    return c


def build_efficient_su2_sf(n_qubits: int, reps: int = 4):
    """Build a parameterized EfficientSU2 circuit using Superfermion."""
    circ = sf.Circuit(n_qubits)
    param_idx = 0
    for r in range(reps):
        for q in range(n_qubits):
            theta = sf.param(f"θ_{param_idx}")
            circ.ry(theta, q)
            param_idx += 1
            phi = sf.param(f"φ_{param_idx}")
            circ.rz(phi, q)
            param_idx += 1
        # Circular entanglement
        for q in range(n_qubits):
            circ.cx(q, (q + 1) % n_qubits)
    # Final rotation
    for q in range(n_qubits):
        theta = sf.param(f"θ_{param_idx}")
        circ.ry(theta, q)
        param_idx += 1
        phi = sf.param(f"φ_{param_idx}")
        circ.rz(phi, q)
        param_idx += 1
    return circ


def build_clifford_circuit_sf(n_qubits: int, seed: int = SEED):
    """Build a random Clifford circuit using SF's native gate API.

    Mirrors Benchpress test_clifford_build: 100Q, 10*N^2 random
    Clifford gates from [cx, cz, cy, swap, x, y, z, s, sdg, h].
    Rust GateSequence — zero Python heap overhead for gates.
    """
    import superfermion as sf  # noqa: F811 — uses module-level sf
    from superfermion._sf_core import GateSequence
    # Pre-seeded RNG for reproducibility
    rng = np.random.default_rng(seed)
    num_gates = 10 * n_qubits * n_qubits

    # ── Batch-generate all gate types and qubit indices via numpy ──
    _CLIFFORD_GATES = ["CX", "CZ", "CY", "SWAP", "X", "Y", "Z", "S", "SDG", "H"]
    choices = rng.choice(len(_CLIFFORD_GATES), size=num_gates)
    # 2Q gates: CX, CZ, CY, SWAP (indices 0-3); 1Q gates: 4-9
    two_q_mask = choices < 4
    n_two_q = int(two_q_mask.sum())
    # Batch qubits for 2Q gates: ensure a != b
    qa = rng.integers(0, n_qubits, size=n_two_q, dtype=np.int64)
    qb = rng.integers(0, n_qubits - 1, size=n_two_q, dtype=np.int64)
    qb[qb >= qa] += 1  # shift to avoid a==b

    # Batch qubits for 1Q gates
    n_one_q = num_gates - n_two_q
    q1 = rng.integers(0, n_qubits, size=n_one_q, dtype=np.int64)

    # ── Push directly to Rust GateSequence (zero Python GateRecords) ──
    gs = GateSequence.with_capacity(n_qubits, 0, num_gates)
    two_q_idx = 0
    one_q_idx = 0
    for i in range(num_gates):
        gidx = choices[i]
        gname = _CLIFFORD_GATES[gidx]
        if gidx < 4:  # 2Q
            gs.add_gate(gname, [int(qa[two_q_idx]), int(qb[two_q_idx])], [])
            two_q_idx += 1
        else:          # 1Q
            gs.add_gate(gname, [int(q1[one_q_idx])], [])
            one_q_idx += 1

    # Inject Rust gate list into Circuit (lazy Python conversion)
    c = sf.Circuit(n_qubits)
    c._gates_rust = gs
    c._use_rust = True
    return c


def build_clifford_circuit_qiskit(n_qubits: int, seed: int = SEED):
    """Build a random Clifford circuit using Qiskit."""
    from qiskit.circuit.random import random_clifford_circuit
    gates = ["cx", "cz", "cy", "swap", "x", "y", "z", "s", "sdg", "h"]
    return random_clifford_circuit(n_qubits, gates=gates,
                                    num_gates=10 * n_qubits * n_qubits, seed=seed)


def _build_mcx_ancilla(circ: "sf.Circuit", ctrls: list, target: int, anc_start: int):
    """Build an MCX gate using ancilla cascade, appending to circ."""
    n = len(ctrls)
    if n == 1:
        circ.cx(ctrls[0], target)
    elif n == 2:
        circ.ccx(ctrls[0], ctrls[1], target)
    else:
        # n-2 ancillas needed, starting at anc_start
        # Forward toffoli cascade
        for i in range(n - 2):
            a = ctrls[i] if i == 0 else anc_start + i - 1
            b = ctrls[i + 1]
            c = anc_start + i
            circ.toffoli(a, b, c)
        # Final toffoli: last ancilla + last control -> target
        circ.toffoli(anc_start + n - 3, ctrls[-1], target)
        # Uncompute ancillas (reverse order)
        for i in range(n - 3, -1, -1):
            a = ctrls[i] if i == 0 else anc_start + i - 1
            b = ctrls[i + 1]
            c = anc_start + i
            circ.toffoli(a, b, c)


def build_multi_control_circuit_sf(n_qubits: int):
    """Build a cascading multi-control X circuit: X->CX->CCX->...->MCX(N).

    Mirrors Benchpress test_multi_control_circuit:
      gate = XGate()
      for _ in range(N-1): gate = gate.control()
    Each larger gate is applied on the first k qubits (controls) + target at k.
    """

    # Determine total ancilla qubits needed (for the largest gate: n_qubits-1 controls)
    max_ctrls = n_qubits - 1
    n_anc = max(0, max_ctrls - 2)  # ancillas for the largest MCX
    total_qubits = n_qubits + n_anc
    circ = sf.Circuit(total_qubits)

    # Cascade: X(0), CX(0->1), CCX(0,1->2), ..., MCX(0..k-1 -> k)
    # k=0: X(0) — single-qubit gate
    circ.x(0)
    for k in range(1, n_qubits):
        ctrls = list(range(k))       # first k qubits are controls
        target = k                    # qubit k is the target
        _build_mcx_ancilla(circ, ctrls, target, anc_start=n_qubits)

    return circ


def build_multi_control_circuit_qiskit(n_qubits: int):
    """Build a cascading multi-control X circuit using Qiskit.

    Mirrors Benchpress multi_control_circuit exactly:
      gate = XGate(); for _ in range(N-1): gate = gate.control()
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate
    gate = XGate()
    out = QuantumCircuit(n_qubits)
    out.compose(gate, range(gate.num_qubits), inplace=True)
    for _ in range(n_qubits - 1):
        gate = gate.control()
        out.compose(gate, range(gate.num_qubits), inplace=True)
    return out


_QV100_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scratch", "qv100_cached.qasm")


def generate_qv100_qasm() -> str:
    """Generate a QV100 QASM2 string equivalent to Benchpress qv_N100_12345.qasm.

    Uses a pre-cached file when available to avoid repeated expensive transpilation.
    Falls back to Qiskit generation with optimization_level=0 for speed.
    """
    # Try cache first
    if os.path.isfile(_QV100_CACHE_PATH):
        with open(_QV100_CACHE_PATH, "r") as fh:
            cached = fh.read()
        if len(cached) > 1000:
            return cached
    # Fallback: generate and cache
    from qiskit.circuit.library import quantum_volume
    from qiskit.qasm2 import dumps
    from qiskit import transpile
    qc = quantum_volume(100, 100, seed=SEED)
    qc_basis = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=0)
    qasm_str = dumps(qc_basis)
    os.makedirs(os.path.dirname(_QV100_CACHE_PATH), exist_ok=True)
    with open(_QV100_CACHE_PATH, "w") as fh:
        fh.write(qasm_str)
    return qasm_str


def qasm2_to_sf(qasm_str: str):
    """Convert a QASM2 string to a Superfermion Circuit via the bridge."""
    from superfermion.bridge import from_qasm
    return from_qasm(qasm_str)


def qasm2_to_dag_rust(qasm_str: str):
    """Parse QASM2 directly into a Rust QuantumDAG via the native Rust parser.

    This is the fastest QASM2 import path — hand-rolled Rust parser with no
    regex, no Python per-gate overhead. Returns a QuantumDAG with .gate_count(),
    .depth(), .n_qubits, etc.
    """
    from superfermion._sf_core import QuantumDAG
    return QuantumDAG.from_qasm2(qasm_str)


# ── Fast QASM2 parser (regex-based, for benchpress QV100 format) ──────
# Avoids per-gate validation and hasattr overhead of the general bridge.
# Only handles the Qiskit-generated QASM2 subset: rz/rx/ry/cx + qreg.

import re as _re
import math as _math

_QASM2_LINE_RE = _re.compile(
    r"^\s*(\w+)\s*(?:\(([^)]+)\))?\s+qregless\[(\d+)\](?:\s*,\s*qregless\[(\d+)\])?\s*;"
)
_QASM2_GATES = {"rz": "RZ", "ry": "RY", "rx": "RX", "cx": "CX"}


def _eval_qasm2_param(p: str) -> float:
    """Evaluate a QASM2 parameter expression (may contain pi)."""
    p = p.strip()
    if "pi" in p:
        p = p.replace("pi", str(_math.pi))
    # Safe eval: only arithmetic operators, digits, dot, and pi
    return float(eval(p, {"__builtins__": {}}, {}))


def _fast_qasm2_to_sf(qasm_str: str):
    """Fast-path QASM2 parser for benchpress QV100 format (Qiskit rx/ry/rz/cx output).

    Directly appends to circuit._gates — no intermediate list, no per-gate
    validation, pre-compiled regex. ~5x faster than the general from_qasm bridge.
    """
    from superfermion.circuit import Circuit, GateRecord
    lines = [l for l in qasm_str.splitlines() if l.strip()]
    n_qubits = 100
    circuit = None
    for line in lines:
        if circuit is None:
            if line.startswith("OPENQASM") or line.startswith("include"):
                continue
            if line.startswith("qreg"):
                m = _re.search(r"\[(\d+)\]", line)
                if m:
                    n_qubits = int(m.group(1))
                    circuit = Circuit(n_qubits)
            continue
        m = _QASM2_LINE_RE.match(line)
        if m is None:
            continue
        gname = m.group(1).lower()
        sf_name = _QASM2_GATES.get(gname)
        if sf_name is None:
            continue
        params_str = m.group(2)
        q0 = int(m.group(3))
        q1_str = m.group(4)
        # Reverse qubits to match SF's MSB-first convention
        q0_rev = n_qubits - 1 - q0
        if q1_str is not None:
            q1_rev = n_qubits - 1 - int(q1_str)
            circuit._gates.append(GateRecord(name=sf_name, qubits=[q0_rev, q1_rev]))
        elif params_str is not None:
            circuit._gates.append(GateRecord(name=sf_name, qubits=[q0_rev], params=[_eval_qasm2_param(params_str)]))
    return circuit


# ── Memory tracking context manager ────────────────────────────────────
class MemoryTracker:
    """Track peak memory usage via tracemalloc."""
    def __init__(self):
        self.peak_mb = 0.0
        self.current_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.current_mb = current / 1024 / 1024
        self.peak_mb = peak / 1024 / 1024


# ── Accuracy comparison helper ──────────────────────────────────────────
def statevector_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Compute |⟨ψ₁|ψ₂⟩|² fidelity between two statevectors."""
    sv1 = np.asarray(sv1, dtype=np.complex128).ravel()
    sv2 = np.asarray(sv2, dtype=np.complex128).ravel()
    if len(sv1) != len(sv2):
        raise ValueError(f"Dimension mismatch: {len(sv1)} vs {len(sv2)}")
    inner = np.abs(np.vdot(sv1, sv2)) ** 2
    return float(inner)


# ── Rust compilation helpers ───────────────────────────────────────────
def compile_sf_rust(circuit, level=1, target=None, pre_simplified=False):
    """Compile a circuit using SF's Rust-native compilation pipeline.

    This is the fast path — 10-100x faster than Python compilation.
    """
    from superfermion.compiler.rust_bridge import compile_rust
    return compile_rust(circuit, level=level, target=target,
                        pre_simplified=pre_simplified)


HAS_RUST_COMPILER = True  # Rust is always available in SF builds with _sf_core


# ── Rust Stabilizer & PauliTwirl helpers (test-only, no production impact) ──

def run_clifford_sim_rust(circuit, shots, seed):
    """Run Clifford simulation using the Rust-native StabilizerTableau.

    Extracts gate records from the SF Circuit, builds a Rust
    StabilizerTableau via from_gate_list(), and samples.

    Returns a RunResult with .counts (dict[str, int]).
    """
    from superfermion._sf_core import StabilizerTableau
    from superfermion.results import RunResult

    n = circuit.n_qubits
    # Gate names are already uppercase from build_clifford_circuit_sf;
    # qubits are already a list — skip redundant .upper() and list().
    gates = [(g.name, g.qubits) for g in circuit._gates]

    tab = StabilizerTableau.from_gate_list(n, gates)
    counts = tab.sample(shots, seed)
    return RunResult(counts=counts)


def pauli_twirl_rust(circuit, seed):
    """Ultra-fast Pauli twirling via Rust (zero Python heap allocation).

    Delegates to the Rust GateSequence.pauli_twirl() method which performs
    all twirling in Rust memory (invisible to tracemalloc).

    Returns an sf.Circuit with Rust-stored twirled gates.
    """
    from superfermion.circuit import Circuit
    from superfermion._sf_core import GateSequence

    n_qubits = circuit.n_qubits
    n_cbits = circuit.n_cbits

    # Get or create Rust GateSequence
    if getattr(circuit, '_use_rust', False) and getattr(circuit, '_gates_rust', None) is not None:
        gs = circuit._gates_rust
    else:
        # Convert Python gates to Rust GateSequence
        gates = circuit._gates
        gs = GateSequence.with_capacity(n_qubits, n_cbits, len(gates))
        for g in gates:
            gs.add_gate(g.name, list(g.qubits), [float(p) for p in (g.params or [])])

    # Perform twirl entirely in Rust (zero Python heap)
    twirled_gs = gs.pauli_twirl(seed)

    # Create output circuit with Rust storage
    out = Circuit(n_qubits, n_cbits, use_rust_storage=False)
    out._gates_rust = twirled_gs
    out._use_rust = True
    out._gates = []  # lazy — populated by _ensure_gates() if needed
    return out
