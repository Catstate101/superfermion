"""Cross-validation: every SF backend vs Qiskit-Aer AND PennyLane statevector.

For each gate and each backend:
- Build an identical circuit in SF, Qiskit, and PennyLane.
- Compare statevectors (fidelity > 1-1e-5) where the backend can return one.
- Compare sampled counts (TVD < 0.03 at 50k shots) for all backends.
- Endianness: SF is MSB (qubit 0 = leftmost bit), Qiskit is LSB.
  Normalise Qiskit counts by reversing each bitstring key.
- PennyLane uses MSB convention (same as SF), no reversal needed.

Run:
    PYTHONIOENCODING=utf-8 python -m pytest tests/test_accuracy_vs_qiskit.py -v
"""
from __future__ import annotations
import os, sys, math, unittest
from typing import Dict, Optional, Tuple
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SHOTS = 50_000
TVD_TOL = 0.03
FIDELITY_TOL = 1e-5

# ── Availability flags ────────────────────────────────────────────────
try:
    import cupy as cp
    _CUPY_AVAILABLE = cp.cuda.is_available()
except Exception:
    _CUPY_AVAILABLE = False

try:
    import pennylane as qml
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def _counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}

def _sv_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    return abs(np.dot(a.conj(), b)) ** 2

def _qiskit_counts_normalise(counts: Dict[str, int]) -> Dict[str, int]:
    """Reverse Qiskit bitstrings (LSB→MSB) to match SF convention."""
    out: Dict[str, int] = {}
    for k, v in counts.items():
        out[k[::-1]] = out.get(k[::-1], 0) + v
    return out


def _qiskit_sv_normalise(sv: np.ndarray, n: int) -> np.ndarray:
    """Reverse qubit axis ordering of a Qiskit statevector so that
    index i has the same basis-state meaning as in SF (MSB-first)."""
    from qiskit.quantum_info import Statevector
    return Statevector(sv).reverse_qargs().data


def _sf_run(backend_name: str, sf_circ, shots: int, seed: int = 42):
    """Run an SF circuit on a named backend; return (counts, sv_or_None)."""
    import superfermion as sf
    b = sf.get_backend(backend_name)
    r = b.run(sf_circ, shots=shots, seed=seed)
    sv = np.asarray(r.statevector) if r.statevector is not None else None
    return r.counts, sv


def _qiskit_run(qk_circ, shots: int, seed: int = 42):
    """Run a Qiskit circuit on AerSimulator statevector; return (counts_normalised, sv)."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    sv = Statevector(qk_circ).data
    sv_norm = _qiskit_sv_normalise(sv, qk_circ.num_qubits)
    sim = AerSimulator(method="statevector")
    qk_circ_meas = qk_circ.copy()
    qk_circ_meas.measure_all()
    tc = transpile(qk_circ_meas, sim)
    job = sim.run(tc, shots=shots, seed_simulator=seed)
    raw = dict(job.result().get_counts())
    counts = _qiskit_counts_normalise(raw)
    return counts, sv_norm


def _pennylane_run(qk_circ, shots: int, seed: int = 42) -> Tuple[Dict[str, int], np.ndarray]:
    """Run an equivalent circuit on PennyLane default.qubit.

    PennyLane uses the same MSB convention as SF so no bitstring reversal needed.
    We build the PL tape from the Qiskit circuit's gate list to guarantee
    circuit identity.  The statevector is computed via a separate analytic
    device (shots=None).
    """
    import pennylane as qml

    n = qk_circ.num_qubits

    # ── Build PennyLane ops from Qiskit gates ──────────────────────────
    def _qk_to_pl_ops(circuit):
        """Convert Qiskit gate list to PennyLane operations.

        Wire mapping: Qiskit qubit i → PennyLane wire i (direct, no reversal).
        Both SF and PL encode bit[0] as wire/qubit 0 in their sample output,
        so the raw gate indices are the same across both frameworks.
        """
        ops = []
        for instruction in circuit.data:
            name = instruction.operation.name
            p = [float(x) for x in instruction.operation.params]
            wires = [circuit.find_bit(q).index for q in instruction.qubits]
            if name == "h":        ops.append(qml.Hadamard(wires[0]))
            elif name == "x":      ops.append(qml.PauliX(wires[0]))
            elif name == "y":      ops.append(qml.PauliY(wires[0]))
            elif name == "z":      ops.append(qml.PauliZ(wires[0]))
            elif name == "s":      ops.append(qml.S(wires[0]))
            elif name == "sdg":    ops.append(qml.adjoint(qml.S)(wires[0]))
            elif name == "t":      ops.append(qml.T(wires[0]))
            elif name == "tdg":    ops.append(qml.adjoint(qml.T)(wires[0]))
            elif name == "sx":     ops.append(qml.SX(wires[0]))
            elif name == "rx":     ops.append(qml.RX(p[0], wires[0]))
            elif name == "ry":     ops.append(qml.RY(p[0], wires[0]))
            elif name == "rz":     ops.append(qml.RZ(p[0], wires[0]))
            elif name == "p":      ops.append(qml.PhaseShift(p[0], wires[0]))
            elif name == "u":      ops.append(qml.U3(p[0], p[1], p[2], wires[0]))
            elif name in ("cx", "cnot"): ops.append(qml.CNOT(wires))
            elif name == "cy":     ops.append(qml.CY(wires))
            elif name == "cz":     ops.append(qml.CZ(wires))
            elif name == "swap":   ops.append(qml.SWAP(wires))
            elif name == "iswap":  ops.append(qml.ISWAP(wires))
            elif name == "rxx":    ops.append(qml.IsingXX(p[0], wires))
            elif name == "ryy":    ops.append(qml.IsingYY(p[0], wires))
            elif name == "rzz":    ops.append(qml.IsingZZ(p[0], wires))
            elif name == "ccx":    ops.append(qml.Toffoli(wires))
            # ignore measure_all barrier etc.
        return ops

    ops = _qk_to_pl_ops(qk_circ)
    wires = list(range(n))

    # Statevector (analytic)
    dev_sv = qml.device("default.qubit", wires=wires)
    @qml.qnode(dev_sv)
    def _sv_circuit():
        for op in ops:
            qml.apply(op)
        return qml.state()

    sv = np.asarray(_sv_circuit(), dtype=np.complex128)

    # Sampled counts
    dev_sample = qml.device("default.qubit", wires=wires, shots=shots)
    @qml.qnode(dev_sample)
    def _sample_circuit():
        for op in ops:
            qml.apply(op)
        return qml.sample()

    # PennyLane returns an (shots, n) array of 0/1 bits in wire order (wire 0 first)
    np.random.seed(seed)
    raw_samples = np.asarray(_sample_circuit())
    counts: Dict[str, int] = {}
    for row in raw_samples:
        bs = "".join(map(str, row.astype(int)))
        counts[bs] = counts.get(bs, 0) + 1

    return counts, sv


# ──────────────────────────────────────────────────────────────────────
# Base class with per-gate helpers
# ──────────────────────────────────────────────────────────────────────

class _AccuracyBase:
    """Mixin — concrete subclasses must also inherit unittest.TestCase."""
    BACKEND: str = ""

    # -- gate circuit builders (returns SF circuit + matching Qiskit circuit) --

    @staticmethod
    def _bell(n: int = 2):
        import superfermion as sf
        from qiskit import QuantumCircuit
        sf_c = sf.Circuit(n); sf_c.h(0); sf_c.cx(0, 1)
        qk_c = QuantumCircuit(n); qk_c.h(0); qk_c.cx(0, 1)
        return sf_c, qk_c

    @staticmethod
    def _s_t_circuit(n: int = 2):
        import superfermion as sf
        from qiskit import QuantumCircuit
        sf_c = sf.Circuit(n)
        sf_c.h(0); sf_c.s(0); sf_c.t(1); sf_c.sdg(0); sf_c.tdg(1)
        sf_c.sx(0); sf_c.cx(0, 1)
        qk_c = QuantumCircuit(n)
        qk_c.h(0); qk_c.s(0); qk_c.t(1); qk_c.sdg(0); qk_c.tdg(1)
        qk_c.sx(0); qk_c.cx(0, 1)
        return sf_c, qk_c

    @staticmethod
    def _rotation_circuit(n: int = 3):
        import superfermion as sf
        from qiskit import QuantumCircuit
        angles = [0.7, 1.2, -0.4, 0.9, 2.1, -1.7]
        sf_c = sf.Circuit(n)
        qk_c = QuantumCircuit(n)
        for i in range(n):
            a = angles[i % len(angles)]
            sf_c.rx(a, i); sf_c.ry(a*0.7, i); sf_c.rz(a*0.3, i)
            qk_c.rx(a, i); qk_c.ry(a*0.7, i); qk_c.rz(a*0.3, i)
        sf_c.cx(0, 1); sf_c.cz(1, 2)
        qk_c.cx(0, 1); qk_c.cz(1, 2)
        return sf_c, qk_c

    @staticmethod
    def _cy_iswap_circuit(n: int = 3):
        import superfermion as sf
        from qiskit import QuantumCircuit
        sf_c = sf.Circuit(n)
        qk_c = QuantumCircuit(n)
        sf_c.h(0); sf_c.h(1); sf_c.h(2)
        qk_c.h(0); qk_c.h(1); qk_c.h(2)
        sf_c.cy(0, 1); sf_c.iswap(1, 2)
        qk_c.cy(0, 1); qk_c.iswap(1, 2)
        return sf_c, qk_c

    @staticmethod
    def _rxx_ryy_rzz_circuit(n: int = 2):
        import superfermion as sf
        from qiskit import QuantumCircuit
        theta = 0.8
        sf_c = sf.Circuit(n)
        qk_c = QuantumCircuit(n)
        sf_c.h(0); sf_c.rxx(theta, 0, 1)
        qk_c.h(0); qk_c.rxx(theta, 0, 1)
        sf_c2 = sf.Circuit(n)
        qk_c2 = QuantumCircuit(n)
        sf_c2.h(0); sf_c2.ryy(theta, 0, 1)
        qk_c2.h(0); qk_c2.ryy(theta, 0, 1)
        return (sf_c, qk_c), (sf_c2, qk_c2)

    @staticmethod
    def _u3_p_circuit(n: int = 2):
        import superfermion as sf
        from qiskit import QuantumCircuit
        sf_c = sf.Circuit(n)
        qk_c = QuantumCircuit(n)
        sf_c.u3(0.5, 0.3, -0.2, 0); sf_c.p(1.1, 1); sf_c.cx(0, 1)
        qk_c.u(0.5, 0.3, -0.2, 0); qk_c.p(1.1, 1); qk_c.cx(0, 1)
        return sf_c, qk_c

    @staticmethod
    def _ccx_circuit(n: int = 3):
        import superfermion as sf
        from qiskit import QuantumCircuit
        sf_c = sf.Circuit(n)
        qk_c = QuantumCircuit(n)
        sf_c.h(0); sf_c.h(1); sf_c.ccx(0, 1, 2)
        qk_c.h(0); qk_c.h(1); qk_c.ccx(0, 1, 2)
        return sf_c, qk_c

    # -- assertion helpers --

    def _assert_match(self, label: str, sf_circ, qk_circ, check_sv: bool = True):
        """Compare SF backend vs Qiskit-Aer (primary ground truth)."""
        sf_counts, sf_sv = _sf_run(self.BACKEND, sf_circ, SHOTS)
        qk_counts, qk_sv = _qiskit_run(qk_circ, SHOTS)

        sf_probs = _counts_to_probs(sf_counts)
        qk_probs = _counts_to_probs(qk_counts)
        tvd = _tvd(sf_probs, qk_probs)
        self.assertLess(
            tvd, TVD_TOL,
            f"[{self.BACKEND}] {label}: TVD={tvd:.4f} > {TVD_TOL}\n"
            f"  SF top-5:  {sorted(sf_probs.items(), key=lambda x:-x[1])[:5]}\n"
            f"  QK top-5:  {sorted(qk_probs.items(), key=lambda x:-x[1])[:5]}",
        )
        if check_sv and sf_sv is not None:
            fid = _sv_fidelity(sf_sv, qk_sv)
            self.assertGreater(
                fid, 1.0 - FIDELITY_TOL,
                f"[{self.BACKEND}] {label}: statevector fidelity={fid:.8f}",
            )

    @unittest.skipUnless(_PL_AVAILABLE, "PennyLane not installed")
    def _assert_match_pl(self, label: str, sf_circ, qk_circ, check_sv: bool = True):
        """Compare SF backend vs PennyLane default.qubit (second ground truth)."""
        sf_counts, sf_sv = _sf_run(self.BACKEND, sf_circ, SHOTS)
        pl_counts, pl_sv = _pennylane_run(qk_circ, SHOTS)

        sf_probs = _counts_to_probs(sf_counts)
        pl_probs = _counts_to_probs(pl_counts)
        tvd = _tvd(sf_probs, pl_probs)
        self.assertLess(
            tvd, TVD_TOL,
            f"[{self.BACKEND}] {label} (vs PL): TVD={tvd:.4f} > {TVD_TOL}\n"
            f"  SF top-5:  {sorted(sf_probs.items(), key=lambda x:-x[1])[:5]}\n"
            f"  PL top-5:  {sorted(pl_probs.items(), key=lambda x:-x[1])[:5]}",
        )
        if check_sv and sf_sv is not None:
            fid = _sv_fidelity(sf_sv, pl_sv)
            self.assertGreater(
                fid, 1.0 - FIDELITY_TOL,
                f"[{self.BACKEND}] {label} (vs PL): statevector fidelity={fid:.8f}",
            )

    # -- test methods (vs Qiskit-Aer) --

    def test_01_bell_state(self):
        self._assert_match("Bell H+CX", *self._bell())

    def test_02_s_t_sdg_tdg_sx(self):
        self._assert_match("S/T/SDG/TDG/SX", *self._s_t_circuit())

    def test_03_rotations_rx_ry_rz_cx_cz(self):
        self._assert_match("RX/RY/RZ/CX/CZ", *self._rotation_circuit())

    def test_04_cy_iswap(self):
        self._assert_match("CY/ISWAP", *self._cy_iswap_circuit())

    def test_05_rxx(self):
        (sf_c, qk_c), _ = self._rxx_ryy_rzz_circuit()
        self._assert_match("RXX", sf_c, qk_c)

    def test_06_ryy(self):
        _, (sf_c, qk_c) = self._rxx_ryy_rzz_circuit()
        self._assert_match("RYY", sf_c, qk_c)

    def test_07_u3_p(self):
        self._assert_match("U3/P", *self._u3_p_circuit())

    def test_08_ccx_toffoli(self):
        self._assert_match("CCX/Toffoli", *self._ccx_circuit(), check_sv=False)

    # -- test methods (vs PennyLane) --

    def test_pl_01_bell_state(self):
        self._assert_match_pl("Bell H+CX", *self._bell())

    def test_pl_02_s_t_sdg_tdg_sx(self):
        self._assert_match_pl("S/T/SDG/TDG/SX", *self._s_t_circuit())

    def test_pl_03_rotations_rx_ry_rz_cx_cz(self):
        self._assert_match_pl("RX/RY/RZ/CX/CZ", *self._rotation_circuit())

    def test_pl_04_cy_iswap(self):
        self._assert_match_pl("CY/ISWAP", *self._cy_iswap_circuit())

    def test_pl_05_rxx(self):
        (sf_c, qk_c), _ = self._rxx_ryy_rzz_circuit()
        self._assert_match_pl("RXX", sf_c, qk_c)

    def test_pl_06_ryy(self):
        _, (sf_c, qk_c) = self._rxx_ryy_rzz_circuit()
        self._assert_match_pl("RYY", sf_c, qk_c)

    def test_pl_07_u3_p(self):
        self._assert_match_pl("U3/P", *self._u3_p_circuit())

    def test_pl_08_ccx_toffoli(self):
        self._assert_match_pl("CCX/Toffoli", *self._ccx_circuit(), check_sv=False)


# ──────────────────────────────────────────────────────────────────────
# One subclass per testable backend
# ──────────────────────────────────────────────────────────────────────

class TestStatevectorBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "statevector"

class TestJAXBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "jax"

class TestSingularityBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "singularity"

class TestSupremacyBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "supremacy"

class TestJAXMPSBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "jax_mps"

class TestMPSBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "mps"

@unittest.skipUnless(_CUPY_AVAILABLE, "CuPy / CUDA not available")
class TestCudaBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "cuda"

@unittest.skipUnless(_CUPY_AVAILABLE, "CuPy / CUDA not available")
class TestCudaMPSBackend(_AccuracyBase, unittest.TestCase):
    BACKEND = "cuda_mps"


if __name__ == "__main__":
    unittest.main(verbosity=2)
