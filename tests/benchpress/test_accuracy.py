"""
Benchpress Scientific-Accuracy Tests — Superfermion vs Qiskit
=============================================================

Modeled after Qiskit/benchpress validation concepts. Benchpress verifies
not only speed, but that different SDKs produce the correct quantum states.

We use pytest-benchmark for the structure, but the core assertions here
are scientific: statevector fidelity > 0.9999 and expectation value matching.

Run:
    python -m pytest tests/benchpress/test_accuracy.py -v --benchmark-disable
"""

import pytest
import numpy as np
import math

import superfermion as sf

from tests.benchpress.conftest import (
    SEED,
    HAS_QISKIT,
    build_ghz_sf,
    build_ghz_qiskit,
    build_qft_sf,
    build_qft_qiskit,
    build_qv_circuit_sf,
    build_qv_circuit_qiskit,
    build_clifford_circuit_sf,
    build_clifford_circuit_qiskit,
    build_multi_control_circuit_sf,
    build_multi_control_circuit_qiskit,
    generate_qv100_qasm,
    qasm2_to_sf,
    statevector_fidelity,
)

# Skip entire suite if Qiskit isn't installed, as we need it for ground truth
pytestmark = pytest.mark.skipif(not HAS_QISKIT, reason="Requires Qiskit for ground truth")


class TestScientificAccuracy:
    """Verifies that Superfermion produces physically identical results to Qiskit."""

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_ghz_fidelity(self, sf_backend, n):
        """Statevector fidelity for GHZ circuit should be 1.0."""
        # 1. Superfermion
        circ_sf = build_ghz_sf(n)
        sv_sf = sf.run(circ_sf, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()

        # 2. Qiskit
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        qc = build_ghz_qiskit(n).reverse_bits()
        qc.save_statevector()
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc, sim)
        sv_qiskit = np.array(sim.run(tqc).result().get_statevector())

        # 3. Compare
        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"GHZ({n}) {sf_backend} fidelity: {fidelity}"

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_qft_fidelity(self, sf_backend, n):
        """Statevector fidelity for QFT circuit should be 1.0."""
        circ_sf = build_qft_sf(n)
        sv_sf = sf.run(circ_sf, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()

        from qiskit import transpile
        from qiskit_aer import AerSimulator
        qc = build_qft_qiskit(n).reverse_bits()
        qc.save_statevector()
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc, sim)
        sv_qiskit = np.array(sim.run(tqc).result().get_statevector())

        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"QFT({n}) {sf_backend} fidelity: {fidelity}"

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("depth", [10, 20])
    def test_qv_random_fidelity(self, sf_backend, depth):
        """Statevector fidelity for random QV circuits (10Q) should be 1.0."""
        n = 10
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        
        # Ground truth Qiskit circuit
        qc = build_qv_circuit_qiskit(n, depth, seed=SEED)
        
        # Decompose into standard basis so we can port it to SF
        qc_basis = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx'])
        
        # Port to SF
        circ_sf = sf.Circuit(n)
        for instr in qc_basis.data:
            name = instr.operation.name
            qargs = [qc_basis.find_bit(q).index for q in instr.qubits]
            params = instr.operation.params
            
            if name == 'rx':
                circ_sf.rx(params[0], qargs[0])
            elif name == 'ry':
                circ_sf.ry(params[0], qargs[0])
            elif name == 'rz':
                circ_sf.rz(params[0], qargs[0])
            elif name == 'cx':
                circ_sf.cx(qargs[0], qargs[1])
                
        sv_sf = sf.run(circ_sf, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()

        qc_basis_rev = qc_basis.reverse_bits()
        qc_basis_rev.save_statevector()
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc_basis_rev, sim)
        sv_qiskit = np.array(sim.run(tqc).result().get_statevector())

        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"QV({n},{depth}) fidelity: {fidelity}"

    def test_gate_matrix_definitions(self):
        """Verify that basic gate unitaries match Qiskit definitions (phase conventions)."""
        from qiskit.circuit.library import RXGate, RYGate, RZGate, HGate, CXGate
        from superfermion.simulator import _get_gate_matrix
        from superfermion.circuit import GateRecord
        
        # H Gate
        sf_h = _get_gate_matrix(GateRecord("H", [0]))
        qk_h = HGate().to_matrix()
        assert np.allclose(sf_h, qk_h)

        # RX(pi/4)
        theta = np.pi / 4
        sf_rx = _get_gate_matrix(GateRecord("RX", [0], [theta]))
        qk_rx = RXGate(theta).to_matrix()
        assert np.allclose(sf_rx, qk_rx)
        
        # RY(pi/3)
        theta = np.pi / 3
        sf_ry = _get_gate_matrix(GateRecord("RY", [0], [theta]))
        qk_ry = RYGate(theta).to_matrix()
        assert np.allclose(sf_ry, qk_ry)
        
        # RZ(pi/2)
        theta = np.pi / 2
        sf_rz = _get_gate_matrix(GateRecord("RZ", [0], [theta]))
        qk_rz = RZGate(theta).to_matrix()
        assert np.allclose(sf_rz, qk_rz)
        
        # CX (CNOT) - note Qiskit's to_matrix uses little-endian ordering for qubits
        sf_cx = _get_gate_matrix(GateRecord("CNOT", [0, 1]))
        qk_cx = CXGate().to_matrix()
        swap_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        qk_cx_big_endian = swap_matrix @ qk_cx @ swap_matrix
        assert np.allclose(sf_cx, qk_cx_big_endian)

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    def test_expectation_value_accuracy(self, sf_backend):
        """Verify that expectation values compute to the same physical result."""
        n = 4
        circ = build_qft_sf(n)
        sv_sf = sf.run(circ, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()
        
        obs_matrix = np.eye(2**n, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        for _ in range(n):
            obs_matrix = np.kron(obs_matrix if _ > 0 else np.eye(1), Z) if _ == 0 else np.kron(obs_matrix, Z)
            
        from superfermion.simulator import expectation_value
        exp_sf = expectation_value(sv_sf, obs_matrix)
        
        from qiskit.quantum_info import Statevector, SparsePauliOp
        qc = build_qft_qiskit(n)
        sv_qk = Statevector(qc)
        obs_qk = SparsePauliOp(["Z"*n])
        exp_qk = sv_qk.expectation_value(obs_qk).real
        
        assert np.isclose(exp_sf, exp_qk, atol=1e-5), f"Expectation {sf_backend}: {exp_sf} vs {exp_qk}"

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [4, 8])
    def test_clifford_fidelity(self, sf_backend, n):
        """Statevector fidelity for random Clifford circuits should be 1.0.

        SF and Qiskit use independent RNGs even with identical seeds,
        so we build Qiskit's circuit from SF's gate instructions directly
        to guarantee identical circuits.
        """
        circ_sf = build_clifford_circuit_sf(n, seed=SEED)
        sv_sf = sf.run(circ_sf, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()

        # Build Qiskit circuit from the SAME gate instructions (SF and Qiskit
        # use independent RNGs - same seed produces different random circuits)
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(n)
        for g in circ_sf._gates:
            name = g.name.upper()
            q = g.qubits
            if name == "CX":
                qc.cx(q[0], q[1])
            elif name == "CZ":
                qc.cz(q[0], q[1])
            elif name == "CY":
                qc.cy(q[0], q[1])
            elif name == "SWAP":
                qc.swap(q[0], q[1])
            elif name == "X":
                qc.x(q[0])
            elif name == "Y":
                qc.y(q[0])
            elif name == "Z":
                qc.z(q[0])
            elif name == "S":
                qc.s(q[0])
            elif name == "SDG":
                qc.sdg(q[0])
            elif name == "H":
                qc.h(q[0])
        # SF is big-endian; Qiskit Statevector is little-endian.
        # reverse_bits() swaps the qubit ordering to match SF.
        sv_qiskit = np.array(Statevector(qc.reverse_bits()))

        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"Clifford({n}) {sf_backend} fidelity: {fidelity}"

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    def test_multi_control_fidelity(self, sf_backend):
        """Verify SF's cascading MCX produces the correct statevector.

        SF uses ancilla qubits for MCX decomposition, which are always |0>.
        We extract the data-qubit subspace by subsampling.
        We use Statevector directly to avoid transpile decomposition effects.
        """
        n = 4
        circ_sf = build_multi_control_circuit_sf(n)
        sv_sf_raw = np.asarray(sf.run(circ_sf, backend=sf_backend).statevector, dtype=np.complex128).ravel()
        # SF circuit has extra ancilla qubits beyond n. Ancillas are always |0>,
        # so the data statevector = sv[::2] (ancilla is the LSB qubit).
        n_data = 2 ** n
        sv_sf = sv_sf_raw[::2 ** (circ_sf.n_qubits - n)].copy()
        if len(sv_sf) > n_data:
            sv_sf = sv_sf[:n_data]

        from qiskit.quantum_info import Statevector
        qc = build_multi_control_circuit_qiskit(n)
        sv_qiskit = np.array(Statevector(qc))

        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"MCX({n}) {sf_backend} fidelity: {fidelity}"

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    def test_qasm2_import_fidelity(self, sf_backend):
        """Verify QASM2 import in SF produces the same statevector as Qiskit."""
        from qiskit import QuantumCircuit, qasm2 as qk_qasm2, transpile
        from qiskit_aer import AerSimulator
        qc_test = QuantumCircuit(8)
        qc_test.h(0)
        for i in range(7):
            qc_test.cx(i, i + 1)
        qasm_test = qk_qasm2.dumps(qc_test)

        circ_sf = qasm2_to_sf(qasm_test)
        sv_sf = sf.run(circ_sf, backend=sf_backend).statevector
        sv_sf = np.asarray(sv_sf, dtype=np.complex128).ravel()

        qc_test_rev = qc_test.reverse_bits()
        qc_test_rev.save_statevector()
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc_test_rev, sim)
        sv_qiskit = np.array(sim.run(tqc).result().get_statevector())

        fidelity = statevector_fidelity(sv_sf, sv_qiskit)
        assert np.isclose(fidelity, 1.0, atol=1e-5), f"QASM2 import {sf_backend} fidelity: {fidelity}"


# ═══════════════════════════════════════════════════════════════════════
# BigInt QASM2 Parsing
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_QISKIT, reason="Requires Qiskit for ground truth")
class TestBigIntQASM2:
    """Verify that SF's QASM2 parser correctly handles big integer parameters."""

    def test_bigint_qasm2_parsing(self):
        """Parse a QASM2 file with a big integer parameter > 2^53."""
        from superfermion.bridge import from_qasm

        # QASM2 with a big integer in the RZ parameter
        big_val = 2**53 + 1  # 9007199254740993 — too large for exact float
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx({big_val}) q[0];
cx q[0], q[1];
"""

        circ = from_qasm(qasm)
        assert circ.n_qubits == 2
        assert circ.gate_count == 2

        # Verify the big integer was preserved, not truncated
        rx_gate = circ._gates[0]
        assert rx_gate.name.upper() == "RX"
        assert rx_gate.params[0] == big_val
        assert isinstance(rx_gate.params[0], int)

    def test_bigint_qasm2_equivalence(self):
        """Verify that bigint QASM2 circuits produce the same statevector as Qiskit."""
        from superfermion.bridge import from_qasm
        from qiskit import QuantumCircuit
        from qiskit.qasm2 import dumps as qk_dumps

        # Create a circuit with big integer params in Qiskit
        big_val = 2**53 + 1
        qc = QuantumCircuit(2)
        qc.rx(big_val, 0)
        qc.cx(0, 1)
        qasm = qk_dumps(qc)

        # Parse with SF
        circ_sf = from_qasm(qasm)

        # Verify gate count matches
        assert circ_sf.gate_count >= 2

        # Verify statevector equivalence
        sv_sf = sf.run(circ_sf, backend="simulator").statevector

        from qiskit_aer import AerSimulator
        from qiskit import transpile
        qc_rev = qc.reverse_bits()
        qc_rev.save_statevector()
        sim = AerSimulator(method="statevector")
        tqc = transpile(qc_rev, sim)
        sv_qiskit = np.array(sim.run(tqc).result().get_statevector())

        fid = statevector_fidelity(np.asarray(sv_sf), sv_qiskit)
        assert np.isclose(fid, 1.0, atol=1e-5), f"BigInt QASM2 fidelity: {fid}"
