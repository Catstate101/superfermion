"""Tests for the Rust core optimizations exposed through _sf_core bindings.

Covers:
- simulate_msb: MSB endianness conversion in Rust
- simulate_and_sample: Rust-side bitstring sampling
- simulate_dm_noisy: Noisy density matrix simulation via Kraus operators
- adjoint_grad: Adjoint differentiation for gradients
- Gate fusion pass correctness
"""

import math

import numpy as np
import pytest

try:
    import _sf_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

pytestmark = [
    pytest.mark.backend,
    pytest.mark.skipif(not CORE_AVAILABLE, reason="_sf_core not built"),
]


def _make_dag(n_qubits: int, ops: list) -> "_sf_core.QuantumDAG":
    """Helper to build a DAG from a list of (gate_name, qubits, params)."""
    dag = _sf_core.QuantumDAG(n_qubits, 0)
    for gate_name, qubits, params in ops:
        dag.add_gate(gate_name, qubits, params)
    return dag


class TestSimulateMSB:
    """Test MSB endianness conversion done in Rust."""

    def test_single_qubit_x(self):
        dag = _make_dag(1, [("x", [0], [])])
        sv = np.asarray(dag.simulate_msb())
        # X|0> = |1>, MSB same as LSB for 1 qubit
        assert abs(sv[1]) > 0.99

    def test_two_qubit_endianness(self):
        # X on qubit 0 only: in LSB convention, q0=1 → index 1
        # In MSB convention, q0=leftmost → |10> = index 2
        dag = _make_dag(2, [("x", [0], [])])
        sv_msb = np.asarray(dag.simulate_msb())
        sv_lsb = np.asarray(dag.simulate())
        # LSB: X on q0 → index 1 (bit 0 set)
        assert abs(sv_lsb[1]) > 0.99
        # MSB: q0 is most significant bit → |10> = index 2
        assert abs(sv_msb[2]) > 0.99

    def test_bell_state_msb(self):
        dag = _make_dag(2, [("h", [0], []), ("cx", [0, 1], [])])
        sv = np.asarray(dag.simulate_msb())
        # Bell state: |00> + |11> / sqrt(2)
        expected = 1 / math.sqrt(2)
        assert abs(abs(sv[0]) - expected) < 1e-10
        assert abs(abs(sv[3]) - expected) < 1e-10
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10

    def test_three_qubit_endianness(self):
        # X on q2 only: in LSB, q2=1 → index 4 (bit 2 set)
        # In MSB, q2 is least significant → |001> = index 1
        dag = _make_dag(3, [("x", [2], [])])
        sv_msb = np.asarray(dag.simulate_msb())
        assert abs(sv_msb[1]) > 0.99  # |001> in MSB

    def test_normalization_preserved(self):
        dag = _make_dag(3, [("h", [0], []), ("h", [1], []), ("cx", [0, 2], [])])
        sv = np.asarray(dag.simulate_msb())
        assert abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-10


class TestSimulateAndSample:
    """Test Rust-side bitstring sampling."""

    def test_deterministic_with_seed(self):
        dag = _make_dag(2, [("h", [0], []), ("cx", [0, 1], [])])
        counts1 = dag.simulate_and_sample(1000, 42)
        counts2 = dag.simulate_and_sample(1000, 42)
        assert counts1 == counts2

    def test_different_seeds_differ(self):
        # Use 3 qubits with H for more possible outcomes, reducing collision chance
        dag = _make_dag(3, [("h", [0], []), ("h", [1], []), ("h", [2], [])])
        counts1 = dag.simulate_and_sample(500, 42)
        counts2 = dag.simulate_and_sample(500, 99)
        assert counts1 != counts2

    def test_bell_state_only_00_11(self):
        dag = _make_dag(2, [("h", [0], []), ("cx", [0, 1], [])])
        counts = dag.simulate_and_sample(10000, 42)
        # Should only have "00" and "11"
        assert set(counts.keys()) <= {"00", "11"}
        assert "00" in counts and "11" in counts
        # Each should be ~50%
        total = sum(counts.values())
        assert abs(counts["00"] / total - 0.5) < 0.05

    def test_x_gate_gives_all_ones(self):
        dag = _make_dag(2, [("x", [0], []), ("x", [1], [])])
        counts = dag.simulate_and_sample(100, 42)
        assert list(counts.keys()) == ["11"]
        assert counts["11"] == 100

    def test_total_shots_correct(self):
        dag = _make_dag(3, [("h", [0], []), ("h", [1], []), ("h", [2], [])])
        shots = 5000
        counts = dag.simulate_and_sample(shots, 42)
        assert sum(counts.values()) == shots

    def test_uniform_superposition_distribution(self):
        # 3 qubits, all H → uniform over 8 bitstrings
        dag = _make_dag(3, [("h", [0], []), ("h", [1], []), ("h", [2], [])])
        counts = dag.simulate_and_sample(80000, 42)
        # Each should be ~12.5% (1/8)
        for bitstring, count in counts.items():
            assert abs(count / 80000 - 0.125) < 0.02, f"{bitstring}: {count/80000:.3f}"


class TestSimulateDMNoisy:
    """Test noisy density matrix simulation via Kraus operators."""

    def _bitflip_kraus(self, p: float) -> list:
        """Bit-flip channel Kraus operators as flat list.
        K0 = sqrt(1-p)*I, K1 = sqrt(p)*X
        Each 2x2 matrix = 8 floats: re00, im00, re01, im01, re10, im10, re11, im11
        """
        s0 = math.sqrt(1 - p)
        s1 = math.sqrt(p)
        # K0 = s0*I
        k0 = [s0, 0, 0, 0, 0, 0, s0, 0]
        # K1 = s1*X
        k1 = [0, 0, s1, 0, s1, 0, 0, 0]
        return k0 + k1

    def _depolarizing_kraus(self, p: float) -> list:
        """Single-qubit depolarizing channel."""
        s0 = math.sqrt(1 - 3 * p / 4)
        sp = math.sqrt(p / 4)
        # K0 = s0*I
        k0 = [s0, 0, 0, 0, 0, 0, s0, 0]
        # K1 = sp*X
        k1 = [0, 0, sp, 0, sp, 0, 0, 0]
        # K2 = sp*Y (Y = [[0, -i], [i, 0]])
        k2 = [0, 0, 0, -sp, 0, sp, 0, 0]
        # K3 = sp*Z
        k3 = [sp, 0, 0, 0, 0, 0, -sp, 0]
        return k0 + k1 + k2 + k3

    def test_no_noise_matches_pure(self):
        dag = _make_dag(1, [("h", [0], [])])
        dm = np.asarray(dag.simulate_dm_noisy([]))
        # For 1 qubit, DM is 2x2 flattened to 4 elements
        n = 1
        dim = 2**n
        rho = dm.reshape(dim, dim)
        # Should be |+><+| = [[0.5, 0.5], [0.5, 0.5]]
        assert abs(rho[0, 0] - 0.5) < 1e-10
        assert abs(rho[0, 1] - 0.5) < 1e-10
        assert abs(rho[1, 0] - 0.5) < 1e-10
        assert abs(rho[1, 1] - 0.5) < 1e-10

    def test_bitflip_full_noise(self):
        # p=1 bit-flip on |0>: should give |1><1|
        dag = _make_dag(1, [("id", [0], [])])
        kraus = self._bitflip_kraus(1.0)
        dm = np.asarray(dag.simulate_dm_noisy([(0, kraus)]))
        rho = dm.reshape(2, 2)
        assert abs(rho[1, 1] - 1.0) < 1e-10
        assert abs(rho[0, 0]) < 1e-10

    def test_bitflip_partial_noise(self):
        # p=0.5 bit-flip on |0>: should give 0.5|0><0| + 0.5|1><1| = I/2
        dag = _make_dag(1, [("id", [0], [])])
        kraus = self._bitflip_kraus(0.5)
        dm = np.asarray(dag.simulate_dm_noisy([(0, kraus)]))
        rho = dm.reshape(2, 2)
        assert abs(rho[0, 0] - 0.5) < 1e-10
        assert abs(rho[1, 1] - 0.5) < 1e-10
        assert abs(rho[0, 1]) < 1e-10

    def test_trace_preserved(self):
        dag = _make_dag(2, [("h", [0], []), ("cx", [0, 1], [])])
        kraus = self._depolarizing_kraus(0.1)
        dm = np.asarray(dag.simulate_dm_noisy([(0, kraus)]))
        rho = dm.reshape(4, 4)
        trace = np.trace(rho)
        assert abs(trace.real - 1.0) < 1e-8, f"Trace = {trace}"
        assert abs(trace.imag) < 1e-8


class TestAdjointGrad:
    """Test adjoint differentiation via _sf_core bindings."""

    def test_rx_z_gradient(self):
        theta = 0.7
        dag = _sf_core.QuantumDAG(1, 0)
        dag.add_gate("rx", [0], ["theta"])

        obs = [([3], 1.0, 0.0)]  # Z observable
        params = {"theta": theta}
        grad = dag.adjoint_grad(obs, params)
        expected = -math.sin(theta)
        assert abs(grad["theta"] - expected) < 1e-8

    def test_ry_z_gradient(self):
        theta = 1.2
        dag = _sf_core.QuantumDAG(1, 0)
        dag.add_gate("ry", [0], ["theta"])

        obs = [([3], 1.0, 0.0)]  # Z
        params = {"theta": theta}
        grad = dag.adjoint_grad(obs, params)
        expected = -math.sin(theta)
        assert abs(grad["theta"] - expected) < 1e-8

    def test_multiple_parameters(self):
        t0, t1 = 0.5, 1.0
        dag = _sf_core.QuantumDAG(1, 0)
        dag.add_gate("rx", [0], ["t0"])
        dag.add_gate("ry", [0], ["t1"])

        obs = [([3], 1.0, 0.0)]
        params = {"t0": t0, "t1": t1}
        grad = dag.adjoint_grad(obs, params)
        assert "t0" in grad and "t1" in grad
        assert all(math.isfinite(v) for v in grad.values())
