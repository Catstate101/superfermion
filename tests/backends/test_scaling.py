"""Scaling tests — verify larger circuits complete with valid outputs via sf.run()."""

import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.backend

SEED = 42


def _ghz_circuit(n: int) -> Circuit:
    circuit = Circuit(n).h(0)
    for i in range(n - 1):
        circuit.cnot(i, i + 1)
    return circuit


def _assert_bitstring_lengths(counts: dict, n_qubits: int) -> None:
    assert counts, "expected non-empty measurement counts"
    for bitstring in counts:
        assert len(bitstring) == n_qubits, (
            f"bitstring {bitstring!r} has length {len(bitstring)}, expected {n_qubits}"
        )


class TestStatevectorScaling:
    def test_15_qubit_circuit_completes(self):
        n = 15
        circuit = _ghz_circuit(n)
        result = sf.run(circuit, device="cpu", shots=200, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
        assert result.state is not None

    def test_10_qubit_ghz_completes(self):
        n = 10
        circuit = _ghz_circuit(n)
        result = sf.run(circuit, device="cpu", shots=500, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
        all_zero = format(0, f"0{n}b")
        all_one = format((1 << n) - 1, f"0{n}b")
        assert set(result.counts.keys()).issubset({all_zero, all_one})


class TestMPSScaling:
    def test_20_qubit_mps_completes(self):
        n = 20
        circuit = _ghz_circuit(n)
        result = sf.run(circuit, device="cpu", method="mps", shots=100, seed=SEED, bond_dim=32)
        _assert_bitstring_lengths(result.counts, n)
        assert result.state is not None


class TestStabilizerScaling:
    def test_50_qubit_clifford(self):
        n = 50
        circuit = _ghz_circuit(n)
        result = sf.run(circuit, device="cpu", method="stabilizer", shots=100, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
        assert result.state is not None


@pytest.mark.slow
class TestLargeCircuitScaling:
    def test_20_qubit_statevector(self):
        n = 20
        circuit = _ghz_circuit(n)
        result = sf.run(circuit, device="cpu", shots=100, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
