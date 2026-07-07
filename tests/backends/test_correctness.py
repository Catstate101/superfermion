"""Correctness tests for quantum simulation via sf.run() across all methods."""

import math

import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.backend

SEED = 42
SHOTS = 1000
METHODS = ["statevector", "density_matrix", "mps", "stabilizer"]
BALANCE_MARGIN = 0.10


def _run(method: str, circuit: Circuit, shots: int = SHOTS):
    return sf.run(circuit, device="cpu", method=method, shots=shots, seed=SEED)


def _assert_balanced_two_outcome(counts, key_a, key_b, shots, margin=BALANCE_MARGIN):
    total = sum(counts.values())
    assert total == shots, f"expected {shots} shots, got {total}"
    frac_a = counts.get(key_a, 0) / shots
    frac_b = counts.get(key_b, 0) / shots
    assert 0.5 - margin <= frac_a <= 0.5 + margin, f"{key_a}: {frac_a:.3f}"
    assert 0.5 - margin <= frac_b <= 0.5 + margin, f"{key_b}: {frac_b:.3f}"


@pytest.mark.parametrize("method", METHODS)
class TestBellState:
    def test_bell_distribution(self, method, bell_circuit):
        result = _run(method, bell_circuit)
        _assert_balanced_two_outcome(result.counts, "00", "11", SHOTS)

    def test_bell_has_state(self, method, bell_circuit):
        result = _run(method, bell_circuit, shots=0)
        assert result.state is not None


@pytest.mark.parametrize("method", METHODS)
class TestGHZState:
    def test_ghz_three_qubit_distribution(self, method, ghz_circuit):
        circuit = ghz_circuit(3)
        result = _run(method, circuit)
        _assert_balanced_two_outcome(result.counts, "000", "111", SHOTS)


@pytest.mark.parametrize("method", ["statevector", "density_matrix", "mps"])
class TestParameterizedRX:
    def test_rx_pi_flips_to_one(self, method):
        theta = sf.param("theta")
        circuit = Circuit(1).rx(theta, 0).bind({"theta": math.pi})
        result = _run(method, circuit)
        assert result.counts.get("1", 0) == SHOTS


class TestStabilizerClifford:
    def test_bell_clifford(self, bell_circuit):
        result = _run("stabilizer", bell_circuit)
        _assert_balanced_two_outcome(result.counts, "00", "11", SHOTS)

    def test_clifford_single_qubit_gates(self):
        circuit = Circuit(2).h(0).s(0).cz(0, 1)
        result = _run("stabilizer", circuit, shots=100)
        assert sum(result.counts.values()) == 100

    def test_non_clifford_raises(self):
        circuit = Circuit(1).rx(0.5, 0)
        with pytest.raises(RuntimeError, match="non-Clifford"):
            _run("stabilizer", circuit)
