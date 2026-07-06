"""Backend correctness tests for core quantum operations."""

import math

import pytest

import superfermion as sf
from superfermion.backends.factory import get_backend
from superfermion.circuit import Circuit


pytestmark = pytest.mark.backend

SEED = 42
SHOTS = 1000
CORE_BACKENDS = ["statevector", "singularity"]
BALANCE_MARGIN = 0.10  # each outcome in 40–60% range for 50/50 distributions


def _run(backend_name: str, circuit: Circuit, shots: int = SHOTS):
    backend = get_backend(backend_name)
    return backend.run(circuit, shots=shots, seed=SEED)


def _assert_balanced_two_outcome(
    counts: dict,
    key_a: str,
    key_b: str,
    shots: int,
    margin: float = BALANCE_MARGIN,
) -> None:
    total = sum(counts.values())
    assert total == shots, f"expected {shots} shots, got {total}"
    frac_a = counts.get(key_a, 0) / shots
    frac_b = counts.get(key_b, 0) / shots
    assert 0.5 - margin <= frac_a <= 0.5 + margin, f"{key_a}: {frac_a:.3f}"
    assert 0.5 - margin <= frac_b <= 0.5 + margin, f"{key_b}: {frac_b:.3f}"


def _assert_two_outcome_probs(
    probs: dict,
    key_a: str,
    key_b: str,
    tol: float = 0.01,
) -> None:
    assert abs(probs.get(key_a, 0.0) - 0.5) < tol, probs
    assert abs(probs.get(key_b, 0.0) - 0.5) < tol, probs


@pytest.mark.parametrize("backend_name", CORE_BACKENDS)
class TestBellState:
    def test_bell_distribution(self, backend_name, bell_circuit):
        result = _run(backend_name, bell_circuit, shots=SHOTS)
        _assert_balanced_two_outcome(result.counts, "00", "11", SHOTS)

    def test_bell_exact_probabilities(self, backend_name, bell_circuit):
        result = _run(backend_name, bell_circuit, shots=0)
        if result.probabilities:
            _assert_two_outcome_probs(result.probabilities, "00", "11")


@pytest.mark.parametrize("backend_name", CORE_BACKENDS)
class TestGHZState:
    def test_ghz_three_qubit_distribution(self, backend_name, ghz_circuit):
        circuit = ghz_circuit(3)
        result = _run(backend_name, circuit, shots=SHOTS)
        _assert_balanced_two_outcome(result.counts, "000", "111", SHOTS)

    def test_ghz_exact_probabilities(self, backend_name, ghz_circuit):
        circuit = ghz_circuit(3)
        result = _run(backend_name, circuit, shots=0)
        if result.probabilities:
            _assert_two_outcome_probs(result.probabilities, "000", "111")


@pytest.mark.parametrize("backend_name", CORE_BACKENDS)
class TestParameterizedRX:
    def test_rx_pi_flips_to_one(self, backend_name):
        theta = sf.param("theta")
        circuit = Circuit(1).rx(theta, 0).bind({"theta": math.pi})
        result = _run(backend_name, circuit, shots=SHOTS)
        assert result.counts.get("1", 0) == SHOTS
        if result.probabilities:
            assert result.probabilities.get("1", 0.0) > 0.99


@pytest.mark.backend
class TestStabilizerClifford:
    """Stabilizer backend supports Clifford gates only (H, CNOT, S, CZ)."""

    @pytest.fixture(autouse=True)
    def _require_stabilizer(self):
        try:
            get_backend("stabilizer")
        except Exception as exc:
            pytest.skip(f"stabilizer backend unavailable: {exc}")

    def test_bell_clifford(self, bell_circuit):
        result = _run("stabilizer", bell_circuit, shots=SHOTS)
        _assert_balanced_two_outcome(result.counts, "00", "11", SHOTS)

    def test_ghz_clifford(self, ghz_circuit):
        circuit = ghz_circuit(3)
        result = _run("stabilizer", circuit, shots=SHOTS)
        _assert_balanced_two_outcome(result.counts, "000", "111", SHOTS)

    def test_clifford_single_qubit_gates(self):
        circuit = Circuit(2).h(0).s(0).cz(0, 1)
        result = _run("stabilizer", circuit, shots=100)
        assert sum(result.counts.values()) == 100
