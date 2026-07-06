"""Backend scaling tests — verify larger circuits complete with valid outputs."""

import pytest

from superfermion.backends.factory import get_backend
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


class TestSingularityScaling:
    def test_10_qubit_ghz_completes(self):
        n = 10
        backend = get_backend("singularity")
        circuit = _ghz_circuit(n)
        result = backend.run(circuit, shots=500, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
        assert set(result.counts.keys()).issubset(
            {format(0, f"0{n}b"), format((1 << n) - 1, f"0{n}b")}
        )


class TestStatevectorScaling:
    def test_15_qubit_circuit_completes(self):
        n = 15
        backend = get_backend("statevector")
        circuit = _ghz_circuit(n)
        result = backend.run(circuit, shots=200, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
        assert result.statevector is not None


@pytest.mark.slow
class TestLargeCircuitScaling:
    @pytest.mark.parametrize("backend_name", ["singularity", "statevector"])
    def test_20_qubit_ghz(self, backend_name):
        n = 20
        backend = get_backend(backend_name)
        circuit = _ghz_circuit(n)
        result = backend.run(circuit, shots=100, seed=SEED)
        _assert_bitstring_lengths(result.counts, n)
