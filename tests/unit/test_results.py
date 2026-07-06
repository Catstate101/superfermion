"""Unit tests for RunResult serialization and construction."""

import numpy as np
import pytest

from superfermion.circuit import Circuit
from superfermion.results import RunResult


pytestmark = pytest.mark.unit


class TestRunResultConstruction:
    def test_basic_fields(self, bell_circuit):
        counts = {"00": 500, "11": 500}
        probs = {"00": 0.5, "11": 0.5}
        sv = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        meta = {"backend": "statevector"}

        result = RunResult(
            counts=counts,
            probabilities=probs,
            statevector=sv,
            shots=1000,
            circuit=bell_circuit,
            metadata=meta,
        )

        assert result.counts == counts
        assert result.probabilities == probs
        assert result.shots == 1000
        assert result.circuit is bell_circuit
        assert result.metadata == meta
        np.testing.assert_allclose(result.statevector, sv)

    def test_defaults(self):
        result = RunResult()
        assert result.counts == {}
        assert result.probabilities == {}
        assert result.statevector is None
        assert result.shots == 0
        assert result.metadata == {}


class TestRunResultSerialization:
    def test_to_dict_from_dict_roundtrip(self, bell_circuit):
        sv = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        original = RunResult(
            counts={"00": 512, "11": 488},
            probabilities={"00": 0.512, "11": 0.488},
            statevector=sv,
            shots=1000,
            circuit=bell_circuit,
            metadata={"device": "cpu"},
        )

        d = original.to_dict()
        restored = RunResult.from_dict(d)

        assert restored.counts == original.counts
        assert restored.probabilities == original.probabilities
        assert restored.shots == original.shots
        assert restored.metadata == original.metadata
        np.testing.assert_allclose(restored.statevector, sv)

    def test_from_dict_without_statevector(self):
        d = {
            "counts": {"0": 100},
            "probabilities": {"0": 1.0},
            "shots": 100,
            "metadata": {"note": "no sv"},
        }
        result = RunResult.from_dict(d)
        assert result.statevector is None
        assert result.counts == {"0": 100}

    def test_to_dict_omits_statevector_when_none(self):
        d = RunResult(shots=10).to_dict()
        assert "statevector_real" not in d
        assert "statevector_imag" not in d


class TestRunResultEdgeCases:
    def test_empty_counts(self):
        result = RunResult(counts={}, shots=0)
        assert result.counts == {}
        assert len(result.counts) == 0

    def test_single_outcome(self):
        result = RunResult(counts={"0": 1000}, probabilities={"0": 1.0}, shots=1000)
        assert sum(result.counts.values()) == 1000

    def test_large_circuit_statevector(self):
        n = 4
        dim = 2 ** n
        sv = np.zeros(dim, dtype=np.complex128)
        sv[0] = 1.0
        circuit = Circuit(n)
        result = RunResult(
            counts={"0000": 1000},
            statevector=sv,
            shots=1000,
            circuit=circuit,
        )
        d = result.to_dict()
        restored = RunResult.from_dict(d)
        assert restored.statevector.shape == (dim,)
