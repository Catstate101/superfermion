"""Grover search domain tests."""

import pytest

from superfermion.algorithms.grover import GroverOracle, grover_search


pytestmark = pytest.mark.domain


class TestGroverSearch:
    def test_grover_oracle_construction(self):
        oracle = GroverOracle.mark_state("101")
        assert oracle.marked == ["101"]
        assert oracle.n_qubits == 3

    def test_grover_finds_marked_state_3_qubits(self):
        marked = "101"
        oracle = GroverOracle.mark_state(marked)
        result = grover_search(oracle, n_qubits=3, device="cpu")

        assert result["top_bitstring"] == marked
        assert result["probability"] > 0.5

    def test_grover_search_runs_and_returns_result(self):
        oracle = GroverOracle.mark_state("101")
        result = grover_search(oracle, n_qubits=3, device="cpu")

        assert result["n_qubits"] == 3
        assert result["iterations"] >= 1
        assert "top_bitstring" in result
        assert "probability" in result
        assert len(result["top_bitstring"]) == 3


class TestGroverCountsKeyConvention:
    """Regression tests for D11a/D11b (counts-key convention).

    ``marked`` strings and ``top_bitstring`` use the same q0-last convention
    as ``Result.counts``, so a non-palindromic mark must round-trip and the
    sampled label must have exactly ``n_qubits`` characters (ancillas traced
    out).
    """

    def test_non_palindromic_mark_round_trips(self):
        marked = "0111"
        result = grover_search(GroverOracle.mark_state(marked), n_qubits=4, device="cpu")
        assert result["top_bitstring"] == marked
        assert result["probability"] > 0.5

    def test_sampled_label_has_n_data_bits(self):
        result = grover_search(
            GroverOracle.mark_state("0111"), n_qubits=4, device="cpu", shots=20000
        )
        assert len(result["top_bitstring"]) == 4

    def test_sampled_and_statevector_labels_agree(self):
        oracle = GroverOracle.mark_state("0111")
        sv = grover_search(oracle, n_qubits=4, device="cpu")
        sampled = grover_search(oracle, n_qubits=4, device="cpu", shots=20000)
        assert sampled["top_bitstring"] == sv["top_bitstring"]
