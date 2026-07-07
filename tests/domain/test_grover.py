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
