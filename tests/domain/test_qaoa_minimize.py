"""QAOA domain tests — MaxCut on a 4-node ring graph."""

import pytest

from superfermion.algorithms.variational import QAOA


pytestmark = [pytest.mark.domain, pytest.mark.timeout(30)]


class TestQAOAMaxCut:
    """QAOA on a simple 4-node cycle graph."""

    @pytest.fixture
    def ring_qaoa(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return QAOA(4, edges, p_layers=2, device="cpu")

    def test_qaoa_finds_reasonable_maxcut(self, ring_qaoa):
        result = ring_qaoa.minimize(iterations=100, seed=42)
        max_cut = result.metadata["max_cut_value"]
        # Optimal cut on a 4-cycle is 4; allow generous tolerance for noisy optimization.
        assert max_cut >= 3.0
        assert result.optimal_value > 0.0

    def test_qaoa_best_bitstring_is_valid(self, ring_qaoa):
        result = ring_qaoa.minimize(iterations=100, seed=42)
        bitstring = result.metadata["best_bitstring"]
        assert len(bitstring) == 4
        assert all(b in "01" for b in bitstring)
        assert result.metadata["max_cut_value"] == ring_qaoa._cut_value(bitstring)
