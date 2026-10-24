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


class TestQAOACountsKeyConvention:
    """Regression tests for D12 (QAOA metadata on a non-symmetric graph).

    The 4-cycle and the docstring triangle are invariant under index reversal,
    so they cannot expose the convention bug; a star graph can.
    """

    STAR = [(0, 1), (0, 2), (0, 3)]

    def test_star_cut_value_uses_counts_key(self):
        qaoa = QAOA(4, self.STAR, p_layers=1, device="cpu")
        # Counts key "1110" means q0=q1=q2=1, q3=0: all three edges are cut.
        assert qaoa._cut_value("1110") == 3.0

    def test_star_metadata_matches_true_cut(self):
        qaoa = QAOA(4, self.STAR, p_layers=2, device="cpu")
        result = qaoa.minimize(iterations=200, seed=42)
        best = result.metadata["best_bitstring"]
        n = 4
        true_cut = sum(1.0 for i, j in self.STAR if int(best[n - 1 - i]) != int(best[n - 1 - j]))
        assert result.metadata["max_cut_value"] == true_cut

    def test_cycle_control_is_reversal_invariant(self):
        edges = [(i, (i + 1) % 8) for i in range(8)]
        qaoa = QAOA(8, edges, p_layers=1, device="cpu")
        assert qaoa._cut_value("10101010") == 8.0
