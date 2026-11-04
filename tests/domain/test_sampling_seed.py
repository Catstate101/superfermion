"""Sampler seeding (audit STAB-1/2/3).

``sf.run`` used to seed every method with the literal 42, so repeated
unseeded runs returned bit-identical counts and an error bar built from
them was meaningless.  Omitting ``seed=`` still does that -- it is pinned
here so the change to a default cannot happen silently -- while
``seed=None`` now draws fresh entropy and reports it back in
``result.metadata["seed"]`` so the draw stays replayable.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion.devices.rust_device import _DEFAULT_SEED, _resolve_seed


pytestmark = [pytest.mark.domain, pytest.mark.timeout(120)]

METHODS = ["statevector", "mps", "stabilizer"]


def _ghz(n: int = 6) -> sf.Circuit:
    """GHZ-n with a terminal measurement: P(000000) = P(111111) = 1/2."""
    circuit = sf.Circuit(n)
    circuit.h(0)
    for qubit in range(1, n):
        circuit.cx(0, qubit)
    for qubit in range(n):
        circuit.measure(qubit)
    return circuit


class TestSeedResolution:
    """``_resolve_seed`` is the single place the seed is decided."""

    def test_an_absent_seed_keeps_the_historical_default(self):
        assert _DEFAULT_SEED == 42
        assert _resolve_seed({}) == (_DEFAULT_SEED, False)

    def test_seed_none_draws_a_32_bit_seed(self):
        drawn = [_resolve_seed({"seed": None}) for _ in range(32)]
        assert all(drawn_flag is True for _, drawn_flag in drawn)
        assert all(isinstance(seed, int) and 0 <= seed < 2**32 for seed, _ in drawn)
        assert len({seed for seed, _ in drawn}) > 1

    def test_an_explicit_seed_is_not_marked_as_drawn(self):
        assert _resolve_seed({"seed": 7}) == (7, False)
        assert _resolve_seed({"seed": "9"}) == (9, False)

    def test_a_non_numeric_seed_falls_back_to_the_default(self):
        assert _resolve_seed({"seed": object()}) == (_DEFAULT_SEED, False)


class TestDefaultSeedIsDeterministic:
    """STAB-1, pinned: omitting ``seed=`` stays reproducible."""

    @pytest.mark.parametrize("method", METHODS)
    def test_repeated_unseeded_runs_are_identical(self, method):
        circuit = _ghz()
        counts = [sf.run(circuit, method=method, shots=500).counts for _ in range(3)]
        assert counts[0] == counts[1] == counts[2]

    def test_an_absent_seed_leaves_no_metadata_stamp(self):
        assert "seed" not in sf.run(_ghz(), shots=100).metadata

    def test_an_explicit_seed_is_reproducible_and_unstamped(self):
        circuit = _ghz()
        first = sf.run(circuit, method="stabilizer", shots=300, seed=11)
        second = sf.run(circuit, method="stabilizer", shots=300, seed=11)
        assert first.counts == second.counts
        assert "seed" not in first.metadata


class TestSeedNoneDrawsFreshSamples:
    """``seed=None`` is the documented way to get independent repeats."""

    @pytest.mark.parametrize("method", METHODS)
    def test_seed_none_varies_and_records_the_seed(self, method):
        circuit = _ghz()
        results = [
            sf.run(circuit, method=method, shots=400, seed=None) for _ in range(6)
        ]
        seeds = [result.metadata.get("seed") for result in results]

        assert all(isinstance(seed, int) for seed in seeds)
        assert len(set(seeds)) == len(seeds)

        # GHZ-6 is a fair coin, so six draws collide with probability ~1e-10.
        zero_counts = [result.counts.get("000000", 0) for result in results]
        assert len(set(zero_counts)) > 1

    def test_the_recorded_seed_replays_the_run(self):
        circuit = _ghz()
        drawn = sf.run(circuit, shots=1000, seed=None)
        replay = sf.run(circuit, shots=1000, seed=drawn.metadata["seed"])
        assert replay.counts == drawn.counts

    def test_seed_none_no_longer_raises(self):
        # It used to raise TypeError: 'NoneType' object cannot be interpreted
        # as an integer.
        assert sf.run(_ghz(), shots=50, seed=None).counts

    def test_seed_none_works_on_a_dynamic_circuit(self):
        circuit = sf.Circuit(2)
        circuit.h(0)
        circuit.measure(0)
        circuit.c_if(0, 1)
        circuit.x(1)
        result = sf.run(circuit, shots=200, seed=None)
        assert isinstance(result.metadata.get("seed"), int)


class TestVarianceRecoversWithVaryingSeeds:
    """STAB-2/3: the spread across seeds, measured rather than assumed."""

    def test_statevector_spread_matches_the_binomial_prediction(self):
        circuit = _ghz()
        samples = np.array(
            [
                sf.run(circuit, method="statevector", shots=200, seed=seed).counts.get(
                    "000000", 0
                )
                for seed in range(200)
            ]
        )
        # Binomial(200, 1/2): mean 100, sd 7.071.
        assert abs(samples.mean() - 100.0) < 3.0
        assert 0.75 < samples.std() / 7.071 < 1.25

    def test_stabilizer_spread_is_pinned_below_binomial(self):
        """Known, measured, and NOT fixed here.

        The Rust tableau sampler has a variance deficit (~0.71x binomial at
        200 seeds, chi2 ~142 on 199 df).  A Rust-side fix would move this
        ratio toward 1.0, which the upper bound allows; a gross regression
        is what this pins.
        """
        circuit = _ghz()
        samples = np.array(
            [
                sf.run(circuit, method="stabilizer", shots=200, seed=seed).counts.get(
                    "000000", 0
                )
                for seed in range(200)
            ]
        )
        ratio = samples.std() / 7.071
        assert abs(samples.mean() - 100.0) < 4.0
        assert 0.55 < ratio < 1.05
