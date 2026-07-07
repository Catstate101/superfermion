"""End-to-end tests for stabilizer simulation through sf.run()."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.e2e


class TestStabilizerBell:
    def test_bell_sampling(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="stabilizer", shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        total = sum(result.counts.values())
        assert total == 10000
        ratio = result.counts.get("00", 0) / total
        assert 0.4 < ratio < 0.6


class TestStabilizerGHZ:
    @pytest.mark.parametrize("n", [3, 5, 8, 10])
    def test_ghz_stabilizer(self, ghz_circuit, n):
        circuit = ghz_circuit(n)
        result = sf.run(circuit, device="cpu", method="stabilizer", shots=10000)
        all_zero = "0" * n
        all_one = "1" * n
        assert set(result.counts.keys()).issubset({all_zero, all_one})
        total = sum(result.counts.values())
        ratio = result.counts.get(all_zero, 0) / total
        assert 0.4 < ratio < 0.6


class TestStabilizerCliffordGates:
    def test_hadamard_only(self):
        qc = Circuit(1).h(0)
        result = sf.run(qc, device="cpu", method="stabilizer", shots=10000)
        assert set(result.counts.keys()).issubset({"0", "1"})
        ratio = result.counts.get("0", 0) / 10000
        assert 0.4 < ratio < 0.6

    def test_x_gate(self):
        qc = Circuit(1).x(0)
        result = sf.run(qc, device="cpu", method="stabilizer", shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_s_sdg_identity(self):
        qc = Circuit(1).x(0).s(0).sdg(0)
        result = sf.run(qc, device="cpu", method="stabilizer", shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_cz_gate(self):
        qc = Circuit(2).h(0).h(1).cz(0, 1).h(0).h(1)
        result = sf.run(qc, device="cpu", method="stabilizer", shots=10000)
        assert sum(result.counts.values()) == 10000

    def test_swap_gate(self):
        qc = Circuit(2).x(0).swap(0, 1)
        result = sf.run(qc, device="cpu", method="stabilizer", shots=1000)
        assert result.counts.get("01", 0) == 1000


class TestStabilizerNonCliffordRejects:
    def test_t_gate_raises(self):
        qc = Circuit(1).t(0)
        with pytest.raises(RuntimeError, match="non-Clifford"):
            sf.run(qc, device="cpu", method="stabilizer", shots=100)

    def test_rx_raises(self):
        qc = Circuit(1).rx(0.5, 0)
        with pytest.raises(RuntimeError, match="non-Clifford"):
            sf.run(qc, device="cpu", method="stabilizer", shots=100)


class TestStabilizerVsExact:
    def test_tvd_bell(self, bell_circuit):
        stab_result = sf.run(bell_circuit, device="cpu", method="stabilizer", shots=100000)
        sv_result = sf.run(bell_circuit, device="cpu", method="statevector", shots=0)
        exact_probs = sv_result.get_probabilities()
        stab_total = sum(stab_result.counts.values())
        stab_probs = {k: v / stab_total for k, v in stab_result.counts.items()}
        all_keys = set(exact_probs) | set(stab_probs)
        tvd = 0.5 * sum(abs(exact_probs.get(k, 0) - stab_probs.get(k, 0)) for k in all_keys)
        assert tvd < 0.02
