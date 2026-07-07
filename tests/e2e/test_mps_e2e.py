"""End-to-end tests for MPS simulation through sf.run()."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.e2e


class TestMPSStatevector:
    def test_bell_state_mps(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="mps", shots=0)
        sv = result.statevector
        assert sv is not None
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        fidelity = abs(np.dot(np.conj(expected), sv)) ** 2
        assert fidelity > 0.99

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_ghz_mps(self, ghz_circuit, n):
        circuit = ghz_circuit(n)
        result = sf.run(circuit, device="cpu", method="mps", shots=0)
        sv = result.statevector
        probs = np.abs(sv) ** 2
        assert probs[0] > 0.49
        assert probs[-1] > 0.49
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)


class TestMPSSampling:
    def test_ghz_sampling(self, ghz_circuit):
        circuit = ghz_circuit(4)
        result = sf.run(circuit, device="cpu", method="mps", shots=10000)
        assert set(result.counts.keys()).issubset({"0000", "1111"})
        total = sum(result.counts.values())
        assert total == 10000

    def test_bell_sampling_distribution(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="mps", shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        ratio = result.counts.get("00", 0) / 10000
        assert 0.4 < ratio < 0.6


class TestMPSBondDim:
    def test_custom_bond_dim(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="mps", shots=0, bond_dim=128)
        assert result.statevector is not None
        assert result.metadata.get("bond_dim") == 128

    def test_default_bond_dim_in_metadata(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="mps", shots=0)
        assert result.metadata.get("bond_dim") == 64


class TestMPSLargeCircuit:
    def test_product_state_20q(self):
        n = 20
        qc = Circuit(n)
        for q in range(n):
            qc.h(q)
        result = sf.run(qc, device="cpu", method="mps", shots=1000)
        assert sum(result.counts.values()) == 1000


class TestMPSMatchesStatevector:
    def test_bell_fidelity(self, bell_circuit):
        sv_result = sf.run(bell_circuit, device="cpu", method="statevector", shots=0)
        mps_result = sf.run(bell_circuit, device="cpu", method="mps", shots=0)
        sv = sv_result.statevector
        mps_sv = mps_result.statevector
        fidelity = abs(np.dot(np.conj(sv), mps_sv)) ** 2
        assert fidelity > 0.999
