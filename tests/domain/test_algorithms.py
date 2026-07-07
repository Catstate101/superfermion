"""Variational algorithm domain tests."""

import numpy as np
import pytest

from superfermion.observables.core import SparsePauliOp


pytestmark = pytest.mark.domain

variational = pytest.importorskip(
    "superfermion.algorithms.variational",
    reason="variational algorithms module unavailable",
)
VQE = variational.VQE
QAOA = variational.QAOA

HardwareEfficientAnsatz = pytest.importorskip(
    "superfermion.qml.templates",
    reason="qml templates unavailable",
).HardwareEfficientAnsatz


class TestVQE:
    def test_vqe_initializes(self):
        hamiltonian = SparsePauliOp.from_dict({"ZZ": -1.0, "XX": -0.5})
        ansatz = HardwareEfficientAnsatz(2, n_layers=1)
        vqe = VQE(ansatz, hamiltonian, device="cpu")
        assert vqe.ansatz is ansatz
        assert vqe.hamiltonian is hamiltonian
        assert vqe._device == "cpu"
        assert vqe._method == "statevector"
        assert len(vqe._param_names) == ansatz.n_parameters


class TestQAOA:
    def test_qaoa_initializes(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        qaoa = QAOA(3, edges, p_layers=2, device="cpu")
        assert qaoa.n_qubits == 3
        assert qaoa.p_layers == 2
        assert len(qaoa.edges) == 3
        assert qaoa._device == "cpu"
        assert qaoa._method == "statevector"

    def test_qaoa_builds_ansatz_circuit(self):
        edges = [(0, 1), (1, 2)]
        qaoa = QAOA(3, edges, p_layers=1, device="cpu")
        circuit = qaoa._build_circuit(
            gamma=np.array([0.1]),
            beta=np.array([0.2]),
        )
        assert circuit.n_qubits == 3
        assert circuit.gate_count > 0
        assert qaoa.cost_hamiltonian is not None
