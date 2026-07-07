"""VQE domain tests — H2 ground-state energy minimization."""

import pytest

from superfermion.algorithms.variational import VQE
from superfermion.chemistry.ansatz import uccsd_ansatz
from superfermion.chemistry.hamiltonians import get_molecular_hamiltonian


pytestmark = [pytest.mark.domain, pytest.mark.timeout(30)]


class TestVQEH2:
    """VQE on parity-reduced H2 (STO-3G, 2 qubits)."""

    @pytest.fixture
    def h2_setup(self):
        H = get_molecular_hamiltonian("H2")
        ansatz = uccsd_ansatz(2, 2)
        return H, ansatz

    def test_vqe_converges_h2_adjoint(self, h2_setup):
        H, ansatz = h2_setup
        vqe = VQE(ansatz, H, device="cpu", diff_method="adjoint")
        result = vqe.minimize(iterations=100, seed=42)
        assert result.optimal_value < -1.0

    def test_vqe_converges_h2_param_shift(self, h2_setup):
        H, ansatz = h2_setup
        vqe = VQE(ansatz, H, device="cpu", diff_method="param_shift")
        result = vqe.minimize(iterations=100, seed=42)
        assert result.optimal_value < -1.0
