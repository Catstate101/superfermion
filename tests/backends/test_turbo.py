"""Gate fusion tests for the turbo optimization pipeline."""

import numpy as np
import pytest

from superfermion.backends.factory import get_backend
from superfermion.circuit import Circuit


SEED = 42

try:
    from superfermion.backends.turbo import fuse_single_qubit_gates

    TURBO_AVAILABLE = True
except ImportError:
    TURBO_AVAILABLE = False


pytestmark = [
    pytest.mark.backend,
    pytest.mark.skipif(not TURBO_AVAILABLE, reason="turbo module unavailable"),
]


class TestSingleQubitFusion:
    def test_consecutive_gates_are_fused(self):
        circuit = Circuit(1).h(0).x(0)
        fused = fuse_single_qubit_gates(circuit)
        assert fused.gate_count < circuit.gate_count
        assert fused.gate_count == 1

    def test_fusion_preserves_semantics(self):
        circuit = Circuit(1).h(0).x(0)
        fused = fuse_single_qubit_gates(circuit)
        backend = get_backend("statevector")
        original = backend.run(circuit, shots=0, seed=SEED)
        optimized = backend.run(fused, shots=0, seed=SEED)
        assert np.allclose(original.statevector, optimized.statevector)

    def test_fusion_preserves_measurement_statistics(self):
        circuit = Circuit(2).h(0).x(0).cnot(0, 1)
        fused = fuse_single_qubit_gates(circuit)
        backend = get_backend("statevector")
        r_orig = backend.run(circuit, shots=1000, seed=SEED)
        r_fused = backend.run(fused, shots=1000, seed=SEED)
        assert r_orig.counts.keys() == r_fused.counts.keys()
        for key in r_orig.counts:
            assert abs(r_orig.counts[key] - r_fused.counts[key]) <= 50

    def test_multi_qubit_gates_not_fused_away(self, bell_circuit):
        fused = fuse_single_qubit_gates(bell_circuit)
        assert fused.gate_count == bell_circuit.gate_count
