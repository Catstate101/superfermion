"""Regression tests for the 0.1.15 audit fixes D1, D2, D3, D4 and D10.

Each test reproduces the defect's observable contract:

* D1 — a 4-qubit opaque unitary must actually be applied on statevector.
* D2 — UNITARY matrices must survive ``to_json``/``from_json`` and .sfc.
* D3 — readout-only noise must affect sampled counts on sv/MPS.
* D4 — stabilizer ``shots=0`` must expose exact probabilities.
* D10 — MPS outputs after non-adjacent 2q gates use qubit order.
"""

import json
import warnings

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion._fixes import (
    LargeUnitaryDAG,
    mps_routing_permutation,
    remap_counts,
    unpermute_statevector,
)
from superfermion.noise import NoiseModel
from superfermion.serialization.circuit_format import (
    circuit_from_bytes,
    circuit_to_bytes,
)


pytestmark = pytest.mark.unit


CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def _peaked_four_qubit_unitary(seed: int = 11):
    """Random 16x16 unitary whose first column peaks at |0111>."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    psi = np.zeros(16, dtype=complex)
    psi[7] = 0.95
    psi += 0.05 * vector
    psi /= np.linalg.norm(psi)
    raw = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    raw[:, 0] = psi
    matrix, _ = np.linalg.qr(raw)
    matrix[:, 0] = psi
    return matrix


# ── D1: >3-qubit opaque unitary silently dropped ────────────────────────────

class TestD1LargeUnitary:
    def test_four_qubit_unitary_applied_statevector(self):
        matrix = _peaked_four_qubit_unitary()
        circuit = Circuit(4).unitary(matrix, [0, 1, 2, 3])

        expected = np.asarray(circuit.to_unitary())[:, 0]
        observed = np.asarray(
            circuit.to_ir().simulate_to_state("statevector").numpy()
        )

        assert np.allclose(observed, expected, atol=1e-12)
        assert int(np.argmax(np.abs(observed))) == 14

    def test_four_qubit_unitary_via_run(self):
        matrix = _peaked_four_qubit_unitary()
        circuit = Circuit(4).unitary(matrix, [0, 1, 2, 3])

        expected = np.asarray(circuit.to_unitary())[:, 0]
        result = sf.run(circuit, method="statevector", shots=0)
        assert result.statevector is not None
        assert np.allclose(result.statevector, expected, atol=1e-12)

    def test_four_qubit_unitary_sampled_fast_path(self):
        matrix = _peaked_four_qubit_unitary()
        circuit = Circuit(4).unitary(matrix, [0, 1, 2, 3])

        expected = np.asarray(circuit.to_unitary())[:, 0]
        top = format(int(np.argmax(np.abs(expected))), "04b")
        result = sf.run(
            circuit,
            method="statevector",
            shots=20000,
            seed=5,
            return_statevector=False,
        )
        assert result.counts
        assert max(result.counts, key=result.counts.get) == top

    def test_four_qubit_unitary_unsupported_methods_raise(self):
        matrix = _peaked_four_qubit_unitary()
        circuit = Circuit(4).unitary(matrix, [0, 1, 2, 3])
        with pytest.raises(ValueError, match="three qubits"):
            sf.run(circuit, method="mps", shots=10)
        with pytest.raises(ValueError, match="three qubits"):
            sf.run(circuit, method="density_matrix", shots=10)

    def test_three_qubit_unitary_keeps_raw_dag(self):
        circuit = Circuit(3).unitary(np.eye(8, dtype=complex), [0, 1, 2])
        assert not isinstance(circuit.to_ir(), LargeUnitaryDAG)

    def test_flag_survives_decomposition_path(self):
        circuit = Circuit(4).unitary(_peaked_four_qubit_unitary(), [0, 1, 2, 3])
        circuit.u3(0.1, 0.2, 0.3, 0)  # forces _prepare_dag decomposition
        expected = np.asarray(circuit.to_unitary())[:, 0]
        result = sf.run(circuit, method="statevector", shots=0)
        assert np.allclose(result.statevector, expected, atol=1e-12)


# ── D2: UNITARY matrix dropped by JSON serialization ────────────────────────

class TestD2UnitarySerialization:
    def test_to_from_json_preserves_matrix(self):
        circuit = Circuit(2).h(0).unitary(CNOT_MAT, [0, 1]).cnot(1, 0)
        restored = Circuit.from_json(circuit.to_json())

        restored._ensure_gates()
        unitaries = [g for g in restored._gates if g.name == "UNITARY"]
        assert len(unitaries) == 1
        assert unitaries[0].matrix is not None
        assert np.allclose(unitaries[0].matrix, CNOT_MAT)
        assert np.allclose(
            np.asarray(restored.to_unitary()), np.asarray(circuit.to_unitary())
        )

    def test_sfc_round_trip_preserves_matrix(self):
        circuit = Circuit(2).h(0).unitary(CNOT_MAT, [0, 1])
        restored = circuit_from_bytes(circuit_to_bytes(circuit))
        assert np.allclose(
            np.asarray(restored.to_unitary()), np.asarray(circuit.to_unitary())
        )

    def test_missing_matrix_is_rejected(self):
        circuit = Circuit(1).unitary(
            np.array([[0, 1], [1, 0]], dtype=complex), [0]
        )
        payload = json.loads(circuit.to_json())
        for gate in payload["gates"]:
            if str(gate["name"]).upper() == "UNITARY":
                gate.pop("matrix", None)
        with pytest.raises(ValueError, match="no 'matrix' field"):
            Circuit.from_json(json.dumps(payload))

    def test_matrix_on_named_gate_is_rejected(self):
        payload = {
            "n_qubits": 1,
            "n_cbits": 1,
            "gates": [{"name": "H", "qubits": [0], "matrix": [[[1.0, 0.0]]]}],
        }
        with pytest.raises(ValueError, match="only UNITARY"):
            Circuit.from_json(json.dumps(payload))

    def test_rust_storage_round_trip_preserves_matrix(self):
        """D2: the Rust ``GateSequence`` path must not drop the matrix."""
        circuit = Circuit(2, use_rust_storage=True)
        circuit.unitary(CNOT_MAT, [0, 1])

        payload = json.loads(circuit.to_json())
        assert "matrix" in payload["gates"][0]

        restored = Circuit.from_json(circuit.to_json())
        restored._ensure_gates()
        unitaries = [g for g in restored._gates if g.name == "UNITARY"]
        assert len(unitaries) == 1
        assert np.allclose(unitaries[0].matrix, CNOT_MAT)

    def test_rust_storage_large_unitary_uses_proxy(self):
        """D1 + D2: a >3-qubit unitary under Rust storage keeps both paths."""
        matrix = _peaked_four_qubit_unitary()
        circuit = Circuit(4, use_rust_storage=True).unitary(
            matrix, [0, 1, 2, 3]
        )

        dag = circuit.to_ir()
        assert isinstance(dag, LargeUnitaryDAG)
        observed = np.asarray(dag.simulate_to_state("statevector").numpy())
        expected = np.asarray(circuit.to_unitary())[:, 0]
        assert np.allclose(observed, expected, atol=1e-12)
        assert json.loads(circuit.to_json())["gates"][0]["matrix"]


# ── D3: readout error ignored on statevector / MPS ──────────────────────────

def _column_flip_rates(counts):
    total = sum(counts.values())
    n_bits = len(next(iter(counts)))
    rates = np.zeros(n_bits)
    for key, count in counts.items():
        for index, char in enumerate(key):
            if char == "1":
                rates[index] += count
    return rates / total


class TestD3ReadoutOnPureStates:
    @pytest.mark.parametrize("method", ["statevector", "mps"])
    def test_readout_applied(self, method):
        noise = NoiseModel()
        noise.add_readout_error(0.25)
        circuit = Circuit(4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = sf.run(
                circuit,
                method=method,
                shots=40000,
                seed=11,
                noise_model=noise,
            )
        rates = _column_flip_rates(result.counts)
        assert np.allclose(rates, 0.25, atol=0.02)

    def test_readout_applied_on_fast_path(self):
        noise = NoiseModel()
        noise.add_readout_error(0.25)
        result = sf.run(
            Circuit(4),
            method="statevector",
            shots=40000,
            seed=11,
            noise_model=noise,
            return_statevector=False,
        )
        rates = _column_flip_rates(result.counts)
        assert np.allclose(rates, 0.25, atol=0.02)

    def test_gate_noise_still_warns(self):
        noise = NoiseModel()
        noise.add_depolarizing(0.01, 1)
        with pytest.warns(RuntimeWarning, match="gate noise"):
            sf.run(
                Circuit(2).h(0).cnot(0, 1),
                method="statevector",
                shots=100,
                noise_model=noise,
            )


# ── D4: stabilizer shots=0 returns no probabilities ─────────────────────────

class TestD4StabilizerProbabilities:
    def test_shots_zero_probabilities(self):
        circuit = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
        result = sf.run(circuit, method="stabilizer", shots=0, seed=7)
        assert result.counts == {}
        assert result.probabilities == pytest.approx({"000": 0.5, "111": 0.5})
        assert result.metadata["probability_path"] == "statevector-fallback"

    def test_shots_positive_unchanged(self):
        circuit = Circuit(3).h(0).cnot(0, 1)
        result = sf.run(circuit, method="stabilizer", shots=4000, seed=7)
        assert set(result.counts) <= {"000", "011"}
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-12

    def test_shots_zero_above_cap_raises(self):
        circuit = Circuit(21).h(0).cnot(0, 1)
        with pytest.raises(ValueError, match="capped at 20"):
            sf.run(circuit, method="stabilizer", shots=0)


# ── D10: MPS outputs in physical-site order after long-range gates ──────────

class TestD10MPSOutputOrder:
    @staticmethod
    def _long_range_circuit():
        return (
            Circuit(4)
            .h(0)
            .h(1)
            .h(2)
            .h(3)
            .cnot(0, 3)
            .ry(0.31, 0)
            .ry(0.62, 1)
            .ry(0.93, 2)
            .ry(1.24, 3)
        )

    def test_evolve_mps_statevector_qubit_order(self):
        circuit = self._long_range_circuit()
        dag = circuit.to_ir()
        mps = dag.evolve_mps(64)
        assert list(mps.perm()) != list(range(4))

        expected = np.asarray(dag.simulate_to_state("statevector", "cpu").numpy())
        observed = np.asarray(mps.to_statevector())
        assert np.allclose(observed, expected, atol=1e-12)

    def test_mps_counts_match_statevector(self):
        circuit = Circuit(4).h(0).cnot(0, 3)
        result = sf.run(circuit, method="mps", shots=20000, seed=5, bond_dim=64)
        assert set(result.counts) == {"0000", "1001"}

    def test_mps_result_state_uses_qubit_order(self):
        circuit = self._long_range_circuit()
        result = sf.run(
            circuit, method="mps", shots=1000, seed=5, bond_dim=64
        )
        assert result.metadata.get("mps_perm") not in (None, list(range(4)))

        expected = np.asarray(
            circuit.to_ir().simulate_to_state("statevector", "cpu").numpy()
        )
        assert np.allclose(result.state.numpy(), expected, atol=1e-12)

    def test_mps_result_state_sample_uses_qubit_order(self):
        circuit = Circuit(4).h(0).cnot(0, 3)
        result = sf.run(circuit, method="mps", shots=500, seed=5, bond_dim=64)
        assert result.metadata.get("mps_perm") not in (None, list(range(4)))
        assert set(result.state.sample(2000, 3)) == {"0000", "1001"}

    def test_adjacent_only_mps_has_no_remap(self):
        circuit = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
        result = sf.run(circuit, method="mps", shots=0, bond_dim=32)
        assert "mps_perm" not in result.metadata
        assert type(result.state).__name__ == "State"

    def test_routing_replication_matches_core(self):
        rng = np.random.default_rng(2026)
        for n in (3, 4, 5, 6):
            for _ in range(6):
                circuit = Circuit(n)
                for _ in range(8):
                    kind = int(rng.integers(0, 4))
                    if kind == 0:
                        circuit.h(int(rng.integers(0, n)))
                    elif kind == 1:
                        circuit.ry(float(rng.uniform(-2, 2)), int(rng.integers(0, n)))
                    else:
                        a = int(rng.integers(0, n))
                        b = int(rng.integers(0, n - 1))
                        if b >= a:
                            b += 1
                        circuit.cnot(a, b)
                dag = circuit.to_ir()
                expected = list(dag.evolve_mps(32).perm())
                observed = mps_routing_permutation(dag.to_gate_records(), n)
                assert observed == expected

    def test_helpers_are_inverses(self):
        perm = [0, 2, 3, 1]
        state = np.arange(16, dtype=complex)
        permuted = unpermute_statevector(state, perm)
        recovered = unpermute_statevector(
            permuted, [perm.index(i) for i in range(4)]
        )
        assert np.allclose(recovered, state)

        counts = {"0000": 3, "1001": 5}
        assert remap_counts(remap_counts(counts, perm), [perm.index(i) for i in range(4)]) == counts
