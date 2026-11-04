"""Bridge/result/limits tests — RES-1, RES-2, MEM-1, INT-1.

Every assertion here is derived from a measurement made against this build,
not from a convention assumed in advance:

* RES-1: ``x(0)`` samples ``'01'`` and ``expval('ZI') == -1``, so character
  ``n-1-q`` is qubit ``q``; ``measure(q, cbit=c)`` never reached the output.
* RES-2: the adjoint gradient of ``⟨ZI⟩`` under ``ry(t)`` is checked against
  a central finite difference of this package's own expectation values.
* MEM-1: a 40-qubit statevector is 16 TiB, which the Rust allocator answers
  with a process abort (exit ``0xC0000409``) instead of an exception.
* INT-1: ``pauli_matrix`` builds ``kron(str[0], str[1], ...)`` while
  ``expval`` reads character ``q`` as qubit ``q`` -- the two are only the
  same for symmetric labels.
"""

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion import _limits
from superfermion._classical_order import (
    apply_to_result,
    char_index_for_qubit,
    measured_classical_map,
    rekey,
)
from superfermion.chemistry.interop import QISKIT_AVAILABLE, to_qiskit_labels
from superfermion.observables.core import Hamiltonian, PauliString, SparsePauliOp
from superfermion.results import RunResult

pytestmark = pytest.mark.domain

SEED = 42
METHODS = ("statevector", "mps", "density_matrix", "stabilizer")


def _swapped_measurement_circuit(n_cbits: int = 2):
    """``x(0); measure(0, cbit=1); measure(1, cbit=0)`` -- the audit case."""
    circuit = sf.Circuit(2, n_cbits=n_cbits)
    circuit.x(0)
    circuit.measure(0, cbit=1)
    circuit.measure(1, cbit=0)
    return circuit


class TestClassicalBitOrdering:
    """RES-1: counts follow the classical bit the circuit asked for."""

    def test_counts_follow_the_explicit_cbit(self):
        result = sf.run(_swapped_measurement_circuit(), shots=200, seed=SEED)
        assert result.counts == {"10": 200}

    def test_probabilities_follow_the_counts(self):
        result = sf.run(_swapped_measurement_circuit(), shots=200, seed=SEED)
        assert result.probabilities == {"10": 1.0}
        assert result.probabilities_array.tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_metadata_records_the_map(self):
        result = sf.run(_swapped_measurement_circuit(), shots=20, seed=SEED)
        assert result.metadata["counts_keyed_by"] == "classical_bits"
        assert result.metadata["qubit_char_index"] == {0: 0, 1: 1}

    def test_default_cbit_keeps_the_historical_keys(self):
        circuit = sf.Circuit(2, n_cbits=2)
        circuit.x(0)
        circuit.measure(0)
        circuit.measure(1)
        result = sf.run(circuit, shots=100, seed=SEED)
        assert result.counts == {"01": 100}
        assert "counts_keyed_by" not in result.metadata

    def test_partial_measurement_is_untouched(self):
        circuit = sf.Circuit(2, n_cbits=2)
        circuit.x(0)
        circuit.measure(0, cbit=1)
        result = sf.run(circuit, shots=100, seed=SEED)
        assert result.counts == {"01": 100}
        assert "counts_keyed_by" not in result.metadata

    def test_extra_classical_bits_are_untouched(self):
        circuit = sf.Circuit(2, n_cbits=4)
        circuit.x(0)
        circuit.measure(0, cbit=1)
        circuit.measure(1, cbit=0)
        result = sf.run(circuit, shots=100, seed=SEED)
        # n_cbits (4) != n_qubits (2): the string width would change.
        assert result.counts == {"01": 100}
        assert "counts_keyed_by" not in result.metadata

    def test_gate_after_measurement_is_untouched(self):
        """The sampled string is the final qubit value, not the latch."""
        circuit = sf.Circuit(2, n_cbits=2)
        circuit.x(0)
        circuit.measure(0, cbit=1)
        circuit.x(0)
        circuit.measure(1, cbit=0)
        result = sf.run(circuit, shots=100, seed=SEED)
        assert result.counts == {"00": 100}
        assert "counts_keyed_by" not in result.metadata

    def test_repeated_measurement_is_untouched(self):
        circuit = sf.Circuit(2, n_cbits=2)
        circuit.x(0)
        circuit.measure(0, cbit=1)
        circuit.measure(0, cbit=0)
        assert measured_classical_map(circuit) is None

    def test_three_qubit_rotation(self):
        circuit = sf.Circuit(3, n_cbits=3)
        circuit.x(0)
        circuit.measure(0, cbit=2)
        circuit.measure(1, cbit=1)
        circuit.measure(2, cbit=0)
        result = sf.run(circuit, shots=50, seed=SEED)
        assert result.counts == {"100": 50}

    @pytest.mark.parametrize("method", METHODS)
    def test_every_backend_agrees(self, method):
        circuit = _swapped_measurement_circuit()
        result = sf.run(circuit, shots=200, seed=SEED, method=method)
        assert result.counts == {"10": 200}
        assert result.metadata["counts_keyed_by"] == "classical_bits"

    def test_expectation_from_counts_uses_the_map(self):
        """The map must not decouple the observable from its qubit."""
        result = sf.run(
            _swapped_measurement_circuit(),
            shots=4000,
            seed=SEED,
            return_statevector=False,
        )
        assert result.state is None  # genuinely the counts path
        assert result.counts == {"10": 4000}
        assert result.expectation("ZI") == pytest.approx(-1.0)
        assert result.expectation("IZ") == pytest.approx(1.0)
        assert result.variance("ZI") == pytest.approx(0.0)

    def test_rekey_is_a_bijection(self):
        outcomes = {"000": 1, "001": 2, "010": 3, "011": 4,
                    "100": 5, "101": 6, "110": 7, "111": 8}
        mapping = {0: 2, 1: 1, 2: 0}
        swapped = rekey(outcomes, mapping, 3)
        assert sum(swapped.values()) == sum(outcomes.values())
        assert len(swapped) == len(outcomes)
        assert swapped["000"] == outcomes["000"]
        assert swapped["100"] == outcomes["001"]
        # This map reverses the string, so it is its own inverse.
        assert rekey(swapped, mapping, 3) == outcomes

    def test_rekey_leaves_unexpected_widths_alone(self):
        outcomes = {"0101": 3}
        assert rekey(outcomes, {0: 1, 1: 0}, 2) == outcomes

    def test_char_index_is_the_identity_for_the_identity_map(self):
        assert char_index_for_qubit({0: 0, 1: 1, 2: 2}, 3) == {0: 2, 1: 1, 2: 0}

    def test_apply_to_result_is_a_no_op_without_a_map(self):
        circuit = sf.Circuit(2, n_cbits=2)
        circuit.measure(0)
        circuit.measure(1)
        result = RunResult(counts={"01": 5}, metadata={})
        returned = apply_to_result(circuit, result)
        assert returned is result
        assert result.counts == {"01": 5}
        assert result.metadata == {}

    def test_the_rust_buffer_is_not_converted(self):
        class _RustBucket:
            @staticmethod
            def to_gate_records():
                return [("H", [0], []), ("MEASURE", [0], [])]

        class _Stub:
            n_qubits = 2
            n_cbits = 2
            _use_rust = True
            _gates = ()
            _gates_rust = _RustBucket()

            def _ensure_gates(self):  # pragma: no cover - must not be called
                raise AssertionError("converted the deferred Rust buffer")

        assert measured_classical_map(_Stub()) is None

    def test_a_lazy_probability_table_stays_lazy(self):
        """Relabelling must not force the 2^n table it is deferring."""
        from superfermion.results import LazyDict

        result = RunResult(counts={"01": 3}, metadata={})
        result.probabilities = LazyDict(lambda: {"01": 0.5, "00": 0.5})
        apply_to_result(_swapped_measurement_circuit(), result)
        assert result.probabilities._build is not None  # still deferred
        assert result.probabilities == {"10": 0.5, "00": 0.5}
        assert result.counts == {"10": 3}


class TestGradientAPI:
    """RES-2: ``RunResult.grad`` works off the public observable spellings."""

    @staticmethod
    def _parametric_result():
        circuit = sf.Circuit(2)
        circuit.ry(sf.param("t"), 0)
        circuit.cx(0, 1)
        return circuit, sf.run(circuit, shots=0, params={"t": 0.7})

    def _finite_difference(self, circuit, step=1e-5):
        def value(t):
            return sf.run(circuit, shots=0, params={"t": t}).expectation("ZI")

        return (value(0.7 + step) - value(0.7 - step)) / (2 * step)

    def test_grad_without_a_dag_matches_finite_differences(self):
        circuit, result = self._parametric_result()
        expected = self._finite_difference(circuit)
        assert abs(expected) > 0.1  # the check would be vacuous otherwise
        got = result.grad("ZI")
        assert set(got) == {"t"}
        assert got["t"] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "observable",
        [
            "ZI",
            {"ZI": 1.0},
            PauliString("ZI"),
            Hamiltonian([PauliString("ZI")]),
            SparsePauliOp(["ZI"]),
            ["ZI"],
            [("ZI", 1.0, 0.0)],
            [("ZI", 1.0 + 0.0j)],
        ],
        ids=[
            "str", "dict", "PauliString", "Hamiltonian", "SparsePauliOp",
            "list-of-str", "label-terms", "label-complex",
        ],
    )
    def test_every_public_observable_form(self, observable):
        circuit, result = self._parametric_result()
        expected = self._finite_difference(circuit)
        assert result.grad(observable)["t"] == pytest.approx(expected, abs=1e-6)

    def test_grad_accepts_a_circuit_as_the_dag(self):
        circuit, result = self._parametric_result()
        expected = self._finite_difference(circuit)
        assert result.grad("ZI", dag=circuit)["t"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_grad_accepts_explicit_values(self):
        circuit, result = self._parametric_result()
        expected = self._finite_difference(circuit)
        assert result.grad("ZI", param_values={"t": 0.7})["t"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_grad_accepts_a_sequence_of_values(self):
        circuit, result = self._parametric_result()
        expected = self._finite_difference(circuit)
        assert result.grad("ZI", param_values=[0.7])["t"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_grad_rejects_a_mis_sized_value_sequence(self):
        _, result = self._parametric_result()
        with pytest.raises(ValueError, match="parameter"):
            result.grad("ZI", param_values=[0.7, 0.8])

    def test_grad_without_a_state_raises(self):
        result = RunResult(counts={"00": 10})
        with pytest.raises(RuntimeError, match="no state"):
            result.grad("ZI")

    def test_grad_on_a_parameter_free_circuit_raises(self):
        """F/V-6: no parameters to differentiate is an error, not ``{}``.

        The empty dict used to be indistinguishable from a gradient that
        genuinely is zero, so a training loop whose circuit had been bound by
        accident read a zero gradient and stopped making progress.
        """
        circuit = sf.Circuit(2)
        circuit.h(0)
        result = sf.run(circuit, shots=0)
        result._sf_grad_context = {}  # as a hand-built result would be
        with pytest.raises(ValueError, match="symbolic parameters"):
            result.grad("ZI")

    def test_bound_circuit_grad_names_the_cause_and_the_fix(self):
        """F/V-6: the message says why, not just that it failed."""
        circuit = sf.Circuit(2)
        circuit.ry(sf.param("t"), 0)
        result = sf.run(circuit.bind({"t": 0.7}), shots=0)
        with pytest.raises(ValueError) as err:
            result.grad("ZI")
        text = str(err.value)
        assert "already bound" in text and ".bind(" in text
        assert "sf.run(circuit, params=" in text

    def test_grad_without_parameter_values_points_at_the_fix(self):
        """A symbolic circuit plus a state, but no bound values to use."""
        bound = sf.run(sf.Circuit(2).ry(0.7, 0), shots=0)
        symbolic = sf.Circuit(2)
        symbolic.ry(sf.param("t"), 0)
        manual = RunResult(state=bound.state, circuit=symbolic)
        with pytest.raises(ValueError, match="param_values"):
            manual.grad("ZI")

    def test_variance_accepts_a_sequence_of_labels(self):
        result = sf.run(sf.Circuit(2).x(0), shots=0)
        assert result.variance(["ZI"]) == pytest.approx(0.0)

    def test_expectation_sums_a_sequence_of_labels(self):
        result = sf.run(sf.Circuit(2).x(0), shots=0)
        # x(0): <ZI> = -1, so two copies sum to -2.
        assert result.expectation(["ZI", "ZI"]) == pytest.approx(-2.0)

    def test_expectation_accepts_label_coefficient_pairs(self):
        result = sf.run(sf.Circuit(2).x(0), shots=0)
        assert result.expectation([("ZI", 2.0)]) == pytest.approx(-2.0)
        assert result.expectation([("ZI", 1.0, 1.0)]) == pytest.approx(-1.0)

    def test_engine_term_form_is_passed_through(self):
        result = sf.run(sf.Circuit(2).x(0), shots=0)
        assert result.expectation([([3, 0], 1.0, 0.0)]) == pytest.approx(-1.0)


class TestStateMemoryGuard:
    """MEM-1: refuse an impossible state instead of aborting the process."""

    def test_forty_qubit_statevector_raises(self):
        with pytest.raises(MemoryError, match="GiB"):
            sf.run(sf.Circuit(40).h(0), shots=1)

    def test_density_matrix_is_guarded_too(self):
        with pytest.raises(MemoryError):
            sf.run(sf.Circuit(20).h(0), shots=1, method="density_matrix")

    def test_the_error_names_a_way_out(self):
        with pytest.raises(MemoryError) as excinfo:
            sf.run(sf.Circuit(40).h(0), shots=1)
        message = str(excinfo.value)
        assert "method='mps'" in message
        assert "max_state_gib" in message

    def test_a_run_that_fits_is_not_affected(self):
        result = sf.run(sf.Circuit(12).h(0), shots=10, seed=SEED)
        assert sum(result.counts.values()) == 10

    def test_max_state_gib_can_tighten(self):
        with pytest.raises(MemoryError):
            sf.run(sf.Circuit(3).h(0), shots=10, max_state_gib=1e-9)

    def test_max_state_gib_can_relax(self):
        result = sf.run(sf.Circuit(3).h(0), shots=10, max_state_gib=1.0, seed=SEED)
        assert sum(result.counts.values()) == 10

    def test_max_state_gib_zero_disables(self, monkeypatch):
        assert _limits.limit_from_kwargs({"max_state_gib": 0}) == math.inf
        assert _limits.state_memory_limit_bytes() < math.inf
        monkeypatch.setenv("SF_STATE_MEMORY_LIMIT_GIB", "0")
        assert _limits.state_memory_limit_bytes() == math.inf

    def test_bytes_for_state_counts_exact_amplitudes(self):
        assert _limits.bytes_for_state(0) == 16
        assert _limits.bytes_for_state(3) == 16 * 8
        assert _limits.bytes_for_state(40) == 16 * 2**40
        assert _limits.bytes_for_state(4, "density_matrix") == 16 * 4**4
        assert _limits.bytes_for_state(6, "unitary") == 16 * 4**6
        assert _limits.bytes_for_state(30, "mps") is None
        assert _limits.bytes_for_state(30, "stabilizer") is None

    def test_guard_returns_the_estimate_when_it_fits(self):
        assert _limits.guard_state_allocation(4, limit_bytes=1 << 20) == 16 * 16

    def test_mps_is_not_held_to_the_statevector_ceiling(self):
        assert _limits.guard_state_allocation(
            60, "mps", limit_bytes=1 << 20
        ) is None

    def test_physical_memory_is_readable_here(self):
        total = _limits.physical_memory_bytes()
        if total is None:
            pytest.skip("platform does not report physical memory")
        assert total > 1 << 20

    def test_to_unitary_is_guarded(self):
        with pytest.raises(MemoryError):
            sf.Circuit(20).h(0).to_unitary()

    def test_small_to_unitary_still_works(self):
        matrix = sf.Circuit(2).h(0).to_unitary()
        assert matrix.shape == (4, 4)
        assert np.allclose(np.abs(matrix[0, :2]), [1.0, 1.0] / np.sqrt(2))

    def test_env_override_moves_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("SF_STATE_MEMORY_LIMIT_GIB", "0.000001")
        assert _limits.state_memory_limit_bytes() == pytest.approx(
            1e-6 * 2**30
        )
        with pytest.raises(MemoryError):
            _limits.guard_state_allocation(20)


class TestInteropOrdering:
    """INT-1: the reversal is now a parameter, not a false claim."""

    @staticmethod
    def _operator():
        return Hamiltonian(
            [PauliString("ZI", coeff=1.0), PauliString("IZ", coeff=2.0)]
        )

    def test_labels_are_unchanged_by_default(self):
        labels, coeffs = to_qiskit_labels(self._operator())
        assert labels == ["ZI", "IZ"]
        assert coeffs == [1.0 + 0.0j, 2.0 + 0.0j]

    def test_reverse_flips_each_label(self):
        labels, coeffs = to_qiskit_labels(self._operator(), reverse=True)
        assert labels == ["IZ", "ZI"]
        assert coeffs == [1.0 + 0.0j, 2.0 + 0.0j]

    def test_padding_happens_before_the_reversal(self):
        labels, _ = to_qiskit_labels(
            Hamiltonian([PauliString("Z")]), n_qubits=3, reverse=True
        )
        assert labels == ["IIZ"]

    def test_reversal_is_an_involution(self):
        labels, _ = to_qiskit_labels(self._operator())
        back, _ = to_qiskit_labels(
            Hamiltonian([PauliString(label[::-1]) for label in labels]),
            reverse=True,
        )
        assert back == labels

    def test_reversal_moves_the_operator_to_the_other_wire(self):
        """``pauli_matrix`` is kron in character order, so it shows the wire."""
        identity = np.eye(2, dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        from superfermion.chemistry.subspace import pauli_matrix

        forward = np.asarray(
            pauli_matrix(Hamiltonian([PauliString("ZI")])), dtype=complex
        )
        reversed_label = to_qiskit_labels(
            Hamiltonian([PauliString("ZI")]), reverse=True
        )[0][0]
        backward = np.asarray(
            pauli_matrix(Hamiltonian([PauliString(reversed_label)])),
            dtype=complex,
        )
        assert np.allclose(forward, np.kron(z, identity))
        assert np.allclose(backward, np.kron(identity, z))

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit not installed")
    def test_round_trip_through_qiskit(self):  # pragma: no cover - no qiskit
        from superfermion.chemistry.interop import from_qiskit_op, to_qiskit_op

        original = self._operator()
        op = to_qiskit_op(original)
        assert from_qiskit_op(op).terms[0].pauli_str == "ZI"
        op_reversed = to_qiskit_op(original, reverse=True)
        assert from_qiskit_op(op_reversed, reverse=True).terms[0].pauli_str == "ZI"
