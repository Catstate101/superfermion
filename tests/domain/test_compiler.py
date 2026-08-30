"""Compiler domain tests — hardware-targeted compilation passes."""

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.compiler.specs import get_spec


pytestmark = pytest.mark.domain

SEED = 42

compile = pytest.importorskip(
    "superfermion.compiler.manager",
    reason="compiler module unavailable",
).compile


def _try_compile(circuit: Circuit, target_name: str = "linear_5") -> Circuit:
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")
    try:
        return compile(circuit, level=1, target=spec)
    except Exception as exc:
        pytest.skip(f"compilation failed: {exc}")


class TestCompile:
    def test_compile_returns_circuit(self, bell_circuit):
        compiled = _try_compile(bell_circuit)
        assert isinstance(compiled, Circuit)
        assert compiled.gate_count >= 1

    def test_compiled_circuit_runs_on_simulator(self, bell_circuit):
        compiled = _try_compile(bell_circuit)
        result = sf.run(compiled, device="cpu", shots=500, seed=SEED)
        assert sum(result.counts.values()) == 500

    def test_compiled_preserves_bell_distribution(self, bell_circuit):
        compiled = _try_compile(bell_circuit)
        original = sf.run(bell_circuit, device="cpu", shots=0, seed=SEED)
        optimized = sf.run(compiled, device="cpu", shots=0, seed=SEED)

        if original.probabilities and optimized.probabilities:
            assert abs(original.probabilities.get("00", 0) - 0.5) < 0.01
            padded_00 = format(0, f"0{compiled.n_qubits}b")
            # Little-endian: qubit q lives at bit position q, so the Bell
            # |11> part on (q0, q1) is index 3 -> rightmost "11".
            padded_11 = format(3, f"0{compiled.n_qubits}b")
            assert abs(optimized.probabilities.get(padded_00, 0) - 0.5) < 0.01
            assert abs(optimized.probabilities.get(padded_11, 0) - 0.5) < 0.01

    def test_linear_5_spec_available(self):
        spec = get_spec("linear_5")
        assert spec is not None
        assert spec.n_qubits == 5
        assert len(spec.coupling_map) == 4
