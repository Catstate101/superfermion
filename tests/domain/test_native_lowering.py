"""Regression tests for the native-basis lowering shims (HW-1 .. HW-5).

Two arms of the Rust ``BasisTranslationPass`` are numerically wrong:

* ``OpType::Ry`` on an ``SX`` device -- overlap ``0.977668245`` with ``Ry(t)``.
* ``cx_decomp`` on an ``ECR`` device -- overlap ``0.707106781`` with ``CX``.

:mod:`superfermion.compiler.native_lowering` repairs both in Python, around the
Rust call, so the emitted circuit is unitary-equivalent to the input and stays
inside the target's declared basis.
"""

import numpy as np
import pytest

from superfermion.circuit import Circuit
from superfermion.compiler import native_lowering as nl
from superfermion.compiler.specs import SPECS, get_spec

pytestmark = pytest.mark.domain

compile = pytest.importorskip(
    "superfermion.compiler.manager",
    reason="compiler module unavailable",
).compile


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def overlap(circuit_a: Circuit, circuit_b: Circuit) -> float:
    """Phase-insensitive unitary overlap in ``[0, 1]``."""
    u = circuit_a.to_unitary()
    v = circuit_b.to_unitary()
    assert u.shape == v.shape
    return float(abs(np.trace(u.conj().T @ v)) / u.shape[0])


def gate_names(circuit: Circuit):
    return {gate["name"].upper() for gate in circuit.to_gate_list()}


def single(gate: str, angle: float = 0.3) -> Circuit:
    circuit = Circuit(1)
    if gate in {"rx", "ry", "rz", "p"}:
        getattr(circuit, gate)(angle, 0)
    else:
        getattr(circuit, gate)(0)
    return circuit


def pair(gate: str, first: int = 0, second: int = 1) -> Circuit:
    circuit = Circuit(2)
    getattr(circuit, gate)(first, second)
    return circuit


# ──────────────────────────────────────────────────────────────────────────
# detection
# ──────────────────────────────────────────────────────────────────────────
class TestDetection:
    def test_ry_lowering_required_on_sx_targets(self):
        assert nl.needs_ry_lowering(["id", "rz", "sx", "x"])
        assert nl.needs_ry_lowering(get_spec("ibm_eagle").compile_basis)
        assert nl.needs_ry_lowering(get_spec("ibm_heron").compile_basis)

    def test_ry_lowering_not_required_when_ry_is_native(self):
        assert not nl.needs_ry_lowering(["id", "rz", "sx", "x", "ry"])

    def test_ry_lowering_not_required_without_sx(self):
        # The RZ/RX arm is the one used here, and it is exact.
        assert not nl.needs_ry_lowering(["rx", "rz", "cz"])
        assert not nl.needs_ry_lowering(get_spec("rigetti_ankaa").compile_basis)

    def test_ry_lowering_not_required_for_fully_native_targets(self):
        assert not nl.needs_ry_lowering(["all"])

    def test_ecr_lowering_required_for_heron(self):
        spec = get_spec("ibm_heron")
        assert nl.needs_ecr_lowering(spec.compile_basis, spec.native_gates)

    def test_ecr_lowering_not_required_when_cx_is_native(self):
        spec = get_spec("ibm_eagle")
        assert not nl.needs_ecr_lowering(spec.compile_basis, spec.native_gates)

    def test_ecr_lowering_not_required_for_other_targets(self):
        for name in ("rigetti_ankaa", "ionq_aria", "linear_5", "jax", "cluster"):
            spec = get_spec(name)
            assert not nl.needs_ecr_lowering(spec.compile_basis, spec.native_gates)


# ──────────────────────────────────────────────────────────────────────────
# the lowering identities themselves
# ──────────────────────────────────────────────────────────────────────────
class TestIdentities:
    def test_cx_via_ecr_matches_cx_up_to_global_phase(self):
        for control, target in ((0, 1), (1, 0)):
            rebuilt = Circuit(2)
            rebuilt.extend_raw(nl.cx_via_ecr(control, target))
            reference = pair("cx", control, target)
            assert overlap(rebuilt, reference) > 1.0 - 1e-9

    def test_cx_via_ecr_only_uses_native_superconducting_gates(self):
        rebuilt = Circuit(2)
        rebuilt.extend_raw(nl.cx_via_ecr(0, 1))
        assert gate_names(rebuilt) == {"RZ", "SX", "X", "ECR"}

    def test_lower_ry_is_exact(self):
        original = single("ry")
        lowered, count = nl.lower_ry(original)
        assert count == 1
        assert "RY" not in gate_names(lowered)
        assert np.allclose(lowered.to_unitary(), original.to_unitary(), atol=1e-12)

    def test_lower_ry_keeps_the_angle(self):
        circuit = Circuit(1)
        circuit.ry(1.234, 0)
        lowered, _ = nl.lower_ry(circuit)
        angles = [gate["params"][0] for gate in lowered.to_gate_list()]
        assert any(abs(float(angle) - 1.234) < 1e-12 for angle in angles)

    def test_lower_ry_is_a_no_op_without_ry(self):
        circuit = single("h")
        lowered, count = nl.lower_ry(circuit)
        assert count == 0
        assert lowered is circuit

    def test_lower_cx_to_ecr_is_a_no_op_without_cx(self):
        circuit = pair("cz")
        lowered, count = nl.lower_cx_to_ecr(circuit)
        assert count == 0
        assert lowered is circuit


# ──────────────────────────────────────────────────────────────────────────
# end-to-end: the compiled circuit must be exactly equivalent
# ──────────────────────────────────────────────────────────────────────────
TARGETS = ("ibm_eagle", "ibm_heron", "rigetti_ankaa", "ionq_aria", "linear_5")

ONE_QUBIT_GATES = ("h", "x", "y", "z", "s", "t", "sx", "rx", "ry", "rz")
TWO_QUBIT_GATES = ("cx", "cz", "swap")


@pytest.mark.parametrize("target_name", TARGETS)
@pytest.mark.parametrize("gate", ONE_QUBIT_GATES)
def test_single_qubit_gate_survives_compilation(gate, target_name):
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")
    original = single(gate)
    compiled = compile(original, level=1, target=spec)
    assert overlap(original, compiled) > 1.0 - 1e-9


@pytest.mark.parametrize("target_name", TARGETS)
@pytest.mark.parametrize("gate", TWO_QUBIT_GATES)
def test_two_qubit_gate_survives_compilation(gate, target_name):
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")
    original = pair(gate)
    compiled = compile(original, level=1, target=spec)
    assert overlap(original, compiled) > 1.0 - 1e-9


@pytest.mark.parametrize("target_name", TARGETS)
def test_mixed_circuit_survives_compilation(target_name):
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")

    original = Circuit(3)
    original.h(0)
    original.ry(0.3, 0)
    original.cx(0, 1)
    original.cz(1, 2)
    original.swap(0, 2)
    original.ry(0.7, 2)
    original.rx(0.5, 1)

    compiled = compile(original, level=1, target=spec)
    assert overlap(original, compiled) > 1.0 - 1e-9


def test_heron_emits_no_controlled_x():
    """``cx_decomp`` is avoided entirely, so no CX reaches an ECR-only device."""
    spec = get_spec("ibm_heron")
    compiled = compile(pair("cx"), level=1, target=spec)
    assert not gate_names(compiled) & {"CX", "CNOT"}
    assert "ECR" in gate_names(compiled)


# ──────────────────────────────────────────────────────────────────────────
# HW-1 / HW-3: the declared basis must describe reality
# ──────────────────────────────────────────────────────────────────────────
def test_native_gates_subset_of_basis_gates():
    for name, spec in SPECS.items():
        native = {gate.lower() for gate in spec.native_gates}
        basis = {gate.lower() for gate in spec.basis_gates}
        assert native <= basis, f"{name}: {native - basis} missing from basis"


def test_compile_basis_covers_native_gates():
    for name, spec in SPECS.items():
        native = {gate.lower() for gate in spec.native_gates}
        compile_basis = {gate.lower() for gate in spec.compile_basis}
        assert native <= compile_basis, f"{name}: {native - compile_basis} missing"


@pytest.mark.parametrize("target_name", TARGETS)
def test_compiled_output_stays_inside_the_declared_basis(target_name):
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")

    compiled = compile(pair("cx"), level=1, target=spec)
    assert spec.circuit_gates_outside(compiled) == []


# ──────────────────────────────────────────────────────────────────────────
# HW-4 / HW-5: width and layout reporting
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target_name", TARGETS)
def test_compilation_keeps_the_register_width(target_name):
    spec = get_spec(target_name)
    if spec is None:
        pytest.skip(f"hardware spec {target_name!r} not found")

    original = Circuit(3)
    original.h(0)
    original.cx(0, 1)
    original.cx(1, 2)

    compiled = compile(original, level=1, target=spec)
    assert compiled.n_qubits == original.n_qubits
    assert compiled.to_unitary().shape == original.to_unitary().shape


def test_compile_reports_the_layout_it_used():
    spec = get_spec("ibm_heron")
    original = Circuit(3)
    original.h(0)
    original.cx(0, 2)

    compiled = compile(original, level=1, target=spec)
    layout = compiled._metadata.get("layout")
    assert layout is not None
    assert layout["routed"] is True
    assert len(layout["initial_layout"]) == original.n_qubits
    # The layout is restored before the circuit is returned.
    assert layout["final_layout"] == layout["initial_layout"]
    assert compiled._metadata.get("target") == spec.name
    assert compiled._metadata.get("level") == 1


def test_compile_without_a_target_reports_no_layout():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    compiled = compile(circuit, level=1, target=None)
    assert "layout" not in compiled._metadata
