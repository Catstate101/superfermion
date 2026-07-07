"""Comprehensive QEC domain tests — codes, decoders, and manager lifecycle."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.domain

qec = pytest.importorskip("superfermion.qec", reason="qec module unavailable")

RepetitionCode = qec.RepetitionCode
ShorCode = qec.ShorCode
SteaneCode = qec.SteaneCode
BaconShorCode = qec.BaconShorCode
GenericCSSCode = qec.GenericCSSCode
SurfaceCode2D = qec.SurfaceCode2D
HypercubeCode4D = qec.HypercubeCode4D
ToricCode2D = qec.ToricCode2D
ColorCode = qec.ColorCode
HoneycombCode = qec.HoneycombCode
QECManager = qec.QECManager
MWPMDecoder = qec.MWPMDecoder
UnionFindDecoder = qec.UnionFindDecoder


def _assert_builds_circuit(code, min_qubits: int = 1):
    circuit = code.build()
    assert isinstance(circuit, Circuit)
    assert circuit.n_qubits >= min_qubits
    assert circuit.gate_count >= 0


class TestLinearCodes:
    def test_repetition_code_construction_and_build(self):
        code = RepetitionCode(n=3, code_type="bit")
        assert code.n == 3
        assert code.code_type == "bit"
        circuit = code.build()
        assert circuit.n_qubits == code.n + 2  # n data + 2 ancilla
        assert circuit.gate_count > 0

    def test_repetition_code_phase_flip(self):
        code = RepetitionCode(n=5, code_type="phase")
        circuit = code.build()
        assert circuit.n_qubits == 7
        gate_names = {g["name"] for g in circuit.to_gate_list()}
        assert "H" in gate_names

    def test_shor_code_build(self):
        code = ShorCode()
        circuit = code.build()
        assert circuit.n_qubits == 9
        assert circuit.gate_count > 0

    def test_steane_code_build(self):
        code = SteaneCode()
        circuit = code.build()
        assert circuit.n_qubits == 7
        assert circuit.gate_count > 0

    def test_bacon_shor_code_construction_and_build(self):
        code = BaconShorCode(L=3)
        assert code.L == 3
        expected_qubits = code.L**2 + 2 * (code.L - 1) * code.L
        circuit = code.build()
        assert circuit.n_qubits == expected_qubits

    def test_generic_css_code_construction_and_build(self):
        hx = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
        hz = np.array([[1, 0, 1]], dtype=np.int32)
        code = GenericCSSCode(hx, hz)
        assert code.n == 3
        circuit = code.build()
        assert circuit.n_qubits == code.n + hx.shape[0] + hz.shape[0]


class TestTopologicalCodes:
    def test_surface_code_properties_and_build(self):
        code = SurfaceCode2D(distance=3)
        assert code.d == 3
        assert code.n_data == 9
        assert code.n_ancilla == 8
        circuit = code.build()
        assert circuit.n_qubits == code.n_data + code.n_ancilla
        assert circuit.gate_count > 0

    def test_hypercube_code_properties_and_build(self):
        code = HypercubeCode4D(size=2)
        assert code.n_data == 16
        assert code.n_syndromes == 32
        circuit = code.build()
        assert circuit.n_qubits == code.n_data + code.n_syndromes
        assert circuit.gate_count > 0

    def test_toric_code_properties_and_build(self):
        code = ToricCode2D(size=3)
        assert code.n_data == 18
        assert code.n_ancilla == 18
        circuit = code.build()
        assert circuit.n_qubits == code.n_data + code.n_ancilla

    def test_color_code_properties_and_build(self):
        code = ColorCode(distance=3)
        assert code.d == 3
        assert code.n_data == (3 * 3**2 + 1) // 4
        circuit = code.build()
        assert circuit.n_qubits == code.n_data + 10

    def test_honeycomb_code_build(self):
        code = HoneycombCode()
        circuit = code.build()
        assert circuit.n_qubits == 20


class TestDecoders:
    @pytest.fixture
    def repetition_decoder_setup(self):
        n_data = 3
        syndrome_map = [[0, 1], [1, 2]]
        return n_data, syndrome_map

    def test_mwpm_decoder_on_repetition_syndrome(self, repetition_decoder_setup):
        n_data, syndrome_map = repetition_decoder_setup
        decoder = MWPMDecoder(n_data, syndrome_map)
        syndrome = np.array([0, 1], dtype=np.int32)
        correction = decoder.decode(syndrome)
        assert isinstance(correction, list)
        for item in correction:
            assert isinstance(item, tuple)
            assert len(item) == 2
            qubit, pauli = item
            assert 0 <= qubit < n_data
            assert pauli in ("X", "Y", "Z")

    def test_union_find_decoder_on_repetition_syndrome(self, repetition_decoder_setup):
        n_data, syndrome_map = repetition_decoder_setup
        decoder = UnionFindDecoder(n_data, syndrome_map)
        syndrome = np.array([1, 0], dtype=np.int32)
        correction = decoder.decode(syndrome)
        assert isinstance(correction, list)

    def test_mwpm_decoder_no_error_syndrome(self, repetition_decoder_setup):
        n_data, syndrome_map = repetition_decoder_setup
        decoder = MWPMDecoder(n_data, syndrome_map)
        correction = decoder.decode(np.array([0, 0], dtype=np.int32))
        assert isinstance(correction, list)


class TestQECManager:
    @pytest.fixture
    def manager(self):
        return QECManager()

    @pytest.mark.parametrize(
        "code_name,kwargs",
        [
            ("repetition", {"n": 3}),
            ("shor", {}),
            ("steane", {}),
            ("bacon_shor", {"L": 3}),
            ("surface_2d", {"distance": 3}),
            ("hypercube_4d", {}),
            ("toric_2d", {"size": 3}),
            ("color_code", {"distance": 3}),
            ("honeycomb", {}),
        ],
    )
    def test_get_code_registry(self, manager, code_name, kwargs):
        code = manager.get_code(code_name, **kwargs)
        _assert_builds_circuit(code)

    def test_get_code_unknown_raises(self, manager):
        with pytest.raises(ValueError, match="not supported"):
            manager.get_code("nonexistent_code")

    def test_run_logical_lifecycle(self, manager):
        result = manager.run_logical_lifecycle("repetition", error_type="X", error_qubit=0)
        assert result["phase"] == "Recovery Complete"
        assert result["success"] is True
        assert "syndrome_detected" in result
        assert "error_corrected" in result

    def test_simulate_fault_tolerant_workflow(self, manager):
        result = manager.simulate_fault_tolerant_workflow("steane")
        assert result["code"] == "steane"
        assert result["qubits"] == 7
        assert "syndrome_sample" in result
        assert result["fidelity_estimate"] > 0

    def test_run_4d_discovery_audit(self, manager):
        result = manager.run_4d_discovery_audit()
        assert result["dimension"] == "4D"
        assert result["status"] == "PASS"
