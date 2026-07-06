"""
Unit test for QEC Surface Code implementation.
"""

import superfermion as sf
from superfermion.qec.codes.surface import SurfaceCode

def test_surface_code_scaling():
    sc = SurfaceCode(distance=3)
    assert sc.n_data == 9
    assert sc.n_measure == 8
    assert sc.n_total == 17

def test_syndrome_extraction_circuit():
    sc = SurfaceCode(distance=3)
    c = sc.build_syndrome_extraction()
    
    # Check if gates were added
    assert c.gate_count > 0
    assert c.n_qubits == 17
    
    # Check for H and CX gates specifically
    qasm = c.to_qasm3()
    assert "h q[" in qasm
    assert "cx q[" in qasm
    assert "measure q[" in qasm

if __name__ == "__main__":
    test_surface_code_scaling()
    test_syndrome_extraction_circuit()
    print("QEC Surface Code: PASS")
