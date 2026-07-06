"""
Test script for Superfermion Circuit Serialization.
"""

from __future__ import annotations

import superfermion as sf
import os

def test_json_serialization():
    print("Testing JSON Serialization...")
    
    # 1. Build a circuit with mixed gates and params
    c = sf.Circuit(2).h(0).cx(0, 1).rx(0.5, 0).rz(1.5, 1)
    orig_json = c.to_json()
    print(f"Serialized JSON size: {len(orig_json)} bytes")
    
    # 2. Reconstruct from JSON
    c_new = sf.Circuit.from_json(orig_json)
    
    # 3. Verify parity
    print(f"Original: {c}")
    print(f"Reconstructed: {c_new}")
    
    assert c_new.n_qubits == c.n_qubits
    assert c_new.depth == c.depth
    assert len(c_new._gates) == len(c._gates)
    
    for g1, g2 in zip(c._gates, c_new._gates):
        assert g1.name == g2.name
        assert g1.qubits == g2.qubits
        assert g1.params == g2.params

    print("[PASS] JSON serialization integrity verified.")

def test_qasm3_export():
    print("\nTesting OpenQASM 3.0 Export...")
    c = sf.Circuit(2).h(0).cx(0, 1)
    qasm = c.to_qasm3()
    print("Exported QASM 3.0:")
    print("-" * 20)
    print(qasm)
    print("-" * 20)
    
    assert "OPENQASM 3.0;" in qasm
    assert "qubit[2] q;" in qasm
    assert "h q[0];" in qasm
    assert "cx q[0], q[1];" in qasm
    print("[PASS] QASM 3.0 export verified.")

if __name__ == "__main__":
    try:
        test_json_serialization()
        test_qasm3_export()
        print("\nCircuits are officially serializable and portable.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
