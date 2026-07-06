"""
Test CLI + Bridges (Phase 4: Session 25-26).
"""

from __future__ import annotations
import sys


def test_cli_info():
    """Test the CLI info command."""
    print("Testing CLI info command...")
    from superfermion.cli import cmd_info
    cmd_info()
    print("[PASS] CLI info command verified.")


def test_cli_version():
    """Test the CLI version command."""
    print("\nTesting CLI version command...")
    from superfermion.cli import cmd_version
    cmd_version()
    print("[PASS] CLI version command verified.")


def test_qasm_bridge():
    """Test QASM import."""
    print("\nTesting QASM bridge...")
    from superfermion.bridge import from_qasm
    
    qasm = """
    OPENQASM 3.0;
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    """
    
    circuit = from_qasm(qasm)
    assert circuit.n_qubits == 2
    assert circuit.gate_count == 2
    print(f"  QASM -> Circuit: {circuit.n_qubits} qubits, {circuit.gate_count} gates")
    
    # Verify round-trip: Circuit -> QASM -> Circuit
    qasm_out = circuit.to_qasm3()
    circuit2 = from_qasm(qasm_out)
    assert circuit2.gate_count == circuit.gate_count
    print(f"  Round-trip verified: {circuit2.gate_count} gates")
    
    print("[PASS] QASM bridge verified.")


def test_qasm_parametric():
    """Test QASM parametric import."""
    print("\nTesting QASM parametric import...")
    from superfermion.bridge import from_qasm
    
    qasm = """
    OPENQASM 3.0;
    qubit[1] q;
    rx(1.5707) q[0];
    ry(3.1416) q[0];
    """
    
    circuit = from_qasm(qasm)
    assert circuit.n_qubits == 1
    assert circuit.gate_count == 2
    print(f"  Parametric QASM: {circuit.gate_count} gates")
    print("[PASS] QASM parametric bridge verified.")


def test_qasm_complex():
    """Test more complex QASM circuits."""
    print("\nTesting complex QASM circuit...")
    from superfermion.bridge import from_qasm
    
    qasm = """
    OPENQASM 3.0;
    qubit[3] q;
    h q[0];
    h q[1];
    h q[2];
    cx q[0], q[1];
    cx q[1], q[2];
    rz(0.785) q[0];
    ry(1.571) q[1];
    """
    
    circuit = from_qasm(qasm)
    assert circuit.n_qubits == 3
    assert circuit.gate_count == 7
    print(f"  Complex QASM: {circuit.n_qubits} qubits, {circuit.gate_count} gates, depth={circuit.depth}")
    print("[PASS] Complex QASM bridge verified.")


if __name__ == "__main__":
    try:
        test_cli_info()
        test_cli_version()
        test_qasm_bridge()
        test_qasm_parametric()
        test_qasm_complex()
        print("\nSession 25-26: CLI + Bridges officially ready.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
