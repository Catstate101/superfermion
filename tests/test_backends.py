"""
Superfermion Quantum Compiler & Transpilation Session.
"""

from __future__ import annotations

import superfermion as sf
import numpy as np

def test_multi_backend_dispatch():
    print("Testing Multi-Backend Dispatch & Registry...")
    
    # Check that we can list backends
    backends = sf.list_backends()
    print(f"Available backends: {backends}")
    assert "statevector" in backends
    assert "cuda" in backends
    assert "mps" in backends
    
    # Create a circuit
    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    
    # Run with default backend
    res1 = sf.run(circuit)
    print(f"Default run result: {res1}")
    assert res1.shots == 1000
    
    # Run with explicit statevector backend
    res2 = sf.run(circuit, backend="statevector", shots=500)
    print(f"Explicit statevector result: {res2}")
    assert res2.shots == 500
    
    # MPS is now fully implemented — verify it runs.
    res3 = sf.run(circuit, backend="mps", shots=256)
    print(f"MPS result: {res3}")
    assert res3.shots == 256

    print("[PASS] Multi-Backend Dispatch tests passed!")

if __name__ == "__main__":
    try:
        test_multi_backend_dispatch()
        print("\nSession 3 (Multi-Backend) integration verified.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
