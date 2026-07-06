"""
Test script for the Superfermion Python Compiler.
"""

from __future__ import annotations

import superfermion as sf
import numpy as np

def test_gate_cancellation():
    print("Testing Gate Cancellation...")
    
    # H * H should be identity
    c1 = sf.Circuit(1).h(0).h(0)
    print(f"Original gates: {len(c1._gates)}")
    assert len(c1._gates) == 2
    
    compiled = sf.compile(c1)
    print(f"Compiled gates: {len(compiled._gates)}")
    assert len(compiled._gates) == 0
    print("[PASS] H*H cancellation passed!")
    
    # X*X*X should be just X
    c2 = sf.Circuit(1).x(0).x(0).x(0)
    compiled2 = sf.compile(c2)
    print(f"Compiled gates (X*X*X): {len(compiled2._gates)}")
    assert len(compiled2._gates) == 1
    print("[PASS] X*X*X reduction passed!")

def test_swap_decomposition():
    print("\nTesting SWAP Decomposition...")
    
    c = sf.Circuit(2).swap(0, 1)
    print(f"Original gates: {len(c._gates)}")
    assert len(c._gates) == 1
    
    compiled = sf.compile(c)
    print(f"Compiled gates: {len(compiled._gates)}")
    # SWAP -> 3 CNOTs
    assert len(compiled._gates) == 3
    print(f"Gate set: {[g.name for g in compiled._gates]}")
    assert compiled._gates[0].name == "CNOT"
    print("[PASS] SWAP decomposition passed!")

if __name__ == "__main__":
    try:
        test_gate_cancellation()
        test_swap_decomposition()
        print("\nPython Compiler (Fallback) logic verified!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
