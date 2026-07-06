"""
Test script for QAOA implementation — MaxCut on 2 nodes.
"""

from __future__ import annotations

from superfermion.algorithms.variational import QAOA

def test_qaoa_maxcut():
    print("Testing QAOA on MaxCut (2-node graph)...")

    # MaxCut on 2-node graph with single edge (0,1)
    qaoa = QAOA(n_qubits=2, edges=[(0, 1)], p_layers=1)

    # Run the optimization
    results = qaoa.minimize(iterations=100)

    max_cut = results.optimal_value
    print(f"  Max Cut Value: {max_cut:.6f}")

    # For MaxCut on 2 nodes with single edge, the max cut is 1.0
    # QAOA with p=1 should reach this or get close
    assert max_cut > 0.8
    print("[PASS] QAOA MaxCut test passed!")

if __name__ == "__main__":
    try:
        test_qaoa_maxcut()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
