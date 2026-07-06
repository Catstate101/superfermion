"""
Test Encodings and Ansatze modules.
"""

from __future__ import annotations
import jax.numpy as jnp
import superfermion as sf
from superfermion.qml.encoding import angle_encoding, basis_encoding, iqp_encoding
from superfermion.qml.ansatz import hardware_efficient, strongly_entangling, two_local


def test_encodings():
    print("Testing Encodings...")
    
    # Angle encoding
    data = jnp.array([0.1, 0.2, 0.3])
    c_angle = angle_encoding(3, data, rotation="RY")
    assert c_angle.n_qubits == 3
    assert c_angle.gate_count == 3
    print(f"  Angle encoding: 3 qubits, {c_angle.gate_count} gates")
    
    # Basis encoding
    c_basis = basis_encoding(3, 5)  # 5 is 101 in binary
    assert c_basis.gate_count == 2  # X on q0 and q2
    print(f"  Basis encoding (5 -> 101): {c_basis.gate_count} X gates")
    
    # IQP encoding
    c_iqp = iqp_encoding(2, jnp.array([0.5, 0.5, 0.5]))
    # Reps=1: 2xH, 2xRZ, 1xRZZInteraction (2xCX + 1xRZ) = 7 gates total
    assert c_iqp.n_qubits == 2
    print(f"  IQP encoding: {c_iqp.gate_count} gates")
    
    print("[PASS] Encodings verified.")


def test_ansatze():
    print("\nTesting Ansatze...")
    
    # HEA
    c_hea = hardware_efficient(2, layers=1)
    # Layer 0: 2xRY, 2xRZ, 1xCX. Final layer: 2xRY. Total = 7 gates.
    assert c_hea.n_qubits == 2
    assert len(c_hea.parameters) == 6 # (2 RY + 2 RZ) * 1 + 2 final RY
    print(f"  Hardware Efficient: {c_hea.gate_count} gates, {len(c_hea.parameters)} params")
    
    # Strongly Entangling
    c_sea = strongly_entangling(2, layers=2)
    # Each layer: 3 rotations per qubit (RY, RZ, RY) + 2 CX (cyclic 0->1, 1->0).
    # 2 layers: 2 * (3*2 + 2) = 16 gates.
    assert c_sea.gate_count == 16
    assert len(c_sea.parameters) == 12 # 3*2 * 2
    print(f"  Strongly Entangling: {c_sea.gate_count} gates, {len(c_sea.parameters)} params")
    
    # Two-Local
    c_tl = two_local(3, rotation_gates=["RY"], entanglement_gates="CZ", reps=2, entanglement="linear")
    # 3 reps of (rotation layer + entanglement layer) + 1 final rotation layer.
    # Actually my implementation does: r in range(reps + 1).
    # If r < reps: add entanglement.
    # Rotation layers: 3 qubits * 1 gate * (2 + 1) = 9 rotations.
    # Entanglement layers: 2 qubits-pairs * 1 gate * 2 = 4 CZs.
    # Total = 13 gates.
    assert c_tl.gate_count == 13
    assert len(c_tl.parameters) == 9
    print(f"  Two-Local (linear): {c_tl.gate_count} gates, {len(c_tl.parameters)} params")
    
    print("[PASS] Ansatze verified.")


if __name__ == "__main__":
    try:
        test_encodings()
        test_ansatze()
        print("\nQML Components (Phase 5/6) Verified.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
