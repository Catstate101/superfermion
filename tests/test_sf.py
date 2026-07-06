import superfermion as sf
import numpy as np

def test_bell_state():
    print("Testing Bell State...")
    # Create a Bell state circuit
    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    print(circuit)
    print(circuit.draw())
    
    # Run the simulation
    result = sf.run(circuit, shots=1000)
    print(f"Counts: {result.counts}")
    
    # Check if we got mostly 00 and 11
    counts = result.counts
    total_bell = counts.get('00', 0) + counts.get('11', 0)
    print(f"Total Bell outcomes (00 + 11): {total_bell}/1000")
    assert total_bell > 900
    print("[PASS] Bell State test passed!")

def test_parameterized():
    print("\nTesting Parameterized Circuit...")
    theta = sf.param("theta")
    circuit = sf.Circuit(1).rx(theta, 0)
    
    # Try running without binding - should fail
    try:
        sf.run(circuit)
        print("FAIL: Should have raised RuntimeError for unbound params")
        assert False
    except RuntimeError as e:
        print(f"[PASS] Correctly caught unbound parameter: {e}")
        
    # Bind and run
    bound = circuit.bind({"theta": np.pi}) # Rx(pi) |0> -> -i|1>
    result = sf.run(bound, shots=100)
    print(f"Rx(pi) Counts: {result.counts}")
    assert result.counts.get('1', 0) == 100
    print("[PASS] Parameterized test passed!")

if __name__ == "__main__":
    try:
        test_bell_state()
        test_parameterized()
        print("\nAll Python tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
