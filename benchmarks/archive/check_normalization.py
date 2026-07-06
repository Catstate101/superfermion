
import superfermion as sf
from superfermion.circuit import Circuit
import os

print("--- GATE CASE NORMALIZATION TEST ---")
c = sf.Circuit(1)
c._add_gate("h", [0])
c._add_gate("Rx", [0], [3.14])
c.measure(0, 0)

all_upper = True
for g in c._gates:
    print(f"Gate: {g.name}")
    if g.name != g.name.upper():
        all_upper = False

if all_upper:
    print("RESULT: SUCCESS - All gates normalized to UPPERCASE")
else:
    print("RESULT: FAILED - Case normalization issue found")

print("\n--- SIMULATOR COMPATIBILITY TEST ---")
try:
    from superfermion.simulator import simulate_statevector
    state = simulate_statevector(c)
    print("RESULT: SUCCESS - Simulator handled normalized names")
except Exception as e:
    print(f"RESULT: FAILED - Simulator error: {e}")

print("\n--- BRIDGE EXPORT TEST ---")
try:
    from superfermion.bridge import to_qiskit
    qc = to_qiskit(c)
    print("RESULT: SUCCESS - Bridge exported normalized names")
except Exception as e:
    print(f"RESULT: FAILED - Bridge error: {e}")
