"""
Demonstrating NATIVE Superfermion Plotting.
This script uses the built-in result.plot() method.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import superfermion as sf
import numpy as np

print("=== Superfermion Native Plotting Demo ===")

# 1. Create a circuit
c = sf.Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
print(f"Executing: {c}")

# 2. Run simulation
result = sf.run(c, shots=1024)

# 3. Use the NATIVE plot() method provided by Superfermion's RunResult class
print("\nSaving native plot to: notebooks/native_result_plot.png")
result.plot(save_path=os.path.join('notebooks', 'native_result_plot.png'))

print("\nOK: Native plot generated. Superfermion handles quantum measurement plots automatically.")
print("For custom ML Curves (Loss/Accuracy), the Superfermion-Matplotlib integration is used.")
