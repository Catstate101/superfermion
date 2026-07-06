"""Tutorial 8 — Hardware compilation & transpile.

Shows how to:
  * build a logical circuit,
  * describe a target device with ``HardwareSpec``,
  * run ``sf.compile`` to decompose / route / cancel gates.
"""
from __future__ import annotations

import superfermion as sf
from superfermion.compiler.manager import HardwareSpec


def main() -> dict:
    # Logical 3-qubit GHZ circuit
    c = sf.Circuit(3).h(0).cx(0, 1).cx(1, 2)

    # Toy target device: linear 3-qubit chain, CX-only entangler
    target = HardwareSpec(
        name="toy_linear",
        n_qubits=3,
        native_gates=["RZ", "SX", "CX"],
        coupling_map=[(0, 1), (1, 2)],
    )

    compiled = sf.compile(c, level=1, target=target)

    report = {
        "logical_gates":  c.gate_count,
        "compiled_gates": compiled.gate_count,
        "logical_depth":  c.depth,
        "compiled_depth": compiled.depth,
    }
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    main()
