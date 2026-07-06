"""Tutorial 7 — Logical qubit lifecycle with QEC.

Walks through encode → inject error → measure syndrome → correct on the
Steane [[7,1,3]] code using ``QECManager``.
"""
from __future__ import annotations

from superfermion.qec import QECManager


def main() -> bool:
    mgr = QECManager()

    # Run the full lifecycle: encode |0_L>, inject X on qubit 0,
    # decode, verify logical state preserved.
    result = mgr.run_logical_lifecycle("steane", error_type="X", error_qubit=0)

    print("Steane code logical lifecycle:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return bool(result["success"])


if __name__ == "__main__":
    main()
