"""SecurityValidator — qubit limits, depth limits, and DoS protection."""

from __future__ import annotations

import superfermion as sf


class SecurityValidator:
    """Validates circuits for security and resource constraints.

    Enforces qubit caps, depth limits, and quota checks independently
    of routing logic.
    """

    MAX_QUBITS = 40
    MAX_DEPTH = 10000

    def validate(self, circuit: sf.Circuit) -> bool:
        """Check if the circuit is safe to execute.

        Raises:
            PermissionError: If the circuit exceeds limits.
        """
        if circuit.n_qubits > self.MAX_QUBITS:
            raise PermissionError(
                f"Circuit exceeds maximum security limit ({self.MAX_QUBITS} qubits). "
                "Contact enterprise support."
            )

        if circuit.depth > self.MAX_DEPTH:
            raise PermissionError("Circuit depth too high. Performance risk detected.")

        return True
