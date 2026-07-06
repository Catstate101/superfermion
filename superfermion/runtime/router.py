"""BackendRouter — pure routing logic, separated from security and fallback concerns."""

from __future__ import annotations

from typing import Optional

import superfermion as sf


class BackendRouter:
    """Determines the most efficient backend for a given circuit.

    Decisions are based purely on qubit count, accelerator availability,
    and cluster mode — no security or quota checks.
    """

    SV_MAX = 28
    MPS_MIN = 29
    CLUSTER_MIN = 35
    QPU_MIN = 127

    def select(self, circuit: sf.Circuit, requested_target: Optional[str] = None) -> str:
        """Select the best backend for the circuit.

        Args:
            circuit: The circuit to route.
            requested_target: If set and not "auto", overrides all routing.
        """
        n = circuit.n_qubits

        if requested_target and requested_target != "auto":
            return requested_target

        import jax
        try:
            accel = jax.default_backend()
        except Exception:
            accel = "cpu"

        import os
        is_cluster = os.environ.get("SF_CLUSTER_MODE") == "1" or os.environ.get("SLURM_JOB_ID") is not None

        if n <= self.SV_MAX:
            if is_cluster:
                return "cluster"
            return "jax"
        elif n <= self.QPU_MIN:
            if accel in ("gpu", "tpu"):
                return "jax_mps"
            return "mps"
        else:
            sf.utils.info(f"Qubit count {n} beyond classical limits. Routing to cloud hardware...")
            return "ibm"
