"""FallbackOrchestrator — offline fallback chain: hardware → GPU sim → CPU sim."""

from __future__ import annotations

import os
from typing import Optional

import superfermion as sf


class FallbackOrchestrator:
    """Guarantees a result can always be produced by walking the fallback chain."""

    SV_MAX = 28
    QPU_MIN = 127

    def resolve(self, circuit: sf.Circuit, preferred_backend: str = "auto") -> str:
        """Return the first available backend in the fallback chain.

        Chain: preferred → hardware (IBM/IonQ) → GPU sim → CPU sim
        """
        n = circuit.n_qubits

        if preferred_backend != "auto" and self._backend_available(preferred_backend):
            return preferred_backend

        # Try hardware if at QPU scale
        if n >= self.QPU_MIN:
            for hw in ("ibm", "ionq"):
                if self._backend_available(hw):
                    return hw

        # Try GPU-accelerated
        try:
            import jax
            if jax.default_backend() in ("gpu", "tpu"):
                return "jax" if n <= self.SV_MAX else "jax_mps"
        except Exception:
            pass

        # CPU fallback
        return "statevector" if n <= self.SV_MAX else "mps"

    def _backend_available(self, name: str) -> bool:
        """Check if a backend can be instantiated or has credentials configured."""
        try:
            from superfermion.backends.registry import BackendRegistry
            BackendRegistry.get_backend(name)
            return True
        except (ImportError, ValueError):
            pass
        env_var = {"ibm": "SF_IBM_TOKEN", "ionq": "SF_IONQ_API_KEY"}.get(name, "")
        return bool(os.environ.get(env_var, ""))
