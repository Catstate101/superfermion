"""Tests for the three backend fixes applied 2026-04-15.

1. sf.singularity — seed was passed both positionally and via **kwargs at n>=32
   (TypeError: ... got multiple values for argument).
   Fix: kwargs.pop("seed", ...) in SingularityBackend.run.

2. sf.supremacy_core — run() always returned counts={} regardless of shots, so
   the sampled-counts path was completely dead (TVD vs reference was pinned at
   0.5 on every cell). Fix: sample from dense SV when n<=25, otherwise do a
   left-to-right MPS sampler.

3. sf.jax_mps — _sample_batched_jax split the RNG key per site but never
   threaded the split keys into the scan; the SAME key was reused at every
   site, so bits at different qubits were fully correlated. GHZ coincidentally
   looked correct (degenerate conditionals). Fix: pass per-site subkeys as
   scan inputs.

Run:
    PYTHONIOENCODING=utf-8 python -m pytest tests/test_backend_fixes.py -v
or directly:
    PYTHONIOENCODING=utf-8 python tests/test_backend_fixes.py
"""
from __future__ import annotations

import os
import sys
import math
import unittest
from typing import Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _counts_to_probs(counts: Dict[str, int], n_qubits: int) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _ideal_probs_from_sv(sv: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """BE convention: qubit 0 is MSB of the bitstring."""
    probs = np.abs(sv.astype(np.complex128)) ** 2
    probs = probs / probs.sum()
    fmt = f"0{n_qubits}b"
    return {format(i, fmt): float(p) for i, p in enumerate(probs) if p > 0.0}


def _tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


class TestSingularitySeedKwargFix(unittest.TestCase):
    """Bug 1 regression — running at n>=25 must not raise TypeError on seed."""

    def test_singularity_n_eq_26_does_not_raise(self):
        # n=26 falls into the 24<n<=32 branch, which previously triggered the
        # "multiple values for argument 'seed'" TypeError.
        from superfermion.backends.singularity import SingularityBackend
        from superfermion.circuit import Circuit

        n = 26
        c = Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)

        backend = SingularityBackend()
        # Explicit seed kwarg is the exact scenario that tripped the old bug.
        result = backend.run(c, shots=64, seed=123)
        self.assertIsNotNone(result)
        self.assertEqual(sum(result.counts.values()), 64)
        # GHZ state: only |0...0> and |1...1> should have mass.
        for bitstring in result.counts:
            self.assertIn(bitstring, {"0" * n, "1" * n})


class TestSingularityDenseSvMemoryGuard(unittest.TestCase):
    """Bug 1b — Rust regime advertised 24<N<=32 without checking RAM.
    After the fix, the router must route to MPS when a 2^n SV would blow
    the memory budget. This guards the user's stated preference for
    graceful CPU/RAM fallback over OOM crashes."""

    def test_dense_sv_fits_gates_on_available_ram(self):
        from superfermion.backends.routing.singularity_router import SingularityRouter
        # n=20 is tiny (~50 MB SV) — must always fit regardless of host.
        self.assertTrue(SingularityRouter.dense_sv_fits(20))
        # n=40 is 17.6 TB of SV — must never fit regardless of host.
        self.assertFalse(SingularityRouter.dense_sv_fits(40))


class TestSupremacySamplingFix(unittest.TestCase):
    """Bug 2 regression — counts must be non-empty and match the statevector."""

    def test_ghz_counts_populated_and_near_ideal(self):
        from superfermion.backends.supremacy_core import SupremacyBackend
        from superfermion.circuit import Circuit

        n = 4
        c = Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)

        backend = SupremacyBackend()
        shots = 20000
        result = backend.run(c, shots=shots, seed=7)

        # Before the fix this was {} — empty counts.
        self.assertGreater(len(result.counts), 0, "supremacy produced no counts")
        self.assertEqual(sum(result.counts.values()), shots)

        # Compare sampled distribution against ideal SV probs.
        sv = np.asarray(result.statevector)
        ideal = _ideal_probs_from_sv(sv, n)
        sampled = _counts_to_probs(result.counts, n)
        # TVD noise floor at 20k shots is ~1/sqrt(N) ≈ 0.007; 0.03 is generous.
        tvd = _tvd(sampled, ideal)
        self.assertLess(tvd, 0.03, f"TVD vs ideal too high: {tvd}")

    def test_random_circuit_counts_match_sv(self):
        """Asymmetric circuit — catches the TVD=0.5 empty-counts bug distinctly."""
        from superfermion.backends.supremacy_core import SupremacyBackend
        from superfermion.circuit import Circuit

        n = 5
        c = Circuit(n)
        c.h(0); c.h(1); c.h(2); c.h(3); c.h(4)
        c.cx(0, 1); c.cx(1, 2); c.cx(2, 3); c.cx(3, 4)
        c.rz(0.7, 2); c.ry(0.3, 0); c.rx(-0.5, 4)

        backend = SupremacyBackend()
        shots = 20000
        result = backend.run(c, shots=shots, seed=11)

        self.assertEqual(sum(result.counts.values()), shots)
        sv = np.asarray(result.statevector)
        ideal = _ideal_probs_from_sv(sv, n)
        sampled = _counts_to_probs(result.counts, n)
        tvd = _tvd(sampled, ideal)
        self.assertLess(tvd, 0.03, f"TVD vs ideal too high: {tvd}")


class TestJaxMpsSamplingFix(unittest.TestCase):
    """Bug 3 regression — per-site RNG keys must produce correct marginals."""

    def test_random_circuit_observable_matches_ideal(self):
        """Bug manifested as obs_linf 0.5-0.95 on random/qft/vqe circuits."""
        from superfermion.backends.jax_mps import JAXMPSBackend
        from superfermion.circuit import Circuit

        n = 4
        c = Circuit(n)
        c.h(0); c.h(1); c.h(2); c.h(3)
        c.cx(0, 1); c.cx(2, 3); c.cx(1, 2)
        c.rz(0.9, 0); c.ry(-0.4, 2)

        backend = JAXMPSBackend()
        shots = 20000
        result = backend.run(c, shots=shots, seed=5)

        self.assertEqual(sum(result.counts.values()), shots)
        sampled = _counts_to_probs(result.counts, n)

        # Compare to dense statevector reference (from a brute-force sim path).
        from superfermion.backends.turbo import simulate_statevector_turbo
        sv = np.asarray(simulate_statevector_turbo(c, seed=5), dtype=np.complex128)
        ideal = _ideal_probs_from_sv(sv, n)

        # Observable-Z_i on each qubit — shot-noise-robust correctness proxy.
        # BE: qubit i at bitstring[i], so bit=='0' is +1, bit=='1' is -1.
        def z_expect(probs: Dict[str, float], q: int) -> float:
            s = 0.0
            for bs, p in probs.items():
                s += (1.0 if bs[q] == "0" else -1.0) * p
            return s

        for q in range(n):
            z_s = z_expect(sampled, q)
            z_i = z_expect(ideal, q)
            self.assertLess(
                abs(z_s - z_i), 0.06,
                f"<Z_{q}> mismatch: sampled={z_s:.4f} ideal={z_i:.4f}",
            )

    def test_ghz_still_works(self):
        """GHZ was the one circuit the buggy version accidentally sampled correctly.
        Make sure we did not regress it."""
        from superfermion.backends.jax_mps import JAXMPSBackend
        from superfermion.circuit import Circuit

        n = 4
        c = Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)

        result = JAXMPSBackend().run(c, shots=20000, seed=3)
        for bs in result.counts:
            self.assertIn(bs, {"0" * n, "1" * n})
        p0 = result.counts.get("0" * n, 0) / 20000
        self.assertTrue(0.45 < p0 < 0.55, f"GHZ |0..0> probability off: {p0}")


class TestMemPalaceCliUnicodeFix(unittest.TestCase):
    """Bug 4 regression — `python -m mempalace --help` under cp1252 must not crash."""

    def _mempalace_python(self):
        """Return the path to whichever Python has mempalace installed.

        Searches sys.executable first, then sibling Python installations on
        PATH. Returns None if none found (test will be skipped).
        """
        import subprocess, shutil
        candidates = [sys.executable]
        for name in ("python", "python3", "py"):
            found = shutil.which(name)
            if found and found not in candidates:
                candidates.append(found)
        # Also probe known Windows install paths on this machine
        import glob as _glob
        for pat in [r"C:\Python*\python.exe",
                    r"C:\Users\*\AppData\Local\Programs\Python\Python*\python.exe"]:
            for p in _glob.glob(pat):
                if p not in candidates:
                    candidates.append(p)
        for py in candidates:
            try:
                r = subprocess.run(
                    [py, "-c", "import mempalace"],
                    capture_output=True, timeout=10,
                )
                if r.returncode == 0:
                    return py
            except Exception:
                continue
        return None

    def test_help_under_cp1252_succeeds(self):
        import subprocess
        py = self._mempalace_python()
        if py is None:
            self.skipTest("mempalace not installed in any detected Python")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"
        r = subprocess.run(
            [py, "-m", "mempalace", "--help"],
            capture_output=True, env=env, timeout=30,
        )
        self.assertEqual(r.returncode, 0, f"stderr: {r.stderr[:400]}")


class TestMPSBondSaturationWarnings(unittest.TestCase):
    """Bond-saturation telemetry in sf.supremacy and sf.jax_mps.

    Each MPS backend must:
    - Report bond_saturated=False and fidelity_lower_bound>=0.99 on a
      lossless circuit (GHZ at adequate bond).
    - Report bond_saturated=True  and fidelity_lower_bound<0.99 and emit
      RuntimeWarning when max_bond is too tight for the circuit.
    """

    def _build_volume_law_circuit(self, n: int = 8, depth: int = 6, seed: int = 0):
        """A random U3+CX ladder that builds volume-law entanglement quickly."""
        rng = np.random.default_rng(seed)
        c = __import__("superfermion").circuit.Circuit(n)
        for _ in range(depth):
            for q in range(n):
                c.rx(float(rng.uniform(0, 6.28)), q)
                c.rz(float(rng.uniform(0, 6.28)), q)
            for q in range(0, n - 1, 2):
                c.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                c.cx(q, q + 1)
        return c

    # --- sf.supremacy ---

    def test_supremacy_ghz_no_saturation(self):
        from superfermion.backends.supremacy_core import SupremacyBackend
        from superfermion.circuit import Circuit
        n = 6
        c = Circuit(n); c.h(0)
        for i in range(n - 1): c.cx(i, i + 1)
        r = SupremacyBackend(options={"max_bond": 64}).run(c, shots=500, seed=0)
        self.assertFalse(r.metadata["bond_saturated"])
        self.assertAlmostEqual(r.metadata["fidelity_lower_bound"], 1.0, places=6)

    def test_supremacy_tight_bond_warns_and_saturates(self):
        import warnings
        from superfermion.backends.supremacy_core import SupremacyBackend
        c = self._build_volume_law_circuit(n=8, depth=6, seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = SupremacyBackend(options={"max_bond": 4}).run(c, shots=100, seed=2)
        self.assertTrue(r.metadata["bond_saturated"],
                        "expected bond_saturated=True at max_bond=4 on volume-law circuit")
        self.assertLess(r.metadata["fidelity_lower_bound"], 0.99,
                        "expected fidelity<0.99 at max_bond=4")
        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w),
                        "expected RuntimeWarning on bond saturation")

    # --- sf.jax_mps ---

    def test_jax_mps_ghz_no_saturation(self):
        from superfermion.backends.jax_mps import JAXMPSBackend
        from superfermion.circuit import Circuit
        n = 6
        c = Circuit(n); c.h(0)
        for i in range(n - 1): c.cx(i, i + 1)
        r = JAXMPSBackend(options={"max_bond_dim": 32}).run(c, shots=500, seed=0)
        self.assertFalse(r.metadata["bond_saturated"])
        self.assertGreaterEqual(r.metadata["fidelity_lower_bound"], 0.99)

    def test_jax_mps_tight_bond_warns_and_saturates(self):
        import warnings
        from superfermion.backends.jax_mps import JAXMPSBackend
        c = self._build_volume_law_circuit(n=8, depth=6, seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = JAXMPSBackend(options={"max_bond_dim": 4}).run(c, shots=100, seed=2)
        self.assertTrue(r.metadata["bond_saturated"],
                        "expected bond_saturated=True at max_bond_dim=4")
        self.assertLess(r.metadata["fidelity_lower_bound"], 0.99,
                        "expected fidelity<0.99 at max_bond_dim=4")
        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w),
                        "expected RuntimeWarning on bond saturation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
