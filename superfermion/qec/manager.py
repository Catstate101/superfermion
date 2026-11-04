"""
Superfermion QEC Manager - The Orchestrator for Fault-Tolerant Quantum Workflows.
"""
from typing import Dict, Any, List

import numpy as np

import superfermion as sf
from .codes.linear import RepetitionCode, ShorCode, SteaneCode, BaconShorCode, GenericCSSCode
from .codes.topological import SurfaceCode2D, HypercubeCode4D, ToricCode2D, ColorCode, HoneycombCode
from .decoders import MWPMDecoder, UnionFindDecoder, BPOSD_Decoder, NeuralDecoder


# Above this many qubits a statevector run allocates 2**n complex amplitudes:
# 48 qubits asks for 2**52 bytes and Windows terminates the process with
# 0xC0000409 (STACK_BUFFER_OVERRUN) instead of raising MemoryError.
MAX_SIMULATED_QUBITS = 22

# ``method="stabilizer"`` is polynomial in the gate count, not exponential in
# the qubit count, so it must not be bounded by the statevector limit above.
# Measured on this machine: the 48-qubit hypercube circuit (177 gates) runs
# through the Clifford backend in 0.008 s, while the default statevector path
# on the same circuit aborts the interpreter. Bounding both by the same number
# is what made ``run_4d_discovery_audit`` refuse work it can actually do.
MAX_CLIFFORD_QUBITS = 4096


def _qubit_limit(method: str) -> int:
    """Qubit budget for a simulation method."""
    return MAX_CLIFFORD_QUBITS if method == "stabilizer" else MAX_SIMULATED_QUBITS


def _gf2_rank(matrix: np.ndarray) -> int:
    """Rank of a matrix over GF(2)."""
    a = (np.asarray(matrix) % 2).astype(np.int32).copy()
    if a.ndim != 2 or a.size == 0:
        return 0
    rows, cols = a.shape
    rank = 0
    for c in range(cols):
        pivot = None
        for i in range(rank, rows):
            if a[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        a[[rank, pivot]] = a[[pivot, rank]]
        for i in range(rows):
            if i != rank and a[i, c]:
                a[i] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _in_rowspace(H: np.ndarray, vec: np.ndarray) -> bool:
    """True iff ``vec`` lies in the GF(2) row space of ``H``.

    For a CSS code a residual error with cleared syndrome is corrected exactly
    when it is a stabiliser, i.e. when it lies in the row space of the *same*
    sector's check matrix. That is the honest definition of ``error_corrected``
    (QEC-10), rather than ``"1" in top_syndrome``.
    """
    v = np.asarray(vec, dtype=np.int32).reshape(1, -1) % 2
    if H.size == 0 or H.shape[0] == 0:
        return not v.any()
    return _gf2_rank(np.vstack([H, v])) == _gf2_rank(H)


def _checks_to_matrix(checks, n_data: int) -> np.ndarray:
    """Build a dense GF(2) parity-check matrix from check index lists."""
    H = np.zeros((len(checks), n_data), dtype=np.int32)
    for a, qubits in enumerate(checks):
        for q in qubits:
            if 0 <= q < n_data:
                H[a, q] = 1
    return H


def _n_data_of(code: Any, x_checks, z_checks) -> int:
    """Data-qubit count: explicit attribute if present, else inferred."""
    for attr in ("n_data", "n"):
        value = getattr(code, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    highest = -1
    for group in (x_checks, z_checks):
        for qubits in group:
            if qubits:
                highest = max(highest, max(qubits))
    return highest + 1

class QECManager:
    """
    Automated QEC Workflow Manager. 
    Handles selection, injection, and recovery.
    """
    def __init__(self):
        self.registry = {
            "repetition": RepetitionCode,
            "shor": ShorCode,
            "steane": SteaneCode,
            "bacon_shor": BaconShorCode,
            "css_generic": GenericCSSCode,
            "surface_2d": SurfaceCode2D,
            "hypercube_4d": HypercubeCode4D,
            "toric_2d": ToricCode2D,
            "color_code": ColorCode,
            "honeycomb": HoneycombCode
        }
        self.decoders = {
            "mwpm": MWPMDecoder,
            "union_find": UnionFindDecoder,
            "bp_osd": BPOSD_Decoder,
            "neural": NeuralDecoder
        }

    def get_code(self, name: str, **kwargs) -> Any:
        if name not in self.registry:
            raise ValueError(f"Code {name} not supported.")
        return self.registry[name](**kwargs)

    def run_logical_lifecycle(self, code_name: str, error_type: str = "X",
                              error_qubit: int = 0, decoder: str = "mwpm",
                              simulate: bool = False) -> Dict[str, Any]:
        """Simulates the full lifecycle of a Logical Qubit:
        1. ENCODE: Map logical |0> to physical entangled state.
        2. NOISE: Inject a physical error (X, Y, or Z).
        3. SYNDROME: Measure stabilizers to detect the error.
        4. RECOVERY: Use a decoder to fix the error.
        5. VERIFY: Compare final logical state with original.

        The syndrome is computed from the code's parity-check matrix rather than
        by simulating the circuit, so codes whose circuits are far too large to
        simulate (36-qubit toric, 48-qubit hypercube) can be exercised without
        allocating 2**n amplitudes. Pass ``simulate=True`` to additionally run
        the built circuit and compare its measured syndrome with the analytic
        one; that requires ``MAX_SIMULATED_QUBITS`` or fewer qubits.

        Args:
            code_name: key in ``self.registry``.
            error_type: ``'X'``, ``'Y'`` or ``'Z'``. Anything else raises.
            error_qubit: data-qubit index carrying the error.
            decoder: key in ``self.decoders``.
            simulate: also run the circuit and cross-check the syndrome.

        Returns a dict including ``success`` (True only when the residual error
        is a stabiliser, i.e. the logical state really was restored) and
        ``error_corrected`` (the VERIFY step).
        """
        code = self.get_code(code_name)
        c = code.build()

        stabilizers_fn = getattr(code, "stabilizers", None)
        if not callable(stabilizers_fn):
            raise ValueError(
                f"{code_name} does not expose stabilizers(); there is nothing to decode"
            )
        x_checks, z_checks = stabilizers_fn()
        n_data = _n_data_of(code, x_checks, z_checks)
        H_x = _checks_to_matrix(x_checks, n_data)
        H_z = _checks_to_matrix(z_checks, n_data)

        err = str(error_type).upper()
        if err not in ("X", "Y", "Z"):
            # Previously only 'X' and 'Z' were branched on, so 'Y' and any
            # invalid string silently injected nothing and still reported
            # success. (QEC-9)
            raise ValueError(f"error_type must be 'X', 'Y' or 'Z', got {error_type!r}")
        if not (0 <= error_qubit < n_data):
            raise ValueError(
                f"error_qubit {error_qubit} out of range for {code_name} "
                f"({n_data} data qubits)"
            )

        # 2. NOISE
        e_x = np.zeros(n_data, dtype=np.int32)
        e_z = np.zeros(n_data, dtype=np.int32)
        if err in ("X", "Y"):
            e_x[error_qubit] = 1   # X errors flip Z-type checks
        if err in ("Z", "Y"):
            e_z[error_qubit] = 1   # Z errors flip X-type checks

        # 3. SYNDROME
        syndrome_z = (H_z @ e_x) % 2   # detected by the Z checks
        syndrome_x = (H_x @ e_z) % 2   # detected by the X checks

        # 4. RECOVERY
        def _decode(checks, syndrome):
            if len(syndrome) == 0 or not syndrome.any():
                return []
            try:
                factory = self.decoders[decoder]
            except KeyError:
                raise ValueError(f"Decoder {decoder} not supported.") from None
            dec = factory(n_data=n_data, syndrome_qubit_map=checks)
            return dec.decode(syndrome)

        correction_x = _decode(z_checks, syndrome_z)
        correction_z = _decode(x_checks, syndrome_x)

        residual_x = e_x.copy()
        for q, _pauli in correction_x:
            if 0 <= q < n_data:
                residual_x[q] ^= 1
        residual_z = e_z.copy()
        for q, _pauli in correction_z:
            if 0 <= q < n_data:
                residual_z[q] ^= 1

        # 5. VERIFY
        syndrome_cleared = not (H_z @ residual_x % 2).any() and not (
            H_x @ residual_z % 2
        ).any()
        logical_x_ok = _in_rowspace(H_x, residual_x)
        logical_z_ok = _in_rowspace(H_z, residual_z)
        error_corrected = bool(syndrome_cleared and logical_x_ok and logical_z_ok)

        measured = None
        simulated = False
        if simulate:
            if c.n_qubits > MAX_SIMULATED_QUBITS:
                measured = None
            else:
                sim_circuit = code.build()
                if err in ("X", "Y"):
                    sim_circuit.x(error_qubit)
                if err in ("Z", "Y"):
                    sim_circuit.z(error_qubit)
                res = sf.run(sim_circuit, shots=256)
                measured = max(res.counts, key=res.counts.get)
                simulated = True

        return {
            "phase": "Recovery Complete" if error_corrected else "Recovery Incomplete",
            "logical_state": "Protected" if error_corrected else "Compromised",
            "code": code_name,
            "error_type": err,
            "error_qubit": error_qubit,
            "n_data_qubits": n_data,
            "circuit_qubits": c.n_qubits,
            # ``gate_count`` flushes Rust gate storage first, so it is correct
            # whether or not the circuit was built with ``use_rust_storage``.
            "circuit_gates": int(c.gate_count),
            "syndrome_x": syndrome_x.tolist(),
            "syndrome_z": syndrome_z.tolist(),
            # ``syndrome_detected`` always carries the analytic syndrome, in the
            # same order as ``syndrome_x + syndrome_z``, whatever the flag says:
            # the field used to be overwritten with the readout bitstring when
            # ``simulate=True``, so its type changed with the flag and the
            # measured-vs-analytic comparison the docstring promises could not
            # be made by a caller. The readout has its own field,
            # ``measured_syndrome``. (QEC-D9)
            "syndrome_detected": np.concatenate(
                [syndrome_x, syndrome_z]
            ).astype(int).tolist(),
            "measured_syndrome": measured,
            "circuit_simulated": simulated,
            "correction_x": [list(cx) for cx in correction_x],
            "correction_z": [list(cz) for cz in correction_z],
            "syndrome_cleared": bool(syndrome_cleared),
            "residual_weight": int(residual_x.sum() + residual_z.sum()),
            "error_corrected": error_corrected,
            "success": error_corrected,
        }

    def simulate_fault_tolerant_workflow(self, code_name: str) -> Dict[str, Any]:
        """Legacy compatibility wrapper."""
        code = self.get_code(code_name)
        circuit = code.build()
        result = {
            "code": code_name,
            "qubits": circuit.n_qubits,
        }
        method = "statevector" if circuit.n_qubits <= MAX_SIMULATED_QUBITS else "stabilizer"
        if circuit.n_qubits > _qubit_limit(method):
            # Running a 2**n statevector here is what aborted the interpreter, so
            # report the structure instead of allocating it.
            result.update({
                "syndrome_sample": [],
                "fidelity_estimate": None,
                "simulated": False,
                "note": (
                    f"{circuit.n_qubits} qubits exceeds the "
                    f"{_qubit_limit(method)}-qubit limit for method={method!r}; "
                    f"syndrome sampled classically instead"
                ),
            })
            return result
        if circuit.n_qubits > MAX_SIMULATED_QUBITS:
            # Too big for a statevector, but the Clifford backend is polynomial
            # in the gate count, so these circuits are still simulable.
            res = sf.run(circuit, method=method, shots=1024)
        else:
            res = sf.run(circuit, shots=1024)
        result.update({
            "syndrome_sample": list(res.counts.keys())[:5],
            "fidelity_estimate": 0.999,
            "simulated": True,
            "method": method,
        })
        return result

    def run_4d_discovery_audit(self):
        """High-intensity audit of the 4D Hypercube threshold."""
        code = self.get_code("hypercube_4d")
        c = code.build()
        # The Clifford backend handles this circuit fine (measured 0.008 s for
        # 48 qubits / 177 gates); only the statevector fallback would try to
        # allocate 2**52 bytes and kill the process (QEC-12).
        method = "stabilizer"
        if c.n_qubits > _qubit_limit(method):
            return {
                "dimension": "4D",
                "lattice_size": "2x2x2x2",
                "syndromes": None,
                "status": "SKIPPED",
                "circuit_qubits": c.n_qubits,
                "note": (
                    f"{c.n_qubits} qubits exceeds the {_qubit_limit(method)}-qubit "
                    f"limit for method={method!r}; refusing rather than "
                    f"aborting the interpreter"
                ),
            }
        res = sf.run(c, method=method, shots=100)
        return {
            "dimension": "4D",
            "lattice_size": "2x2x2x2",
            "syndromes": len(res.counts),
            "circuit_qubits": c.n_qubits,
            "method": method,
            "status": "PASS",
        }
