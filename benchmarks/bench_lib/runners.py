"""Per-framework run wrappers.

Every runner takes a QASM string + shots + n_qubits, returns a RunRecord
with uniformly-formatted timing, memory, counts (BE convention) and
optional statevector (BE convention).

Important: SuperFermion's `bridge.from_qasm` silently reverses qubit
indices (n_qubits - 1 - i) as part of its BE<->LE bridging for Qiskit
compatibility. Since our harness already normalizes Qiskit outputs at
the boundary, we MUST NOT double-reverse on the SF side. We therefore
build SF circuits via a minimal no-reverse QASM parser defined here.

Timing protocol:
  - 2 untimed warm-up runs (flush JIT / topology cache).
  - N_TIMED timed runs, each measured with time.perf_counter().
  - Report mean, std, min, p50, p95.

Memory protocol:
  - tracemalloc peak during ONE additional dedicated run (Python heap).
  - psutil RSS delta around the same run (captures Rust/JAX arenas).
"""
from __future__ import annotations

import gc
import os
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import psutil

from bench.normalize import (
    normalize_counts,
    reverse_bitstring_keys,
    reverse_statevector_endianness,
)

N_WARMUP = 2
N_TIMED = 10


# ---------- no-reverse QASM -> SF Circuit ----------
# Minimal parser that intentionally does NOT mirror qubit indices, unlike
# superfermion.bridge.from_qasm. Supports the gate vocabulary produced by
# bench/circuits.py: h, x, y, z, s, t, rx, ry, rz, cx, cz, swap, ccx.

import math as _math
import re as _re


def _sf_circuit_from_qasm_no_reverse(qasm: str):
    """Build an SF Circuit with qubit indices used directly (BE frame)."""
    import superfermion as sf  # local import keeps module load fast
    n_qubits = None
    circuit = None
    for raw in qasm.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
            continue
        if line.startswith("qubit[") or line.startswith("qreg"):
            m = _re.search(r"\[(\d+)\]", line)
            if m:
                n_qubits = int(m.group(1))
                circuit = sf.Circuit(n_qubits)
            continue
        if line.startswith("creg") or line.startswith("bit["):
            continue
        if circuit is None:
            continue
        m = _re.match(r"(\w+)(?:\(([^)]*)\))?\s+(.+);", line)
        if not m:
            continue
        gate_name = m.group(1).lower()
        params_str = m.group(2)
        qubits_str = m.group(3)
        qubit_indices = [int(x) for x in _re.findall(r"\[(\d+)\]", qubits_str)]
        params = []
        if params_str:
            for p in params_str.split(","):
                p = p.strip()
                try:
                    params.append(float(p))
                except ValueError:
                    params.append(eval(p.replace("pi", str(_math.pi))))
        gate_map = {
            "h": "h", "x": "x", "y": "y", "z": "z",
            "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
            "sx": "sx",
            "rx": "rx", "ry": "ry", "rz": "rz",
            "u": "u", "u3": "u3",
            "p": "p", "u1": "p",
            "cx": "cx", "cnot": "cx",
            "cz": "cz",
            "swap": "swap",
            "ccx": "ccx",
        }
        sf_name = gate_map.get(gate_name)
        if not sf_name or not hasattr(circuit, sf_name):
            continue
        method = getattr(circuit, sf_name)
        if params:
            method(*params, *qubit_indices)
        else:
            method(*qubit_indices)
    if circuit is None:
        raise ValueError("Could not parse QASM: no qubit declaration found")
    return circuit


@dataclass
class RunRecord:
    framework: str
    backend: str
    circuit: str
    n_qubits: int
    shots: int

    ok: bool = False
    error: str = ""
    skipped_reason: str = ""

    # timing (ms)
    t_mean_ms: float = 0.0
    t_std_ms: float = 0.0
    t_min_ms: float = 0.0
    t_p50_ms: float = 0.0
    t_p95_ms: float = 0.0

    # memory (bytes)
    py_peak_bytes: int = 0
    rss_delta_bytes: int = 0

    # outputs (Big-Endian)
    probs: Dict[str, float] = field(default_factory=dict)
    statevector_present: bool = False

    def to_dict(self, include_sv: bool = False, sv: Optional[np.ndarray] = None) -> Dict[str, Any]:
        d = asdict(self)
        if include_sv and sv is not None:
            # Store as (real, imag) pairs only for small states
            if sv.size <= 2 ** 12:
                d["statevector_re"] = [float(x.real) for x in sv]
                d["statevector_im"] = [float(x.imag) for x in sv]
        return d


def _stats_ms(times_s: List[float]) -> Dict[str, float]:
    arr = np.asarray(times_s, dtype=float) * 1000.0
    return {
        "t_mean_ms": float(arr.mean()),
        "t_std_ms": float(arr.std(ddof=0)),
        "t_min_ms": float(arr.min()),
        "t_p50_ms": float(np.median(arr)),
        "t_p95_ms": float(np.percentile(arr, 95)),
    }


def _time_and_mem(run_once: Callable[[], Any], run_once_for_counts: Callable[[], Any]) -> Dict[str, Any]:
    """Generic timing + memory harness.
    `run_once` is used for all timing/memory runs.
    `run_once_for_counts` returns the artefact we actually keep.
    """
    # Warm-up
    for _ in range(N_WARMUP):
        run_once()
    # Timed
    times: List[float] = []
    for _ in range(N_TIMED):
        gc.disable()
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
    stats = _stats_ms(times)
    # Memory: one dedicated run with tracemalloc + RSS
    proc = psutil.Process(os.getpid())
    gc.collect()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    artefact = run_once_for_counts()
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss_after = proc.memory_info().rss
    stats["py_peak_bytes"] = int(py_peak)
    stats["rss_delta_bytes"] = int(max(0, rss_after - rss_before))
    stats["_artefact"] = artefact
    return stats


# ---------- SuperFermion ----------

def run_superfermion(
    sf_backend: str, qasm: str, n_qubits: int, shots: int,
    want_statevector: bool,
) -> tuple[RunRecord, Optional[np.ndarray]]:
    import superfermion as sf

    rec = RunRecord("superfermion", sf_backend, "", n_qubits, shots)

    # Use no-reverse parser so SF lands gates on the same physical wires
    # as PennyLane's BE frame. Qiskit output is reversed separately.
    circuit = _sf_circuit_from_qasm_no_reverse(qasm)

    def _run():
        return sf.run(circuit, backend=sf_backend, shots=shots, seed=42)

    try:
        stats = _time_and_mem(_run, _run)
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
        return rec, None

    result = stats.pop("_artefact")
    for k, v in stats.items():
        setattr(rec, k, v)

    # Counts -> already BigEndian for SF
    counts = getattr(result, "counts", None) or {}
    rec.probs = normalize_counts(counts, n_qubits)

    sv = None
    if want_statevector:
        sv = getattr(result, "statevector", None)
        if sv is not None:
            sv = np.asarray(sv, dtype=complex).reshape(-1)
            rec.statevector_present = True

    rec.ok = True
    return rec, sv


# ---------- Qiskit-Aer ----------

def run_qiskit_aer(
    method: str, qasm: str, n_qubits: int, shots: int,
    want_statevector: bool,
) -> tuple[RunRecord, Optional[np.ndarray]]:
    """method: 'statevector' or 'matrix_product_state'."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    rec = RunRecord("qiskit_aer", method, "", n_qubits, shots)
    try:
        qc = QuantumCircuit.from_qasm_str(qasm)
    except Exception as e:
        rec.error = f"QASM parse: {type(e).__name__}: {e}"
        return rec, None

    # Keep a separate circuit for statevector extraction (save_statevector instruction).
    qc_counts = qc.copy()
    if not any(instr.operation.name == "measure" for instr in qc_counts.data):
        qc_counts.measure_all()
    qc_counts_transpiled = None

    qc_sv = None
    qc_sv_transpiled = None
    if want_statevector and method == "statevector":
        qc_sv = qc.copy()
        qc_sv.save_statevector()

    backend = AerSimulator(method=method)

    # Transpile once so it's not included in timing
    qc_counts_transpiled = transpile(qc_counts, backend)
    if qc_sv is not None:
        qc_sv_transpiled = transpile(qc_sv, backend)

    def _run_counts():
        res = backend.run(qc_counts_transpiled, shots=shots, seed_simulator=42).result()
        return res.get_counts()

    try:
        stats = _time_and_mem(_run_counts, _run_counts)
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
        return rec, None
    counts = stats.pop("_artefact")
    for k, v in stats.items():
        setattr(rec, k, v)

    # Qiskit counts are Little-Endian; reverse before normalizing
    rec.probs = normalize_counts(reverse_bitstring_keys(counts), n_qubits)

    sv = None
    if want_statevector and qc_sv_transpiled is not None:
        try:
            res = backend.run(qc_sv_transpiled, shots=1, seed_simulator=42).result()
            sv_le = np.asarray(res.get_statevector(), dtype=complex).reshape(-1)
            sv = reverse_statevector_endianness(sv_le, n_qubits)
            rec.statevector_present = True
        except Exception:
            # statevector extraction is a nice-to-have, not fatal
            pass

    rec.ok = True
    return rec, sv


# ---------- PennyLane ----------

def run_pennylane(
    device_name: str, qasm: str, n_qubits: int, shots: int,
    want_statevector: bool,
) -> tuple[RunRecord, Optional[np.ndarray]]:
    """device_name: 'default.qubit' or 'lightning.qubit'."""
    import pennylane as qml

    rec = RunRecord("pennylane", device_name, "", n_qubits, shots)

    # Build ops from QASM via qml's loader if available, else translate manually.
    try:
        loader = qml.from_qasm(qasm)
    except Exception as e:
        rec.error = f"qml.from_qasm: {type(e).__name__}: {e}"
        return rec, None

    # ---- Counts run ----
    try:
        dev_counts = qml.device(device_name, wires=n_qubits, shots=shots)

        @qml.qnode(dev_counts)
        def circ_counts():
            loader(wires=range(n_qubits))
            return qml.counts()
    except Exception as e:
        rec.error = f"qnode counts: {type(e).__name__}: {e}"
        return rec, None

    def _run_counts():
        return circ_counts()

    try:
        stats = _time_and_mem(_run_counts, _run_counts)
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
        return rec, None

    counts_raw = stats.pop("_artefact")
    for k, v in stats.items():
        setattr(rec, k, v)

    # qml.counts returns dict of bitstring -> count. PennyLane is BigEndian
    # (wire 0 leftmost) — no reversal needed.
    counts = {str(k): int(v) for k, v in counts_raw.items()}
    rec.probs = normalize_counts(counts, n_qubits)

    sv = None
    if want_statevector:
        try:
            dev_sv = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev_sv)
            def circ_sv():
                loader(wires=range(n_qubits))
                return qml.state()

            sv = np.asarray(circ_sv(), dtype=complex).reshape(-1)
            rec.statevector_present = True
        except Exception:
            pass

    rec.ok = True
    return rec, sv
