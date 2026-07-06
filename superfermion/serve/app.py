"""
Superfermion Serve — The Quantum API Gateway.

Provides REST and WebSocket endpoints for remote circuit execution,
autonomous agent interaction, code execution, CLI commands, and hardware job management.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import superfermion as sf
from superfermion.serve.auth import authenticate, check_qubit_limit
from superfermion.runtime.arbiter import arbiter

app = FastAPI(
    title="Superfermion Quantum API",
    description="High-performance quantum simulation API with code execution, CLI, and circuit endpoints.",
    version=sf.__version__
)

# CORS — allow frontend dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class CircuitSchema(BaseModel):
    n_qubits: int
    gates: List[Dict[str, Any]]
    
class RunRequest(BaseModel):
    circuit: Optional[CircuitSchema] = None
    qasm: Optional[str] = None
    backend: str = "jax"
    shots: int = 1000
    target: Optional[str] = None

class ExecuteRequest(BaseModel):
    code: str
    backend: str = "statevector"
    timeout: int = 30

class CLIRequest(BaseModel):
    command: str
    timeout: int = 60

# --- In-Memory Job Store (MVP) ---
JOBS: Dict[str, Any] = {}

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "version": sf.__version__,
        "singularity_active": True,
        "supported_backends": ["jax", "statevector", "cuda", "cluster", "ibm_eagle", "ionq_aria"]
    }

@app.post("/v1/run")
async def run_circuit(request: RunRequest, user: Dict[str, Any] = Security(authenticate)):
    """Execute a circuit with Auth, Security Validation, and Auto-Routing."""
    try:
        # 1. Reconstruct Circuit
        if request.qasm:
            import superfermion.bridge as bridge
            circuit = bridge.from_qasm(request.qasm)
        elif request.circuit:
            circuit = sf.Circuit(request.circuit.n_qubits)
            # (Gate parsing would go here)
        else:
            raise HTTPException(400, "Provide either 'circuit' schema or 'qasm' string.")
            
        # 2. Security & Quota Checks
        check_qubit_limit(circuit.n_qubits, user["tier"])
        arbiter.validate_security(circuit)
        
        # 3. Dynamic Routing (The 'System')
        # If no backend specified, arbiter chooses the best one
        backend_to_use = request.backend if request.backend != "jax" else arbiter.select_best_backend(circuit, request.target)
        
        # 4. Dispatch via Runtime
        from superfermion.runtime import runtime
        job = runtime.run(circuit, backend=backend_to_use, shots=request.shots, target=request.target)
        
        # 5. Store and Return
        job_id = job.job_id
        JOBS[job_id] = job
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "backend_used": backend_to_use,
            "user": user["user"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        sf.utils.error(f"API Run Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Retrieve the result or status of a quantum job."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
        
    job = JOBS[job_id]
    result = job.result()
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "counts": result.counts if hasattr(result, 'counts') else None,
        "metadata": result.metadata if hasattr(result, 'metadata') else {}
    }


@app.get("/v1/jobs")
async def list_jobs(limit: int = 50):
    """List recent jobs with status."""
    recent = list(JOBS.values())[-limit:]
    return {
        "total": len(JOBS),
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "backend": j.backend if hasattr(j, 'backend') else None,
            }
            for j in reversed(recent)
        ]
    }


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running/pending job."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found.")
    job = JOBS[job_id]
    if hasattr(job, 'cancel'):
        job.cancel()
    JOBS[job_id] = job
    return {"job_id": job_id, "status": job.status.value, "cancelled": True}

@app.post("/v1/execute")
async def execute_code(request: ExecuteRequest):
    """Execute arbitrary superfermion Python code in-process.

    Runs code directly in the FastAPI process (which already has superfermion
    + JAX imported), avoiding the 10-15s subprocess cold-start penalty.
    Pre-imports available in namespace: ``sf``, ``np``, ``jnp``.
    """
    t0 = time.perf_counter()

    # ── Capture user variables defined during exec ─────────────────────
    _pre_keys = set(locals()) | {"sf", "np", "jnp"}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result_value = None
    success = True

    import numpy as np

    # Already imported at module level: superfermion as sf, jax.numpy as jnp
    exec_ns: Dict[str, Any] = {
        "sf": sf,
        "np": np,
        "jnp": jnp,
    }

    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        try:
            exec(request.code, exec_ns)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # ── Extract user-defined variables ────────────────────────────
        result_value = {}
        skip_types = {
            "module", "type", "function", "builtin_function_or_method",
            "method", "method-wrapper",
        }
        for key, val in exec_ns.items():
            if key in _pre_keys or key.startswith("_"):
                continue
            tname = type(val).__name__
            if tname in skip_types:
                continue
            try:
                if hasattr(val, "tolist"):
                    val = val.tolist()
                elif isinstance(val, (int, float, str, bool, list, dict, tuple, type(None), complex)):
                    pass
                elif isinstance(val, (bytes, bytearray)):
                    val = "<binary data>"
                else:
                    val = str(val)
            except Exception:
                val = str(val)
            result_value[key] = val

    except Exception as exc:
        stderr_buf.write(traceback.format_exc())
        result_value = {"__error__": str(exc)}
        success = False

    execution_time_ms = (time.perf_counter() - t0) * 1000

    return {
        "success": success,
        "stdout": stdout_buf.getvalue().strip(),
        "stderr": stderr_buf.getvalue().strip() if stderr_buf.getvalue() else "",
        "result": result_value,
        "execution_time_ms": round(execution_time_ms, 2),
    }


@app.post("/v1/cli")
async def run_cli(request: CLIRequest):
    """Execute a superfermion CLI command and return its output.

    Supported commands: sf backends, sf validate, sf benchmark, sf version,
    sf vqe, sf qaoa, sf chemistry, sf qec.
    """
    t0 = time.perf_counter()

    # Security: only allow 'sf' commands, block shell metacharacters
    cmd = request.command.strip()
    if not cmd.startswith("sf "):
        raise HTTPException(400, "Only 'sf' CLI commands are allowed.")

    # Basic injection prevention
    dangerous = ["|", ";", "&", "`", "$", "(", ")", "{", "}", "<", ">", "\n", "\r"]
    for char in dangerous:
        if char in cmd:
            raise HTTPException(400, f"Character '{char}' is not allowed in CLI commands.")

    try:
        parts = cmd.split()
        result = subprocess.run(
            [sys.executable, "-m", "superfermion.cli"] + parts[1:],
            capture_output=True,
            text=True,
            timeout=request.timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or ".",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "SF_NO_BANNER": "1"},
        )

        execution_time_ms = (time.perf_counter() - t0) * 1000

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "execution_time_ms": round(execution_time_ms, 2),
        }

    except subprocess.TimeoutExpired:
        execution_time_ms = (time.perf_counter() - t0) * 1000
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {request.timeout}s",
            "exit_code": -1,
            "execution_time_ms": round(execution_time_ms, 2),
        }
    except Exception as e:
        execution_time_ms = (time.perf_counter() - t0) * 1000
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Server error: {str(e)}",
            "exit_code": -1,
            "execution_time_ms": round(execution_time_ms, 2),
        }


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker / monitoring."""
    return {"status": "healthy", "version": sf.__version__}


# --- Helper ---

def _indent_code(code: str, indent: int = 4) -> str:
    """Indent each line of code by the given number of spaces."""
    prefix = " " * indent
    return "\n".join(prefix + line if line.strip() else "" for line in code.splitlines())


@app.websocket("/v1/monitor")
async def stream_monitor(websocket: WebSocket):
    """Live stream of quantum metrics and singularity status."""
    await websocket.accept()
    try:
        while True:
            # Mock live telemetry
            import time
            await websocket.send_json({
                "timestamp": time.time(),
                "qpu_load": 0.25,
                "entanglement_density": 0.89,
                "singularity_health": "stable"
            })
            # Wait for 1s
            import asyncio
            await asyncio.sleep(1)
    except Exception:
        await websocket.close()
