"""
Environment Detection — Auto-detect runtime (Jupyter, Colab, VS Code, CLI).
"""

from __future__ import annotations

import os
import sys
from enum import Enum, auto
from typing import Optional


class Environment(Enum):
    """Detected runtime environment."""
    JUPYTER = auto()
    COLAB = auto()
    VSCODE = auto()
    CLI = auto()
    UNKNOWN = auto()


def detect_environment() -> Environment:
    """Auto-detect the current runtime environment.
    
    Returns:
        The detected Environment enum value.
    """
    # Google Colab detection
    try:
        import google.colab  # noqa
        return Environment.COLAB
    except ImportError:
        pass
    
    # Jupyter detection (IPython kernel)
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell_name = type(shell).__name__
            if "ZMQInteractiveShell" in shell_name:
                return Environment.JUPYTER
    except (ImportError, NameError):
        pass
    
    # VS Code detection
    if os.environ.get("TERM_PROGRAM") == "vscode" or os.environ.get("VSCODE_PID"):
        return Environment.VSCODE
    
    # CLI fallback
    if sys.stdin and sys.stdin.isatty():
        return Environment.CLI
    
    return Environment.UNKNOWN


def get_display_backend() -> str:
    """Get the best display backend for the current environment.
    
    Returns:
        One of: 'html', 'svg', 'text'
    """
    env = detect_environment()
    if env in (Environment.JUPYTER, Environment.COLAB):
        return "html"
    elif env == Environment.VSCODE:
        return "svg"
    else:
        return "text"


# Singleton cached value
_CACHED_ENV: Optional[Environment] = None

def current_environment() -> Environment:
    """Get the current environment (cached after first call)."""
    global _CACHED_ENV
    if _CACHED_ENV is None:
        _CACHED_ENV = detect_environment()
    return _CACHED_ENV


# Rich display helpers for Jupyter/Colab
def display_circuit_html(circuit) -> str:
    """Generate an HTML representation of a circuit for Jupyter."""
    gates_html = ""
    for i, gate in enumerate(circuit._gates):
        color = "#4CAF50" if len(gate.qubits) == 1 else "#2196F3"
        gates_html += (
            f'<span style="background:{color};color:white;padding:2px 6px;'
            f'border-radius:3px;margin:1px;font-size:12px">'
            f'{gate.name}({",".join(str(q) for q in gate.qubits)})</span> '
        )
    
    return f"""
    <div style="font-family:monospace;padding:10px;background:#1a1a2e;
                border-radius:8px;color:#e0e0e0;margin:5px 0">
        <div style="color:#00d4ff;font-weight:bold;margin-bottom:5px">
            Circuit: {circuit._name or 'unnamed'} | 
            Qubits: {circuit.n_qubits} | 
            Depth: {circuit.depth} | 
            Gates: {circuit.gate_count}
        </div>
        <div>{gates_html}</div>
    </div>
    """


def display_result_html(result) -> str:
    """Generate an HTML representation of an AlgorithmResult for Jupyter."""
    history_spark = ""
    if hasattr(result, 'history') and result.history:
        # Mini sparkline using Unicode block chars
        mn, mx = min(result.history), max(result.history)
        rng = mx - mn if mx != mn else 1
        blocks = " ".join(
            f"{v:.4f}" for v in result.history[-5:]
        )
        history_spark = f"<div style='color:#888;font-size:11px'>Last 5: {blocks}</div>"
    
    fidelity_line = ""
    if hasattr(result, 'final_fidelity') and result.final_fidelity is not None:
        fidelity_line = f"<div>Fidelity: <b>{result.final_fidelity:.6f}</b></div>"
    
    return f"""
    <div style="font-family:monospace;padding:10px;background:#1a1a2e;
                border-radius:8px;color:#e0e0e0;margin:5px 0">
        <div style="color:#00ff88;font-weight:bold">AlgorithmResult</div>
        <div>Optimal Value: <b>{result.optimal_value:.6f}</b></div>
        {fidelity_line}
        {history_spark}
    </div>
    """
