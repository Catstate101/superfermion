"""Single source of truth for gate unitary matrices.

Replaces the duplicated ~150-line if-chains in:
  - ``superfermion.circuit.GateRecord.to_unitary()``
  - ``superfermion.backends.simulator.StatevectorBackend._get_gate_matrix()``
"""

from __future__ import annotations

import math
from typing import List, Union

import numpy as np


def gate_unitary_matrix(
    name: str,
    params: List[Union[float, complex, int]] = None,
    use_complex_trig: bool = False,
) -> np.ndarray:
    """Return the unitary matrix for a gate.

    Args:
        name: Gate name (case-insensitive).
        params: Gate parameters (angles, etc.).
        use_complex_trig: If True, uses ``np.cos/np.sin`` for complex-valued
            parameters (needed by StatevectorBackend). If False, uses
            ``math.cos/math.sin`` for faster real-valued paths.

    Returns:
        Unitary matrix as ``np.ndarray`` of ``np.complex128``.

    Raises:
        ValueError: If the gate name is unknown.
    """
    name = name.upper()
    if params is None:
        params = []

    if use_complex_trig:
        params = [
            complex(p) if not (isinstance(p, str) or hasattr(p, 'value')) else 0.0+0j
            for p in params
        ]
        _cos, _sin = np.cos, np.sin
        _sqrt2_inv = 1.0 / np.sqrt(2.0)
    else:
        params = [
            float(p) if not (isinstance(p, str) or hasattr(p, 'value')) else 0.0
            for p in params
        ]
        _cos, _sin = math.cos, math.sin
        _sqrt2_inv = 1.0 / math.sqrt(2.0)

    # ── 1-qubit gates ──
    if name == "ID" or name == "I":
        return np.eye(2, dtype=np.complex128)
    if name == "H":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) * _sqrt2_inv
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if name == "SDG":
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if name == "T":
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    if name == "TDG":
        return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)
    if name == "SX":
        return np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)
    if name == "SXDG":
        return np.array([[0.5-0.5j, 0.5+0.5j], [0.5+0.5j, 0.5-0.5j]], dtype=np.complex128)

    # ── Parameterised 1-qubit gates ──
    if name == "RX":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    if name == "RY":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        theta = params[0] if params else 0.0
        return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex128)
    if name in ("R1", "P"):
        phi = params[0] if params else 0.0
        return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=np.complex128)
    if name in ("U", "U3"):
        theta, phi, lam = (params if len(params) >= 3 else (0, 0, 0))
        ct, st = _cos(theta/2), _sin(theta/2)
        return np.array([
            [ct, -np.exp(1j*lam)*st],
            [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]
        ], dtype=np.complex128)
    if name == "U1":
        lam = params[0] if params else 0.0
        return np.array([[1, 0], [0, np.exp(1j*lam)]], dtype=np.complex128)
    if name == "U2":
        phi, lam = (params[0], params[1]) if len(params) >= 2 else (0, 0)
        s = 1.0 / math.sqrt(2)
        return np.array([[s, -np.exp(1j*lam)*s], [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*s]], dtype=np.complex128)

    # ── 2-qubit gates ──
    if name in ("CX", "CNOT"):
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if name == "SWAP":
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128)
    if name == "CY":
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=np.complex128)
    if name == "ISWAP":
        return np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=np.complex128)
    if name == "ECR":
        s = 1.0 / math.sqrt(2)
        return np.array([[0,0,s,1j*s],[0,0,1j*s,s],[s,-1j*s,0,0],[-1j*s,s,0,0]], dtype=np.complex128)

    # Parameterised 2-qubit gates
    if name in ("CU", "CU3"):
        theta, phi, lam = params if len(params) >= 3 else (0, 0, 0)
        c, s = _cos(theta/2), _sin(theta/2)
        u3 = np.array([
            [c, -np.exp(1j*lam)*s],
            [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]
        ], dtype=np.complex128)
        res = np.eye(4, dtype=np.complex128)
        res[2:, 2:] = u3
        return res
    if name == "CP":
        phi = params[0] if params else 0.0
        res = np.eye(4, dtype=np.complex128)
        res[3, 3] = np.exp(1j*phi)
        return res
    if name == "CRX":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        res = np.eye(4, dtype=np.complex128)
        res[2:, 2:] = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
        return res
    if name == "CRY":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        res = np.eye(4, dtype=np.complex128)
        res[2:, 2:] = np.array([[c, -s], [s, c]], dtype=np.complex128)
        return res
    if name == "CRZ":
        theta = params[0] if params else 0.0
        res = np.eye(4, dtype=np.complex128)
        res[2:, 2:] = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)]).astype(np.complex128)
        return res
    if name == "CH":
        res = np.eye(4, dtype=np.complex128)
        res[2:, 2:] = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        return res
    if name == "RXX":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        return np.array([[c,0,0,-1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[-1j*s,0,0,c]], dtype=np.complex128)
    if name == "RYY":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        return np.array([[c,0,0,1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[1j*s,0,0,c]], dtype=np.complex128)
    if name == "RZZ":
        theta = params[0] if params else 0.0
        em, ep = np.exp(-1j*theta/2), np.exp(1j*theta/2)
        return np.diag([em, ep, ep, em]).astype(np.complex128)
    if name == "RZX":
        theta = params[0] if params else 0.0
        c, s = _cos(theta/2), _sin(theta/2)
        return np.array([[c,-1j*s,0,0],[-1j*s,c,0,0],[0,0,c,1j*s],[0,0,1j*s,c]], dtype=np.complex128)

    # ── 3-qubit gates ──
    if name in ("CCX", "TOFFOLI"):
        m = np.eye(8, dtype=np.complex128)
        m[6,6]=0; m[7,7]=0; m[6,7]=1; m[7,6]=1
        return m
    if name == "CSWAP":
        m = np.eye(8, dtype=np.complex128)
        m[5,5]=0; m[6,6]=0; m[5,6]=1; m[6,5]=1
        return m

    raise ValueError(f"Unknown gate: '{name}'")
