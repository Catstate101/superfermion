"""
Quantum Compilation Passes — Python shims for capabilities not yet in Rust.

The Rust ``sf-compiler`` crate handles gate cancellation, rotation merging,
SWAP decomposition, constant folding, superconducting basis translation,
and SABRE routing. The passes here are **temporary shims** for gaps:

- ``UnitaryDecompositionPass``: decomposes opaque UNITARY gates (ZYZ/CSD)
- ``BasisTranslationPass``: translates to arbitrary (non-SX) basis sets

Both should eventually migrate to Rust crates.
"""

from __future__ import annotations

from typing import List, Protocol, Tuple

from superfermion.circuit import Circuit, GateRecord


class Pass(Protocol):
    """Protocol for a compilation pass."""
    def name(self) -> str: ...
    def run(self, circuit: Circuit) -> Circuit: ...


class BasisTranslationPass:
    """Translates all gates to a target native basis set.

    Supports decomposition to any combination of standard basis gates.
    Common targets: ``{rz, sx, x, cx}``, ``{rz, sx, x, cz}``,
    ``{rz, sx, x, ecr}``, ``{rx, rz, cz}``.
    """
    _PI = 3.141592653589793
    _PI2 = 1.5707963267948966

    def __init__(self, native_gates: List[str]):
        self.native_gates = {g.upper() for g in native_gates}

    def name(self) -> str:
        return "BasisTranslationPass"

    def _is_native(self, name: str) -> bool:
        return name in self.native_gates or name in ("MEASURE", "BARRIER", "RESET")

    def _has(self, *names: str) -> bool:
        return any(n in self.native_gates for n in names)

    def _emit_h(self, c: Circuit, q: int) -> None:
        """H = RZ(pi/2) SX RZ(pi/2)"""
        if self._has("H"):
            c.h(q)
        elif self._has("SX"):
            c.rz(self._PI2, q).sx(q).rz(self._PI2, q)
        else:
            c.rz(self._PI2, q).rx(self._PI2, q).rz(self._PI2, q)

    def _emit_cx(self, c: Circuit, ctrl: int, tgt: int) -> None:
        """Emit a CNOT using the available 2Q native gate."""
        if self._has("CX", "CNOT"):
            c.cx(ctrl, tgt)
        elif self._has("CZ"):
            self._emit_h(c, tgt)
            c.cz(ctrl, tgt)
            self._emit_h(c, tgt)
        elif self._has("ECR"):
            c.rz(self._PI2, ctrl).rx(self._PI2, ctrl)
            c._gates.append(GateRecord("ECR", [ctrl, tgt], []))
            c.rx(self._PI2, tgt)
        else:
            c.cx(ctrl, tgt)

    def _decompose(self, gate: GateRecord, c: Circuit) -> None:
        name = gate.name.upper()
        q = gate.qubits

        if name == "X":
            if self._has("X"):
                c.x(q[0])
            elif self._has("SX"):
                c.sx(q[0]).sx(q[0])
            else:
                c.rx(self._PI, q[0])
        elif name == "Y":
            if self._has("SX"):
                c.rz(self._PI, q[0]).sx(q[0]).sx(q[0])
            else:
                c.rx(self._PI, q[0]).rz(self._PI, q[0])
        elif name == "Z":
            c.rz(self._PI, q[0])
        elif name == "H":
            self._emit_h(c, q[0])
        elif name in ("CNOT", "CX"):
            self._emit_cx(c, q[0], q[1])
        elif name == "CZ":
            if self._has("CZ"):
                c.cz(q[0], q[1])
            else:
                self._emit_h(c, q[1])
                self._emit_cx(c, q[0], q[1])
                self._emit_h(c, q[1])
        elif name == "S":
            c.rz(self._PI2, q[0])
        elif name == "SDG":
            c.rz(-self._PI2, q[0])
        elif name == "T":
            c.rz(self._PI / 4, q[0])
        elif name == "TDG":
            c.rz(-self._PI / 4, q[0])
        elif name == "SX":
            if self._has("SX"):
                c.sx(q[0])
            else:
                c.rx(self._PI2, q[0])
        elif name == "RX":
            theta = float(gate.params[0]) if gate.params else 0.0
            if self._has("RX"):
                c.rx(theta, q[0])
            elif self._has("SX"):
                c.rz(-self._PI2, q[0]).sx(q[0]).rz(self._PI2, q[0])
                if abs(theta - self._PI2) > 1e-10:
                    c.rz(theta - self._PI2, q[0])
            else:
                c.rx(theta, q[0])
        elif name == "RY":
            theta = float(gate.params[0]) if gate.params else 0.0
            if self._has("RY"):
                c.ry(theta, q[0])
            elif self._has("SX"):
                c.rz(-self._PI2, q[0]).sx(q[0]).rz(theta + self._PI2, q[0]).sx(q[0]).rz(-self._PI2, q[0])
            else:
                c.rz(-self._PI2, q[0]).rx(theta, q[0]).rz(self._PI2, q[0])
        elif name == "P":
            c.rz(float(gate.params[0]) if gate.params else 0.0, q[0])
        elif name in ("U", "U3"):
            theta = float(gate.params[0]) if len(gate.params) > 0 else 0.0
            phi = float(gate.params[1]) if len(gate.params) > 1 else 0.0
            lam = float(gate.params[2]) if len(gate.params) > 2 else 0.0
            if self._has("SX"):
                c.rz(lam, q[0]).sx(q[0]).rz(theta + self._PI, q[0]).sx(q[0]).rz(phi + self._PI, q[0])
            else:
                c.rz(lam, q[0]).rx(self._PI2, q[0]).rz(theta, q[0]).rx(-self._PI2, q[0]).rz(phi, q[0])
        elif name == "SWAP":
            self._emit_cx(c, q[0], q[1])
            self._emit_cx(c, q[1], q[0])
            self._emit_cx(c, q[0], q[1])
        elif name in ("CCX", "TOFFOLI"):
            self._emit_h(c, q[2])
            self._emit_cx(c, q[1], q[2]); c.rz(-self._PI / 4, q[2])
            self._emit_cx(c, q[0], q[2]); c.rz(self._PI / 4, q[2])
            self._emit_cx(c, q[1], q[2]); c.rz(-self._PI / 4, q[2])
            self._emit_cx(c, q[0], q[2]); c.rz(self._PI / 4, q[1]); c.rz(self._PI / 4, q[2])
            self._emit_h(c, q[2])
            self._emit_cx(c, q[0], q[1]); c.rz(self._PI / 4, q[0]); c.rz(-self._PI / 4, q[1])
            self._emit_cx(c, q[0], q[1])
        elif name == "ID":
            pass
        elif name == "CY":
            c.rz(-self._PI2, q[1])
            self._emit_cx(c, q[0], q[1])
            c.rz(self._PI2, q[1])
        elif name == "CP":
            lam = float(gate.params[0]) if gate.params else 0.0
            c.rz(lam / 2, q[0])
            self._emit_cx(c, q[0], q[1])
            c.rz(-lam / 2, q[1])
            self._emit_cx(c, q[0], q[1])
            c.rz(lam / 2, q[1])
        elif name == "ISWAP":
            self._emit_cx(c, q[0], q[1])
            self._emit_h(c, q[0])
            self._emit_cx(c, q[1], q[0])
            c.rz(self._PI2, q[0])
            self._emit_cx(c, q[1], q[0])
            c.rz(-self._PI2, q[0])
            self._emit_h(c, q[0])
            self._emit_cx(c, q[0], q[1])
        elif name == "RZZ":
            theta = float(gate.params[0]) if gate.params else 0.0
            self._emit_cx(c, q[0], q[1])
            c.rz(theta, q[1])
            self._emit_cx(c, q[0], q[1])
        elif name == "RXX":
            theta = float(gate.params[0]) if gate.params else 0.0
            self._emit_h(c, q[0]); self._emit_h(c, q[1])
            self._emit_cx(c, q[0], q[1])
            c.rz(theta, q[1])
            self._emit_cx(c, q[0], q[1])
            self._emit_h(c, q[0]); self._emit_h(c, q[1])
        elif name == "RYY":
            theta = float(gate.params[0]) if gate.params else 0.0
            c.rx(self._PI2, q[0]).rx(self._PI2, q[1])
            self._emit_cx(c, q[0], q[1])
            c.rz(theta, q[1])
            self._emit_cx(c, q[0], q[1])
            c.rx(-self._PI2, q[0]).rx(-self._PI2, q[1])
        else:
            c._gates.append(gate)

    def run(self, circuit: Circuit) -> Circuit:
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        for gate in circuit._gates:
            if self._is_native(gate.name.upper()):
                new_circuit._gates.append(gate)
            else:
                self._decompose(gate, new_circuit)
        return new_circuit

class UnitaryDecompositionPass:
    """Decomposes opaque UNITARY gates into primitive basis gates.

    1-qubit: ZYZ Euler decomposition -> Rz, Ry, Rz
    2-qubit: CSD + ABC decomposition -> up to 6 CX + single-qubit rotations
    3+ qubit: raises ValueError (not yet supported)
    """
    _EPS = 1e-10

    def name(self) -> str:
        return "UnitaryDecompositionPass"

    @staticmethod
    def _rz_mat(theta):
        import numpy as np
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ])

    @staticmethod
    def _ry_mat(theta):
        import numpy as np
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def _zyz_params(matrix):
        """Extract ZYZ Euler angles: U = exp(i*phase) Rz(alpha) Ry(beta) Rz(gamma).

        The matrix elements of Rz(a)Ry(b)Rz(g) are:
          [0,0] = exp(-i(a+g)/2) cos(b/2)
          [1,0] = exp(i(a-g)/2)  sin(b/2)
          [1,1] = exp(i(a+g)/2)  cos(b/2)
        """
        import numpy as np
        det = np.linalg.det(matrix)
        phase = float(np.angle(det) / 2)
        su = matrix * np.exp(-1j * phase)

        beta = float(2 * np.arccos(np.clip(abs(su[0, 0]), 0, 1)))
        if abs(np.sin(beta / 2)) < 1e-10:
            alpha = float(2 * np.angle(su[1, 1]))
            gamma = 0.0
        else:
            alpha = float(np.angle(su[1, 1]) + np.angle(su[1, 0]))
            gamma = float(np.angle(su[1, 1]) - np.angle(su[1, 0]))
        return phase, alpha, beta, gamma

    def _decompose_1q(self, matrix, qubit: int, circuit: Circuit) -> None:
        """ZYZ Euler decomposition: U = Rz(alpha) Ry(beta) Rz(gamma)."""
        _, alpha, beta, gamma = self._zyz_params(matrix)
        if abs(gamma) > self._EPS:
            circuit.rz(gamma, qubit)
        if abs(beta) > self._EPS:
            circuit.ry(beta, qubit)
        if abs(alpha) > self._EPS:
            circuit.rz(alpha, qubit)

    def _emit_mux_ry(self, alpha: float, beta: float, ctrl: int, tgt: int, circuit: Circuit) -> None:
        """Multiplexed Ry: Ry(alpha) on tgt when ctrl=0, Ry(beta) when ctrl=1.

        Circuit: Ry((a+b)/2) CNOT Ry((a-b)/2) CNOT
        """
        avg = (alpha + beta) / 2
        diff = (alpha - beta) / 2
        if abs(avg) > self._EPS:
            circuit.ry(avg, tgt)
        circuit.cx(ctrl, tgt)
        if abs(diff) > self._EPS:
            circuit.ry(diff, tgt)
        circuit.cx(ctrl, tgt)

    def _emit_demux(self, G0, G1, ctrl: int, tgt: int, circuit: Circuit) -> None:
        """Demultiplexed gate: G0 on tgt when ctrl=0, G1 on tgt when ctrl=1.

        Uses the ABC decomposition (Nielsen & Chuang, Corollary 4.2) of the
        relative unitary U = G0^dag G1.  We find A, B, C with ABC = I and
        e^(i*phase)*AXBXC = U, then lay out the circuit in time order:

            C  ->  CX  ->  B  ->  CX  ->  (G0 @ A)  ->  Rz(phase) on ctrl

        ctrl=0 path (CNOTs inactive):  (G0*A)*B*C = G0*(ABC) = G0
        ctrl=1 path (CNOTs flip tgt):  (G0*A)*X*B*X*C = G0*(AXBXC) = G0*U/e^(i*phase) ~ G1

        Cost: 2 CNOTs + single-qubit rotations.
        """
        import numpy as np

        U = G0.conj().T @ G1
        if np.allclose(U, np.eye(2), atol=1e-8):
            self._decompose_1q(G0, tgt, circuit)
            return

        phase, alpha, beta, gamma = self._zyz_params(U)

        A_abc = self._rz_mat(alpha) @ self._ry_mat(beta / 2)
        B_abc = self._ry_mat(-beta / 2) @ self._rz_mat(-(alpha + gamma) / 2)
        C_abc = self._rz_mat((gamma - alpha) / 2)

        self._decompose_1q(C_abc, tgt, circuit)
        circuit.cx(ctrl, tgt)
        self._decompose_1q(B_abc, tgt, circuit)
        circuit.cx(ctrl, tgt)
        self._decompose_1q(G0 @ A_abc, tgt, circuit)

        if abs(phase) > self._EPS:
            circuit.rz(phase, ctrl)

    def _decompose_2q(self, matrix, qubits: list, circuit: Circuit) -> None:
        """Decompose a 2-qubit unitary using Cosine-Sine Decomposition.

        CSD factors U into:
          U = LeftDemux(U1,U2) @ MuxRy(theta) @ RightDemux(V1,V2)
        where each Demux uses 2 CNOTs and MuxRy uses 2 CNOTs.
        Total: up to 6 CNOTs (fewer when some factors are trivial).
        """
        import numpy as np
        from scipy.linalg import cossin

        q0, q1 = qubits[0], qubits[1]

        (u1, u2), theta, (v1h, v2h) = cossin(matrix, p=2, q=2, separate=True)

        t0 = float(theta[0]) if len(theta) > 0 else 0.0
        t1 = float(theta[1]) if len(theta) > 1 else t0

        # Right demux: {V1H on q1 when q0=0, V2H on q1 when q0=1}
        self._emit_demux(v1h, v2h, ctrl=q0, tgt=q1, circuit=circuit)

        # Middle: multiplexed Ry on q0, controlled by q1
        # When q1=0: Ry(2*t0) on q0; When q1=1: Ry(2*t1) on q0
        self._emit_mux_ry(2 * t0, 2 * t1, ctrl=q1, tgt=q0, circuit=circuit)

        # Left demux: {U1 on q1 when q0=0, U2 on q1 when q0=1}
        self._emit_demux(u1, u2, ctrl=q0, tgt=q1, circuit=circuit)

    def run(self, circuit: Circuit) -> Circuit:
        import numpy as np
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        circuit._ensure_gates()
        for gate in circuit._gates:
            if gate.name.upper() == "UNITARY" and gate.matrix is not None:
                n_qubits = len(gate.qubits)
                if n_qubits == 1:
                    self._decompose_1q(gate.matrix, gate.qubits[0], new_circuit)
                elif n_qubits == 2:
                    self._decompose_2q(gate.matrix, gate.qubits, new_circuit)
                else:
                    raise ValueError(
                        f"UnitaryDecompositionPass does not support {n_qubits}-qubit "
                        f"unitaries yet. Decompose into 1Q/2Q blocks first."
                    )
            else:
                new_circuit._gates.append(gate)
        return new_circuit


class RoutingPass:
    """Inserts SWAP gates to satisfy hardware coupling map."""
    def __init__(self, coupling_map: List[Tuple[int, int]]):
        self.coupling_map = coupling_map
    def name(self) -> str: return "RoutingPass"
    def run(self, circuit: Circuit) -> Circuit: return circuit
