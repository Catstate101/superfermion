"""Exact analytic oracle for every gate and every parameter slot.

Every matrix in :data:`GATES` is re-derived here from its mathematical
definition (Pauli exponentials, controlled blocks, basis permutations) instead
of being copied from the implementation's own tables. The comparison target is
``Circuit.to_unitary()``, the Rust path, so a sign, a factor of two, a swapped
parameter or a reversed control/target shows up as a non-zero difference --
with no Qiskit, PennyLane or qulacs installed.

Conventions used by this file (both are documented public behaviour):

* ``qubit k`` is bit ``k`` of the statevector index (little-endian, the same as
  ``sf.run(...).statevector`` and Qiskit's ``Operator``).
* a gate's *local* matrix is written in textbook layout, ``qubits[0]`` first /
  most significant, i.e. ``|q0 q1 ...>``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.domain

# All four rotation angles are distinct so that a swapped or dropped parameter
# slot cannot hide behind a coincidence.
THETA = 0.7
PHI = 1.1
LAM = -0.4
GAMMA = 0.9

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ── Analytic building blocks ────────────────────────────────────────────────


def expm_pauli(pauli: np.ndarray, angle: float) -> np.ndarray:
    """``exp(-i·angle·P/2)`` for a Pauli operator ``P``."""
    return math.cos(angle / 2) * np.eye(pauli.shape[0]) - 1j * math.sin(angle / 2) * pauli


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """The OpenQASM ``U(θ, φ, λ)`` single-qubit unitary."""
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [ct, -np.exp(1j * lam) * st],
            [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
        ],
        dtype=complex,
    )


def controlled(u: np.ndarray) -> np.ndarray:
    """``diag(1, 1) ⊕ u`` — identity unless the first (control) qubit is |1⟩."""
    m = np.eye(2 * u.shape[0], dtype=complex)
    m[u.shape[0] :, u.shape[0] :] = u
    return m


def permutation(mapping: dict[int, int], size: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=complex)
    for col, row in mapping.items():
        m[row, col] = 1.0
    return m


def toffoli() -> np.ndarray:
    """Flip ``q2`` when ``q0 = q1 = 1`` (textbook index bits ``q0 q1 q2``)."""
    m = np.zeros((8, 8), dtype=complex)
    for col in range(8):
        flip = 1 if (col >> 2) & 1 and (col >> 1) & 1 else 0
        m[col ^ flip, col] = 1.0
    return m


def fredkin() -> np.ndarray:
    """Swap ``q1`` and ``q2`` when ``q0 = 1``."""
    m = np.zeros((8, 8), dtype=complex)
    for col in range(8):
        q0, q1, q2 = (col >> 2) & 1, (col >> 1) & 1, col & 1
        if q0:
            q1, q2 = q2, q1
        m[(q0 << 2) | (q1 << 1) | q2, col] = 1.0
    return m


def embed_local(local: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """Place a textbook-ordered *local* matrix on *qubits* of an *n*-qubit space.

    ``qubits[0]`` is the most significant bit of the local index; ``qubit k``
    is bit ``k`` of the big index (little-endian statevector).
    """
    local = np.asarray(local, dtype=complex)
    k = len(qubits)
    size = 1 << k
    total = 1 << n
    out = np.zeros((total, total), dtype=complex)
    others = [q for q in range(n) if q not in qubits]

    for rest in range(1 << len(others)):
        base = 0
        for i, q in enumerate(others):
            if (rest >> i) & 1:
                base |= 1 << q
        for a in range(size):
            for b in range(size):
                amp = local[a, b]
                if amp == 0:
                    continue
                src = dst = base
                for m, q in enumerate(qubits):
                    shift = k - 1 - m  # qubits[0] is the most significant
                    if (b >> shift) & 1:
                        src |= 1 << q
                    if (a >> shift) & 1:
                        dst |= 1 << q
                out[dst, src] += amp
    return out


# ── Every gate the fluent API can build ─────────────────────────────────────

SQ = 1 / math.sqrt(2)

# name -> (n_qubits, qubits, builder, analytic local matrix)
GATES = {
    # 1 qubit, no parameters
    "id": (1, [0], lambda c: c.id(0), I2),
    "h": (1, [0], lambda c: c.h(0), np.array([[1, 1], [1, -1]], dtype=complex) * SQ),
    "x": (1, [0], lambda c: c.x(0), X),
    "y": (1, [0], lambda c: c.y(0), Y),
    "z": (1, [0], lambda c: c.z(0), Z),
    "s": (1, [0], lambda c: c.s(0), np.diag([1, 1j])),
    "sdg": (1, [0], lambda c: c.sdg(0), np.diag([1, -1j])),
    "t": (1, [0], lambda c: c.t(0), np.diag([1, np.exp(1j * math.pi / 4)])),
    "tdg": (1, [0], lambda c: c.tdg(0), np.diag([1, np.exp(-1j * math.pi / 4)])),
    "sx": (1, [0], lambda c: c.sx(0), np.exp(1j * math.pi / 4) * expm_pauli(X, math.pi / 2)),
    # 1 qubit, one parameter
    "rx": (1, [0], lambda c: c.rx(THETA, 0), expm_pauli(X, THETA)),
    "ry": (1, [0], lambda c: c.ry(THETA, 0), expm_pauli(Y, THETA)),
    "rz": (1, [0], lambda c: c.rz(THETA, 0), expm_pauli(Z, THETA)),
    "p": (1, [0], lambda c: c.p(PHI, 0), np.diag([1, np.exp(1j * PHI)])),
    "r1": (1, [0], lambda c: c.r1(PHI, 0), np.diag([1, np.exp(1j * PHI)])),
    # 1 qubit, three parameters
    "u": (1, [0], lambda c: c.u(THETA, PHI, LAM, 0), u3(THETA, PHI, LAM)),
    "u3": (1, [0], lambda c: c.u3(THETA, PHI, LAM, 0), u3(THETA, PHI, LAM)),
    # 2 qubits, no parameters
    "cnot": (2, [0, 1], lambda c: c.cnot(0, 1), controlled(X)),
    "cx": (2, [0, 1], lambda c: c.cx(0, 1), controlled(X)),
    "cz": (2, [0, 1], lambda c: c.cz(0, 1), controlled(Z)),
    "cy": (2, [0, 1], lambda c: c.cy(0, 1), controlled(Y)),
    "swap": (2, [0, 1], lambda c: c.swap(0, 1), permutation({0: 0, 1: 2, 2: 1, 3: 3}, 4)),
    "iswap": (
        2,
        [0, 1],
        lambda c: c.iswap(0, 1),
        np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    ),
    "ecr": (
        2,
        [0, 1],
        lambda c: c.ecr(0, 1),
        SQ * np.array([[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]]),
    ),
    # 2 qubits, one parameter
    "cp": (2, [0, 1], lambda c: c.cp(PHI, 0, 1), np.diag([1, 1, 1, np.exp(1j * PHI)])),
    "crx": (2, [0, 1], lambda c: c.crx(THETA, 0, 1), controlled(expm_pauli(X, THETA))),
    "cry": (2, [0, 1], lambda c: c.cry(THETA, 0, 1), controlled(expm_pauli(Y, THETA))),
    "crz": (2, [0, 1], lambda c: c.crz(THETA, 0, 1), controlled(expm_pauli(Z, THETA))),
    "rxx": (2, [0, 1], lambda c: c.rxx(THETA, 0, 1), expm_pauli(np.kron(X, X), THETA)),
    "ryy": (2, [0, 1], lambda c: c.ryy(THETA, 0, 1), expm_pauli(np.kron(Y, Y), THETA)),
    "rzz": (2, [0, 1], lambda c: c.rzz(THETA, 0, 1), expm_pauli(np.kron(Z, Z), THETA)),
    # 2 qubits, four parameters
    "cu": (2, [0, 1], lambda c: c.cu(THETA, PHI, LAM, 0, 1, GAMMA),
           controlled(np.exp(1j * GAMMA) * u3(THETA, PHI, LAM))),
    "cu3": (2, [0, 1], lambda c: c.cu3(THETA, PHI, LAM, 0, 1, GAMMA),
            controlled(np.exp(1j * GAMMA) * u3(THETA, PHI, LAM))),
    # 3 qubits
    "ccx": (3, [0, 1, 2], lambda c: c.ccx(0, 1, 2), toffoli()),
    "toffoli": (3, [0, 1, 2], lambda c: c.toffoli(0, 1, 2), toffoli()),
    "cswap": (3, [0, 1, 2], lambda c: c.cswap(0, 1, 2), fredkin()),
    "fredkin": (3, [0, 1, 2], lambda c: c.fredkin(0, 1, 2), fredkin()),
}


@pytest.mark.parametrize("name", sorted(GATES))
def test_gate_matches_analytic_definition(name):
    """``Circuit.to_unitary`` equals the textbook matrix, element for element."""
    n, qubits, build, local = GATES[name]
    circuit = build(sf.Circuit(n))
    expected = embed_local(local, qubits, n)
    obtained = np.asarray(circuit.to_unitary(), dtype=complex)

    assert obtained.shape == expected.shape == (1 << n, 1 << n)
    assert np.allclose(obtained, expected, atol=1e-12), (
        f"{name}: max|diff| = {np.max(np.abs(obtained - expected)):.3e}"
    )


@pytest.mark.parametrize("name", sorted(GATES))
def test_gate_is_unitary(name):
    n, _, build, _ = GATES[name]
    u = np.asarray(build(sf.Circuit(n)).to_unitary(), dtype=complex)
    assert np.allclose(u.conj().T @ u, np.eye(1 << n), atol=1e-12)


@pytest.mark.parametrize("name", sorted(GATES))
def test_record_matrix_is_the_textbook_matrix(name):
    """``GateRecord.to_unitary`` is textbook-ordered; ``Circuit.to_unitary`` is not.

    The record keeps ``qubits[0]`` as the most significant bit (``|q0 q1>``),
    while the circuit matrix addresses ``qubit k`` as bit ``k``. Bit-reversing
    the record reproduces the circuit matrix exactly -- the relationship
    documented on ``GateRecord.to_unitary``.
    """
    n, _, build, local = GATES[name]
    circuit = build(sf.Circuit(n))
    records = circuit._ensure_gates() if circuit._gates is None else circuit._gates
    assert len(records) == 1

    record = np.asarray(records[0].to_unitary(), dtype=complex)
    assert np.allclose(record, np.asarray(local, dtype=complex), atol=1e-12)

    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)]
    reversed_record = record[np.ix_(perm, perm)]
    assert np.allclose(
        reversed_record, np.asarray(circuit.to_unitary(), dtype=complex), atol=1e-12
    )


# ── Parameter slots ─────────────────────────────────────────────────────────


def slot_circuit(gate: str, params: list[float]):
    if gate == "cu":
        # cu(theta, phi, lam, control, target, gamma)
        return sf.Circuit(2).cu(*params[:3], 0, 1, params[3])
    return getattr(sf.Circuit(1), gate)(*params, 0)


@pytest.mark.parametrize(
    "gate, base_params",
    [("rx", [THETA]), ("ry", [THETA]), ("rz", [THETA]), ("p", [PHI]),
     ("r1", [PHI]), ("u", [THETA, PHI, LAM]), ("cu", [THETA, PHI, LAM, GAMMA])],
)
def test_each_parameter_slot_changes_the_matrix(gate, base_params):
    """A parameter routed to the wrong slot cannot hide behind a coincidence."""
    reference = np.asarray(slot_circuit(gate, base_params).to_unitary(), dtype=complex)
    for slot in range(len(base_params)):
        nudged = list(base_params)
        nudged[slot] += 0.37
        moved = np.asarray(slot_circuit(gate, nudged).to_unitary(), dtype=complex)
        assert not np.allclose(reference, moved, atol=1e-6), (
            f"{gate}: parameter slot {slot} does not reach the unitary"
        )


def test_p_and_r1_are_the_same_gate():
    a = np.asarray(sf.Circuit(1).p(PHI, 0).to_unitary(), dtype=complex)
    b = np.asarray(sf.Circuit(1).r1(PHI, 0).to_unitary(), dtype=complex)
    assert np.allclose(a, b, atol=0.0)


def test_cry_is_exactly_a_controlled_ry_block():
    """The IR has no ``CRy`` opcode; ``cry`` emits ``cu(θ, 0, 0)``."""
    a = np.asarray(sf.Circuit(2).cry(THETA, 0, 1).to_unitary(), dtype=complex)
    b = np.asarray(sf.Circuit(2).cu(THETA, 0.0, 0.0, 0, 1).to_unitary(), dtype=complex)
    assert np.allclose(a, b, atol=0.0)


# ── Placement, composition and the custom-unitary escape hatch ──────────────


@pytest.mark.parametrize(
    "name, qubits",
    [
        ("cnot", [2, 0]),
        ("crz", [2, 0]),
        ("cp", [1, 2]),
        ("swap", [2, 1]),
        ("h", [2]),
        ("rx", [1]),
    ],
)
def test_gate_on_non_adjacent_qubits(name, qubits):
    """Qubit indices are honoured exactly, not just the first ``k`` lines."""
    n = 3
    _, _, _, local = GATES[name]
    from superfermion.circuit import Circuit

    circuit = getattr(Circuit(n), name)(*(_gate_args(name, qubits)), *qubits)
    expected = embed_local(local, qubits, n)
    assert np.allclose(np.asarray(circuit.to_unitary(), dtype=complex), expected, atol=1e-12)


def _gate_args(name: str, qubits: list[int]) -> list[float]:
    """The parameter list for *name* (empty for the fixed gates)."""
    return {"rx": [THETA], "crz": [THETA], "cp": [PHI]}.get(name, [])


def test_unitary_composes_right_to_left():
    circuit = sf.Circuit(2).h(0).cnot(0, 1).rz(THETA, 1)
    single = [
        np.asarray(sf.Circuit(2).h(0).to_unitary(), dtype=complex),
        np.asarray(sf.Circuit(2).cnot(0, 1).to_unitary(), dtype=complex),
        np.asarray(sf.Circuit(2).rz(THETA, 1).to_unitary(), dtype=complex),
    ]
    expected = single[2] @ single[1] @ single[0]
    assert np.allclose(np.asarray(circuit.to_unitary(), dtype=complex), expected, atol=1e-12)


def test_custom_unitary_gate_matches_its_matrix():
    local = np.array([[0, 1], [1, 0]], dtype=complex)  # X
    circuit = sf.Circuit(2).unitary(local, [0])
    assert np.allclose(
        np.asarray(circuit.to_unitary(), dtype=complex), embed_local(local, [0], 2), atol=1e-12
    )
