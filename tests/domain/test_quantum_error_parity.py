"""Custom quantum errors + per-instruction filters — Qiskit Aer parity.

Every feature added to ``superfermion.noise`` is pinned against an
independent oracle, never against itself:

  * custom Kraus set        ↔ ``qiskit_aer.noise.kraus_error``
  * coherent unitary        ↔ ``coherent_unitary_error``
  * mixed unitary           ↔ ``mixed_unitary_error``
  * reset channel           ↔ ``reset_error`` (full superoperator, and
                              end-to-end on a gate instruction)
  * per-instruction filter  ↔ ``NoiseModel.add_quantum_error(err, instrs, qubits)``
  * ``compose``/``power``/``tensor`` ↔ the same ``QuantumError`` algebra
  * static ZZ crosstalk     ↔ ``coherent_unitary_error(exp(-i θ/2 Z⊗Z))``
  * cycle-aware crosstalk   ↔ an independent numpy cycle model (Aer has no
                              notion of concurrent-instruction coupling)

Each parity assertion is accompanied by a "the effect is real" guard, so a
pair of accidentally noiseless states can never pass as parity.

Layout note: ``metadata['density_matrix']`` is the q0-first public matrix,
converted to the little-endian (Qiskit) basis with ``_to_little_endian``.

Reach note: ``reset`` now stays on the density-matrix engine — the exact
CPTP map ``ρ -> |0><0|_q ⊗ Tr_q ρ`` is applied by ``apply_reset`` — so the
reset channel is pinned both as a superoperator and end-to-end on a
``reset`` instruction.  Every other method (statevector, MPS) still routes
a reset circuit to the trajectory engine.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion.devices.noisy_dm_state import _to_little_endian
from superfermion.noise import ErrorChannel, kraus_reset

pytestmark = pytest.mark.domain

qiskit = pytest.importorskip("qiskit")
aer_noise = pytest.importorskip("qiskit_aer.noise")
AerSimulator = pytest.importorskip("qiskit_aer").AerSimulator

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SuperOp  # noqa: E402

AerNoiseModel = aer_noise.NoiseModel

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PARITY = 1e-10


# ── harness ──────────────────────────────────────────────────────────
def _sf_rho(circ, model, n):
    """SF density matrix in the little-endian (Qiskit) basis."""
    res = sf.run(circ, method="density_matrix", shots=0, noise_model=model)
    return _to_little_endian(np.asarray(res.metadata["density_matrix"]), n)


def _clean_rho(circ, n):
    res = sf.run(circ, method="density_matrix", shots=0)
    return _to_little_endian(np.asarray(res.metadata["density_matrix"]), n)


def _aer_rho(qc, model):
    qc = qc.copy()
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=model)
    return np.asarray(sim.run(qc).result().data()["density_matrix"])


def _diff(a, b):
    return float(np.max(np.abs(a - b)))


def _superop(kraus):
    """Superoperator of a Kraus set, row-major vec convention."""
    return sum(np.kron(k, k.conj()) for k in kraus)


def _assert_parity(sf_rho, aer_rho, clean_rho, tag=""):
    """Aer equality AND a non-trivial effect — never a vacuous comparison."""
    assert _diff(sf_rho, aer_rho) < PARITY, f"{tag}: SF != Aer"
    assert _diff(aer_rho, clean_rho) > 1e-3, f"{tag}: the noise did nothing"


def _qk_bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _sf_bell():
    return sf.Circuit(2).h(0).cnot(0, 1)


def _sf_phase_probe():
    """A 2q-gate circuit whose state is NOT an eigenstate of ZZ or of AD(q0).

    After ``x(0); h(1); cx(0,1)`` the pair sits in (|11> + |10>)/sqrt(2):
    odd and even ZZ parity, and q0 is populated, so a diagonal 2q error and
    a damping error on wire 0 both change the density matrix.  A Bell state
    would be an eigenstate of both and every parity check would be vacuous.
    """
    return sf.Circuit(2).x(0).h(1).cnot(0, 1)


def _qk_phase_probe():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    qc.cx(0, 1)
    return qc


# amplitude-damping Kraus set, hand-written (not imported from the API)
G = 0.3
K_AD0 = np.array([[1, 0], [0, np.sqrt(1 - G)]], dtype=np.complex128)
K_AD1 = np.array([[0, np.sqrt(G)], [0, 0]], dtype=np.complex128)
AD = [K_AD0, K_AD1]


class TestCustomKraus:
    def test_matches_aer(self):
        model = sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["h"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(AD), "h", [0])
        _assert_parity(
            _sf_rho(_sf_bell(), model, 2),
            _aer_rho(_qk_bell(), aer),
            _clean_rho(_sf_bell(), 2),
            "add_kraus",
        )

    def test_two_qubit_kraus_matches_aer(self):
        """Damping on wire 0 of the pair — also pins the (0,1) kron order."""
        pair = [np.kron(K_AD0, I2), np.kron(K_AD1, I2)]
        model = sf.NoiseModel().add_kraus(pair, qubits=(0, 1), instructions=["cx"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(pair), "cx", [0, 1])
        _assert_parity(
            _sf_rho(_sf_phase_probe(), model, 2),
            _aer_rho(_qk_phase_probe(), aer),
            _clean_rho(_sf_phase_probe(), 2),
            "add_kraus 2q",
        )

    def test_qubit_count_inferred_from_shape(self):
        assert ErrorChannel.from_kraus(AD).n_qubits == 1
        assert ErrorChannel.from_kraus([np.eye(4)]).n_qubits == 2

    def test_non_trace_preserving_set_is_rejected(self):
        bad = [np.array([[1, 0], [0, 0.5]], dtype=np.complex128)]
        with pytest.raises(ValueError, match="sum K\\^dagger K"):
            sf.NoiseModel().add_kraus(bad)

    def test_wrong_shape_is_rejected(self):
        bad = [np.eye(3, dtype=np.complex128)]
        with pytest.raises(ValueError, match="2x2 \\(1 qubit\\) or 4x4"):
            sf.NoiseModel().add_kraus(bad)

    def test_empty_set_is_rejected(self):
        with pytest.raises(ValueError, match="at least one Kraus"):
            sf.NoiseModel().add_kraus([])


class TestCoherentUnitary:
    def test_matches_aer(self):
        u = np.array([[1, 0], [0, np.exp(1j * 0.4)]], dtype=np.complex128)
        model = sf.NoiseModel().add_unitary(u, instructions=["h"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.coherent_unitary_error(u), "h", [0])
        _assert_parity(
            _sf_rho(_sf_bell(), model, 2),
            _aer_rho(_qk_bell(), aer),
            _clean_rho(_sf_bell(), 2),
            "add_unitary",
        )

    def test_two_qubit_ordering_matches_aer(self):
        """``qubits=(0, 1)`` must mean Qiskit's ``[0, 1]`` (q0 = high factor).

        The diagonal is deliberately asymmetric (phase on |11> only), so a
        reversed convention would move the phase to |10> and fail.
        """
        u = np.diag([1, 1, 1, 1j]).astype(np.complex128)
        model = sf.NoiseModel().add_unitary(u, qubits=(0, 1), instructions=["cx"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.coherent_unitary_error(u), "cx", [0, 1])
        rho = _sf_rho(_sf_bell(), model, 2)
        _assert_parity(rho, _aer_rho(_qk_bell(), aer), _clean_rho(_sf_bell(), 2), "2q order")
        # the phase really sits on |11> and the population is untouched
        assert abs(rho[3, 3] - 0.5) < 1e-12
        assert rho[3, 3].real > 0.49

    def test_non_unitary_is_rejected(self):
        with pytest.raises(ValueError, match="not unitary"):
            sf.NoiseModel().add_unitary(np.array([[1, 0], [0, 2]], dtype=np.complex128))


class TestMixedUnitary:
    def test_matches_aer(self):
        u1 = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        u2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        model = sf.NoiseModel().add_mixed_unitary([u1, u2], [0.7, 0.3], instructions=["h"])
        aer = AerNoiseModel()
        aer.add_quantum_error(
            aer_noise.mixed_unitary_error([(u1, 0.7), (u2, 0.3)]), "h", [0]
        )
        _assert_parity(
            _sf_rho(_sf_bell(), model, 2),
            _aer_rho(_qk_bell(), aer),
            _clean_rho(_sf_bell(), 2),
            "add_mixed_unitary",
        )

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            sf.NoiseModel().add_mixed_unitary([X, Z], [0.4, 0.4])

    def test_negative_probability_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            sf.NoiseModel().add_mixed_unitary([X, Z], [1.2, -0.2])


class TestResetError:
    def test_superoperator_equals_aer(self):
        """Five-term Kraus set vs ``SuperOp(reset_error(p0, p1))``."""
        p0, p1 = 0.1, 0.2
        expected = np.array(
            [
                [1 - p1, 0, 0, p0],
                [0, 1 - p0 - p1, 0, 0],
                [0, 0, 1 - p0 - p1, 0],
                [p1, 0, 0, 1 - p0],
            ],
            dtype=np.complex128,
        )
        assert _diff(_superop(kraus_reset(p0, p1)), expected) < 1e-12
        assert (
            _diff(_superop(kraus_reset(p0, p1)), SuperOp(aer_noise.reset_error(p0, p1)).data)
            < 1e-12
        )

    def test_kraus_set_is_trace_preserving(self):
        assert ErrorChannel.from_reset(0.1, 0.2).is_trace_preserving
        assert ErrorChannel.from_reset(0.0, 1.0).is_trace_preserving

    def test_channel_matches_aer_on_a_gate(self):
        qk = QuantumCircuit(1)
        qk.h(0)
        model = sf.NoiseModel().add_reset_error(0.1, 0.2, instructions=["h"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.reset_error(0.1, 0.2), "h", [0])
        _assert_parity(
            _sf_rho(sf.Circuit(1).h(0), model, 1),
            _aer_rho(qk, aer),
            _clean_rho(sf.Circuit(1).h(0), 1),
            "reset error",
        )

    def test_default_instructions_target_reset(self):
        model = sf.NoiseModel().add_reset_error(0.1)
        assert model.single_qubit_channels[-1].instructions == ("Reset",)

    def test_populations_after_pure_reset(self):
        """|1> under reset(0.1, 0.2): P(1) = 0.2 + 0.7*1 = 0.9."""
        qk = QuantumCircuit(1)
        qk.x(0)
        model = sf.NoiseModel().add_reset_error(0.1, 0.2, instructions=["x"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.reset_error(0.1, 0.2), "x", [0])
        rho_sf = _sf_rho(sf.Circuit(1).x(0), model, 1)
        assert rho_sf[1, 1].real == pytest.approx(0.9, abs=1e-12)
        assert _diff(rho_sf, _aer_rho(qk, aer)) < PARITY


class TestInstructionTargeting:
    def test_filtered_channel_is_silent_on_other_gates(self):
        model = sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["x"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(AD), "x", [0])
        qk = QuantumCircuit(2)
        qk.h(0)
        rho = _sf_rho(sf.Circuit(2).h(0), model, 2)
        assert _diff(rho, _clean_rho(sf.Circuit(2).h(0), 2)) < 1e-12
        assert _diff(rho, _aer_rho(qk, aer)) < PARITY

    def test_filtered_channel_fires_only_on_its_gate(self):
        model = sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["x"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(AD), "x", [0])
        qk = QuantumCircuit(2)
        qk.h(0)
        qk.x(0)
        circ = sf.Circuit(2).h(0).x(0)
        _assert_parity(
            _sf_rho(circ, model, 2), _aer_rho(qk, aer), _clean_rho(circ, 2), "filter on x"
        )

    def test_filter_wins_over_per_touch_placement(self):
        """Default placement is per-touch; an explicit filter still narrows it.

        Without the filter the channel would also fire on the ``cx``, so the
        filtered run must differ from the unfiltered one.
        """
        circ = _sf_bell()
        filtered = _sf_rho(
            circ,
            sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["h"]),
            2,
        )
        per_touch = _sf_rho(circ, sf.NoiseModel().add_kraus(AD, qubits=0), 2)
        assert _diff(filtered, per_touch) > 1e-3
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(AD), "h", [0])
        assert _diff(filtered, _aer_rho(_qk_bell(), aer)) < PARITY

    def test_two_non_commuting_filters_merge_on_a_shared_gate(self):
        """Different filters ⇒ Python cannot pre-compose; Rust merges per gate.

        Z-phase on ``ry`` and X on ``ry,x`` both fire on ``ry``.  With
        non-commuting unitaries the add order is observable (0.36 apart), so
        this pins both the merge and its ordering.
        """
        zph = np.diag([1, np.exp(1j * 0.7)]).astype(np.complex128)
        circ = sf.Circuit(1).ry(0.6, 0)
        qk = QuantumCircuit(1)
        qk.ry(0.6, 0)

        first = (
            sf.NoiseModel()
            .add_unitary(zph, qubits=0, instructions=["ry"])
            .add_unitary(X, qubits=0, instructions=["ry", "x"])
        )
        second = (
            sf.NoiseModel()
            .add_unitary(X, qubits=0, instructions=["ry", "x"])
            .add_unitary(zph, qubits=0, instructions=["ry"])
        )
        aer1 = AerNoiseModel()
        aer1.add_quantum_error(aer_noise.coherent_unitary_error(zph), "ry", [0])
        aer1.add_quantum_error(aer_noise.coherent_unitary_error(X), ["ry", "x"], [0])
        aer2 = AerNoiseModel()
        aer2.add_quantum_error(aer_noise.coherent_unitary_error(X), ["ry", "x"], [0])
        aer2.add_quantum_error(aer_noise.coherent_unitary_error(zph), "ry", [0])

        rho1 = _sf_rho(circ, first, 1)
        rho2 = _sf_rho(circ, second, 1)
        assert _diff(rho1, _aer_rho(qk, aer1)) < PARITY
        assert _diff(rho2, _aer_rho(qk, aer2)) < PARITY
        assert _diff(rho1, rho2) > 1e-3, "add order must be observable here"

    def test_two_qubit_filter_matches_aer(self):
        pair = [np.kron(K_AD0, I2), np.kron(K_AD1, I2)]
        model = sf.NoiseModel().add_kraus(pair, qubits=(0, 1), instructions=["cx"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.kraus_error(pair), "cx", [0, 1])
        _assert_parity(
            _sf_rho(_sf_phase_probe(), model, 2),
            _aer_rho(_qk_phase_probe(), aer),
            _clean_rho(_sf_phase_probe(), 2),
            "2q filter",
        )

    def test_two_qubit_reversed_pair_is_silent(self):
        pair = [np.kron(K_AD0, I2), np.kron(K_AD1, I2)]
        model = sf.NoiseModel().add_kraus(pair, qubits=(1, 0), instructions=["cx"])
        circ = _sf_phase_probe()
        rho = _sf_rho(circ, model, 2)
        # silent: the state must be exactly the clean one
        assert _diff(rho, _clean_rho(circ, 2)) < 1e-12

    def test_instruction_matching_is_case_insensitive(self):
        lower = _sf_rho(_sf_bell(), sf.NoiseModel().add_kraus(AD, 0, ["cx"]), 2)
        upper = _sf_rho(_sf_bell(), sf.NoiseModel().add_kraus(AD, 0, ["CX"]), 2)
        assert np.array_equal(lower, upper)

    def test_unknown_instruction_is_rejected(self):
        with pytest.raises(ValueError, match="unknown instruction"):
            sf.NoiseModel().add_kraus(AD, instructions=["not_a_gate"])

    def test_decomposed_instruction_is_rejected(self):
        """``u3`` is lowered before simulation, so a filter on it never fires."""
        with pytest.raises(ValueError, match="decomposed before simulation"):
            sf.NoiseModel().add_kraus(AD, instructions=["u3"])

    def test_empty_instruction_list_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            sf.NoiseModel().add_kraus(AD, instructions=[])


class TestErrorChannelAlgebra:
    def test_compose_matches_aer(self):
        ad = ErrorChannel.from_kraus(AD)
        assert _diff(_superop(ad.compose(ad).kraus), SuperOp(
            aer_noise.kraus_error(AD).compose(aer_noise.kraus_error(AD))
        ).data) < 1e-12

    def test_power_matches_aer(self):
        """Aer rejects ``power(0)``; SF defines it as the identity channel."""
        ad = ErrorChannel.from_kraus(AD)
        aer_err = aer_noise.kraus_error(AD)
        for n in (1, 3):
            assert _diff(
                _superop(ad.power(n).kraus), SuperOp(aer_err.power(n)).data
            ) < 1e-12, n

    def test_tensor_matches_aer(self):
        """Distinct factors so the kron order is pinned, not symmetric."""
        ad = ErrorChannel.from_kraus(AD)
        zph = ErrorChannel.from_unitary(Z)
        aer_prod = aer_noise.kraus_error(AD).tensor(
            aer_noise.coherent_unitary_error(Z)
        )
        assert _diff(_superop(ad.tensor(zph).kraus), SuperOp(aer_prod).data) < 1e-12
        # and the product is genuinely asymmetric, so the check above can fail
        assert _diff(
            _superop(ad.tensor(zph).kraus), _superop(zph.tensor(ad).kraus)
        ) > 1e-6

    def test_tensor_wire_order_on_the_engine(self):
        """``AD.tensor(Z)`` on (0,1): Z on wire 0, damping on wire 1 (Qiskit)."""
        ad = ErrorChannel.from_kraus(AD)
        zph = ErrorChannel.from_unitary(Z)
        model = sf.NoiseModel().add_quantum_error(
            ad.tensor(zph), qubits=(0, 1), instructions=["cx"]
        )
        aer = AerNoiseModel()
        aer.add_quantum_error(
            aer_noise.kraus_error(AD).tensor(aer_noise.coherent_unitary_error(Z)),
            "cx",
            [0, 1],
        )
        _assert_parity(
            _sf_rho(_sf_phase_probe(), model, 2),
            _aer_rho(_qk_phase_probe(), aer),
            _clean_rho(_sf_phase_probe(), 2),
            "tensor on engine",
        )

    def test_compose_is_self_after_other(self):
        """``a.compose(b)`` = apply b, then a — so S = S_a @ S_b."""
        a = ErrorChannel.from_unitary(Z)
        b = ErrorChannel.from_kraus(AD)
        composed = _superop(a.compose(b).kraus)
        assert _diff(composed, _superop(a.kraus) @ _superop(b.kraus)) < 1e-12
        # sequential construction: b first, then a, equals compose(a, b)
        model_seq = sf.NoiseModel().add_quantum_error(b, instructions=["h"])
        model_seq.add_quantum_error(a, instructions=["h"])
        model_cmp = sf.NoiseModel().add_quantum_error(a.compose(b), instructions=["h"])
        circ = sf.Circuit(1).h(0)
        assert _diff(_sf_rho(circ, model_seq, 1), _sf_rho(circ, model_cmp, 1)) < 1e-12

    def test_compose_rejects_mismatched_qubit_counts(self):
        one = ErrorChannel.from_kraus(AD)
        two = ErrorChannel.from_unitary(np.eye(4))
        with pytest.raises(ValueError, match="same number of qubits"):
            one.compose(two)

    def test_tensor_rejects_beyond_two_qubits(self):
        two = ErrorChannel.from_unitary(np.eye(4))
        with pytest.raises(ValueError, match="beyond 2 qubits"):
            two.tensor(two)

    def test_power_zero_is_the_identity_channel(self):
        ident = ErrorChannel.from_kraus(AD).power(0)
        assert _diff(_superop(ident.kraus), np.eye(4)) < 1e-12

    def test_trace_preservation_survives_algebra(self):
        ad = ErrorChannel.from_kraus(AD)
        assert ad.compose(ad).is_trace_preserving
        assert ad.power(4).is_trace_preserving
        assert ad.tensor(ad).is_trace_preserving

    def test_error_channel_is_exported_at_top_level(self):
        assert sf.ErrorChannel is ErrorChannel


# ── cycle-aware crosstalk: independent numpy cycle model ─────────────
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def _le_full(ops, n):
    """Kron-embed per-wire operators (``ops[q]`` acts on wire q) in LE order."""
    full = np.eye(1, dtype=np.complex128)
    for q in range(n - 1, -1, -1):
        full = np.kron(full, ops[q])
    return full


def _apply_le(rho, ops, n):
    u = _le_full(ops, n)
    return u @ rho @ u.conj().T


def _zero_rho(n):
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def _zz_kick(rho, edge, theta, n, times=1):
    """exp(-i θ/2 Z_a Z_b) applied ``times`` times, LE basis."""
    parity = _le_full([Z if q in edge else I2 for q in range(n)], n)
    u = np.cos(theta / 2) * np.eye(2**n) - 1j * np.sin(theta / 2) * parity
    for _ in range(times):
        rho = u @ rho @ u.conj().T
    return rho


def _cnot01_le():
    """CNOT, control wire 0, target wire 1, LE basis."""
    return np.kron(I2, P0) + np.kron(X, P1)


def _depol_2q_le(rho, p, edge, n):
    """SF total-convention 2q depolarizing: (1-p)ρ + p/15 Σ_{P≠I} PρP."""
    a, b = edge
    acc = (1 - p) * rho
    for pa in (I2, X, Y, Z):
        for pb in (I2, X, Y, Z):
            if pa is I2 and pb is I2:
                continue
            ops = [I2] * n
            ops[a], ops[b] = pa, pb
            acc = acc + (p / 15) * _apply_le(rho, ops, n)
    return acc


class TestCrosstalk:
    def test_static_zz_matches_coherent_unitary(self):
        theta = 0.35
        uzz = (
            np.cos(theta / 2) * np.eye(4)
            - 1j * np.sin(theta / 2) * np.kron(Z, Z)
        ).astype(np.complex128)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, instructions=["cx"])
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.coherent_unitary_error(uzz), "cx", [0, 1])
        _assert_parity(
            _sf_rho(_sf_phase_probe(), model, 2),
            _aer_rho(_qk_phase_probe(), aer),
            _clean_rho(_sf_phase_probe(), 2),
            "crosstalk ZZ",
        )

    def test_crosstalk_scales_with_the_angle(self):
        circ = _sf_phase_probe()
        small = _sf_rho(
            circ, sf.NoiseModel().add_crosstalk([(0, 1)], 0.1, instructions=["cx"]), 2
        )
        large = _sf_rho(
            circ, sf.NoiseModel().add_crosstalk([(0, 1)], 0.8, instructions=["cx"]), 2
        )
        clean = _clean_rho(circ, 2)
        assert 1e-4 < _diff(small, clean) < _diff(large, clean)

    def test_crosstalk_adds_depolarizing(self):
        theta, rate = 0.3, 0.05
        uzz = (
            np.cos(theta / 2) * np.eye(4)
            - 1j * np.sin(theta / 2) * np.kron(Z, Z)
        ).astype(np.complex128)
        # Aer's depolarizing_error uses the lambda convention; SF's default is
        # the total convention, so the qiskit-convention channel is the peer.
        model_q = sf.NoiseModel().add_crosstalk([(0, 1)], theta, instructions=["cx"])
        model_q.add_depolarizing(
            rate, n_qubits=2, convention="qiskit", qubits=(0, 1), instructions=["cx"]
        )
        aer = AerNoiseModel()
        aer.add_quantum_error(aer_noise.coherent_unitary_error(uzz), "cx", [0, 1])
        aer.add_quantum_error(aer_noise.depolarizing_error(rate, 2), "cx", [0, 1])
        _assert_parity(
            _sf_rho(_sf_phase_probe(), model_q, 2),
            _aer_rho(_qk_phase_probe(), aer),
            _clean_rho(_sf_phase_probe(), 2),
            "crosstalk + depolarizing",
        )
        # and the total-convention shorthand (error_rate=) is itself non-trivial
        shorthand = sf.NoiseModel().add_crosstalk(
            [(0, 1)], theta, error_rate=rate, instructions=["cx"]
        )
        assert (
            _diff(_sf_rho(_sf_phase_probe(), shorthand, 2), _clean_rho(_sf_phase_probe(), 2))
            > 1e-3
        )

    def test_empty_edge_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            sf.NoiseModel().add_crosstalk([], 0.1)

    # ── cycle-aware (spectator) mode ─────────────────────────────────
    def test_crosstalk_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="'static' or 'spectator'"):
            sf.NoiseModel().add_crosstalk([(0, 1)], 0.1, mode="sideways")

    def test_spectator_mode_reaches_the_rust_channel_modes(self):
        spectator = sf.NoiseModel().add_crosstalk(
            [(0, 1)], 0.1, error_rate=0.05, mode="spectator"
        )
        assert spectator.to_rust_kraus_2q_modes() == [1, 1]
        static = sf.NoiseModel().add_crosstalk([(0, 1)], 0.1)
        assert static.to_rust_kraus_2q_modes() is None
        plain = sf.NoiseModel().add_depolarizing(0.01, n_qubits=2)
        assert plain.to_rust_kraus_2q_modes() is None

    def test_spectator_leaves_an_idle_edge_alone(self):
        circ = sf.Circuit(3).x(2)  # nothing touches wire 0 or 1
        clean = _clean_rho(circ, 3)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], 0.35, mode="spectator")
        assert _diff(_sf_rho(circ, model, 3), clean) < 1e-12
        # non-vacuous: the same channel fires once a neighbour is busy
        driven = sf.Circuit(3).h(0)
        assert _diff(_sf_rho(driven, model, 3), _clean_rho(driven, 3)) > 1e-3

    def test_spectator_zz_kicks_when_the_partner_idles(self):
        theta = 0.31
        circ = sf.Circuit(3).h(0)  # (|000> + |100>)/sqrt(2): not ZZ-diagonal
        clean = _clean_rho(circ, 3)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, mode="spectator")
        got = _sf_rho(circ, model, 3)
        assert _diff(got, _zz_kick(clean, (0, 1), theta, 3)) < PARITY
        assert _diff(got, clean) > 1e-3

    def test_spectator_counts_driven_endpoints_in_one_cycle(self):
        theta = 0.31
        circ = sf.Circuit(3).h(0).h(1)  # disjoint -> one cycle, two endpoints
        clean = _clean_rho(circ, 3)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, mode="spectator")
        got = _sf_rho(circ, model, 3)
        assert _diff(got, _zz_kick(clean, (0, 1), theta, 3, times=2)) < PARITY
        assert _diff(got, _zz_kick(clean, (0, 1), theta, 3)) > 1e-3

    def test_spectator_kicks_land_at_the_cycle_boundary(self):
        theta = 0.31
        circ = sf.Circuit(2).h(0).cnot(0, 1)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, mode="spectator")
        # cycle 1: h(0) only -> one kick; cycle 2: cnot on the edge -> two.
        want = _zz_kick(_apply_le(_zero_rho(2), [H, I2], 2), (0, 1), theta, 2)
        cx = _cnot01_le()
        want = _zz_kick(cx @ want @ cx.conj().T, (0, 1), theta, 2, times=2)
        got = _sf_rho(circ, model, 2)
        assert _diff(got, want) < PARITY
        assert _diff(got, _clean_rho(circ, 2)) > 1e-3
        # static mode folds onto the gate, whose output is ZZ-diagonal:
        # the exposure during the h(0) cycle is missed entirely
        static = sf.NoiseModel().add_crosstalk([(0, 1)], theta, instructions=["cx"])
        assert _diff(_sf_rho(circ, static, 2), _clean_rho(circ, 2)) < PARITY

    def test_spectator_double_kicks_on_the_edge_gate(self):
        theta = 0.31
        circ = sf.Circuit(2).h(1).cnot(0, 1)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, mode="spectator")
        # cycle 1: h(1) touches only endpoint b -> one unit; cycle 2: the cnot
        # touches both endpoints -> two more.  The control stays |0>, so the
        # state (|00> + |01>)/sqrt(2) survives the cnot unchanged and every
        # kick is a parity-dependent phase: the three units are observable.
        want = _apply_le(_zero_rho(2), [I2, H], 2)
        want = _zz_kick(want, (0, 1), theta, 2, times=1)
        cx = _cnot01_le()
        want = _zz_kick(cx @ want @ cx.conj().T, (0, 1), theta, 2, times=2)
        got = _sf_rho(circ, model, 2)
        assert _diff(got, want) < PARITY
        # one unit for the edge gate instead of two would leave 2, not 3
        assert _diff(got, _zz_kick(_clean_rho(circ, 2), (0, 1), theta, 2, 2)) > 1e-3

    def test_spectator_respects_the_instruction_filter(self):
        theta = 0.31
        circ = sf.Circuit(2).h(0).h(1)
        clean = _clean_rho(circ, 2)
        only_h = sf.NoiseModel().add_crosstalk(
            [(0, 1)], theta, instructions=["h"], mode="spectator"
        )
        assert _diff(
            _sf_rho(circ, only_h, 2), _zz_kick(clean, (0, 1), theta, 2, times=2)
        ) < PARITY
        only_cx = sf.NoiseModel().add_crosstalk(
            [(0, 1)], theta, instructions=["cx"], mode="spectator"
        )
        assert _diff(_sf_rho(circ, only_cx, 2), clean) < 1e-12

    def test_static_mode_still_fires_once_after_the_edge_gate(self):
        theta = 0.31
        circ = _sf_phase_probe()
        clean = _clean_rho(circ, 2)
        model = sf.NoiseModel().add_crosstalk([(0, 1)], theta, instructions=["cx"])
        assert model.to_rust_kraus_2q_modes() is None
        got = _sf_rho(circ, model, 2)
        assert _diff(got, _zz_kick(clean, (0, 1), theta, 2)) < PARITY
        assert _diff(got, clean) > 1e-3

    def test_spectator_depolarizing_inflates_per_unit(self):
        p = 0.2
        circ = sf.Circuit(2).h(0).h(1)  # one cycle, two driven endpoints
        clean = _clean_rho(circ, 2)
        model = sf.NoiseModel().add_crosstalk(
            [(0, 1)], 0.0, error_rate=p, mode="spectator"
        )
        got = _sf_rho(circ, model, 2)
        twice = _depol_2q_le(_depol_2q_le(clean, p, (0, 1), 2), p, (0, 1), 2)
        assert _diff(got, twice) < PARITY
        assert _diff(got, _depol_2q_le(clean, p, (0, 1), 2)) > 1e-3

    def test_spectator_mode_survives_dict_round_trip(self):
        theta = 0.31
        model = sf.NoiseModel().add_crosstalk(
            [(0, 1)], theta, error_rate=0.05, mode="spectator"
        )
        data = model.to_dict()
        assert [c.get("mode") for c in data["two_qubit_channels"]] == [
            "spectator",
            "spectator",
        ]
        rebuilt = sf.NoiseModel.from_dict(data)
        assert rebuilt.to_rust_kraus_2q_modes() == [1, 1]
        circ = _sf_phase_probe()
        assert _diff(_sf_rho(circ, model, 2), _sf_rho(circ, rebuilt, 2)) < 1e-12
        # untouched default models keep the historical dict shape
        plain = sf.NoiseModel().add_depolarizing(0.01).to_dict()
        assert "mode" not in plain["single_qubit_channels"][0]

    def test_spectator_preserves_trace_and_hermiticity(self):
        model = sf.NoiseModel().add_crosstalk(
            [(0, 1)], 0.31, error_rate=0.05, mode="spectator"
        )
        rho = _sf_rho(_sf_phase_probe(), model, 2)
        assert abs(np.trace(rho) - 1.0) < 1e-10
        assert _diff(rho, rho.conj().T) < 1e-12
        assert np.linalg.eigvalsh(rho).min() > -1e-10


class TestSerialization:
    def test_custom_kraus_round_trips(self):
        model = sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["h"])
        rebuilt = sf.NoiseModel.from_dict(model.to_dict())
        circ = sf.Circuit(1).h(0)
        assert np.array_equal(_sf_rho(circ, model, 1), _sf_rho(circ, rebuilt, 1))

    def test_instructions_round_trip(self):
        model = sf.NoiseModel().add_unitary(Z, qubits=1, instructions=["cx"])
        data = model.to_dict()
        assert data["single_qubit_channels"][0]["instructions"] == ["CNOT"]
        rebuilt = sf.NoiseModel.from_dict(data)
        assert rebuilt.single_qubit_channels[0].instructions == ("CNOT",)
        assert rebuilt.single_qubit_channels[0].qubits == (1,)

    def test_reset_channel_round_trips(self):
        model = sf.NoiseModel().add_reset_error(0.1, 0.2)
        rebuilt = sf.NoiseModel.from_dict(model.to_dict())
        assert rebuilt.single_qubit_channels[0].instructions == ("Reset",)
        assert _diff(
            _superop(rebuilt.single_qubit_channels[0].kraus),
            _superop(kraus_reset(0.1, 0.2)),
        ) < 1e-12


class TestFastPath:
    """An unfiltered model must stay on the historical binding call."""

    def test_unfiltered_model_never_passes_instruction_filters(self, monkeypatch):
        from superfermion.devices import rust_device

        seen = []
        original = rust_device._dm_binding_call

        def spy(method_name, dag, noise_ops, noise_ops_2q, placement,
                noise_2q_targets=None, noise_1q_instrs=None, noise_2q_instrs=None,
                noise_2q_modes=None):
            seen.append((noise_1q_instrs, noise_2q_instrs, noise_2q_modes))
            return original(method_name, dag, noise_ops, noise_ops_2q, placement,
                            noise_2q_targets, noise_1q_instrs, noise_2q_instrs,
                            noise_2q_modes)

        monkeypatch.setattr(rust_device, "_dm_binding_call", spy)
        plain = (
            sf.NoiseModel()
            .add_depolarizing(0.01)
            .add_depolarizing(0.02, n_qubits=2)
        )
        sf.run(_sf_bell(), method="density_matrix", shots=0, noise_model=plain)
        assert seen, "the noisy DM binding was never called"
        assert all(a is None and b is None and m is None for a, b, m in seen)

        seen.clear()
        filtered = sf.NoiseModel().add_kraus(AD, qubits=0, instructions=["h"])
        sf.run(_sf_bell(), method="density_matrix", shots=0, noise_model=filtered)
        assert seen, "the noisy DM binding was never called"
        assert any(a is not None for a, _, _ in seen)

        seen.clear()
        spectator = sf.NoiseModel().add_crosstalk([(0, 1)], 0.1, mode="spectator")
        sf.run(_sf_bell(), method="density_matrix", shots=0, noise_model=spectator)
        assert seen, "the noisy DM binding was never called"
        assert any(m is not None for _, _, m in seen)

    def test_unfiltered_output_is_unchanged(self):
        """Pinned value: 1q depolarizing after x on |0> keeps the known law."""
        p = 0.07
        res = sf.run(
            sf.Circuit(1).x(0),
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel().add_depolarizing(p),
        )
        # total-convention 1q depolarizing: P(1) = 1 - 2p/3
        assert res.probabilities.get("1", 0.0) == pytest.approx(1 - 2 * p / 3, abs=1e-12)
