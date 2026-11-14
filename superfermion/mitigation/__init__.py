"""
Error Mitigation — Post-processing techniques for noise reduction.

Implements:
- Zero Noise Extrapolation (ZNE)
- Measurement Error Mitigation (readout correction)
- Classical shadows (randomized Pauli-basis snapshots + shadow expval)
"""
from __future__ import annotations


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
import warnings

import numpy as np

import superfermion as sf

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from superfermion.noise import NoiseModel

# SUP-22: classical-shadow estimators live in the sibling submodule; re-export
# here so ``sf.mitigation.classical_shadow`` / ``ClassicalShadow`` work.
from superfermion.mitigation.classical_shadow import (  # noqa: E402,F401
    ClassicalShadow,
    classical_shadow,
    shadow_expval,
)


def zne(
    circuit: sf.Circuit,
    observable_fn: Callable,
    scale_factors: List[int] = [1, 2, 3],
    device: Any = "cpu",
    method: str = "statevector",
    noise_model: Any = None,
    observable: Any = None,
) -> float:
    """Zero Noise Extrapolation — mitigate errors by running at
    multiple noise levels and extrapolating to zero noise.

    Args:
        circuit: The quantum circuit to run.
        observable_fn: Function that takes a statevector and returns a scalar.
            Used on the noiseless path only (see ``noise_model``).
        scale_factors: Fold depths (1 = original).  A fold at depth ``s`` runs
            each gate ``2s - 1`` times, which is what the noise sees; the
            extrapolation is anchored on that copy count, not on ``s`` (see
            the note below).
        device: Execution target — ``"cpu"``, ``"gpu"``, or provider.
        method: Simulation method for the noiseless path.
        noise_model: Optional ``superfermion.noise.NoiseModel``.  When given,
            every fold runs on ``method='density_matrix'`` *with this model*,
            so the folded gates really do accumulate the gate noise
            ``2s - 1`` times — which is the quantity ZNE extrapolates.
        observable: Observable to evaluate when ``noise_model`` is given.  A
            noisy state is a density matrix, so there is no statevector to
            hand to ``observable_fn``; the value is ``Tr(O rho)`` instead.
            Anything ``RunResult.expectation`` accepts (``"ZZ"``, a list of
            Pauli strings, a ``Hamiltonian``, ...).

    Returns:
        The zero-noise extrapolated expectation value.

    Example:
        A noiseless fold cannot move the answer, so this form is a consistency
        check rather than a mitigation.  Note that ``sf.Hamiltonian`` has no
        ``__matmul__`` -- it is evaluated with ``.expectation(state)``.

        >>> from superfermion.mitigation import zne
        >>> def energy(sv):
        ...     return H.expectation(sv)
        >>> zne(circuit, energy, scale_factors=[1, 2, 3])

        To actually mitigate, give the extra gates noise to amplify:

        >>> from superfermion.noise import NoiseModel
        >>> noise = NoiseModel().add_depolarizing(0.02, qubits=[0, 1])
        >>> zne(circuit, energy, scale_factors=[1, 2, 3],
        ...     noise_model=noise, observable=H)

    MIT-1: folding is unitary-exact — ``_fold_circuit(c, s)`` has the same
    unitary as ``c`` for every ``s`` — so on a *noiseless* simulator all scales
    return the same number and the extrapolation cannot change anything.  That
    is arithmetic, not a defect in the fold: ZNE needs a noise model for the
    extra gates to have something to amplify.  Pass ``noise_model=`` together
    with ``observable=`` for a mitigation that can move the answer; with no
    noise model the plain value is returned and a ``RuntimeWarning`` states
    that the extrapolation was a no-op.

    MIT-3 (ZNE-1): the fold at depth ``s`` runs each gate ``2s - 1`` times, not
    ``s`` times, because a unitarity-preserving fold is
    ``G G^dag G (G^dag G)...`` and so always ends on ``G``.  Extrapolating in
    ``s`` therefore asked for the value at *half a copy*, which is not the
    zero-noise point; the abscissa here is :func:`_amplification`
    (``1, 3, 5, ...``).  Measured on ``h;cx`` under depolarizing noise, this is
    the difference between extrapolating to 1.0203 (worse than the raw 0.98)
    and to 0.9990 against a noiseless 1.0.
    """
    expectations = _folded_expectations(
        circuit,
        observable_fn,
        scale_factors,
        device=device,
        method=method,
        noise_model=noise_model,
        observable=observable,
    )
    if len(expectations) > 1:
        # Folding is unitary-exact, so a noiseless run gives the same number at
        # every scale up to float rounding -- compare with a tolerance rather
        # than bit equality, or the ~1e-16 reassociation noise hides the no-op.
        # The tolerance is scaled by ``max(1, |value|)`` and not by |value|
        # alone: an observable whose true value is 0 (e.g. <Z> on |+>) leaves a
        # reference of ~5e-17, and a purely relative tolerance then shrinks to
        # 5e-29 and never fires even though every fold returned the same
        # number.
        spread = max(expectations) - min(expectations)
        reference = max(abs(value) for value in expectations)
        if spread <= 1e-12 * max(reference, 1.0):
            warnings.warn(
                "zne: every fold scale returned the same expectation "
                f"({expectations[0]!r}), so the extrapolation changed nothing. "
                "Circuit folding is unitary-exact on a noiseless simulator; pass "
                "noise_model=NoiseModel(...) and observable=... to give the extra "
                "gates noise to amplify.",
                RuntimeWarning,
                stacklevel=2,
            )

    return _richardson_extrapolation(
        [_amplification(scale) for scale in scale_factors], expectations
    )


def _amplification(scale: int) -> int:
    """Gate copies a fold at depth ``scale`` runs — the noise-amplification
    variable.

    The fold is ``G G^dag G (G^dag G)...``: it has to end on ``G`` to keep the
    original unitary, so the copy count is odd, ``2*scale - 1``.  That count is
    the variable the accumulated noise is linear in, and it is what the
    extrapolation must be anchored on.  Using ``scale`` itself puts the
    extrapolation target at ``2s - 1 = 0``, i.e. ``s = 1/2``, which lands at
    about ``1/(1-lam)`` instead of 1.  (MIT-D3/ZNE-1)
    """
    return 2 * scale - 1


def _checked_scale(scale: Any) -> int:
    """``scale`` as a positive int, or a ValueError explaining why not.

    ``range(2*scale-1)`` silently produced an *empty* circuit for ``scale=0``
    -- the identity, reported as a fold -- and a ``TypeError`` from ``range``
    for a fractional scale.  Neither says what the caller did wrong.
    """
    value = scale
    if isinstance(value, bool):
        raise ValueError(f"_fold_circuit needs an integer scale, got {scale!r}.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"_fold_circuit needs an integer scale >= 1, got {scale!r}: a fold "
            "depth s runs each gate 2s-1 times, so a fractional depth has no "
            "integer copy count.\n"
            "  Fix: scale_factors=[1, 3, 5] (or any integers >= 1)."
        )
    value = int(value)
    if value < 1:
        raise ValueError(
            f"_fold_circuit needs a scale >= 1, got {scale}: a depth of 0 "
            "would run each gate -1 times, i.e. drop every gate and return an "
            "empty circuit that is not the original unitary.\n"
            "  Fix: scale_factors=[1, 3, 5] (scale=1 is the original circuit)."
        )
    return value


def _expectation_from_noisy(result: Any, observable: Any) -> float:
    """``Tr(O rho)`` from a density-matrix (noisy) result."""
    for carrier in (getattr(result, "state", None), result):
        expectation = getattr(carrier, "expectation", None)
        if expectation is None:
            continue
        try:
            return float(np.real(expectation(observable)))
        except Exception:  # noqa: BLE001 - try the next carrier
            continue
    raise RuntimeError(
        "zne could not evaluate the observable on the noisy result: neither "
        "state.expectation nor RunResult.expectation accepted it.\n"
        "  Fix: pass a Pauli string such as observable='ZZ', a list of Pauli "
        "strings, or a Hamiltonian."
    )


def _folded_expectations(
    circuit: sf.Circuit,
    observable_fn: Callable,
    scale_factors: List[int],
    *,
    device: Any,
    method: str,
    noise_model: Any,
    observable: Any,
) -> List[float]:
    """Expectation at each fold scale, noiseless or under ``noise_model``."""
    expectations: List[float] = []

    for scale in scale_factors:
        scaled_circuit = _fold_circuit(circuit, scale)

        if noise_model is None:
            result = sf.run(scaled_circuit, device=device, method=method, shots=0)
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            expectations.append(float(observable_fn(sv)))
            continue

        if observable is None:
            raise ValueError(
                "zne(noise_model=...) needs observable=... as well: a noisy "
                "run returns a density matrix, so there is no statevector to "
                "pass to observable_fn.\n"
                "  Fix: zne(c, fn, noise_model=nm, observable='ZZ') — the value "
                "is then computed as Tr(O rho)."
            )

        result = sf.run(
            scaled_circuit,
            device=device,
            method="density_matrix",
            shots=0,
            noise_model=noise_model,
        )
        expectations.append(_expectation_from_noisy(result, observable))

    return expectations


#: Gates whose adjoint is ``RZ(phi) RY(theta) RZ(lambda)`` adjointed -- the
#: three-phase unitary ``U(th, ph, la)`` and its controlled opcode ``CU``.  Their
#: adjoint negates every angle **and swaps the two phase angles**
#: (``U(th,ph,la)^dag = U(-th,-la,-ph)``); negating alone is only right for the
#: single-angle rotations.
_THREE_ANGLE_ADJOINTS = frozenset({"U", "CU"})

#: Adjoints the Circuit API already spells as their own gate.
_NAMED_DAGGER = {"S": "sdg", "SDG": "s", "T": "tdg", "TDG": "t"}


def _named_unitary(name: str, params: List[Any]) -> Optional[np.ndarray]:
    """Unitary of ``name(params)``, or ``None`` when the package has none.

    ``None`` covers the two cases where no matrix exists to compare against: a
    non-unitary instruction (``measure``, ``barrier``) and a gate whose angles
    are still symbolic, so they are not numbers yet.
    """
    values: List[float] = []
    for p in params:
        try:
            values.append(float(p))
        except (TypeError, ValueError):
            return None
    from superfermion.gates.matrices import gate_unitary_matrix

    try:
        return np.asarray(gate_unitary_matrix(name, values), dtype=complex)
    except (ValueError, KeyError, TypeError):
        return None


def _adjoint_param_candidates(name: str, params: List[Any]) -> List[List[Any]]:
    """The ways ``name`` can spell its own adjoint, most-specific first."""
    negated = [-p for p in params]
    if len(negated) >= 3 and name.upper() in _THREE_ANGLE_ADJOINTS:
        return [[negated[0], negated[2], negated[1], *negated[3:]], negated]
    return [negated]


def _forward_spec(gate: Any) -> tuple:
    """How to append ``gate`` itself: ``("named", method, params)`` or
    ``("unitary", matrix)``."""
    if gate.matrix is not None:
        return ("unitary", np.asarray(gate.matrix, dtype=complex))
    name = gate.name.lower()
    method = getattr(sf.Circuit, name, None)
    if not callable(method):
        raise ValueError(
            f"_fold_circuit cannot fold gate {gate.name!r}: it is not a "
            "Circuit method and carries no matrix, so re-appending it "
            "would silently drop it and fold a different circuit.\n"
            "  Fix: build the circuit from named gates, or give the gate "
            "a .matrix."
        )
    return ("named", name, list(gate.params))


def _adjoint_spec(gate: Any, forward: tuple) -> tuple:
    """How to append the **adjoint** of ``gate``.

    The fold is ``G G^dag G (G^dag G)...``, so every second copy has to be the
    adjoint and not a repeat of ``G``: repeating is only correct for a gate
    that is its own inverse.  Each choice below is checked against the gate's
    own matrix (``gate_unitary_matrix``), so a gate the package grows later
    cannot be folded on a wrong assumption about its order.  ``MIT-D1``.
    """
    base = forward[1] if forward[0] == "unitary" else _named_unitary(
        gate.name, gate.params)
    if base is not None and np.allclose(base.conj().T, base, atol=1e-12):
        return forward                                    # G^dag == G
    upper = gate.name.upper()
    if upper in _NAMED_DAGGER:
        return ("named", _NAMED_DAGGER[upper], [])
    if base is None:
        if not gate.params:
            return forward                                # measure / barrier
        return ("named", forward[1],
                _adjoint_param_candidates(gate.name, gate.params)[0])
    if forward[0] == "named":
        for candidate in _adjoint_param_candidates(gate.name, gate.params):
            trial = _named_unitary(gate.name, candidate)
            if trial is not None and np.allclose(trial, base.conj().T, atol=1e-12):
                return ("named", forward[1], candidate)
    return ("unitary", base.conj().T)


def _emit(out: sf.Circuit, gate: Any, spec: tuple) -> None:
    """Append one copy described by ``spec`` on ``gate``'s own qubits."""
    qubits = list(gate.qubits)
    start = len(out._gates)
    if spec[0] == "unitary":
        out.unitary(spec[1], qubits)
    else:
        _, name, params = spec
        method = getattr(out, name)
        classical = tuple(getattr(gate, "classical_bits", ()) or ())
        if name == "cu" and len(params) == 4:
            # Circuit.cu(theta, phi, lam, control, target, gamma): the record keeps
            # gamma last, but it is a keyword slot *after* the qubits.  Forwarding
            # the record positionally fed gamma into the control slot and raised
            # "qubit index must be int" on every cu/cu3 -- and on ch/cry, which
            # lower to cu.  (MIT-D2)
            method(params[0], params[1], params[2], *qubits, params[3])
        elif params:
            method(*params, *qubits)
        elif classical:
            method(*qubits, *classical)
        else:
            method(*qubits)
    # Folding repeats a gate; if it was guarded, every copy must run under the
    # same c_if or the unfold would apply it when the condition is false.
    if gate.condition is not None:
        for rec in out._gates[start:]:
            rec.condition = gate.condition


def _fold_circuit(circuit: sf.Circuit, scale: int) -> sf.Circuit:
    """Fold a circuit to amplify noise by a given scale factor.

    Gate folding: replaces each gate G with ``G G^dag G`` (repeated), so the
    folded circuit has exactly the unitary of the original at every scale.
    The adjoint is spelled with the same named gate wherever the package has
    one -- negated angles for ``rx``/``rz``/``p``/... , swapped phase angles
    for ``u``/``cu``, ``sdg``/``tdg`` for ``s``/``t`` -- and as an opaque
    ``unitary`` (the gate's own matrix, adjointed) otherwise, which is how
    ``sx`` and ``iswap`` are covered.

    scale=1: original circuit
    scale=2: each gate -> G * G^dag * G  (3 copies)
    scale=3: each gate -> G * G^dag * G * G^dag * G  (5 copies)
    """
    scale = _checked_scale(scale)
    if scale == 1:
        return circuit

    # Keep the classical register width: a fold that re-appends ``measure``
    # must be able to record the declared ``cbit``.
    try:
        n_cbits = int(getattr(circuit, "n_cbits", circuit.n_qubits))
    except (TypeError, ValueError):
        n_cbits = circuit.n_qubits
    new_circuit = (
        sf.Circuit(circuit.n_qubits)
        if n_cbits == circuit.n_qubits
        else sf.Circuit(circuit.n_qubits, n_cbits=n_cbits)
    )

    for gate in circuit._gates:
        forward = _forward_spec(gate)
        adjoint = _adjoint_spec(gate, forward)
        for rep in range(_amplification(scale)):
            _emit(new_circuit, gate, forward if rep % 2 == 0 else adjoint)

    return new_circuit


def _extract_params(circuit: sf.Circuit) -> List[np.ndarray]:
    """Extract parameter values from circuit gates."""
    params = []
    for gate in circuit._gates:
        if gate.params:
            for p in gate.params:
                params.append(np.array(float(p)))
    return params


def _richardson_extrapolation(
    scale_factors: List[int],
    expectations: List[float]
) -> float:
    """Richardson extrapolation to zero noise."""
    n = len(scale_factors)

    if n == 1:
        return expectations[0]

    if n == 2:
        x1, x2 = scale_factors
        y1, y2 = expectations
        slope = (y2 - y1) / (x2 - x1)
        return y1 - slope * x1

    coeffs = np.polyfit(scale_factors, expectations, min(n - 1, 2))
    return float(np.polyval(coeffs, 0))


def readout_correction(
    counts: Dict[str, int],
    calibration_matrix: np.ndarray = None,
    n_qubits: int = None,
) -> Dict[str, int]:
    """Correct measurement counts by inverting the calibration matrix.

    ``calibration_matrix`` is ``M[i][j] = P(measure i | true j)``, so the
    observed distribution is ``M @ p_true`` and the correction solves that
    system for ``p_true``.  It is inverted with ``np.linalg.pinv``, not
    ``inv``: ``M`` is a sampling estimate and is frequently singular (two
    rows measured identical), where ``inv`` would raise.

    The pseudo-inverse can return a **negative** "probability" whenever the
    counts are inconsistent with the model -- ordinary sampling fluctuation at
    small shot counts, or a matrix that does not describe this device.  A
    negative entry cannot be expressed as a count, so it is clamped to zero
    and the rest renormalised, and a ``RuntimeWarning`` reports how much mass
    was discarded and how negative it went.  That warning is the signal that
    the returned numbers are a *projection* of the correction onto the
    probability simplex rather than an inverse: with ``counts={'0': 0,
    '1': 1000}`` and a matrix whose correction is ``[-1.538, 2.538]``, the
    only representable answer is the noisy input back again, mass ``1.538``
    discarded.  Without the warning that case is indistinguishable from a
    correction that genuinely found nothing to change.

    The shot total is preserved **exactly**: the corrected probabilities are
    allocated to integers by largest remainder, so ``sum(result.values()) ==
    sum(counts.values())``.  Rounding each entry independently instead leaves
    the total short by up to ``2**n / 2`` shots (measured: 92 of 300 random
    cases drifted, by up to 2, at 1000 shots).

    Args:
        counts: Raw measurement counts, keyed by bit string with the leftmost
            character as the highest-numbered qubit -- the layout
            ``RunResult.counts`` writes.
        calibration_matrix: ``2**n x 2**n`` confusion matrix whose columns sum
            to 1.  When ``None`` the counts are returned unchanged: there is
            nothing to correct with.
        n_qubits: Number of qubits (auto-detected from the width of the first
            key).

    Returns:
        Corrected counts.  Keys whose corrected count is zero are omitted, so
        the key set can shrink; the values still sum to the input's total.
    """
    if calibration_matrix is None:
        return counts
    if not counts:
        return {}

    if n_qubits is None:
        n_qubits = len(next(iter(counts)))

    n_states = 2**n_qubits
    total_shots = sum(counts.values())
    if total_shots <= 0:
        return {}

    raw_probs = np.zeros(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        raw_probs[idx] += count / total_shots

    M_inv = np.linalg.pinv(np.asarray(calibration_matrix, dtype=float))
    if M_inv.shape != (n_states, n_states):
        raise ValueError(
            f"calibration_matrix must be {n_states} x {n_states} for "
            f"{n_qubits} qubit(s), got {M_inv.shape}.\n"
            "  Fix: pass the confusion matrix measured on the same number of "
            "qubits -- or set n_qubits to the width the matrix was measured "
            "on."
        )
    corrected_probs = M_inv @ raw_probs

    negative = corrected_probs < 0
    if negative.any():
        n_bad = int(negative.sum())
        discarded = float(-corrected_probs[negative].sum())
        warnings.warn(
            "readout_correction: the pseudo-inverse produced "
            f"{n_bad} negative probabilit{'y' if n_bad == 1 else 'ies'} "
            f"({discarded:.6g} of mass in total, worst "
            f"{float(corrected_probs[negative].min()):.6g}). The counts are "
            "inconsistent with this calibration matrix -- sampling "
            "fluctuation, or a matrix from a different device. They were "
            "clamped to zero and renormalised, so the result is the "
            "closest probability distribution to the correction, not the "
            "correction itself.",
            RuntimeWarning,
            stacklevel=2,
        )

    corrected_probs = np.maximum(corrected_probs, 0)
    total_prob = float(corrected_probs.sum())
    if total_prob <= 0:
        warnings.warn(
            "readout_correction: no positive probability survived the "
            "correction, so every corrected count is zero and nothing can "
            "be renormalised. Returning the raw counts unchanged -- check "
            "that calibration_matrix is column-stochastic and belongs to "
            "this device.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(counts)
    corrected_probs = corrected_probs / total_prob

    # Largest-remainder allocation: floor everything, then hand the leftover
    # shots to the entries with the biggest fractional part.  Rounding each
    # entry on its own does not conserve the shot total.
    target = corrected_probs * total_shots
    allocated = np.floor(target)
    leftover = total_shots - int(allocated.sum())
    if leftover > 0:
        order = np.argsort(-(target - allocated), kind="stable")
        allocated[order[:leftover]] += 1

    corrected_counts = {}
    for i in range(n_states):
        c = int(allocated[i])
        if c > 0:
            corrected_counts[format(i, f'0{n_qubits}b')] = c

    return corrected_counts


def zne_with_calibration(
    circuit: sf.Circuit,
    observable_fn: Callable,
    calibration: Any = None,
    scale_factors: Optional[List[int]] = None,
    device: Any = "cpu",
    method: str = "statevector",
    apply_readout_correction: bool = False,
    observable: Any = None,
) -> Dict[str, Any]:
    """Zero Noise Extrapolation driven by device calibration data.

    Unlike the basic ``zne()`` which uses noise-blind circuit folding,
    this function reads real gate fidelities from a ``CalibrationSet``
    and uses them to construct a calibrated ``NoiseModel``,
    compute hardware-informed noise scale factors, and optionally apply
    readout-error correction.

    Args:
        circuit: Quantum circuit to mitigate.
        observable_fn: Function ``(statevector) -> scalar``.
        calibration: ``CalibrationSet`` or ``NoiseModel``.
        scale_factors: Fold depths (1 = original). If None,
            derived from calibration (default: [1, 2, 3]).  As in ``zne``, a
            fold at depth ``s`` runs each gate ``2s - 1`` times and the
            extrapolation is anchored on that copy count.
        device: Execution target.
        method: Simulation method.
        apply_readout_correction: Request readout correction on measured
            counts. There are none to correct here: every fold is evaluated as
            ``Tr(O rho)`` on the exact density matrix produced under
            ``noise_model``, and a readout error is a property of the
            *measurement*, not of ``rho``, so it never enters that number.
            Passing ``True`` therefore applies nothing, returns
            ``readout_corrected=False``, and emits a ``RuntimeWarning`` saying
            so; call :func:`readout_correction` on real ``result.counts`` from
            a ``shots>0`` run if you need it. (It used to set
            ``readout_corrected=True`` and change no number at all.)
        observable: Observable to evaluate on the noisy folds.  The calibrated
            ``NoiseModel`` is only applied when this is given, because a noisy
            run has no statevector for ``observable_fn`` (see ``zne``).  When
            omitted the call keeps the previous noiseless behaviour and
            reports ``noise_model_used=False``.

    Returns:
        Dict with keys: ``zne_value``, ``raw_values``,
        ``scale_factors``, ``noise_params``, ``readout_corrected``,
        ``noise_model_used``.

        ``readout_corrected`` is always ``False`` -- see
        ``apply_readout_correction``: this path has no counts to correct, so it
        reports the honest value instead of the requested one.
    """
    noise_params = None
    noise_model = None

    if calibration is not None:
        if hasattr(calibration, "extract_noise_params"):
            noise_params = calibration.extract_noise_params()
            noise_model = calibration.to_noise_model()
        elif hasattr(calibration, "single_qubit_channels"):
            noise_model = calibration
            noise_params = _noise_model_to_params(noise_model)
        else:
            noise_params = dict(calibration) if isinstance(calibration, dict) else {}

    if noise_params is None:
        noise_params = {
            "depolarizing_1q": 0.001,
            "depolarizing_2q": 0.01,
            "readout_error": 0.01,
            "noise_factors": [1, 2, 3],
        }

    if scale_factors is None:
        scale_factors = noise_params.get("noise_factors", [1, 2, 3])

    # MIT-1: the calibrated noise model used to be built and then never run --
    # every fold went through the noiseless statevector path, so the "mitigated"
    # value was bit-for-bit the unmitigated one.  It is now applied to each
    # fold whenever an observable is available to read back.
    noise_model_used = noise_model is not None and observable is not None
    expectations = _folded_expectations(
        circuit,
        observable_fn,
        scale_factors,
        device=device,
        method=method,
        noise_model=noise_model if noise_model_used else None,
        observable=observable,
    )

    zne_value = _richardson_extrapolation(
        [_amplification(scale) for scale in scale_factors], expectations
    )

    readout_applied = False
    if apply_readout_correction:
        # The flag used to be reported as applied whenever a readout error was
        # present, while nothing was corrected: ``readout_correction`` was
        # never called and no counts were ever sampled.  Now it says what it
        # did -- nothing -- and explains why. (xq7/V-136)
        error = noise_params.get("readout_error", 0)
        detail = (
            f"the calibration carries readout_error={error!r}"
            if error
            else "the calibration carries no readout_error either"
        )
        warnings.warn(
            "zne_with_calibration(apply_readout_correction=True) corrected "
            "nothing: the folds are evaluated as Tr(O rho) on an exact "
            f"density matrix, and {detail}. A readout error belongs to the "
            "measurement, not to rho, so it is absent from that number and "
            "there is no confusion matrix to invert. readout_corrected is "
            "reported as False.\n"
            "  Fix: run the circuit with shots>0, then pass "
            "result.counts to mitigation.readout_correction(counts, M).",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "zne_value": zne_value,
        "raw_values": expectations,
        "scale_factors": scale_factors,
        "noise_params": noise_params,
        "readout_corrected": readout_applied,
        "noise_model_used": noise_model_used,
    }


def _noise_model_to_params(noise_model: Any) -> Dict[str, Any]:
    """Extract parameter dict from a NoiseModel instance."""
    params: Dict[str, Any] = {
        "noise_factors": [1, 2, 3],
        "depolarizing_1q": 0.0,
        "depolarizing_2q": 0.0,
        "readout_error": 0.0,
        "amplitude_damping": 0.0,
        "phase_damping": 0.0,
    }

    if hasattr(noise_model, "readout_error"):
        params["readout_error"] = noise_model.readout_error

    if hasattr(noise_model, "single_qubit_channels"):
        for ch in noise_model.single_qubit_channels:
            if ch.name == "depolarizing":
                params["depolarizing_1q"] = max(
                    params["depolarizing_1q"], ch.error_rate
                )
            elif ch.name == "amplitude_damping":
                params["amplitude_damping"] = max(
                    params["amplitude_damping"], ch.error_rate
                )
            elif ch.name == "phase_damping":
                params["phase_damping"] = max(
                    params["phase_damping"], ch.error_rate
                )

    if hasattr(noise_model, "two_qubit_channels"):
        for ch in noise_model.two_qubit_channels:
            if ch.name == "depolarizing":
                params["depolarizing_2q"] = max(
                    params["depolarizing_2q"], ch.error_rate
                )

    return params


def calibration_based_noise_model(
    backend_name: str,
    dt: float = 0.222,
) -> "NoiseModel":
    """Build a NoiseModel from backend calibration data."""
    from superfermion.pulse.calibration import CalibrationSet

    cals = CalibrationSet(backend_name, dt=dt)

    for q in range(min(5, 4)):
        cals.add_default_single_qubit(q)
    for q in range(min(5, 4) - 1):
        cals.add_default_two_qubit(q, q + 1)

    return cals.to_noise_model()
