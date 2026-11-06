#!/usr/bin/env python3
"""Prove the gradient kernels work in *this* install, in about twenty seconds.

Why this exists
---------------
Every gradient defect reported against 0.1.22/0.1.23 was reproducible on the
published wheel and invisible in the source tree, and half of the follow-up
conversation was spent working out *which* install had produced a number.  This
script removes both problems: it is one file, it imports only numpy (plus scipy
for the VQE check), it prints the identity of the engine it exercised, and it
fails loudly instead of printing a plausible number.

It duplicates the arithmetic of ``tests/domain/test_gradient_regressions.py``
on purpose -- the gate must run from a wheel where the test tree is not
shipped, and must not depend on pytest.

What it checks (IDs match the 0.1.24 defect ledger)
--------------------------------------------------
A1  ``p(a)`` and ``rz(a)`` gradients match, and neither is the old silent 0.0
A2  ``cp`` / ``cu`` adjoint gradients match central finite differences
B   a pre-bound circuit raises instead of returning zeros
A3  ``res.grad`` / ``adjoint_grad`` / ``State.grad`` agree on the ``p`` gate
C   ``ry(t); ry(t)`` -- parameter-shift sums every occurrence of a name
D   ``cu`` -- parameter-shift handles the {0, +-1/2} spectrum (all four slots)
E   ``str`` / ``dict`` / ``SparsePauliOp`` observables agree
F   ``u_ansatz`` + ``VQE(diff_method="adjoint")`` reaches the ground state

Usage
-----
    python scripts/grad_gate.py
    python scripts/grad_gate.py --expect-version 0.1.26
    python scripts/grad_gate.py --expect-engine-root "$RUNNER_TEMP/venv"
    python scripts/grad_gate.py --json grad_gate.json

Exit status: 0 = every check passed, 1 = a check failed, 2 = nothing could be
exercised (import or identity failure).  Same behaviour on Windows, Linux,
macOS and Colab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

FD_EPS = 1e-5
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# ── Engine identity ──────────────────────────────────────────────────────────


def identity() -> dict:
    """Which interpreter, package and compiled engine is this process using?"""
    import superfermion as sf

    engine = Path(sf._sf_core.__file__).resolve()
    digest = hashlib.sha256()
    with open(engine, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "version": getattr(sf, "__version__", "?"),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "package": str(Path(sf.__file__).resolve()),
        "engine": str(engine),
        "engine_sha256": digest.hexdigest(),
    }


# ── Shared helpers (a copy of the test module, minus pytest) ─────────────────


def finite_difference(circuit, observable, params, name, eps=FD_EPS):
    """Central difference of <O> over every occurrence of ``name``."""
    import superfermion as sf

    total = 0.0
    for k in range(len(circuit.param_occurrences(name))):
        up = circuit.bind_occurrence(params, name, k, value=params[name] + eps)
        down = circuit.bind_occurrence(params, name, k, value=params[name] - eps)
        total += (
            sf.run(up, shots=0).expectation(observable)
            - sf.run(down, shots=0).expectation(observable)
        ) / (2.0 * eps)
    return total


def reviewer_circuit(gate: str):
    """The circuit from the bug report, with ``p`` or ``rz`` on parameter ``a``."""
    import superfermion as sf

    a, b, c = sf.param("a"), sf.param("b"), sf.param("c")
    circ = sf.Circuit(2).ry(0.7, 0).ry(1.1, 1).cnot(0, 1)
    circ = circ.p(a, 0) if gate == "p" else circ.rz(a, 0)
    return circ.rx(b, 0).ry(c, 1).cnot(0, 1)


REVIEWER_PARAMS = {"a": 0.9, "b": -0.4, "c": 1.3}
REVIEWER_OBS = {"YI": 1.0, "IZ": 0.5, "XX": 0.3}


# ── Checks ───────────────────────────────────────────────────────────────────


def check_a1_phase_gate_matches_rz():
    """A1: ``p`` and ``rz`` differ by a global phase, so their gradients must too."""
    from superfermion.qml.gradient.adjoint import adjoint_grad

    g_p = adjoint_grad(reviewer_circuit("p"), REVIEWER_OBS, REVIEWER_PARAMS)["a"]
    g_rz = adjoint_grad(reviewer_circuit("rz"), REVIEWER_OBS, REVIEWER_PARAMS)["a"]
    fd = finite_difference(reviewer_circuit("p"), REVIEWER_OBS, REVIEWER_PARAMS, "a")

    detail = f"p={g_p:+.10f} rz={g_rz:+.10f} fd={fd:+.10f}"
    if g_p == 0.0:
        return FAIL, detail + "  (P gate silently dropped: defect A)", {}
    if abs(g_p - g_rz) > 1e-12:
        return FAIL, detail + "  (p != rz)", {}
    if abs(g_p - fd) > 1e-8:
        return FAIL, detail + "  (adjoint != finite difference)", {}
    return PASS, detail, {"p": g_p, "rz": g_rz, "fd": fd}


def check_a2_controlled_gates_match_fd():
    """A2: ``cp`` and ``cu`` adjoint gradients against central differences."""
    import superfermion as sf
    from superfermion.qml.gradient.adjoint import adjoint_grad

    t = sf.param("t")
    cp_circ = sf.Circuit(2).h(0).h(1).cp(t, 0, 1)
    cp_grad = adjoint_grad(cp_circ, {"XX": 1.0}, {"t": 0.7})["t"]
    cp_fd = finite_difference(cp_circ, {"XX": 1.0}, {"t": 0.7}, "t")

    cu_circ = sf.Circuit(2).h(0).cu(t, 0.0, 0.0, 0, 1)
    cu_grad = adjoint_grad(cu_circ, {"XI": 1.0}, {"t": 0.7})["t"]
    cu_fd = finite_difference(cu_circ, {"XI": 1.0}, {"t": 0.7}, "t")

    detail = (
        f"cp={cp_grad:+.10f}/{cp_fd:+.10f} cu={cu_grad:+.10f}/{cu_fd:+.10f} "
        f"(adjoint/fd)"
    )
    if abs(cp_grad) < 1e-12 or abs(cu_grad) < 1e-12:
        return FAIL, detail + "  (controlled gate silently dropped: defect A)", {}
    if abs(cp_grad - cp_fd) > 1e-8 or abs(cu_grad - cu_fd) > 1e-8:
        return FAIL, detail + "  (adjoint != finite difference)", {}
    return PASS, detail, {"cp": cp_grad, "cu": cu_grad, "cp_fd": cp_fd, "cu_fd": cu_fd}


def check_b_prebound_circuit_raises():
    """B: a bound circuit has no free parameters, so it must raise."""
    import superfermion as sf
    from superfermion.qml.gradient.adjoint import adjoint_grad

    t = sf.param("t")
    bound = sf.Circuit(1).ry(t, 0).ry(t, 0).bind({"t": 0.35})
    try:
        grad = adjoint_grad(bound, {"Z": 1.0}, {"t": 0.35})
    except ValueError as exc:
        message = str(exc)
        if "to_ir()" not in message:
            return FAIL, f"raised ValueError without the to_ir() hint: {message}", {}
        return PASS, "raised ValueError with the to_ir() hint", {}
    return FAIL, f"returned {grad!r} instead of raising (defect B)", {}


def check_a3_entry_points_agree():
    """A3: the three public entry points are one kernel and must agree."""
    import superfermion as sf
    from superfermion.qml.gradient.adjoint import adjoint_grad
    from superfermion.results import _coerce_observable

    circuit = reviewer_circuit("p")
    dag = circuit.to_ir()
    result = sf.run(circuit.bind(REVIEWER_PARAMS), shots=0)

    from_result = result.grad(REVIEWER_OBS, dag, REVIEWER_PARAMS)["a"]
    from_adjoint = adjoint_grad(circuit, REVIEWER_OBS, REVIEWER_PARAMS)["a"]
    from_state = result.state.grad(
        _coerce_observable(REVIEWER_OBS, 2), dag, REVIEWER_PARAMS
    )["a"]

    detail = (
        f"res.grad={from_result:+.10f} adjoint_grad={from_adjoint:+.10f} "
        f"State.grad={from_state:+.10f}"
    )
    if from_result == 0.0:
        return FAIL, detail + "  (P gate silently dropped: defect A)", {}
    if max(abs(from_result - from_adjoint), abs(from_result - from_state)) > 1e-12:
        return FAIL, detail + "  (entry points disagree)", {}
    return PASS, detail, {"res_grad": from_result}


def check_c_shared_parameter_shift():
    """C: two gates on one name need two shifts, summed."""
    import superfermion as sf
    from superfermion.qml.gradient.parameter_shift import parameter_shift_grad

    t = sf.param("t")
    circuit = sf.Circuit(1).ry(t, 0).ry(t, 0)
    params = {"t": 0.35}
    grad = parameter_shift_grad(circuit, {"Z": 1.0}, params)["t"]
    fd = finite_difference(circuit, {"Z": 1.0}, params, "t")

    detail = f"ps={grad:+.12f} fd={fd:+.12f} exact=-1.288435374397"
    if abs(grad) < 1e-12:
        return FAIL, detail + "  (name-global shift: defect C)", {}
    if abs(grad - (-1.2884353743969346)) > 1e-9:
        return FAIL, detail + "  (wrong per-parameter-shift value)", {}
    if abs(grad - fd) > 1e-8:
        return FAIL, detail + "  (parameter-shift != finite difference)", {}
    return PASS, detail, {"ps": grad, "fd": fd}


def check_d_controlled_rotation_shift():
    """D: every ``cu`` slot, including the {0, +-1/2} theta spectrum."""
    import superfermion as sf
    from superfermion.qml.gradient.parameter_shift import parameter_shift_grad

    names = ["theta", "phi", "lam", "gamma"]
    values = [0.7, 0.3, -0.5, 0.2]
    syms = [sf.param(n) for n in names]
    # Target starts in |+y> so that all four slots move the expectation value;
    # a trivially-zero slot could hide a wrong rule.
    circuit = sf.Circuit(2).h(0).ry(0.6, 1).cu(syms[0], syms[1], syms[2], 0, 1, syms[3])
    observable = {"YI": 0.7, "XI": 0.3, "IZ": 0.5, "XX": 0.2}
    params = dict(zip(names, values))

    grad = parameter_shift_grad(circuit, observable, params)
    worst, worst_name = 0.0, ""
    parts = []
    for name in names:
        fd = finite_difference(circuit, observable, params, name)
        parts.append(f"{name}={grad[name]:+.10f}/{fd:+.10f}")
        if abs(grad[name]) < 1e-12:
            return FAIL, " ".join(parts) + f"  ({name} silently zero)", {}
        if abs(grad[name] - fd) > worst:
            worst, worst_name = abs(grad[name] - fd), name

    detail = " ".join(parts) + " (ps/fd)"
    if worst > 1e-8:
        return FAIL, detail + f"  (worst {worst_name} off by {worst:.2e}: defect D)", {}
    return PASS, detail, {"worst_abs_diff": worst}


def check_e_observable_forms():
    """E: ``str``, ``dict`` and ``SparsePauliOp`` are the same observable."""
    import superfermion as sf
    from superfermion.observables.core import SparsePauliOp
    from superfermion.qml.gradient.parameter_shift import parameter_shift_grad

    t = sf.param("t")
    circuit = sf.Circuit(2).ry(t, 0).cnot(0, 1)
    params = {"t": 0.4}

    values = {
        "str": parameter_shift_grad(circuit, "ZI", params)["t"],
        "dict": parameter_shift_grad(circuit, {"ZI": 1.0}, params)["t"],
        "spo": parameter_shift_grad(
            circuit, SparsePauliOp.from_dict({"ZI": 1.0}), params
        )["t"],
        "list": parameter_shift_grad(circuit, ["ZI"], params)["t"],
    }
    detail = " ".join(f"{k}={v:+.10f}" for k, v in values.items())
    if values["str"] == 0.0:
        return FAIL, detail + "  (observable evaluated to zero)", {}
    spread = max(values.values()) - min(values.values())
    if spread > 1e-12:
        return FAIL, detail + f"  (forms disagree by {spread:.2e}: defect E)", {}
    return PASS, detail, values


def check_f_vqe_reaches_ground_state():
    """F: the package's own U-gate template must optimise, not stall."""
    try:
        from superfermion.algorithms.variational import VQE
    except ImportError as exc:  # scipy is an optional extra
        return SKIP, f"VQE unavailable ({exc}); install scipy to run this check", {}
    from superfermion.observables.core import SparsePauliOp
    from superfermion.qml.ansatz import u_ansatz

    vqe = VQE(u_ansatz(1, 1), SparsePauliOp.from_dict({"X": 1.0}), diff_method="adjoint")
    result = vqe.minimize(iterations=200, seed=7)
    detail = f"energy={result.optimal_value:+.9f} exact=-1.000000000"
    if abs(result.optimal_value - (-1.0)) > 1e-6:
        return FAIL, detail + "  (optimiser stalled: all-zero adjoint gradients)", {}
    return PASS, detail, {"energy": float(result.optimal_value)}


def check_g_new_gate_builders():
    """G: r1/crx/cry/crz are buildable and differentiated by both engines."""
    import superfermion as sf
    from superfermion.qml.gradient.adjoint import adjoint_grad
    from superfermion.qml.gradient.parameter_shift import parameter_shift_grad

    gates = ("r1", "crx", "cry", "crz")
    missing = [g for g in gates if not hasattr(sf.Circuit, g)]
    if missing:
        return FAIL, f"no Circuit builder for {missing}", {}

    observable = {"YI": 0.7, "XI": 0.3, "IZ": 0.5, "XX": 0.2}
    params = {"t": 0.83}
    worst, parts = 0.0, []
    for gate in gates:
        t = sf.param("t")
        circuit = sf.Circuit(2).ry(0.7, 0).ry(1.1, 1).cnot(0, 1)
        circuit = circuit.r1(t, 0) if gate == "r1" else getattr(circuit, gate)(t, 0, 1)
        circuit = circuit.rx(0.3, 0).ry(-0.4, 1)

        fd = finite_difference(circuit, observable, params, "t")
        adjoint = adjoint_grad(circuit, observable, params)["t"]
        shifted = parameter_shift_grad(circuit, observable, params)["t"]
        parts.append(f"{gate}={adjoint:+.6f}/{shifted:+.6f}/{fd:+.6f}")
        if adjoint == 0.0:
            return FAIL, " ".join(parts) + f"  ({gate} adjoint silently zero)", {}
        worst = max(worst, abs(adjoint - fd), abs(shifted - fd))

    detail = " ".join(parts) + " (adjoint/ps/fd)"
    if worst > 1e-8:
        return FAIL, detail + f"  (worst off by {worst:.2e})", {}
    return PASS, detail, {"worst_abs_diff": worst}


CHECKS = [
    ("A1 phase gate p == rz (adjoint)", check_a1_phase_gate_matches_rz),
    ("A2 cp/cu adjoint vs finite difference", check_a2_controlled_gates_match_fd),
    ("B  pre-bound circuit raises", check_b_prebound_circuit_raises),
    ("A3 res.grad / adjoint_grad / State.grad", check_a3_entry_points_agree),
    ("C  shared parameter (parameter-shift)", check_c_shared_parameter_shift),
    ("D  controlled rotation (parameter-shift)", check_d_controlled_rotation_shift),
    ("E  str / dict / SparsePauliOp observables", check_e_observable_forms),
    ("F  VQE on u_ansatz converges", check_f_vqe_reaches_ground_state),
    ("G  r1/crx/cry/crz builders", check_g_new_gate_builders),
]


# ── Driver ───────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the gradient kernels of the installed SuperFermion."
    )
    parser.add_argument("--expect-version", help="fail unless sf.__version__ matches")
    parser.add_argument(
        "--expect-engine-root",
        help="fail unless the compiled engine lives under this directory",
    )
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument(
        "--quiet", action="store_true", help="only print failures and the summary"
    )
    args = parser.parse_args(argv)

    try:
        ident = identity()
    except Exception:
        print("grad_gate: cannot import superfermion from this interpreter")
        traceback.print_exc()
        return 2

    print(f"superfermion {ident['version']}  (python {ident['python_version']})")
    print(f"  package  {ident['package']}")
    print(f"  engine   {ident['engine']}")
    print(f"  sha256   {ident['engine_sha256']}")
    print(f"  python   {ident['python']}")

    identity_failures = []
    if args.expect_version and ident["version"] != args.expect_version:
        identity_failures.append(
            f"version is {ident['version']}, expected {args.expect_version}"
        )
    if args.expect_engine_root:
        root = Path(args.expect_engine_root).resolve()
        engine = Path(ident["engine"])
        if not engine.is_relative_to(root):
            identity_failures.append(f"engine {engine} is outside {root}")
        else:
            print(f"  root     {root}  (engine is inside it)")
    for failure in identity_failures:
        print(f"[FAIL] identity: {failure}")
    print()

    report = []
    for label, func in CHECKS:
        try:
            status, detail, values = func()
        except Exception as exc:
            status = FAIL
            detail = f"raised {type(exc).__name__}: {exc}"
            values = {}
        report.append(
            {"check": label, "status": status, "detail": detail, "values": values}
        )
        if not args.quiet or status != PASS:
            print(f"[{status}] {label}")
            print(f"         {detail}")

    failed = [r for r in report if r["status"] == FAIL]
    skipped = [r for r in report if r["status"] == SKIP]
    total = len(report) + len(identity_failures)
    ok = total - len(failed) - len(identity_failures)

    print()
    if failed or identity_failures:
        print(f"grad_gate: FAIL ({ok}/{total} passed)")
        if identity_failures and not args.quiet:
            print("  identity failures:", *identity_failures, sep="\n    ")
    else:
        suffix = f", {len(skipped)} skipped" if skipped else ""
        print(f"grad_gate: PASS ({ok}/{total} passed{suffix})")

    if args.json:
        payload = {
            "identity": ident,
            "expect_version": args.expect_version,
            "expect_engine_root": args.expect_engine_root,
            "identity_failures": identity_failures,
            "checks": report,
            "status": "FAIL" if (failed or identity_failures) else "PASS",
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  report   {args.json}")

    if identity_failures or failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
