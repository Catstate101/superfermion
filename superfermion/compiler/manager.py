"""
Compiler Manager — Central orchestrator for quantum compilation.

Architecture: "Python is the API, Rust Does the Work."

All compilation delegates to the Rust ``_sf_core.Compiler`` pipeline.
Python handles only one temporary shim that Rust does not yet support:
  - ``UnitaryDecompositionPass``: decomposes opaque UNITARY gates (pre-Rust)

Basis translation is now handled entirely in Rust via
``BasisTranslationPass`` and ``GateFusionPass``.

There are no Python fallbacks. ``_sf_core`` is a required dependency.
"""

from __future__ import annotations

from superfermion.circuit import Circuit
from superfermion.compiler.passes import (
    BasisTranslationPass,
    Pass,
    UnitaryDecompositionPass,
)
from superfermion.compiler.specs import HardwareSpec


class PassManager:
    """Manages a sequence of compilation passes.

    Retained for users who need custom pass pipelines (e.g. combining
    the temporary shim passes with their own transforms).
    """

    def __init__(self, passes: list[Pass] | None = None):
        self.passes: list[Pass] = passes or []

    _plugin_passes: dict = {}

    def add_pass(self, pass_obj: Pass):
        """Add a new pass to the pipeline."""
        self.passes.append(pass_obj)

    @classmethod
    def add_plugin_pass(cls, name: str, pass_cls: type) -> None:
        """Register a plugin compiler pass for use in pipelines."""
        cls._plugin_passes[name] = pass_cls

    @classmethod
    def list_plugin_passes(cls) -> list:
        """List all registered plugin passes."""
        return sorted(cls._plugin_passes.keys())

    def run(self, circuit: Circuit) -> Circuit:
        """Run all registered passes sequentially."""
        current_circuit = circuit
        for pass_obj in self.passes:
            current_circuit = pass_obj.run(current_circuit)
        return current_circuit


def _has_unitary_gates(circuit: Circuit) -> bool:
    """Check if circuit contains opaque UNITARY gates."""
    circuit._ensure_gates()
    return any(g.name.upper() == "UNITARY" for g in circuit._gates)


# Gates the Rust compiler can consume (its internal passes know these).
_RUST_KNOWN_GATES = frozenset({
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "ID",
    "CX", "CNOT", "CZ", "SWAP",
    "RX", "RY", "RZ", "SX", "P",
})


def _has_gates_rust_cannot_handle(circuit: Circuit) -> bool:
    """Check if circuit has gates outside the Rust compiler's vocabulary."""
    circuit._ensure_gates()
    passthrough = {"MEASURE", "BARRIER", "RESET"}
    return any(
        g.name.upper() not in _RUST_KNOWN_GATES
        and g.name.upper() not in passthrough
        for g in circuit._gates
    )


def compile(
    circuit: Circuit,
    level: int = 1,
    target: HardwareSpec | None = None,
    twirl_seed: int | None = None,
) -> Circuit:
    """Compile a quantum circuit for a hardware target.

    Delegates to the Rust ``_sf_core.Compiler`` for all optimization,
    decomposition, and routing. Python shims run before/after Rust for
    capabilities not yet in the Rust compiler.

    Args:
        circuit: The input quantum circuit.
        level: Optimization level (0=none, 1=standard, 2=aggressive).
        target: Optional hardware target specification.
        twirl_seed: Seed for the Pauli twirl that level >= 2 applies to 2Q
            gates. Without a seed each call draws a fresh thread-RNG seed, so
            two level-2 compiles of the same input can differ gate for gate
            (the unitary is unaffected: measured at 1e-16 across runs).
            Supplying a seed makes the compiled circuit reproducible.

    Returns:
        Compiled and optimized circuit for the target. The returned circuit has
        the same width as ``circuit`` even when the target is wider, and carries
        the layout it was compiled under in ``circuit._metadata["layout"]``
        (see :func:`superfermion.compiler.layout.get_layout`).
    """
    if level == 0 and not target:
        return circuit

    # A string target is what the docs and examples use
    # (``sf.compile(c, target="ibm_eagle")``), but a ``str`` has none of the
    # ``HardwareSpec`` attributes the pipeline reads: every
    # ``getattr(target, ..., default)`` fell back silently, so the call
    # compiled for no device at all -- no routing, no native basis, no error.
    # Resolve the alias through the same registry the presets live in.
    if isinstance(target, str):
        from superfermion.compiler.specs import get_spec, list_devices

        resolved = get_spec(target)
        if resolved is None:
            raise ValueError(
                f"unknown hardware target {target!r} "
                f"(known targets: {sorted(list_devices())})"
            )
        target = resolved

    # Rust compilation evaluates parameters, which requires concrete
    # values. Symbolic (unbound) parameters must not reach the Rust
    # pipeline - it panics on them (PanicException, SUP-19). Reject up
    # front with a clean, actionable error instead.
    unbound = circuit.to_ir().parameter_names()
    if unbound:
        raise ValueError(
            f"compile() requires bound parameter values (level={level}, "
            f"target={target!r}); unbound parameters: {unbound}. "
            f"Call circuit.bind({{...}}) first, or use compile(level=0) "
            f"to keep parameters symbolic."
        )

    # Shim (temporary): decompose opaque unitaries — Rust compiler
    # does not yet handle UNITARY gates with embedded matrices.
    if _has_unitary_gates(circuit):
        circuit = UnitaryDecompositionPass().run(circuit)

    # Shim (temporary): decompose complex gates the Rust compiler
    # doesn't know (CCX, Toffoli, ISWAP, CY, CP, RXX, RYY, RZZ, U3,
    # etc.) into primitives Rust can handle. Uses BasisTranslationPass
    # targeting the Rust compiler's input vocabulary.
    if _has_gates_rust_cannot_handle(circuit):
        rust_input_basis = list(_RUST_KNOWN_GATES)
        circuit = BasisTranslationPass(rust_input_basis).run(circuit)

    # Layout: the Rust pipeline routes onto the device and then throws the
    # resulting ``QubitMapping`` away, which silently permutes the wires
    # (``cx(0,2)`` came back as ``CX(1,2)``) and rebuilds the result on the
    # device's width instead of the circuit's. Routing here keeps the
    # permutation known and undone, so the compiled unitary equals the input's
    # and the register keeps its size (HW-2, HW-4, HW-5).
    layout_report: dict | None = None
    if target is not None and getattr(target, "coupling_map", None):
        from superfermion.compiler.layout import route_with_layout

        # A gate with three or more wire operands has no single edge to live
        # on, so route_with_layout cannot place it: it is reported in
        # ``unrouted_gates`` and left on its wires.  The lowering that removes
        # it (CSWAP -> CX, CCX, CX, then the Rust input vocabulary) runs inside
        # compile_rust, i.e. *after* routing, so the CX gates it emits were
        # never routed and the compiled circuit kept long-range 2q gates --
        # the connectivity guard then rejected the result (112 of 200 lane-A
        # programs).  Lower first, then route what is left; compile_rust's own
        # prepare() then finds nothing to rewrite, so nothing is lowered twice.
        from superfermion.compiler.native_lowering import prepare as _lower_first

        circuit, _ = _lower_first(circuit, target)
        circuit, layout_report = route_with_layout(circuit, target)

    if target:
        compiled = _rust_compile(
            circuit,
            level=level,
            target=target,
            skip_routing=bool(layout_report and layout_report.get("routed")),
            twirl_seed=twirl_seed,
        )
    else:
        # No target: hardware-neutral optimization only
        compiled = _rust_compile(circuit, level=level, target=None, twirl_seed=twirl_seed)

    if layout_report is not None:
        compiled._metadata["layout"] = layout_report
        compiled._metadata["target"] = getattr(target, "name", None)
        compiled._metadata["level"] = level
    return compiled


def _rust_compile(
    circuit: Circuit,
    level: int,
    target: HardwareSpec | None,
    skip_routing: bool = False,
    twirl_seed: int | None = None,
) -> Circuit:
    """Delegate to the Rust compilation pipeline."""
    from superfermion.compiler.rust_bridge import compile_rust
    return compile_rust(
        circuit,
        level=level,
        target=target,
        skip_routing=skip_routing,
        twirl_seed=twirl_seed,
    )


def apply_noise_suppression(
    circuit: Circuit,
    twirl: bool = True,
    dd_sequence: str = "XY4",
    seed: int | None = None,
) -> Circuit:
    """Apply noise suppression passes (opt-in, separate from compile).

    Pauli twirling and dynamical decoupling are noise-mitigation
    techniques, not transpilation. They are deliberately excluded from
    ``compile()`` and exposed here for explicit use.

    One exception: the Rust pipeline applies a Pauli twirl to 2Q gates when
    ``optimization_level >= 2``, with a fresh random seed unless
    ``compile(..., twirl_seed=N)`` is given. That twirl randomises the gate
    list (the unitary is unchanged); the passes here are the additional,
    explicitly requested ones.

    Args:
        circuit: Compiled circuit (should already be in target basis).
        twirl: Apply Pauli twirling to 2Q gates.
        dd_sequence: Dynamical decoupling sequence name.
        seed: Random seed for twirling reproducibility.

    Returns:
        Circuit with noise suppression gates inserted.
    """
    from superfermion.compiler.advanced import (
        DynamicalDecouplingPass,
        PauliTwirlingPass,
    )

    if twirl:
        circuit = PauliTwirlingPass(seed=seed).run(circuit)
    if dd_sequence:
        circuit = DynamicalDecouplingPass(sequence=dd_sequence).run(circuit)
    return circuit
