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

from typing import List, Optional

from superfermion.circuit import Circuit
from superfermion.compiler.passes import (
    BasisTranslationPass,
    UnitaryDecompositionPass,
    Pass,
)
from superfermion.compiler.specs import HardwareSpec


class PassManager:
    """Manages a sequence of compilation passes.

    Retained for users who need custom pass pipelines (e.g. combining
    the temporary shim passes with their own transforms).
    """

    def __init__(self, passes: Optional[List[Pass]] = None):
        self.passes: List[Pass] = passes or []

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


def compile(circuit: Circuit, level: int = 1, target: Optional[HardwareSpec] = None) -> Circuit:
    """Compile a quantum circuit for a hardware target.

    Delegates to the Rust ``_sf_core.Compiler`` for all optimization,
    decomposition, and routing. Python shims run before/after Rust for
    capabilities not yet in the Rust compiler.

    Args:
        circuit: The input quantum circuit.
        level: Optimization level (0=none, 1=standard, 2=aggressive).
        target: Optional hardware target specification.

    Returns:
        Compiled and optimized circuit for the target. The returned circuit has
        the same width as ``circuit`` even when the target is wider, and carries
        the layout it was compiled under in ``circuit._metadata["layout"]``
        (see :func:`superfermion.compiler.layout.get_layout`).
    """
    if level == 0 and not target:
        return circuit

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
    layout_report: Optional[dict] = None
    if target is not None and getattr(target, "coupling_map", None):
        from superfermion.compiler.layout import route_with_layout

        circuit, layout_report = route_with_layout(circuit, target)

    if target:
        compiled = _rust_compile(
            circuit,
            level=level,
            target=target,
            skip_routing=bool(layout_report and layout_report.get("routed")),
        )
    else:
        # No target: hardware-neutral optimization only
        compiled = _rust_compile(circuit, level=level, target=None)

    if layout_report is not None:
        compiled._metadata["layout"] = layout_report
        compiled._metadata["target"] = getattr(target, "name", None)
        compiled._metadata["level"] = level
    return compiled


def _rust_compile(
    circuit: Circuit,
    level: int,
    target: Optional[HardwareSpec],
    skip_routing: bool = False,
) -> Circuit:
    """Delegate to the Rust compilation pipeline."""
    from superfermion.compiler.rust_bridge import compile_rust
    return compile_rust(circuit, level=level, target=target, skip_routing=skip_routing)


def apply_noise_suppression(
    circuit: Circuit,
    twirl: bool = True,
    dd_sequence: str = "XY4",
    seed: Optional[int] = None,
) -> Circuit:
    """Apply noise suppression passes (opt-in, separate from compile).

    Pauli twirling and dynamical decoupling are noise-mitigation
    techniques, not transpilation. They are deliberately excluded from
    ``compile()`` and exposed here for explicit use.

    Args:
        circuit: Compiled circuit (should already be in target basis).
        twirl: Apply Pauli twirling to 2Q gates.
        dd_sequence: Dynamical decoupling sequence name.
        seed: Random seed for twirling reproducibility.

    Returns:
        Circuit with noise suppression gates inserted.
    """
    from superfermion.compiler.advanced import (
        PauliTwirlingPass,
        DynamicalDecouplingPass,
    )

    if twirl:
        circuit = PauliTwirlingPass(seed=seed).run(circuit)
    if dd_sequence:
        circuit = DynamicalDecouplingPass(sequence=dd_sequence).run(circuit)
    return circuit
