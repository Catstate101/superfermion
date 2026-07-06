"""
Compiler Manager — Central orchestrator for quantum compilation.
"""

from __future__ import annotations

from typing import List, Optional

from superfermion.circuit import Circuit
from superfermion.compiler.passes import (
    GateCancellationPass,
    SwapDecompositionPass,
    BasisTranslationPass,
    RotationMergingPass,
    ConstantFoldingPass,
    Pass,
)
from superfermion.compiler.advanced import (
    SABRERoutingPass,
    DynamicalDecouplingPass,
    PauliTwirlingPass,
)
from superfermion.compiler.advanced_passes import CommutationPass
from superfermion.compiler.specs import HardwareSpec


class PassManager:
    """Manages a sequence of compilation passes."""
    
    def __init__(self, passes: Optional[List[Pass]] = None):
        self.passes: List[Pass] = passes or []

    # Plugin pass registry (class-level, shared across all PassManager instances)
    _plugin_passes: dict = {}

    def add_pass(self, pass_obj: Pass):
        """Add a new pass to the pipeline."""
        self.passes.append(pass_obj)

    @classmethod
    def add_plugin_pass(cls, name: str, pass_cls: type) -> None:
        """Register a plugin compiler pass for use in pipelines.

        Called automatically by ``@register_pass`` decorator.
        """
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


def compile(circuit: Circuit, level: int = 1, target: Optional[HardwareSpec] = None) -> Circuit:
    """
    High-level compilation entry point.
    
    Args:
        circuit: The input quantum circuit.
        level: Optimization level (0=none, 1=standard, 2=aggressive).
        target: Optional hardware target specification.
        
    Returns:
        Compiled and optimized circuit for the target.
    """
    if level == 0 and not target:
        return circuit

    # Assemble the pipeline
    manager = PassManager()
    
    # 1. Hardware Neutral Optimizations
    if level >= 1:
        manager.add_pass(SwapDecompositionPass())
        manager.add_pass(GateCancellationPass())
        manager.add_pass(RotationMergingPass())
        manager.add_pass(ConstantFoldingPass())
        manager.add_pass(CommutationPass())

    # 2. Aggressive Gate-Level Optimizations
    if level >= 2:
        # Re-run cancellation after commutation may have exposed new pairs
        manager.add_pass(GateCancellationPass())
        manager.add_pass(RotationMergingPass())

    # 3. Hardware Specific Optimizations
    if target:
        # Translate to basis gates
        manager.add_pass(BasisTranslationPass(target.native_gates))

        # Route to coupling map using full SABRE routing
        if target.coupling_map:
            manager.add_pass(SABRERoutingPass(target.coupling_map))

        # Apply dynamical decoupling for noise suppression on real hardware
        if level >= 2:
            manager.add_pass(PauliTwirlingPass(seed=42))
            manager.add_pass(DynamicalDecouplingPass(sequence="XY4"))

    return manager.run(circuit)
