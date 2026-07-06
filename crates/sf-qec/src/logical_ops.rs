//! Logical operations on encoded qubits.

use crate::codes::StabilizerCode;
use sf_ir::QuantumDAG;

/// Trait for logical operations.
pub trait LogicalOp {
    fn apply_logical_x(&self, dag: &mut QuantumDAG, code: &dyn StabilizerCode);
    fn apply_logical_z(&self, dag: &mut QuantumDAG, code: &dyn StabilizerCode);
}

pub struct DefaultLogicalOps;

impl LogicalOp for DefaultLogicalOps {
    // TODO: Implement transversality check and gate insertion.
    // For a repetition code, Logical X = X applied to all data qubits.
    // For a repetition code, Logical Z = Z applied to one data qubit.
    // Plan: iterate over stabilizers to verify transversality, then insert
    // the appropriate single-qubit gates on each data qubit.

    fn apply_logical_x(&self, _dag: &mut QuantumDAG, _code: &dyn StabilizerCode) {
        // TODO: Implementation pending. See trait-level TODO for plan.
    }

    fn apply_logical_z(&self, _dag: &mut QuantumDAG, _code: &dyn StabilizerCode) {
        // TODO: Implementation pending. See trait-level TODO for plan.
    }
}
