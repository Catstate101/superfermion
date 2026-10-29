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
    //
    // Until that lands these must panic rather than return. Returning quietly
    // hands the caller back an unchanged DAG -- a circuit that is silently
    // missing the logical operation it asked for. That is the same failure
    // mode the OpenQASM exporter refuses with ValueError instead of dropping
    // an unmappable gate.

    fn apply_logical_x(&self, _dag: &mut QuantumDAG, _code: &dyn StabilizerCode) {
        unimplemented!("DefaultLogicalOps::apply_logical_x: logical X insertion is not implemented")
    }

    fn apply_logical_z(&self, _dag: &mut QuantumDAG, _code: &dyn StabilizerCode) {
        unimplemented!("DefaultLogicalOps::apply_logical_z: logical Z insertion is not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::RepetitionCode;

    /// Both methods used to return `()` without touching the DAG, so a caller
    /// had no way to tell that no logical operation had been inserted. These
    /// pin the loud failure so the stub cannot quietly come back.
    #[test]
    #[should_panic(expected = "logical X insertion is not implemented")]
    fn apply_logical_x_refuses_instead_of_returning_silently() {
        let mut dag = QuantumDAG::new(3, 0);
        let code = RepetitionCode::new(3);
        DefaultLogicalOps.apply_logical_x(&mut dag, &code);
    }

    #[test]
    #[should_panic(expected = "logical Z insertion is not implemented")]
    fn apply_logical_z_refuses_instead_of_returning_silently() {
        let mut dag = QuantumDAG::new(3, 0);
        let code = RepetitionCode::new(3);
        DefaultLogicalOps.apply_logical_z(&mut dag, &code);
    }
}
