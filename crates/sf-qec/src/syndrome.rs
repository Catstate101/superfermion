//! Syndrome extraction — build circuits to measure stabilizers.

use crate::codes::{Stabilizer, StabilizerCode, Pauli};
use sf_ir::{QuantumDAG, OpType};

/// Builds syndrome extraction circuits from stabilizer codes.
pub struct SyndromeExtractor;

impl SyndromeExtractor {
    /// Build a circuit that measures all stabilizers of the code.
    ///
    /// For each stabilizer:
    /// 1. Prepare ancilla in |+⟩ (for X-type) or |0⟩ (for Z-type)
    /// 2. Apply controlled-Pauli gates between ancilla and data qubits
    /// 3. Measure ancilla
    pub fn build_circuit(code: &dyn StabilizerCode) -> QuantumDAG {
        let n_total = code.n_data() + code.n_ancilla();
        let n_cbits = code.n_ancilla();
        let mut dag = QuantumDAG::new(n_total, n_cbits);

        for (idx, stab) in code.stabilizers().iter().enumerate() {
            Self::add_stabilizer_measurement(&mut dag, stab, idx);
        }

        dag
    }

    /// Add gates for measuring a single stabilizer.
    fn add_stabilizer_measurement(dag: &mut QuantumDAG, stab: &Stabilizer, cbit: usize) {
        let ancilla = stab.ancilla;

        // Determine if this is an X-type or Z-type stabilizer
        let has_x = stab.paulis.iter().any(|(_, p)| *p == Pauli::X);

        if has_x {
            // For X-stabilizers: prepare |+⟩ state
            dag.add_op(OpType::H, &[ancilla]);
        }

        // Apply controlled-Pauli interactions
        for &(data_qubit, ref pauli) in &stab.paulis {
            match pauli {
                Pauli::X => {
                    // CNOT with ancilla as control
                    dag.add_op(OpType::CNOT, &[ancilla, data_qubit]);
                }
                Pauli::Z => {
                    // CNOT with data as control, ancilla as target
                    dag.add_op(OpType::CNOT, &[data_qubit, ancilla]);
                }
                Pauli::Y => {
                    // Y = S† · X · S
                    dag.add_op(OpType::Sdg, &[data_qubit]);
                    dag.add_op(OpType::CNOT, &[ancilla, data_qubit]);
                    dag.add_op(OpType::S, &[data_qubit]);
                }
                Pauli::I => {} // No operation
            }
        }

        if has_x {
            dag.add_op(OpType::H, &[ancilla]);
        }

        // Measure ancilla
        dag.add_measure(ancilla, cbit);
    }

    /// Get the number of syndrome bits.
    pub fn n_syndrome_bits(code: &dyn StabilizerCode) -> usize {
        code.stabilizers().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::RepetitionCode;

    #[test]
    fn test_repetition_syndrome_circuit() {
        let code = RepetitionCode::new(3);
        let circuit = SyndromeExtractor::build_circuit(&code);
        // 3 data + 2 ancilla = 5 qubits
        assert_eq!(circuit.n_qubits, 5);
        // Should have CNOT gates and measurements
        assert!(circuit.gate_count() > 0);
    }

    #[test]
    fn test_syndrome_bits() {
        let code = RepetitionCode::new(5);
        assert_eq!(SyndromeExtractor::n_syndrome_bits(&code), 4);
    }
}
