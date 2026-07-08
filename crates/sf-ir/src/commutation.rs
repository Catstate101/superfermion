//! Commutation analysis for quantum gates.
//!
//! Determines whether two gates commute (can be reordered without changing
//! the circuit's unitary). This is used by optimization passes to identify
//! gates that can be moved past each other to enable cancellation.
//!
//! Key commutation rules:
//! - Diagonal gates commute with each other (Rz, S, T, Z, CZ, Rzz, P)
//! - X commutes with CNOT when X is on the target qubit
//! - Rz commutes with CNOT when Rz is on the control qubit
//! - Gates on disjoint qubits always commute
//! - Same-type rotation gates on the same qubit always commute

use crate::ops::OpType;

/// Check if two operations commute when they share at least one qubit.
///
/// `qubits_a` and `qubits_b` are the qubit operands of each gate.
/// Returns `true` if the gates can be reordered without affecting the unitary.
pub fn commutes(op_a: &OpType, qubits_a: &[usize], op_b: &OpType, qubits_b: &[usize]) -> bool {
    // Gates on completely disjoint qubits always commute
    if qubits_a.iter().all(|q| !qubits_b.contains(q)) {
        return true;
    }

    // Barriers and measurements don't commute with anything sharing qubits
    if op_a.is_boundary() || op_b.is_boundary()
        || op_a.is_measurement() || op_b.is_measurement()
        || *op_a == OpType::Barrier || *op_b == OpType::Barrier
        || *op_a == OpType::Reset || *op_b == OpType::Reset
    {
        return false;
    }

    // Same gate type on same qubits always commutes with itself
    if std::mem::discriminant(op_a) == std::mem::discriminant(op_b) && qubits_a == qubits_b {
        return true;
    }

    // Both 1-qubit gates on the same qubit
    if qubits_a.len() == 1 && qubits_b.len() == 1 && qubits_a[0] == qubits_b[0] {
        return commutes_1q_1q(op_a, op_b);
    }

    // 1q gate with 2q gate sharing one qubit
    if qubits_a.len() == 1 && qubits_b.len() == 2 {
        return commutes_1q_2q(op_a, qubits_a[0], op_b, qubits_b);
    }
    if qubits_b.len() == 1 && qubits_a.len() == 2 {
        return commutes_1q_2q(op_b, qubits_b[0], op_a, qubits_a);
    }

    // Both 2-qubit gates on the same qubits
    if qubits_a.len() == 2 && qubits_b.len() == 2 {
        return commutes_2q_2q(op_a, qubits_a, op_b, qubits_b);
    }

    false
}

/// Check commutation of two 1-qubit gates on the same qubit.
fn commutes_1q_1q(a: &OpType, b: &OpType) -> bool {
    // Diagonal 1q gates commute with each other
    if is_z_diagonal_1q(a) && is_z_diagonal_1q(b) {
        return true;
    }

    // Same-axis rotations commute
    if is_rz(a) && is_rz(b) { return true; }
    if is_rx(a) && is_rx(b) { return true; }
    if is_ry(a) && is_ry(b) { return true; }

    false
}

/// Check commutation of a 1-qubit gate with a 2-qubit gate.
fn commutes_1q_2q(gate_1q: &OpType, qubit_1q: usize, gate_2q: &OpType, qubits_2q: &[usize]) -> bool {
    let is_control = qubits_2q[0] == qubit_1q;
    let is_target = qubits_2q[1] == qubit_1q;

    match gate_2q {
        OpType::CNOT => {
            if is_control {
                // Z-basis diagonal gates commute with CNOT on control
                is_z_diagonal_1q(gate_1q)
            } else if is_target {
                // X-basis gates commute with CNOT on target
                is_x_basis_1q(gate_1q)
            } else {
                true
            }
        }
        OpType::CZ => {
            // CZ is symmetric and Z-diagonal; Z-diagonal 1q gates commute on either qubit
            if is_control || is_target {
                is_z_diagonal_1q(gate_1q)
            } else {
                true
            }
        }
        OpType::Rzz(_) => {
            // Rzz is Z-diagonal on both qubits
            if is_control || is_target {
                is_z_diagonal_1q(gate_1q)
            } else {
                true
            }
        }
        OpType::Rxx(_) => {
            // Rxx commutes with X-basis gates on either qubit
            if is_control || is_target {
                is_x_basis_1q(gate_1q)
            } else {
                true
            }
        }
        OpType::Ryy(_) => {
            // Ryy commutes with Y-basis gates on either qubit
            if is_control || is_target {
                is_y_basis_1q(gate_1q)
            } else {
                true
            }
        }
        _ => false,
    }
}

/// Check commutation of two 2-qubit gates.
fn commutes_2q_2q(a: &OpType, qa: &[usize], b: &OpType, qb: &[usize]) -> bool {
    // Both Z-diagonal 2q gates on the same qubits commute
    if is_z_diagonal_2q(a) && is_z_diagonal_2q(b) && qa == qb {
        return true;
    }

    // CZ is symmetric, so CZ on (0,1) commutes with CZ on (1,0)
    if *a == OpType::CZ && *b == OpType::CZ {
        let mut sa = qa.to_vec(); sa.sort();
        let mut sb = qb.to_vec(); sb.sort();
        if sa == sb { return true; }
    }

    false
}

fn is_z_diagonal_1q(op: &OpType) -> bool {
    matches!(op,
        OpType::Rz(_) | OpType::S | OpType::Sdg | OpType::T | OpType::Tdg
        | OpType::Z | OpType::P(_) | OpType::R1(_) | OpType::Id
    )
}

fn is_x_basis_1q(op: &OpType) -> bool {
    matches!(op,
        OpType::X | OpType::Rx(_) | OpType::SX | OpType::SXdg | OpType::Id
    )
}

fn is_y_basis_1q(op: &OpType) -> bool {
    matches!(op, OpType::Y | OpType::Ry(_) | OpType::Id)
}

fn is_rz(op: &OpType) -> bool {
    matches!(op, OpType::Rz(_) | OpType::S | OpType::Sdg | OpType::T | OpType::Tdg | OpType::Z | OpType::P(_))
}

fn is_rx(op: &OpType) -> bool {
    matches!(op, OpType::Rx(_) | OpType::X | OpType::SX | OpType::SXdg)
}

fn is_ry(op: &OpType) -> bool {
    matches!(op, OpType::Ry(_) | OpType::Y)
}

fn is_z_diagonal_2q(op: &OpType) -> bool {
    matches!(op, OpType::CZ | OpType::Rzz(_) | OpType::CP(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Parameter;

    #[test]
    fn test_disjoint_qubits_commute() {
        assert!(commutes(&OpType::H, &[0], &OpType::X, &[1]));
        assert!(commutes(&OpType::CNOT, &[0, 1], &OpType::H, &[2]));
    }

    #[test]
    fn test_z_diag_commute() {
        assert!(commutes(&OpType::Rz(Parameter::Const(0.5)), &[0], &OpType::S, &[0]));
        assert!(commutes(&OpType::T, &[0], &OpType::Z, &[0]));
        assert!(commutes(&OpType::Rz(Parameter::Const(0.5)), &[0], &OpType::Rz(Parameter::Const(1.0)), &[0]));
    }

    #[test]
    fn test_h_rz_dont_commute() {
        assert!(!commutes(&OpType::H, &[0], &OpType::Rz(Parameter::Const(0.5)), &[0]));
    }

    #[test]
    fn test_rz_commutes_with_cnot_on_control() {
        assert!(commutes(&OpType::Rz(Parameter::Const(0.5)), &[0], &OpType::CNOT, &[0, 1]));
    }

    #[test]
    fn test_rz_doesnt_commute_with_cnot_on_target() {
        assert!(!commutes(&OpType::Rz(Parameter::Const(0.5)), &[1], &OpType::CNOT, &[0, 1]));
    }

    #[test]
    fn test_x_commutes_with_cnot_on_target() {
        assert!(commutes(&OpType::X, &[1], &OpType::CNOT, &[0, 1]));
    }

    #[test]
    fn test_x_doesnt_commute_with_cnot_on_control() {
        assert!(!commutes(&OpType::X, &[0], &OpType::CNOT, &[0, 1]));
    }

    #[test]
    fn test_z_diag_commutes_with_cz() {
        assert!(commutes(&OpType::Rz(Parameter::Const(0.5)), &[0], &OpType::CZ, &[0, 1]));
        assert!(commutes(&OpType::S, &[1], &OpType::CZ, &[0, 1]));
    }

    #[test]
    fn test_cz_cz_commute() {
        assert!(commutes(&OpType::CZ, &[0, 1], &OpType::CZ, &[0, 1]));
        assert!(commutes(&OpType::CZ, &[0, 1], &OpType::CZ, &[1, 0]));
    }

    #[test]
    fn test_same_gate_same_qubits_commute() {
        assert!(commutes(&OpType::CNOT, &[0, 1], &OpType::CNOT, &[0, 1]));
    }

    #[test]
    fn test_measure_doesnt_commute() {
        assert!(!commutes(&OpType::Measure, &[0], &OpType::H, &[0]));
    }

    #[test]
    fn test_rzz_commutes_with_rz() {
        assert!(commutes(
            &OpType::Rz(Parameter::Const(0.5)), &[0],
            &OpType::Rzz(Parameter::Const(0.3)), &[0, 1],
        ));
    }
}
