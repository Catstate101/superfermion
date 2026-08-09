//! Gate fusion pass: merge consecutive single-qubit gates on the same qubit
//! into a single U(theta, phi, lambda) gate.
//!
//! This replaces the Python-side `turbo.fuse_single_qubit_gates()` to keep
//! gate reduction entirely in Rust, eliminating one Python→Rust round-trip.

use crate::{CompilerError, Pass};
use nalgebra::Matrix2;
use num_complex::Complex64;
use sf_ir::{OpType, Parameter, QuantumDAG};

/// Fuse consecutive 1-qubit gates on the same qubit into a single U gate.
pub struct GateFusionPass;

impl GateFusionPass {
    pub fn new() -> Self {
        Self
    }

    /// Multiply two 2x2 unitary matrices.
    fn mat_mul(a: &Matrix2<Complex64>, b: &Matrix2<Complex64>) -> Matrix2<Complex64> {
        a * b
    }

    /// Decompose a 2x2 unitary into U(theta, phi, lambda) parameters.
    /// U(t, p, l) = [[cos(t/2), -e^{il}*sin(t/2)],
    ///               [e^{ip}*sin(t/2), e^{i(p+l)}*cos(t/2)]]
    fn decompose_u3(u: &Matrix2<Complex64>) -> (f64, f64, f64) {
        let u00 = u[(0, 0)];
        let u01 = u[(0, 1)];
        let u10 = u[(1, 0)];
        let u11 = u[(1, 1)];

        // theta = 2 * acos(|u00|), clamped for numerical stability
        let abs_u00 = u00.norm().min(1.0);
        let theta = 2.0 * abs_u00.acos();

        if theta < 1e-6 {
            // Near-identity: theta~0, phi+lambda = arg(u11)
            let phase = u11.arg();
            if phase.abs() < 1e-6 {
                return (0.0, 0.0, 0.0);
            }
            return (0.0, 0.0, phase);
        }

        if (theta - std::f64::consts::PI).abs() < 1e-6 {
            // Near X-like: theta~pi
            let phi = u10.arg();
            let lambda = -u01.arg();
            return (std::f64::consts::PI, phi, lambda);
        }

        let phi = u10.arg() - u00.arg();
        let lambda = -(u01.arg() - u00.arg());

        (theta, phi, lambda)
    }
}

impl Pass for GateFusionPass {
    fn name(&self) -> &str {
        "GateFusion"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let n_qubits = dag.n_qubits;

        // For each qubit, collect consecutive 1q gates and fuse them.
        // We rebuild the DAG from scratch with fused gates.
        let instructions: Vec<_> = dag.to_instructions().into_iter().cloned().collect();

        // Track pending 1q gate accumulator per qubit: Option<Matrix2<Complex64>>
        let mut accum: Vec<Option<Matrix2<Complex64>>> = vec![None; n_qubits];
        let mut fused_ops: Vec<(OpType, Vec<usize>)> = Vec::new();

        let flush_qubit = |q: usize,
                           accum: &mut Vec<Option<Matrix2<Complex64>>>,
                           fused_ops: &mut Vec<(OpType, Vec<usize>)>| {
            if let Some(mat) = accum[q].take() {
                let (theta, phi, lambda) = Self::decompose_u3(&mat);
                // Skip if near-identity (threshold accounts for float precision in matrix products)
                if theta.abs() > 1e-6 || phi.abs() > 1e-6 || lambda.abs() > 1e-6 {
                    fused_ops.push((
                        OpType::U(
                            Parameter::Const(theta),
                            Parameter::Const(phi),
                            Parameter::Const(lambda),
                        ),
                        vec![q],
                    ));
                }
            }
        };

        for inst in &instructions {
            if inst.op_type.is_boundary() || inst.op_type == OpType::Barrier {
                continue;
            }

            if inst.qubits.len() == 1 && !inst.op_type.is_measurement() {
                let q = inst.qubits[0];
                let gate_mat = inst.op_type.to_matrix();
                let m = Matrix2::new(
                    gate_mat[(0, 0)],
                    gate_mat[(0, 1)],
                    gate_mat[(1, 0)],
                    gate_mat[(1, 1)],
                );

                accum[q] = Some(match accum[q].take() {
                    Some(existing) => Self::mat_mul(&m, &existing),
                    None => m,
                });
            } else {
                // Multi-qubit gate or measurement: flush all involved qubits
                for &q in inst.qubits.iter() {
                    flush_qubit(q, &mut accum, &mut fused_ops);
                }
                fused_ops.push((inst.op_type.clone(), inst.qubits.to_vec()));
            }
        }

        // Flush remaining accumulators
        for q in 0..n_qubits {
            flush_qubit(q, &mut accum, &mut fused_ops);
        }

        // Rebuild DAG
        let mut new_dag = QuantumDAG::new(dag.n_qubits, dag.n_cbits);
        for (op_type, qubits) in fused_ops {
            if op_type == OpType::Measure {
                if !qubits.is_empty() {
                    new_dag.add_measure(qubits[0], qubits[0]);
                }
            } else {
                new_dag.add_op(op_type, &qubits);
            }
        }

        *dag = new_dag;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::{OpType, Parameter, QuantumDAG};

    #[test]
    fn test_hh_cancels() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 0, "H*H should cancel to identity");
    }

    #[test]
    fn test_rz_rz_merge() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rz(Parameter::Const(0.5)), &[0]);
        dag.add_op(OpType::Rz(Parameter::Const(0.3)), &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        // Two Rz should fuse into one U gate
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn test_2q_interrupts_fusion() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        // CNOT interrupts fusion: H, CNOT, H remain separate (though each H
        // becomes a U gate).
        assert_eq!(dag.gate_count(), 3);
    }

    #[test]
    fn test_three_gates_fuse_to_one() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::S, &[0]);
        dag.add_op(OpType::T, &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 1, "H*S*T should fuse into one U gate");
    }

    #[test]
    fn test_multi_qubit_independent_fusion() {
        // Two independent qubits each get their gates fused
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::S, &[0]);
        dag.add_op(OpType::X, &[1]);
        dag.add_op(OpType::Y, &[1]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        // q0: H*S → 1 gate (or identity check), q1: X*Y → should fuse
        assert!(
            dag.gate_count() <= 2,
            "Each qubit chain should fuse: got {}",
            dag.gate_count()
        );
    }

    #[test]
    fn test_xx_cancels() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::X, &[0]);
        dag.add_op(OpType::X, &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 0, "X*X should cancel to identity");
    }

    #[test]
    fn test_single_gate_unchanged() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        // A single gate should remain (possibly as U equivalent)
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn test_fusion_preserves_2q_gates() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CZ, &[0, 1]);

        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();

        // 2-qubit gates should not be fused by 1q fusion pass
        assert_eq!(dag.gate_count(), 2);
    }

    #[test]
    fn test_decompose_u3_identity() {
        let eye = Matrix2::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        );
        let (theta, phi, lam) = GateFusionPass::decompose_u3(&eye);
        assert!(theta.abs() < 1e-6, "Identity theta={}", theta);
        assert!(
            (phi + lam).abs() < 1e-6 || (phi + lam - 2.0 * std::f64::consts::PI).abs() < 1e-6,
            "Identity phi+lam should be ~0, got phi={}, lam={}",
            phi,
            lam
        );
    }

    #[test]
    fn test_decompose_u3_x_gate() {
        let x_mat = Matrix2::new(
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        let (theta, _phi, _lam) = GateFusionPass::decompose_u3(&x_mat);
        assert!(
            (theta - std::f64::consts::PI).abs() < 1e-6,
            "X gate theta should be pi, got {}",
            theta
        );
    }
}
