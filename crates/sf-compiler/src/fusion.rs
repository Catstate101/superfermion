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

        // theta = 2 * atan2(|u10|, |u00|).  The arccos form used here before
        // is ill-conditioned at theta -> 0: one ulp of rounding in |u00|
        // (1.1e-16) became an angle of 1.5e-8, and that spurious rotation is a
        // first-order error in the fused gate.  atan2 is well-conditioned at
        // both ends and returns exactly 0 when u10 is exactly 0.
        let abs_u00 = u00.norm();
        let abs_u10 = u10.norm();
        let theta = 2.0 * abs_u10.atan2(abs_u00);

        // The near-identity branch must not swallow a real rotation.  It used
        // to fire at theta < 1e-6, which discarded genuine angles: lane F3 had
        // a two-gate product with fused theta = 6.66e-8 that came back as
        // U(0, 0, lambda), a theta/2 = 3.3e-8 error in the matrix.  1e-15 is
        // ~2 ulp of pi, so only the residue of an exact cancellation is
        // collapsed; a phase of 1e-13 is still a real rotation.
        if theta < 1e-15 {
            // sin(theta/2) is below the noise floor, so phi is unconstrained;
            // fix phi = 0 and keep the relative phase on the diagonal.
            let lambda = u11.arg() - u00.arg();
            if lambda.abs() < 1e-15 {
                return (0.0, 0.0, 0.0);
            }
            return (0.0, 0.0, lambda);
        }

        // U(t,p,l) = [[c, -e^{il}s], [e^{ip}s, e^{i(p+l)}c]] with c = cos(t/2), s = sin(t/2).
        // Writing u00 = e^{i*alpha}*c: phi = arg(u10) - alpha and
        // lambda = arg(u01) - alpha - pi (the -pi absorbs the minus sign on u01).
        // At theta ~ pi, alpha = arg(0) = 0 and the formula remains exact.
        let alpha = u00.arg();
        let phi = u10.arg() - alpha;
        let lambda = u01.arg() - alpha - std::f64::consts::PI;

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
                // Drop only a true identity.  The old 1e-6 threshold on the
                // angles discarded fusions of up to 1e-6 rad (5e-7 in the
                // matrix) -- the same defect class as decompose_u3 above.
                if theta.abs() > 1e-15 || phi.abs() > 1e-15 || lambda.abs() > 1e-15 {
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
    fn test_decompose_u3_pure_phase_is_exact() {
        // Regression: a diagonal gate has |u00| = 1 up to one ulp, and
        // ``2*acos(|u00|)`` turned that ulp into theta = 1.5e-8.  Fusion calls
        // this on *products* of gates, so the rounding is the normal case.
        let rz = Matrix2::new(
            Complex64::new((0.7f64 / 2.0).cos(), -(0.7f64 / 2.0).sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new((0.7f64 / 2.0).cos(), (0.7f64 / 2.0).sin()),
        );
        let (theta, _phi, _lam) = GateFusionPass::decompose_u3(&rz);
        assert!(
            theta.abs() < 1e-15,
            "a pure phase must have theta == 0 exactly, got {theta:e}"
        );
    }

    #[test]
    fn test_decompose_u3_small_rotation_is_not_swallowed() {
        // Regression for lane F3: a two-gate QASM ``u`` product had fused
        // theta = 6.66e-8, and the near-identity branch (formerly
        // ``theta < 1e-6``) answered U(0, 0, lambda).  The missing rotation is
        // theta/2 = 3.3e-8 in the matrix -- exactly the measured error.
        let theta = 6.66e-8_f64;
        let (s, c) = ((theta / 2.0).sin(), (theta / 2.0).cos());
        let m = Matrix2::new(
            Complex64::new(c, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(c, 0.0),
        );
        let (t, phi, lam) = GateFusionPass::decompose_u3(&m);
        assert!(
            (t - theta).abs() < 1e-15,
            "theta must survive the decomposition, got {t:e} vs {theta:e}"
        );
        let (cc, ss) = ((t / 2.0).cos(), (t / 2.0).sin());
        let recon = Matrix2::new(
            Complex64::from_polar(cc, 0.0),
            Complex64::from_polar(-ss, lam),
            Complex64::from_polar(ss, phi),
            Complex64::from_polar(cc, phi + lam),
        );
        let mut err = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                err = err.max((recon[(i, j)] - m[(i, j)]).norm());
            }
        }
        assert!(
            err < 1e-15,
            "U(theta, phi, lambda) reconstruction error {err:e}"
        );
    }

    #[test]
    fn test_near_identity_threshold_keeps_small_rotations() {
        // A fused cluster must not be dropped just because its angles are
        // small: flush_qubit used to skip anything below 1e-6 rad.
        let theta = 5.0e-7_f64;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::U(
                Parameter::Const(theta),
                Parameter::Const(0.0),
                Parameter::Const(0.0),
            ),
            &[0],
        );
        let pass = GateFusionPass::new();
        pass.run(&mut dag).unwrap();
        assert_eq!(
            dag.gate_count(),
            1,
            "a {theta:e} rad rotation must not be fused away"
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
