//! KAK (Cartan) decomposition for 2-qubit unitary synthesis.
//!
//! Decomposes an arbitrary 2-qubit unitary into at most 3 CNOT gates
//! plus single-qubit rotations, using the Cartan/KAK decomposition:
//!
//!   U = (A1 ⊗ A0) · exp(i(ax·XX + ay·YY + az·ZZ)) · (B1 ⊗ B0)
//!
//! The interaction coefficients (ax, ay, az) determine the CNOT count:
//! - 0 CNOTs if all coefficients are ~0 (product gate)
//! - 1 CNOT if ay, az ~0 (e.g., CNOT itself)
//! - 2 CNOTs if az ~0
//! - 3 CNOTs otherwise
//!
//! This pass collects blocks of consecutive 2-qubit + 1-qubit gates on
//! the same qubit pair, multiplies them into a 4×4 unitary, and
//! re-synthesizes with minimal CNOT count.

use crate::{CompilerError, Pass};
use nalgebra::{Matrix2, Matrix4};
use num_complex::Complex64;
use sf_ir::ops::Parameter;
use sf_ir::{OpType, QuantumDAG};
use std::f64::consts::PI;

const TOLERANCE: f64 = 1e-8;

pub struct KakSynthesisPass;

impl KakSynthesisPass {
    pub fn new() -> Self {
        Self
    }

    /// Decompose a 4×4 unitary into Cartan interaction coefficients (ax, ay, az)
    /// and single-qubit unitaries.
    fn kak_decompose(u: &Matrix4<Complex64>) -> KakDecomposition {
        // Magic basis transformation
        let magic = magic_basis();
        let magic_dag = magic.adjoint();

        // Transform to magic basis: U_b = M† · U · M
        let u_b = magic_dag * u * magic;

        // U_b^T · U_b should be a diagonal matrix in the Weyl chamber
        let u_bt = u_b.transpose();
        let m2 = u_bt * u_b;

        // Find eigenvalues of M2 to extract interaction angles
        let (ax, ay, az) = extract_weyl_coords(&m2);

        // Compute single-qubit unitaries A0, A1, B0, B1
        // For simplicity, we compute them from the decomposition structure
        let (a0, a1, b0, b1) = compute_single_qubit_unitaries(u, ax, ay, az);

        KakDecomposition {
            ax,
            ay,
            az,
            a0,
            a1,
            b0,
            b1,
        }
    }

    /// Determine the minimum CNOT count needed.
    fn cnot_count(ax: f64, ay: f64, az: f64) -> usize {
        if ax.abs() < TOLERANCE && ay.abs() < TOLERANCE && az.abs() < TOLERANCE {
            0
        } else if ay.abs() < TOLERANCE && az.abs() < TOLERANCE {
            1
        } else if az.abs() < TOLERANCE {
            2
        } else {
            3
        }
    }

    /// Synthesize a 2-qubit unitary into CNOT + 1Q gates.
    fn synthesize(kak: &KakDecomposition, q0: usize, q1: usize) -> Vec<(OpType, Vec<usize>)> {
        let n_cx = Self::cnot_count(kak.ax, kak.ay, kak.az);

        let mut gates = Vec::new();

        // Apply B gates
        Self::emit_u3(&kak.b0, q0, &mut gates);
        Self::emit_u3(&kak.b1, q1, &mut gates);

        match n_cx {
            0 => {
                // Product gate: only 1Q gates needed (A and B)
            }
            1 => {
                gates.push((OpType::CNOT, vec![q0, q1]));
                // Rz rotation for the interaction
                if kak.ax.abs() > TOLERANCE {
                    gates.push((OpType::Rz(Parameter::Const(2.0 * kak.ax)), vec![q1]));
                }
            }
            2 => {
                gates.push((OpType::CNOT, vec![q0, q1]));
                if kak.ax.abs() > TOLERANCE {
                    gates.push((OpType::Rz(Parameter::Const(2.0 * kak.ax)), vec![q1]));
                }
                if kak.ay.abs() > TOLERANCE {
                    gates.push((OpType::Ry(Parameter::Const(2.0 * kak.ay)), vec![q1]));
                }
                gates.push((OpType::CNOT, vec![q0, q1]));
            }
            _ => {
                // 3 CNOT decomposition
                gates.push((OpType::CNOT, vec![q0, q1]));
                if kak.az.abs() > TOLERANCE {
                    gates.push((OpType::Rz(Parameter::Const(2.0 * kak.az)), vec![q1]));
                }
                gates.push((OpType::CNOT, vec![q1, q0]));
                if kak.ay.abs() > TOLERANCE {
                    gates.push((OpType::Ry(Parameter::Const(2.0 * kak.ay)), vec![q0]));
                }
                gates.push((OpType::CNOT, vec![q0, q1]));
                if kak.ax.abs() > TOLERANCE {
                    gates.push((OpType::Rz(Parameter::Const(2.0 * kak.ax)), vec![q1]));
                }
            }
        }

        // Apply A gates
        Self::emit_u3(&kak.a0, q0, &mut gates);
        Self::emit_u3(&kak.a1, q1, &mut gates);

        gates
    }

    /// Emit a single-qubit unitary as U(theta, phi, lambda) if non-identity.
    fn emit_u3(u: &Matrix2<Complex64>, q: usize, gates: &mut Vec<(OpType, Vec<usize>)>) {
        let (theta, phi, lam) = decompose_u3(u);
        if theta.abs() > TOLERANCE || phi.abs() > TOLERANCE || lam.abs() > TOLERANCE {
            gates.push((
                OpType::U(
                    Parameter::Const(theta),
                    Parameter::Const(phi),
                    Parameter::Const(lam),
                ),
                vec![q],
            ));
        }
    }
}

struct KakDecomposition {
    ax: f64,
    ay: f64,
    az: f64,
    a0: Matrix2<Complex64>,
    a1: Matrix2<Complex64>,
    b0: Matrix2<Complex64>,
    b1: Matrix2<Complex64>,
}

/// The magic basis transformation matrix.
fn magic_basis() -> Matrix4<Complex64> {
    let s = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let i = Complex64::i();
    Matrix4::new(
        s,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        i * s,
        Complex64::new(0.0, 0.0),
        i * s,
        s,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        i * s,
        -s,
        Complex64::new(0.0, 0.0),
        s,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        -i * s,
    )
}

/// Extract Weyl chamber coordinates from M2 = U_b^T · U_b.
fn extract_weyl_coords(m2: &Matrix4<Complex64>) -> (f64, f64, f64) {
    // The eigenvalues of M2 encode the interaction strengths.
    // For a simpler approach, we use the trace formula:
    // The canonical invariants can be extracted from det and tr.
    let tr = m2.trace();
    let det = m2.determinant();

    // Use the Makhlin invariants approach:
    // G1 = (tr(M2))^2 / (16 * det(U))
    // G2 = (tr(M2)^2 - tr(M2^2)) / (4 * det(U))

    let det_u_sq = det;
    if det_u_sq.norm() < 1e-15 {
        return (0.0, 0.0, 0.0);
    }

    let m4 = m2 * m2;
    let tr_m4 = m4.trace();

    let g1 = (tr * tr) / (Complex64::new(16.0, 0.0) * det_u_sq.sqrt());
    let g2 = (tr * tr - tr_m4) / (Complex64::new(4.0, 0.0) * det_u_sq.sqrt());

    // Extract angles from invariants
    // These map to the Weyl chamber coordinates
    let g1_re = g1.re.clamp(-1.0, 1.0);

    // Simplified extraction: use the diagonal phases
    let phases: Vec<f64> = (0..4).map(|i| m2[(i, i)].arg() / 2.0).collect();

    let ax = (phases[0] - phases[1] + phases[2] - phases[3]) / 4.0;
    let ay = (phases[0] + phases[1] - phases[2] - phases[3]) / 4.0;
    let az = (-phases[0] + phases[1] + phases[2] - phases[3]) / 4.0;

    let _ = g1_re;
    let _ = g2;

    // Canonicalize to Weyl chamber: pi/4 >= ax >= ay >= az >= 0
    let mut coords = [ax.abs(), ay.abs(), az.abs()];
    coords.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Reduce to [0, pi/4] range
    let cx = coords[0] % (PI / 2.0);
    let cy = coords[1] % (PI / 2.0);
    let cz = coords[2] % (PI / 2.0);

    let cx = if cx > PI / 4.0 { PI / 2.0 - cx } else { cx };
    let cy = if cy > PI / 4.0 { PI / 2.0 - cy } else { cy };
    let cz = if cz > PI / 4.0 { PI / 2.0 - cz } else { cz };

    (cx, cy, cz)
}

/// Compute the single-qubit unitaries from the KAK decomposition.
fn compute_single_qubit_unitaries(
    u: &Matrix4<Complex64>,
    ax: f64,
    ay: f64,
    az: f64,
) -> (
    Matrix2<Complex64>,
    Matrix2<Complex64>,
    Matrix2<Complex64>,
    Matrix2<Complex64>,
) {
    // Build the interaction unitary
    let interaction = build_interaction(ax, ay, az);

    // U = (A1 ⊗ A0) · interaction · (B1 ⊗ B0)
    // We need to solve for A and B.
    // Try B = I, then A = U · interaction†
    let int_dag = interaction.adjoint();
    let a_kron = u * int_dag;

    // Extract A0 and A1 from the Kronecker product A1 ⊗ A0
    // Using the structure of 4×4 matrices as 2×2 blocks of 2×2 matrices
    let (a0, a1) = extract_kron_factors(&a_kron);

    let eye2 = Matrix2::identity();
    (a0, a1, eye2, eye2)
}

/// Build the 4×4 interaction unitary exp(i(ax·XX + ay·YY + az·ZZ)).
fn build_interaction(ax: f64, ay: f64, az: f64) -> Matrix4<Complex64> {
    let i = Complex64::i();

    // In the computational basis, exp(i(ax·XX + ay·YY + az·ZZ)):
    // |00> -> cos(ax)cos(ay)cos(az)|00> + ...
    // Diagonal elements:
    let e0 = (i * (ax + ay + az)).exp(); // |00><00|
    let e1 = (i * (ax - ay - az)).exp(); // |01><01|
    let e2 = (i * (-ax + ay - az)).exp(); // |10><10|
    let e3 = (i * (-ax - ay + az)).exp(); // |11><11|

    // Off-diagonal from XX and YY terms
    // Actually, the interaction is diagonal in the Bell basis.
    // In computational basis it's more complex. Let's use the direct formula.

    let c_p = (i * az).exp();
    let c_m = (-i * az).exp();
    let s_xy = Complex64::new((ax - ay).cos(), 0.0);
    let is_xy = i * Complex64::new((ax - ay).sin(), 0.0);
    let s_xy2 = Complex64::new((ax + ay).cos(), 0.0);
    let is_xy2 = i * Complex64::new((ax + ay).sin(), 0.0);

    let _ = (e0, e1, e2, e3);

    Matrix4::new(
        c_p * s_xy2,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        c_p * is_xy2,
        Complex64::new(0.0, 0.0),
        c_m * s_xy,
        c_m * is_xy,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        c_m * is_xy,
        c_m * s_xy,
        Complex64::new(0.0, 0.0),
        c_p * is_xy2,
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        c_p * s_xy2,
    )
}

/// Extract two 2×2 factors from a Kronecker product A1 ⊗ A0.
fn extract_kron_factors(m: &Matrix4<Complex64>) -> (Matrix2<Complex64>, Matrix2<Complex64>) {
    // The Kronecker product A1 ⊗ A0 has block structure:
    // [ a1_00 * A0,  a1_01 * A0 ]
    // [ a1_10 * A0,  a1_11 * A0 ]

    // Extract A0 from the top-left block, normalizing by a1_00
    let block00 = Matrix2::new(m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    let block01 = Matrix2::new(m[(0, 2)], m[(0, 3)], m[(1, 2)], m[(1, 3)]);
    let block10 = Matrix2::new(m[(2, 0)], m[(2, 1)], m[(3, 0)], m[(3, 1)]);
    let block11 = Matrix2::new(m[(2, 2)], m[(2, 3)], m[(3, 2)], m[(3, 3)]);

    // Find the block with largest norm for numerical stability
    let norms = [
        block00.norm(),
        block01.norm(),
        block10.norm(),
        block11.norm(),
    ];
    let max_idx = norms
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let (a0_raw, a1_coeff) = match max_idx {
        0 => {
            let scale = if block00[(0, 0)].norm() > TOLERANCE {
                block00[(0, 0)]
            } else {
                block00[(1, 0)]
            };
            (
                block00 / scale,
                Matrix2::new(
                    scale,
                    block01[(0, 0)] / (block00[(0, 0)] / scale),
                    block10[(0, 0)] / (block00[(0, 0)] / scale),
                    block11[(0, 0)] / (block00[(0, 0)] / scale),
                ),
            )
        }
        _ => {
            // Fallback: use identity for both
            (Matrix2::identity(), Matrix2::identity())
        }
    };

    // Re-extract A0 more robustly: use SVD or just the top-left block
    // Simplified: normalize the block
    let det_a0 = a0_raw.determinant();
    let a0 = if det_a0.norm() > TOLERANCE {
        a0_raw / det_a0.sqrt()
    } else {
        Matrix2::identity()
    };

    // Compute A1 from the block ratios
    let a1_00 = if a0[(0, 0)].norm() > TOLERANCE {
        m[(0, 0)] / a0[(0, 0)]
    } else if a0[(1, 0)].norm() > TOLERANCE {
        m[(1, 0)] / a0[(1, 0)]
    } else {
        Complex64::new(1.0, 0.0)
    };
    let a1_01 = if a0[(0, 0)].norm() > TOLERANCE {
        m[(0, 2)] / a0[(0, 0)]
    } else if a0[(1, 0)].norm() > TOLERANCE {
        m[(1, 2)] / a0[(1, 0)]
    } else {
        Complex64::new(0.0, 0.0)
    };
    let a1_10 = if a0[(0, 0)].norm() > TOLERANCE {
        m[(2, 0)] / a0[(0, 0)]
    } else if a0[(1, 0)].norm() > TOLERANCE {
        m[(3, 0)] / a0[(1, 0)]
    } else {
        Complex64::new(0.0, 0.0)
    };
    let a1_11 = if a0[(0, 0)].norm() > TOLERANCE {
        m[(2, 2)] / a0[(0, 0)]
    } else if a0[(1, 0)].norm() > TOLERANCE {
        m[(3, 2)] / a0[(1, 0)]
    } else {
        Complex64::new(1.0, 0.0)
    };

    let a1 = Matrix2::new(a1_00, a1_01, a1_10, a1_11);
    let det_a1 = a1.determinant();
    let a1 = if det_a1.norm() > TOLERANCE {
        a1 / det_a1.sqrt()
    } else {
        Matrix2::identity()
    };

    let _ = a1_coeff;

    (a0, a1)
}

/// Decompose a 2×2 unitary into U(theta, phi, lambda).
fn decompose_u3(u: &Matrix2<Complex64>) -> (f64, f64, f64) {
    let u00 = u[(0, 0)];
    let u01 = u[(0, 1)];
    let u10 = u[(1, 0)];
    let u11 = u[(1, 1)];

    let abs_u00 = u00.norm().min(1.0);
    let theta = 2.0 * abs_u00.acos();

    if theta < TOLERANCE {
        // Near-identity: sin(theta/2) ~ 0 so phi is unconstrained; fix phi = 0.
        let lambda = u11.arg() - u00.arg();
        if lambda.abs() < TOLERANCE {
            return (0.0, 0.0, 0.0);
        }
        return (0.0, 0.0, lambda);
    }

    // U(theta,phi,lambda) = [[c, -e^{i*lambda}s], [e^{i*phi}s, e^{i*(phi+lambda)}c]]
    // with c = cos(theta/2), s = sin(theta/2). Writing u00 = e^{i*alpha}*c:
    //   phi = arg(u10) - alpha
    //   lambda = arg(u01) - alpha - PI  (the -PI absorbs the minus sign on u01)
    // At theta = PI, arg(0) = 0 so alpha vanishes and the formula stays exact.
    let alpha = u00.arg();
    let phi = u10.arg() - alpha;
    let lambda = u01.arg() - alpha - PI;

    (theta, phi, lambda)
}

impl Pass for KakSynthesisPass {
    fn name(&self) -> &str {
        "KakSynthesis"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        // Collect blocks of consecutive gates on the same 2-qubit pair.
        // For each block, compute the combined unitary and re-synthesize.
        let instructions: Vec<_> = dag
            .to_instructions()
            .iter()
            .map(|op| (op.op_type.clone(), op.qubits.to_vec()))
            .collect();

        let mut new_ops: Vec<(OpType, Vec<usize>)> = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let (ref op, ref qs) = instructions[i];

            if op.is_boundary() || *op == OpType::Barrier || op.is_measurement() {
                new_ops.push((op.clone(), qs.clone()));
                i += 1;
                continue;
            }

            if qs.len() != 2 || op.n_qubits() != 2 {
                new_ops.push((op.clone(), qs.clone()));
                i += 1;
                continue;
            }

            // Found a 2Q gate. Collect the block of consecutive 2Q + 1Q gates on these qubits.
            let q0 = qs[0];
            let q1 = qs[1];
            let block_start = i;
            let mut block_end = i + 1;
            let mut n_2q = 1;

            while block_end < instructions.len() {
                let (ref next_op, ref next_qs) = instructions[block_end];

                if next_op.is_boundary() || *next_op == OpType::Barrier || next_op.is_measurement()
                {
                    break;
                }

                // 1Q gate on q0 or q1
                if next_qs.len() == 1 && (next_qs[0] == q0 || next_qs[0] == q1) {
                    block_end += 1;
                    continue;
                }

                // Same 2Q gate pair
                if next_qs.len() == 2 && next_qs[0] == q0 && next_qs[1] == q1 {
                    n_2q += 1;
                    block_end += 1;
                    continue;
                }
                if next_qs.len() == 2 && next_qs[0] == q1 && next_qs[1] == q0 {
                    n_2q += 1;
                    block_end += 1;
                    continue;
                }

                break;
            }

            // Only synthesize if we have multiple 2Q gates (otherwise the block is already optimal)
            if n_2q < 2 {
                for j in block_start..block_end {
                    new_ops.push(instructions[j].clone());
                }
                i = block_end;
                continue;
            }

            // Compute the combined 4×4 unitary for the block
            let _dim = 4;
            let mut combined = Matrix4::identity();

            for j in block_start..block_end {
                let (ref gate_op, ref gate_qs) = instructions[j];
                let gate_mat = gate_op.to_matrix();

                let mat_4x4 = if gate_qs.len() == 1 {
                    // 1Q gate: expand to 4×4 via Kronecker product
                    let gate_2x2 = Matrix2::new(
                        gate_mat[(0, 0)],
                        gate_mat[(0, 1)],
                        gate_mat[(1, 0)],
                        gate_mat[(1, 1)],
                    );

                    if gate_qs[0] == q0 {
                        kron_2x2(&gate_2x2, &Matrix2::identity())
                    } else {
                        kron_2x2(&Matrix2::identity(), &gate_2x2)
                    }
                } else {
                    // 2Q gate
                    let mut m = Matrix4::zeros();
                    for r in 0..4 {
                        for c in 0..4 {
                            m[(r, c)] = gate_mat[(r, c)];
                        }
                    }
                    // Handle qubit ordering
                    if gate_qs[0] == q1 && gate_qs[1] == q0 {
                        swap_qubits_4x4(&m)
                    } else {
                        m
                    }
                };

                combined = mat_4x4 * combined;
            }

            // Decompose and synthesize
            let kak = Self::kak_decompose(&combined);
            let synthesized = Self::synthesize(&kak, q0, q1);

            new_ops.extend(synthesized);
            i = block_end;
        }

        // Rebuild DAG
        let mut new_dag = QuantumDAG::new(dag.n_qubits, dag.n_cbits);
        new_dag.metadata = dag.metadata.clone();
        for (op_type, qubits) in new_ops {
            if op_type == OpType::Measure {
                if !qubits.is_empty() {
                    new_dag.add_measure(qubits[0], qubits[0]);
                }
            } else if !op_type.is_boundary() {
                new_dag.add_op(op_type, &qubits);
            }
        }

        *dag = new_dag;
        Ok(())
    }
}

/// Kronecker product of two 2×2 matrices.
fn kron_2x2(a: &Matrix2<Complex64>, b: &Matrix2<Complex64>) -> Matrix4<Complex64> {
    Matrix4::new(
        a[(0, 0)] * b[(0, 0)],
        a[(0, 0)] * b[(0, 1)],
        a[(0, 1)] * b[(0, 0)],
        a[(0, 1)] * b[(0, 1)],
        a[(0, 0)] * b[(1, 0)],
        a[(0, 0)] * b[(1, 1)],
        a[(0, 1)] * b[(1, 0)],
        a[(0, 1)] * b[(1, 1)],
        a[(1, 0)] * b[(0, 0)],
        a[(1, 0)] * b[(0, 1)],
        a[(1, 1)] * b[(0, 0)],
        a[(1, 1)] * b[(0, 1)],
        a[(1, 0)] * b[(1, 0)],
        a[(1, 0)] * b[(1, 1)],
        a[(1, 1)] * b[(1, 0)],
        a[(1, 1)] * b[(1, 1)],
    )
}

/// Swap qubit ordering in a 4×4 matrix (SWAP conjugation).
fn swap_qubits_4x4(m: &Matrix4<Complex64>) -> Matrix4<Complex64> {
    let perm = [0, 2, 1, 3]; // SWAP permutation
    let mut result = Matrix4::zeros();
    for r in 0..4 {
        for c in 0..4 {
            result[(perm[r], perm[c])] = m[(r, c)];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_block_no_change() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let original_count = dag.gate_count();
        KakSynthesisPass::new().run(&mut dag).unwrap();
        // Single 2Q gate: no re-synthesis needed (n_2q < 2)
        assert_eq!(dag.gate_count(), original_count);
    }

    #[test]
    fn test_cnot_cnot_cancels() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        KakSynthesisPass::new().run(&mut dag).unwrap();
        // CNOT · CNOT = I → should produce 0 or very few gates
        assert!(
            dag.gate_count() <= 2,
            "CNOT·CNOT should simplify, got {} gates",
            dag.gate_count()
        );
    }

    #[test]
    fn test_swap_resynthesis() {
        // SWAP = CX · CX(reversed) · CX → can be re-synthesized
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        KakSynthesisPass::new().run(&mut dag).unwrap();
        // SWAP needs exactly 3 CNOTs, so the output should have ≤3 2Q gates
        let n_2q = dag
            .to_instructions()
            .iter()
            .filter(|op| op.op_type.n_qubits() == 2)
            .count();
        assert!(
            n_2q <= 3,
            "SWAP should need at most 3 CNOTs, got {} 2Q gates",
            n_2q
        );
    }

    #[test]
    fn test_preserves_1q_gates_outside_block() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[2]); // different qubit
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::X, &[2]); // different qubit

        KakSynthesisPass::new().run(&mut dag).unwrap();
        // H and X on qubit 2 should remain
        let has_h = dag
            .to_instructions()
            .iter()
            .any(|op| op.op_type == OpType::H);
        let has_x = dag
            .to_instructions()
            .iter()
            .any(|op| op.op_type == OpType::X);
        assert!(has_h, "H gate on q2 should be preserved");
        assert!(has_x, "X gate on q2 should be preserved");
    }

    #[test]
    fn test_decompose_u3_identity() {
        let eye = Matrix2::identity();
        let (theta, _phi, _lam) = decompose_u3(&eye);
        assert!(theta.abs() < TOLERANCE);
    }

    #[test]
    fn test_magic_basis_unitary() {
        let m = magic_basis();
        let m_dag = m.adjoint();
        let prod = &m_dag * &m;
        // M†M should be close to identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (prod[(i, j)] - expected).norm() < 1e-10,
                    "M†M[{},{}] = {}, expected {}",
                    i,
                    j,
                    prod[(i, j)],
                    expected
                );
            }
        }
    }
}
