//! Density-matrix lane: the shapes a pure state cannot hold, held exactly.
//!
//! The statevector and tableau lanes refuse a `Reset` after use and only ever
//! *drop* a readout, because neither can represent a mixed state.  A density
//! matrix can, and it can do so exactly rather than by approximation:
//!
//! * **reset** is the channel `rho -> Tr_q(rho) (x) |0><0|`.  Kraus form:
//!   `K0 = |0><0|`, `K1 = |0><1|` -- the `|1>` branch is re-prepared in `|0>`.
//! * **measurement** with no later conditioned gate is the non-selective
//!   channel `rho -> P0 rho P0 + P1 rho P1` (`K0 = |0><0|`, `K1 = |1><1|`).
//!   That is dephasing, which is exactly what "we looked and did not tell you
//!   the answer" does to the state.
//!
//! What stays refused is feed-forward.  A gate conditioned on a classical bit
//! makes the state a mixture *over branches*, so a single rho is not enough:
//! that needs trajectory sampling (`simulate_dynamic`), not this lane.
//!
//! Scope note: this lane exists for correctness, not for speed.  The earlier
//! feasibility probe measured a statistical tie with the routed `aer` path, so
//! per keep-only-if-win it was left out of the first milestone.  What it buys
//! is the one row the matrix could not answer: `reset` after use.

use std::collections::HashMap;

use nalgebra::DMatrix;
use num_complex::Complex64;
use sf_ir::dag::QuantumDAG;
use sf_ir::dm::DensityMatrixState;
use sf_ir::OpType;

/// One gate/readout compiled for the density-matrix lane.
struct DmOp {
    op_type: OpType,
    qubits: Vec<usize>,
}

/// Compile-once snapshot for [`crate::plan::Mode::Dm`].
pub struct DmProgram {
    pub n_qubits: usize,
    pub parameter_names: Vec<String>,
    operations: Vec<DmOp>,
    /// Pre-summed channel superoperators for `Reset` and `Measure`.  Both are
    /// 4x4 on the closed (ket, bra) blocks of one qubit, so applying them is
    /// one sweep per qubit with no per-call allocation.
    reset_superop: [[Complex64; 4]; 4],
    dephase_superop: [[Complex64; 4]; 4],
}

impl DmProgram {
    pub fn new(dag: &QuantumDAG) -> Result<Self, String> {
        // rho holds 4^n complex amplitudes; refuse before the allocation
        // rather than after the process has already died.
        if dag.n_qubits >= usize::BITS as usize / 2 {
            return Err("cat: density-matrix plan: qubit count overflows the rho buffer".into());
        }
        let mut operations = Vec::new();
        for op in dag.to_instructions() {
            if op.op_type.is_boundary() || matches!(op.op_type, OpType::Barrier | OpType::Id) {
                continue;
            }
            if let Some((cbit, value)) = op.condition {
                return Err(format!(
                    "cat: density-matrix plan: {:?} on qubits {:?} is conditioned on \
                     classical bit {} == {}; a conditioned gate makes the state a \
                     mixture over branches, not one rho - sample the trajectories \
                     with QuantumDAG.simulate_dynamic(shots, seed) in Rust, or \
                     sf.run(circuit, method=\"statevector\", shots=N) from Python",
                    op.op_type,
                    op.qubits.as_slice(),
                    cbit,
                    value
                ));
            }
            let qubits: Vec<usize> = if matches!(op.op_type, OpType::MeasureAll) {
                (0..dag.n_qubits).collect()
            } else {
                op.qubits.iter().copied().collect()
            };
            match &op.op_type {
                // Channels, handled per qubit in `simulate`.
                OpType::Reset | OpType::Measure | OpType::MeasureAll => {}
                OpType::Custom(name, _) => {
                    return Err(format!(
                        "cat: density-matrix plan: custom gate '{name}' carries no matrix, \
                         so there is nothing to apply to rho"
                    ));
                }
                other => {
                    // Any arity is allowed: one- and two-qubit gates take the
                    // shared fused kernel, and a wider gate is applied
                    // axis-by-axis in `apply_wide_unitary`.  That path is
                    // required, not optional -- `apply_unitary` panics above
                    // two qubits.
                    let _ = other;
                    if qubits.is_empty() {
                        continue;
                    }
                }
            }
            operations.push(DmOp {
                op_type: op.op_type.clone(),
                qubits,
            });
        }
        Ok(Self {
            n_qubits: dag.n_qubits,
            parameter_names: dag.parameter_names(),
            operations,
            reset_superop: superop_from_kraus(&reset_kraus()),
            dephase_superop: superop_from_kraus(&dephase_kraus()),
        })
    }

    /// Evolve `|0..0><0..0|` through the program and return the flat rho in
    /// the `data[ket | (bra << n)]` layout the binding wraps.
    pub fn simulate(&self, values: &HashMap<String, f64>) -> Result<Vec<Complex64>, String> {
        // Same contract as `PreparedStatevector::simulate`: every declared
        // parameter must be given and finite, and nothing else may be given.
        for name in &self.parameter_names {
            match values.get(name) {
                Some(value) if value.is_finite() => {}
                Some(_) => return Err(format!("parameter '{name}' must be finite")),
                None => return Err(format!("missing parameter '{name}'")),
            }
        }
        if let Some(name) = values
            .keys()
            .find(|name| !self.parameter_names.contains(name))
        {
            return Err(format!("unknown parameter '{name}'"));
        }

        let mut rho = DensityMatrixState::new(self.n_qubits);
        for template in &self.operations {
            let op_type = template.op_type.bind_params(values);
            match &op_type {
                OpType::Reset => {
                    for &q in &template.qubits {
                        rho.apply_1q_superop(q, &self.reset_superop);
                    }
                }
                OpType::Measure | OpType::MeasureAll => {
                    for &q in &template.qubits {
                        rho.apply_1q_superop(q, &self.dephase_superop);
                    }
                }
                other => {
                    let matrix = other.to_matrix();
                    if template.qubits.len() <= 2 {
                        rho.apply_unitary(&matrix, &template.qubits);
                    } else {
                        // Wider than the shared kernel carries: transform the
                        // ket axis by `u` and the bra axis by `u^dagger`.
                        let n = self.n_qubits;
                        let mut buffer = rho.into_data();
                        apply_wide_unitary(&mut buffer, n, &template.qubits, &matrix);
                        rho = DensityMatrixState::from_data(buffer, n);
                    }
                }
            }
        }
        Ok(rho.into_data())
    }
}

/// Kraus set of the reset channel, `{ |0><0|, |0><1| }`.
fn reset_kraus() -> [[[Complex64; 2]; 2]; 2] {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [
        [[o, z], [z, z]], // K0 = |0><0|
        [[z, o], [z, z]], // K1 = |0><1|: the |1> branch lands back in |0>
    ]
}

/// Kraus set of the non-selective measurement channel, `{ |0><0|, |1><1| }`.
fn dephase_kraus() -> [[[Complex64; 2]; 2]; 2] {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [
        [[o, z], [z, z]], // K0 = |0><0|
        [[z, z], [z, o]], // K1 = |1><1|
    ]
}

/// `M[(r, c)][(r', c')] = sum_k K_k[r, r'] * conj(K_k[c, c'])`, the same
/// composition `DensityMatrixState::kraus_superop_1q` performs.
fn superop_from_kraus(kraus: &[[[Complex64; 2]; 2]; 2]) -> [[Complex64; 4]; 4] {
    let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
    for k in kraus {
        for r in 0..2 {
            for rp in 0..2 {
                let krrp = k[r][rp];
                for c in 0..2 {
                    for cp in 0..2 {
                        m[r * 2 + c][rp * 2 + cp] += krrp * k[c][cp].conj();
                    }
                }
            }
        }
    }
    m
}

/// Apply a `2^k x 2^k` unitary to `k` target qubits of the flat rho.
///
/// `DensityMatrixState::apply_unitary` carries only one- and two-qubit gates
/// and panics above that, so a wider gate -- Toffoli, CSWAP, an opaque
/// three-qubit unitary -- is applied here: `u` on the ket axis, `u^dagger` on
/// the bra axis.  Exact, just a worse constant than the fused kernels.
fn apply_wide_unitary(rho: &mut [Complex64], n: usize, qubits: &[usize], u: &DMatrix<Complex64>) {
    let dim = 1usize << n;
    apply_on_axis(rho, dim, qubits, u, true);
    apply_on_axis(rho, dim, qubits, u, false);
}

/// One side of `rho -> U rho U^dagger`: transform the ket axis by `u` when
/// `ket_axis`, otherwise the bra axis by `u^dagger`.
fn apply_on_axis(
    rho: &mut [Complex64],
    dim: usize,
    qubits: &[usize],
    u: &DMatrix<Complex64>,
    ket_axis: bool,
) {
    let k = qubits.len();
    let block = 1usize << k;
    let stride = if ket_axis { 1 } else { dim };
    let mut mask = 0usize;
    for &q in qubits {
        mask |= 1usize << q;
    }
    // Every index whose target bits are clear owns one group: the positions
    // that differ from it only on the target qubits.  `to_matrix()` writes the
    // local index with `qubits[0]` as the most significant bit (CCX is built
    // in basis `|q0 q1 q2>` and swaps 6 <-> 7), so target `j` owns bit
    // `k - 1 - j` of the local index.
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(dim / block);
    for base in 0..dim {
        if base & mask != 0 {
            continue;
        }
        let mut positions = Vec::with_capacity(block);
        for t in 0..block {
            let mut pos = base;
            for (j, &q) in qubits.iter().enumerate() {
                if (t >> (k - 1 - j)) & 1 == 1 {
                    pos |= 1usize << q;
                }
            }
            positions.push(pos);
        }
        groups.push(positions);
    }
    let mut gathered = vec![Complex64::new(0.0, 0.0); block];
    let mut applied = vec![Complex64::new(0.0, 0.0); block];
    for outer in 0..dim {
        let offset = if ket_axis { outer * dim } else { outer };
        for positions in &groups {
            for (t, &pos) in positions.iter().enumerate() {
                gathered[t] = rho[offset + pos * stride];
            }
            for r in 0..block {
                let mut acc = Complex64::new(0.0, 0.0);
                for (c, &v) in gathered.iter().enumerate() {
                    let m = u[(r, c)];
                    acc += if ket_axis { m * v } else { m.conj() * v };
                }
                applied[r] = acc;
            }
            for (t, &pos) in positions.iter().enumerate() {
                rho[offset + pos * stride] = applied[t];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell(with_reset: bool) -> QuantumDAG {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        if with_reset {
            dag.add_op(OpType::Reset, &[0]);
        }
        dag
    }

    fn probs(rho: &[Complex64], n: usize) -> Vec<f64> {
        let dim = 1 << n;
        (0..dim).map(|i| at(rho, i, i, n).re).collect()
    }

    /// `rho(ket, bra)` in the engine layout `data[ket | (bra << n)]`.  Named
    /// so the tests read as matrix entries rather than as index arithmetic.
    fn at(rho: &[Complex64], ket: usize, bra: usize, n: usize) -> Complex64 {
        rho[ket | (bra << n)]
    }

    /// `|00> + |11>` reset on qubit 0 is the *mixed* state
    /// `0.5|00><00| + 0.5|10><10|` (label `|q1 q0>`) -- q0 is definitely 0,
    /// q1 is still uniform.  No pure vector can hold this; that is why the
    /// other lanes refuse it.
    #[test]
    fn reset_after_use_is_the_exact_mixed_state() {
        let program = DmProgram::new(&bell(true)).expect("reset after use is a channel");
        let rho = program.simulate(&HashMap::new()).unwrap();
        let p = probs(&rho, 2);
        // Basis index is little-endian: bit `q` of the index is qubit `q`, so
        // the two surviving branches are index 0 (|00>) and index 2 (|10>).
        assert!((p[0] - 0.5).abs() < 1e-12, "{p:?}");
        assert!((p[2] - 0.5).abs() < 1e-12, "{p:?}");
        assert!(p[1].abs() < 1e-12 && p[3].abs() < 1e-12, "{p:?}");
        // Purity 0.5 < 1 proves the lane really produced a mixed state.
        let dim = 4;
        let mut purity = 0.0;
        for ket in 0..dim {
            for bra in 0..dim {
                purity += (at(&rho, ket, bra, 2) * at(&rho, bra, ket, 2)).re;
            }
        }
        assert!((purity - 0.5).abs() < 1e-12, "purity {purity}");
    }

    /// The identical circuit *without* the reset keeps its coherence, so the
    /// difference between the two rho values is the reset channel itself.
    #[test]
    fn without_the_reset_the_state_stays_coherent() {
        let program = DmProgram::new(&bell(false)).unwrap();
        let rho = program.simulate(&HashMap::new()).unwrap();
        let coherence = at(&rho, 0, 3, 2).norm();
        assert!((coherence - 0.5).abs() < 1e-12, "{coherence}");
    }

    /// `x(0)` then `reset(0)` is `|00>` with certainty: the decidable case the
    /// pure lanes over-refuse, answered exactly here.
    #[test]
    fn reset_of_a_basis_state_is_deterministic() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::X, &[0]);
        dag.add_op(OpType::Reset, &[0]);
        let rho = DmProgram::new(&dag)
            .unwrap()
            .simulate(&HashMap::new())
            .unwrap();
        let p = probs(&rho, 2);
        assert!((p[0] - 1.0).abs() < 1e-12, "{p:?}");
    }

    /// Toffoli is wider than the shared one-/two-qubit kernel, so it takes the
    /// axis-by-axis path: a pure input must come out pure and match the
    /// statevector lane entry for entry.
    #[test]
    fn a_three_qubit_gate_matches_the_pure_lane() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(OpType::CCX, &[0, 1, 2]);
        let psi = crate::plan::CatPlan::new(&dag, crate::plan::Mode::Sv)
            .expect("a plain unitary is legal in the sv lane")
            .execute_statevector(&HashMap::new())
            .unwrap();
        let rho = DmProgram::new(&dag)
            .unwrap()
            .simulate(&HashMap::new())
            .unwrap();
        for ket in 0..8 {
            for bra in 0..8 {
                let want = psi[ket] * psi[bra].conj();
                let got = at(&rho, ket, bra, 3);
                assert!(
                    (got - want).norm() < 1e-12,
                    "rho({ket},{bra}) = {got}, want {want}"
                );
            }
        }
    }

    /// A readout with no later conditioned gate is dephasing: the populations
    /// survive, the coherence does not.
    #[test]
    fn a_readout_dephases_without_changing_populations() {
        let mut dag = QuantumDAG::new(1, 1);
        dag.add_op(OpType::H, &[0]);
        dag.add_measure(0, 0);
        let rho = DmProgram::new(&dag)
            .unwrap()
            .simulate(&HashMap::new())
            .unwrap();
        assert!((at(&rho, 0, 0, 1).re - 0.5).abs() < 1e-12);
        assert!((at(&rho, 1, 1, 1).re - 0.5).abs() < 1e-12);
        assert!(at(&rho, 0, 1, 1).norm() < 1e-12, "coherence must be gone");
    }

    /// Feed-forward needs branches, so it is refused with a route out.
    #[test]
    fn feed_forward_is_refused() {
        let mut dag = QuantumDAG::new(2, 2);
        dag.add_op(OpType::H, &[0]);
        dag.add_measure(0, 0);
        dag.add_op_conditioned(OpType::X, &[1], 0, 1);
        let msg = match DmProgram::new(&dag) {
            Ok(_) => panic!("feed-forward cannot be one rho"),
            Err(msg) => msg,
        };
        assert!(msg.contains("conditioned"), "{msg}");
        assert!(msg.contains("simulate_dynamic"), "{msg}");
    }
}
