//! The compile-once [`CatPlan`] handle and its execution [`Mode`].
//!
//! A `CatPlan` is built once from a `&QuantumDAG` and reused many times:
//! statevector snapshots, the compiled adjoint program, and the Pauli-frame
//! stabilizer/Clifford tableau all hang off the same handle, so the public
//! surface (`sf.cat(...)`) never changes across milestones.  Density-matrix
//! mode is the one deliberate hold-out — the feasibility probe measured a
//! statistical tie with the routed aer path, so per keep-only-if-win it
//! stays an explicit, non-silent error.

use std::collections::HashMap;

use num_complex::Complex64;
use sf_ir::adjoint::PauliTerm;
use sf_ir::dag::{PreparedStatevector, QuantumDAG};
use sf_ir::OpType;

use crate::adj::AdjProgram;
use crate::paulframe::FrameTableau;

/// Execution regime a [`CatPlan`] is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Sv,
    Dm,
    Adj,
    Stab,
    Cliff,
}

/// Compile-once handle.  The cached snapshot owns everything that does not
/// depend on parameter values; only the cheap per-call bind is repeated.
pub struct CatPlan {
    n_qubits: usize,
    parameter_names: Vec<String>,
    mode: Mode,
    // Statevector snapshot, compiled adjoint program, and the Pauli-frame
    // tableau shared by the stabilizer and Clifford modes.  DM carries no
    // structure yet; building it is an explicit, non-silent error.
    sv: Option<PreparedStatevector>,
    adj: Option<AdjProgram>,
    stab: Option<FrameTableau>,
}

impl CatPlan {
    /// Lower `dag` into a reusable plan for `mode`.
    pub fn new(dag: &QuantumDAG, mode: Mode) -> Result<Self, String> {
        match mode {
            Mode::Sv => {
                let prepared = PreparedStatevector::new(dag)?;
                Ok(Self {
                    n_qubits: prepared.n_qubits,
                    parameter_names: prepared.parameter_names.clone(),
                    mode,
                    sv: Some(prepared),
                    adj: None,
                    stab: None,
                })
            }
            Mode::Dm => Err("cat: density-matrix mode not yet implemented".to_string()),
            Mode::Adj => {
                let adj = AdjProgram::from_dag(dag)?;
                Ok(Self {
                    n_qubits: dag.n_qubits,
                    parameter_names: dag.parameter_names(),
                    mode,
                    sv: None,
                    adj: Some(adj),
                    stab: None,
                })
            }
            Mode::Stab | Mode::Cliff => {
                // Both modes compile to the same Pauli-frame tableau: the
                // Clifford primitives are exactly what `from_gate_list`
                // accepts, so anything else (T gates, constant rotations, a
                // variable `ry`) is an explicit error at build time, never a
                // wrong sample.  Constant-angle rotations are not
                // pattern-matched onto the Clifford set; use SX/SXdg for
                // pi/2 X-rotations.
                let mut gates: Vec<(String, Vec<usize>)> = Vec::new();
                for op in dag.to_instructions() {
                    if op.op_type.is_boundary() {
                        continue;
                    }
                    let name = match &op.op_type {
                        OpType::H => "H",
                        OpType::S => "S",
                        OpType::Sdg => "SDG",
                        OpType::SX => "SX",
                        OpType::SXdg => "SXDG",
                        OpType::X => "X",
                        OpType::Y => "Y",
                        OpType::Z => "Z",
                        OpType::CNOT => "CX",
                        OpType::CZ => "CZ",
                        OpType::CY => "CY",
                        OpType::SWAP => "SWAP",
                        OpType::Id | OpType::Measure | OpType::Reset | OpType::Barrier => continue,
                        other => {
                            return Err(format!(
                                "cat: stabilizer/clifford plan: unsupported gate {other:?} \
                                 (Clifford primitives only)"
                            ))
                        }
                    };
                    gates.push((name.to_string(), op.qubits.to_vec()));
                }
                let tab = FrameTableau::from_gate_list(dag.n_qubits, &gates)?;
                Ok(Self {
                    n_qubits: dag.n_qubits,
                    parameter_names: dag.parameter_names(),
                    mode,
                    sv: None,
                    adj: None,
                    stab: Some(tab),
                })
            }
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Execute the statevector plan with freshly-bound `values`.
    ///
    /// Reuses the vetted `PreparedStatevector` kernel path, so results are
    /// bit-for-bit identical to `dag.bind(values).simulate()`.
    pub fn execute_statevector(
        &self,
        values: &HashMap<String, f64>,
    ) -> Result<Vec<Complex64>, String> {
        let sv = self
            .sv
            .as_ref()
            .ok_or_else(|| "cat: plan is not a statevector plan".to_string())?;
        sv.simulate(values)
    }

    /// Execute the adjoint plan: gradient of ⟨O⟩ for `terms` at `theta`
    /// (aligned with [`parameter_names`](Self::parameter_names)), reusing the
    /// compiled program's resident buffers across calls.
    pub fn execute_gradient(
        &mut self,
        theta: &[f64],
        terms: &[PauliTerm],
    ) -> Result<Vec<f64>, String> {
        let adj = self
            .adj
            .as_mut()
            .ok_or_else(|| "cat: plan is not an adjoint plan".to_string())?;
        adj.grad(theta, terms).map(|g| g.to_vec())
    }

    /// Execute the stabilizer/Clifford plan: `shots` bulk samples from the
    /// compiled Pauli-frame tableau.
    ///
    /// Keys are little-endian bitstrings — character `n-1-q` is qubit `q` —
    /// matching every other SF sampler (`sf.run(..., method="stabilizer")`
    /// and the sf-ir tableau).
    pub fn execute_sample(
        &self,
        shots: usize,
        seed: Option<u64>,
    ) -> Result<HashMap<String, usize>, String> {
        let tab = self
            .stab
            .as_ref()
            .ok_or_else(|| "cat: plan is not a stabilizer plan".to_string())?;
        Ok(tab.sample(shots, seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::{OpType, Parameter, QuantumDAG};

    fn var(name: &str) -> Parameter {
        Parameter::Variable {
            name: name.into(),
            id: 0,
        }
    }

    #[test]
    fn cat_sv_handle_is_bit_exact_vs_direct_simulate() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Ry(var("x")), &[1]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::Rz(var("x")), &[2]);
        dag.add_op(OpType::S, &[1]);

        let plan = CatPlan::new(&dag, Mode::Sv).unwrap();
        assert_eq!(plan.mode(), Mode::Sv);
        assert_eq!(plan.n_qubits(), 3);

        for x in [0.3, -1.1, 2.0, 0.0] {
            let values = HashMap::from([("x".into(), x)]);
            let got = plan.execute_statevector(&values).unwrap();
            let want = dag.bind(&values).simulate();
            assert_eq!(got.len(), want.len());
            for (a, b) in got.iter().zip(want.iter()) {
                assert!(
                    (a - b).norm() < 1e-12,
                    "statevector mismatch at handle reuse"
                );
            }
        }
    }

    #[test]
    fn cat_sv_handle_rejects_unsupported_circuits() {
        // Measurement is not a unitary statevector plan: the shared snapshot
        // builder must reject it at compile time, not return a wrong state.
        let mut dag = QuantumDAG::new(1, 1);
        dag.add_op(OpType::Measure, &[0]);
        assert!(CatPlan::new(&dag, Mode::Sv).is_err());
    }

    fn cliff_op(name: &str) -> OpType {
        match name {
            "H" => OpType::H,
            "S" => OpType::S,
            "SDG" => OpType::Sdg,
            "X" => OpType::X,
            "Y" => OpType::Y,
            "Z" => OpType::Z,
            "CX" => OpType::CNOT,
            "CZ" => OpType::CZ,
            "SWAP" => OpType::SWAP,
            other => panic!("cliff_op: {other}"),
        }
    }

    fn ghz_dag(n: usize) -> QuantumDAG {
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::H, &[0]);
        for i in 1..n {
            dag.add_op(OpType::CNOT, &[0, i]);
        }
        dag
    }

    #[test]
    fn cat_stab_handle_deterministic_keys_are_le() {
        // X(0) must surface as the little-endian key "001" (qubit q is
        // character n-1-q), catching any qubit-order inversion.
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::X, &[0]);
        let plan = CatPlan::new(&dag, Mode::Stab).unwrap();
        let counts = plan.execute_sample(64, Some(5)).unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.get("001"), Some(&64));

        let mut dag2 = QuantumDAG::new(3, 0);
        dag2.add_op(OpType::X, &[2]);
        let plan2 = CatPlan::new(&dag2, Mode::Stab).unwrap();
        let counts2 = plan2.execute_sample(64, Some(5)).unwrap();
        assert_eq!(counts2.get("100"), Some(&64));
    }

    #[test]
    fn cat_stab_handle_random_key_sets_are_le() {
        // H(0) then CX(0,3): the two outcomes differ in qubits 0 and 3, which
        // are characters n-1-0=5 and n-1-3=2 — the key set pins both order
        // and the CX spread.
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 3]);
        let plan = CatPlan::new(&dag, Mode::Stab).unwrap();
        let counts = plan.execute_sample(5000, Some(11)).unwrap();
        let mut got: Vec<String> = counts.keys().cloned().collect();
        got.sort();
        assert_eq!(got, vec!["000000".to_string(), "001001".to_string()]);

        // GHZ-8: exactly two branches, the right keys, near-balanced.
        let plan = CatPlan::new(&ghz_dag(8), Mode::Stab).unwrap();
        let shots = 20_000usize;
        let counts = plan.execute_sample(shots, Some(7)).unwrap();
        assert_eq!(counts.len(), 2);
        let c0 = *counts.get(&"0".repeat(8)).unwrap_or(&0);
        let c1 = *counts.get(&"1".repeat(8)).unwrap_or(&0);
        assert_eq!(c0 + c1, shots);
        assert!(
            (c0 as f64 - shots as f64 / 2.0).abs() <= 5.0 * (shots as f64 / 4.0).sqrt(),
            "GHZ branch imbalance: {c0}/{c1}"
        );
    }

    #[test]
    fn cat_stab_handle_matches_sf_ir_tableau_distribution() {
        // Same brick shape as paulframe's test, but built through the DAG ->
        // plan adapter, so this pins the whole `sf.cat(c, "stabilizer")`
        // path (gate mapping, boundary skipping, qubit order) against the
        // vetted sf-ir tableau sampler.
        let n = 64usize;
        let mut gates: Vec<(String, Vec<usize>)> = Vec::new();
        let mut state = 23u64.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..3 {
            for q in 0..n {
                gates.push(("H".into(), vec![q]));
            }
            for q in (0..n - 1).step_by(2) {
                gates.push(("CX".into(), vec![q, q + 1]));
            }
            for q in 0..n {
                if next() & 1 == 1 {
                    gates.push(("S".into(), vec![q]));
                }
            }
            for q in (1..n - 1).step_by(2) {
                gates.push(("CX".into(), vec![q, q + 1]));
            }
        }

        let mut dag = QuantumDAG::new(n, 0);
        for (name, qubits) in &gates {
            dag.add_op(cliff_op(name), qubits);
        }
        let plan = CatPlan::new(&dag, Mode::Stab).unwrap();

        let shots = 30_000usize;
        let c_ours = plan.execute_sample(shots, Some(3)).unwrap();
        let theirs = sf_ir::stabilizer::StabilizerTableau::from_gate_list(n, &gates).unwrap();
        let c_theirs = theirs.sample(shots, Some(4));

        // Key index via little-endian parse (matches the LE key convention).
        let as_idx = |counts: &HashMap<String, usize>| -> HashMap<u64, f64> {
            counts
                .iter()
                .map(|(k, &v)| (u64::from_str_radix(k, 2).unwrap(), v as f64 / shots as f64))
                .collect()
        };
        let p_ours = as_idx(&c_ours);
        let p_theirs = as_idx(&c_theirs);
        let mut keys: Vec<u64> = p_ours.keys().chain(p_theirs.keys()).copied().collect();
        keys.sort_unstable();
        keys.dedup();
        let mut total = 0.0;
        for k in &keys {
            let a = p_ours.get(k).copied().unwrap_or(0.0);
            let b = p_theirs.get(k).copied().unwrap_or(0.0);
            total += (a - b).abs();
        }
        let k_eff = keys.len().max(1) as f64;
        let thr = 2.0 * 4.0 * (k_eff / (2.0 * std::f64::consts::PI * shots as f64)).sqrt();
        let dist = 0.5 * total;
        assert!(
            dist < thr,
            "plan-vs-sf-ir TVD {dist:.5} >= threshold {thr:.5}"
        );
    }

    #[test]
    fn cat_stab_handle_rejects_non_clifford_gates() {
        // A variable rotation must be a build-time error, not a wrong sample
        // (and must not evaluate an unbound parameter).
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Ry(var("x")), &[0]);
        let err = match CatPlan::new(&dag, Mode::Stab) {
            Ok(_) => panic!("variable rotation must be rejected at build time"),
            Err(e) => e,
        };
        assert!(err.contains("unsupported gate"), "err: {err}");

        // T is non-Clifford too.
        let mut dag2 = QuantumDAG::new(2, 0);
        dag2.add_op(OpType::T, &[0]);
        assert!(CatPlan::new(&dag2, Mode::Stab).is_err());
    }

    #[test]
    fn cat_cliff_handle_agrees_with_stab() {
        let plan = CatPlan::new(&ghz_dag(8), Mode::Cliff).unwrap();
        assert_eq!(plan.mode(), Mode::Cliff);
        let shots = 4096usize;
        let a = plan.execute_sample(shots, Some(17)).unwrap();
        assert_eq!(a.len(), 2, "GHZ collapses to two branches");
        assert_eq!(a.values().sum::<usize>(), shots);

        // Stab mode shares the compiled tableau: same seed, same counts.
        let stab = CatPlan::new(&ghz_dag(8), Mode::Stab).unwrap();
        let b = stab.execute_sample(shots, Some(17)).unwrap();
        assert_eq!(a, b, "same seed through Stab/Cliff must agree");
    }

    #[test]
    fn cat_adj_handle_matches_sf_ir_engine() {
        // Ladder ansatz at n=3, L=1 (P=6), observable Z^3.
        let mut dag = QuantumDAG::new(3, 0);
        let mut slot = 0usize;
        for i in 0..3 {
            dag.add_op(OpType::Ry(var(&format!("t{slot}"))), &[i]);
            slot += 1;
            dag.add_op(OpType::Rz(var(&format!("t{slot}"))), &[i]);
            slot += 1;
        }
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 2]);

        let mut plan = CatPlan::new(&dag, Mode::Adj).unwrap();
        assert_eq!(plan.mode(), Mode::Adj);
        assert_eq!(plan.n_qubits(), 3);
        assert_eq!(plan.parameter_names().len(), 6);

        let terms = vec![sf_ir::adjoint::PauliTerm {
            paulis: vec![3u8; 3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let theta: Vec<f64> = (0..6).map(|i| 0.2 + 0.1 * i as f64).collect();
        let values: HashMap<String, f64> = plan
            .parameter_names()
            .iter()
            .cloned()
            .zip(theta.iter().copied())
            .collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();
        let got = plan.execute_gradient(&theta, &terms).unwrap();
        assert_eq!(got.len(), want.gradients.len());
        for (a, b) in got.iter().zip(want.gradients.iter()) {
            assert!((a - b).abs() < 1e-11, "plan adj mismatch: {a} vs {b}");
        }
    }
}
