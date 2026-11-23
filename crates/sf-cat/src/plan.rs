//! The compile-once [`CatPlan`] handle and its execution [`Mode`].
//!
//! A `CatPlan` is built once from a `&QuantumDAG` and reused many times:
//! statevector snapshots, the compiled adjoint program, the Pauli-frame
//! stabilizer/Clifford tableau, the density-matrix channel program, and the
//! symbolic circuit the MPS lane re-evolves all hang off the same handle, so
//! the public surface (`sf.cat(...)`) never changes across milestones.
//!
//! Density-matrix mode was the last hold-out — the feasibility probe measured
//! a statistical tie with the routed aer path, so per keep-only-if-win it was
//! deferred behind an explicit, non-silent error.  It is kept now because it
//! answers shapes no pure lane can (`Reset` after use), not because it is
//! faster; see [`crate::dmlane`].

use std::collections::HashMap;

use num_complex::Complex64;
use sf_ir::adjoint::PauliTerm;
use sf_ir::dag::{PreparedStatevector, QuantumDAG};
use sf_ir::{MPSState, MethodError, OpType, QuantumStateImpl};

use crate::adj::AdjProgram;
use crate::dmlane::DmProgram;
use crate::paulframe::FrameTableau;

/// Reject circuits the Pauli-frame sampler cannot represent.
///
/// `Mode::Stab`/`Mode::Cliff` evolve one tableau with unitaries and sample it
/// once at the end: there is no collapse, no classical state, no branch.  A
/// skipped `Measure`, `Reset` or conditioned gate therefore used to yield the
/// *unconditional final-state distribution* — a plausible answer to a
/// different question.  This walk keeps every legal pattern (terminal
/// measurements, `measure_all()`, barriers, resets on qubits nothing has
/// touched yet) and rejects the three that change the semantics:
///
/// * any op carrying a classical condition (feed-forward),
/// * a gate acting on a qubit that was already measured,
/// * a `Reset` on a qubit a previous op already touched (reset is
///   collapse-and-reprepare, not a free `|0>`).
///
/// `lane` only labels the message (`"stabilizer/clifford"`, `"adjoint"`), so a
/// user reading the error sees the mode they asked for.
fn reject_dynamic_ops(dag: &QuantumDAG, lane: &str) -> Result<(), String> {
    let n = dag.n_qubits;
    let mut touched = vec![false; n];
    let mut measured = vec![false; n];
    // Name the limitation of *this* lane: an adjoint caller is not running a
    // Pauli-frame sampler, and telling them so would be a second wrong answer.
    let why = if lane == "adjoint" {
        "the adjoint program differentiates the unitary and carries no collapse"
    } else {
        "the Pauli-frame sampler has no collapse or branching"
    };
    for op in dag.to_instructions() {
        if op.op_type.is_boundary() || matches!(op.op_type, OpType::Barrier | OpType::Id) {
            continue;
        }
        if let Some((cbit, value)) = op.condition {
            return Err(format!(
                "cat: {lane} plan: {:?} on qubits {:?} is conditioned on classical \
                 bit {} == {}; {why}, so \
                 feed-forward needs the trajectory sampler - call \
                 sf.run(circuit, method=\"statevector\", shots=N) from Python, or \
                 QuantumDAG.simulate_dynamic(shots, seed) in Rust",
                op.op_type,
                op.qubits.as_slice(),
                cbit,
                value
            ));
        }
        match &op.op_type {
            OpType::Measure | OpType::MeasureAll => {
                if matches!(op.op_type, OpType::MeasureAll) {
                    measured.iter_mut().for_each(|m| *m = true);
                } else {
                    for &q in op.qubits.iter() {
                        measured[q] = true;
                    }
                }
            }
            OpType::Reset => {
                for &q in op.qubits.iter() {
                    if touched[q] || measured[q] {
                        return Err(format!(
                            "cat: {lane} plan: Reset on qubit {} after it was used; \
                             reset means collapse-and-reprepare, which this plan does not \
                             model - use sf.run(circuit, method=\"statevector\", shots=N) \
                             from Python, or QuantumDAG.simulate_dynamic(shots, seed) in Rust",
                            q
                        ));
                    }
                }
            }
            _ => {
                for &q in op.qubits.iter() {
                    if measured[q] {
                        return Err(format!(
                            "cat: {lane} plan: {:?} acts on qubit {} after it was \
                             measured; a mid-circuit measurement collapses the state, so this \
                             circuit needs trajectory sampling - sf.run(circuit, \
                             method=\"statevector\", shots=N) from Python, or \
                             QuantumDAG.simulate_dynamic(shots, seed) in Rust - not the \
                             final-state tableau",
                            op.op_type, q
                        ));
                    }
                    touched[q] = true;
                }
            }
        }
    }
    Ok(())
}

/// True when the circuit carries at least one classical readout.
fn has_measurements(dag: &QuantumDAG) -> bool {
    dag.to_instructions()
        .iter()
        .any(|op| op.op_type.is_measurement())
}

/// A readout may only be *ignored* by the statevector lane when nothing follows
/// it: the state returned is then the state before that trailing readout, which
/// is the standard convention.  A readout with a gate after it is a different
/// matter -- dropping it would return a state in which the measured qubit is
/// still coherent, so the answer would not be the circuit's state at all.  The
/// tableau lanes do not care, because they only ever return a distribution and
/// the marginal over the remaining qubits is unchanged; a state is not.
fn reject_non_trailing_readouts(dag: &QuantumDAG) -> Result<(), String> {
    let mut seen_readout = false;
    for op in dag.to_instructions() {
        if op.op_type.is_boundary() || matches!(op.op_type, OpType::Barrier) {
            continue;
        }
        if op.op_type.is_measurement() {
            seen_readout = true;
            continue;
        }
        if seen_readout {
            return Err(format!(
                "cat: statevector plan: {:?} on qubits {:?} runs after a measurement; \
                 the statevector lane can only drop a *trailing* readout, because the \
                 state it returns would otherwise differ from the collapsed one - \
                 move the readout to the end, or sample the circuit with \
                 sf.run(circuit, method=\"statevector\", shots=N)",
                op.op_type,
                op.qubits.as_slice()
            ));
        }
    }
    Ok(())
}

/// A view of `dag` with the classical readouts removed, in topological order so
/// parameter names keep their original sequence.
fn without_measurements(dag: &QuantumDAG) -> QuantumDAG {
    let mut out = QuantumDAG::new(dag.n_qubits, dag.n_cbits);
    for op in dag.to_instructions() {
        if op.op_type.is_measurement() {
            continue;
        }
        out.add_op(op.op_type.clone(), op.qubits.as_slice());
    }
    out
}

/// True when the circuit carries at least one reset.
fn has_resets(dag: &QuantumDAG) -> bool {
    dag.to_instructions()
        .iter()
        .any(|op| op.op_type == OpType::Reset)
}

/// A view of `dag` with the *leading* resets removed.
///
/// A reset on a qubit no previous operation touched is `|0> -> |0>`, so
/// removing it cannot change the state -- the same argument that lets the
/// adjoint lane drop it, and the same answer Qiskit's
/// `Statevector.from_instruction` gives.  A reset anywhere else is
/// collapse-and-reprepare, which yields a mixed state; this lane returns a
/// single pure vector and cannot hold one, so it stays an error with a route
/// out.
///
/// The `touched`/`measured` bookkeeping is re-derived here rather than borrowed
/// from `reject_dynamic_ops`: the guard and this filter have to be correct
/// independently, or a later relaxation of one would silently make the other
/// drop a real reset.
///
/// `lane` only names the caller in the error text: the statevector and MPS
/// plans share this filter but must not tell the user they were holding the
/// other handle.
fn strip_leading_resets(dag: &QuantumDAG, lane: &str) -> Result<QuantumDAG, String> {
    let nq = dag.n_qubits;
    let mut touched = vec![false; nq];
    let mut measured = vec![false; nq];
    let mut out = QuantumDAG::new(dag.n_qubits, dag.n_cbits);
    for op in dag.to_instructions() {
        if op.op_type == OpType::Reset {
            for &q in op.qubits.iter() {
                if q < nq && (touched[q] || measured[q]) {
                    return Err(format!(
                        "cat: {lane} plan: Reset on qubit {q} after it was used; \
                         reset means collapse-and-reprepare, which produces a mixed state \
                         and this lane returns a single pure vector - sample a trajectory \
                         with sf.run(circuit, method=\"{lane}\", shots=N) instead"
                    ));
                }
            }
            continue;
        }
        if op.op_type.is_measurement() {
            if matches!(op.op_type, OpType::MeasureAll) {
                measured.iter_mut().for_each(|m| *m = true);
            } else {
                for &q in op.qubits.iter() {
                    if q < nq {
                        measured[q] = true;
                    }
                }
            }
        } else if !(op.op_type.is_boundary() || matches!(op.op_type, OpType::Barrier | OpType::Id))
        {
            for &q in op.qubits.iter() {
                if q < nq {
                    touched[q] = true;
                }
            }
        }
        out.add_op(op.op_type.clone(), op.qubits.as_slice());
    }
    Ok(out)
}

/// Execution regime a [`CatPlan`] is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Sv,
    Dm,
    Adj,
    Stab,
    Cliff,
    Mps,
}

impl Mode {
    /// The name `sf.cat` accepts for this mode.
    ///
    /// Errors that name a mode have to use these strings: `Debug` prints
    /// `Dm`/`Stab`, which matches nothing a caller passed in.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Sv => "statevector",
            Mode::Dm => "density_matrix",
            Mode::Adj => "adjoint",
            Mode::Stab => "stabilizer",
            Mode::Cliff => "clifford",
            Mode::Mps => "mps",
        }
    }
}

/// Compile-once handle.  The cached snapshot owns everything that does not
/// depend on parameter values; only the cheap per-call bind is repeated.
pub struct CatPlan {
    n_qubits: usize,
    parameter_names: Vec<String>,
    mode: Mode,
    // Statevector snapshot, compiled adjoint program, the Pauli-frame tableau
    // shared by the stabilizer and Clifford modes, the density-matrix channel
    // program, and the symbolic circuit the MPS lane re-evolves.  Exactly one
    // is `Some`, selected by `mode`.
    sv: Option<PreparedStatevector>,
    adj: Option<AdjProgram>,
    stab: Option<FrameTableau>,
    dm: Option<DmProgram>,
    mps: Option<QuantumDAG>,
}

impl CatPlan {
    /// Lower `dag` into a reusable plan for `mode`.
    pub fn new(dag: &QuantumDAG, mode: Mode) -> Result<Self, String> {
        match mode {
            Mode::Sv => {
                // A readout is dropped when it is trailing, and refused when
                // anything runs after it (see `reject_non_trailing_readouts`).
                // `prepare` builds a unitary, so it is handed a view without the
                // readouts rather than being taught to skip them itself -- that
                // function is shared with the gradient paths, which must keep
                // refusing them.
                reject_dynamic_ops(dag, "statevector")?;
                reject_non_trailing_readouts(dag)?;
                // A *leading* reset is `|0> -> |0>`, so it is dropped the same
                // way the adjoint lane drops it; a reset after use stays an
                // error inside `strip_leading_resets`.
                let filtered = if has_measurements(dag) || has_resets(dag) {
                    let resets_dropped = strip_leading_resets(dag, "statevector")?;
                    Some(without_measurements(&resets_dropped))
                } else {
                    None
                };
                let prepared = PreparedStatevector::new(filtered.as_ref().unwrap_or(dag))?;
                Ok(Self {
                    n_qubits: prepared.n_qubits,
                    parameter_names: prepared.parameter_names.clone(),
                    mode,
                    sv: Some(prepared),
                    adj: None,
                    stab: None,
                    dm: None,
                    mps: None,
                })
            }
            Mode::Dm => {
                // The only lane that can hold a `Reset` after use or a
                // non-selective mid-circuit readout: rho is evolved through
                // the channel itself, not approximated by a pure vector.
                // Feed-forward stays refused -- one rho cannot represent a
                // mixture over branches (see `DmProgram::new`).
                let dm = DmProgram::new(dag)?;
                Ok(Self {
                    n_qubits: dm.n_qubits,
                    parameter_names: dm.parameter_names.clone(),
                    mode,
                    sv: None,
                    adj: None,
                    stab: None,
                    dm: Some(dm),
                    mps: None,
                })
            }
            Mode::Adj => {
                // The adjoint program differentiates the unitary; it has no
                // collapse and no classical state.  A *terminal* measurement
                // is harmless (it cannot change a unitary or the expectation
                // of a Pauli observable), but a gate on a measured qubit or a
                // conditioned gate would make the returned gradient the
                // unitary-only value -- measured: cos(theta)cos(phi) where the
                // true derivative is exactly 0.  Same guard as Stab/Cliff.
                reject_dynamic_ops(dag, "adjoint")?;
                let adj = AdjProgram::from_dag(dag)?;
                Ok(Self {
                    n_qubits: dag.n_qubits,
                    parameter_names: dag.parameter_names(),
                    mode,
                    sv: None,
                    adj: Some(adj),
                    stab: None,
                    dm: None,
                    mps: None,
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
                //
                // Dynamic constructs (mid-circuit measurement, reset after
                // use, feed-forward) are rejected here rather than skipped:
                // the tableau has no collapse, so sampling it anyway would
                // answer a different question than the caller asked.
                reject_dynamic_ops(dag, "stabilizer/clifford")?;
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
                        // Terminal measurement forms carry no tableau update;
                        // `reject_dynamic_ops` has already proved every measure
                        // is terminal, so skipping them here is exact.
                        OpType::Id
                        | OpType::Measure
                        | OpType::MeasureAll
                        | OpType::Reset
                        | OpType::Barrier => continue,
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
                    dm: None,
                    mps: None,
                })
            }
            Mode::Mps => {
                // One pure tensor network, so the statevector lane's guards
                // apply verbatim: a trailing readout is dropped, a leading
                // reset is |0> -> |0>, and anything after a use stays an
                // error.  `reject_dynamic_ops` runs first so a branch or a
                // conditioned gate is refused with the MPS lane named.
                //
                // What is cached is the *symbolic* circuit: gate order, wire
                // routing and parameter names are fixed once here, then each
                // call re-binds in place and re-evolves.  `bond_dim` is a
                // per-call knob -- it changes reachable accuracy, not the
                // program -- so it is not baked into the snapshot.
                reject_dynamic_ops(dag, "mps")?;
                let resets_dropped = strip_leading_resets(dag, "mps")?;
                let owned = without_measurements(&resets_dropped);
                Ok(Self {
                    n_qubits: owned.n_qubits,
                    parameter_names: owned.parameter_names(),
                    mode,
                    sv: None,
                    adj: None,
                    stab: None,
                    dm: None,
                    mps: Some(owned),
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

    /// Execute the density-matrix plan: the flat `rho` (length 4^n) in the
    /// `data[ket | (bra << n)]` layout the binding wraps.
    ///
    /// Unlike the pure lanes this one accepts a `Reset` after use and a
    /// mid-circuit readout: both are channels, and the returned matrix is the
    /// exact mixed state rather than a trajectory.
    pub fn execute_density_matrix(
        &self,
        values: &HashMap<String, f64>,
    ) -> Result<Vec<Complex64>, String> {
        let dm = self
            .dm
            .as_ref()
            .ok_or_else(|| "cat: plan is not a density-matrix plan".to_string())?;
        dm.simulate(values)
    }

    /// Execute the MPS plan and return the tensor network itself.
    ///
    /// `bond_dim` caps the Schmidt rank across every cut: large enough and the
    /// result is the exact statevector (they agree to rounding), small enough
    /// and the evolution truncates, which `MPSState::discarded_weight` reports
    /// rather than hiding.  The returned network keeps the *surviving* weight
    /// (it is not renormalized), so `||psi||^2 == 1 - discarded_weight`.  The
    /// snapshot is re-bound in place on every call, so the compiled gate order
    /// survives a parameter update.
    pub fn execute_mps(
        &mut self,
        values: &HashMap<String, f64>,
        bond_dim: usize,
    ) -> Result<MPSState, String> {
        // Name the parameters that are still symbolic.  Without this the
        // rebind succeeds and `evolve_mps` panics deep in the op evaluator
        // ("Cannot evaluate unbound variable ..."), which reaches Python as a
        // PanicException instead of an error the caller can act on.
        let mut missing: Vec<&str> = self
            .parameter_names
            .iter()
            .filter(|name| !values.contains_key(name.as_str()))
            .map(|name| name.as_str())
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            return Err(format!(
                "cat: mps plan: unbound parameter(s) {} - supply a value for every \
                 name in parameter_names()",
                missing.join(", ")
            ));
        }
        let dag = self
            .mps
            .as_mut()
            .ok_or_else(|| "cat: plan is not an mps plan".to_string())?;
        dag.update_parameters(values);
        Ok(dag.evolve_mps(bond_dim))
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

    /// Execute the MPS plan's gradient: d⟨O⟩/dθ on the truncated chain.
    ///
    /// This is the shift rule on the MPS forward — the same functional
    /// `execute_mps(...).expectation(...)` reports, truncation included — not a
    /// reverse sweep.  The two are different objects: a reverse pass would have
    /// to differentiate through the SVD that truncates, and with the bond at a
    /// binding cap the discarded subspace moves with θ, so the frozen-projection
    /// gradient it computes is not the gradient of this functional.  What the
    /// chain buys here is memory: `bond_dim` bounds the state instead of `2^n`.
    ///
    /// Cost is `2·P` evolutions for an untruncated chain and `4·P` once the
    /// cap bites; [`sf_ir`] decides per parameter (a two-term rule is only the
    /// derivative of the reported functional while nothing is discarded).
    /// `theta` is in [`parameter_names`](Self::parameter_names) order, matching
    /// every other lane's gradient entry point.
    pub fn execute_gradient_mps(
        &mut self,
        theta: &[f64],
        terms: &[PauliTerm],
        bond_dim: usize,
    ) -> Result<Vec<f64>, String> {
        let names = self.parameter_names.clone();
        if theta.len() != names.len() {
            return Err(format!(
                "theta has {} values but the plan has {} parameters",
                theta.len(),
                names.len()
            ));
        }
        let values: HashMap<String, f64> =
            names.iter().cloned().zip(theta.iter().copied()).collect();
        // Every name is bound by construction, so `evolve_mps` below cannot hit
        // an unbound variable: only a wrong `theta` length can reach the error
        // above, and that is named by count.
        let dag = self
            .mps
            .as_ref()
            .ok_or_else(|| "cat: plan is not an mps plan".to_string())?;
        // The state is evolved on a *bound clone*.  Handing the shift rule a
        // dag that this call already bound would leave every op carrying its
        // value instead of a variable, so the exact rule would find no
        // occurrences and every parameter would silently take the 4-evolution
        // stencil (measured: 0 difference in the answer at an untruncated
        // bond, 2x the cost).  The plan keeps its symbolic dag untouched.
        let mut bound = dag.clone_dag();
        bound.update_parameters(&values);
        let state = bound.evolve_mps(bond_dim);
        let handle = sf_ir::MPSStateWrapper::new(state, "cat");
        handle
            .grad(terms, dag, &values)
            .map_err(|e: MethodError| e.to_string())
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
        // Name the mode the handle actually holds: "not a stabilizer plan"
        // alone left the caller guessing which handle they were holding.
        let tab = self.stab.as_ref().ok_or_else(|| {
            format!(
                "cat: sample() needs a stabilizer or clifford plan; this handle was \
                 compiled in '{}' mode",
                self.mode.as_str()
            )
        })?;
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
        // A readout with a gate after it is not a unitary: the plan must reject
        // it at compile time rather than return a state in which the measured
        // qubit is still coherent.
        let mut mid = QuantumDAG::new(1, 1);
        mid.add_op(OpType::H, &[0]);
        mid.add_measure(0, 0);
        mid.add_op(OpType::H, &[0]);
        assert!(CatPlan::new(&mid, Mode::Sv).is_err());

        // A lone *trailing* readout is droppable: the state is |0> exactly.
        let mut trailing = QuantumDAG::new(1, 1);
        trailing.add_op(OpType::Measure, &[0]);
        let plan =
            CatPlan::new(&trailing, Mode::Sv).expect("a trailing readout cannot change the state");
        let sv = plan.execute_statevector(&HashMap::new()).unwrap();
        assert_eq!(sv.len(), 2);
        assert!((sv[0].norm() - 1.0).abs() < 1e-12, "|0> keeps amplitude 1");
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

    /// The `mps` lane answers the same gradient question as `adjoint` while the
    /// bond is large enough to be exact, and keeps answering (finitely, and
    /// deterministically) once it truncates.  Before this entry point existed
    /// the handle raised "plan is not an adjoint plan" on every mps plan.
    #[test]
    fn cat_mps_handle_reports_a_gradient() {
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

        let terms = vec![sf_ir::adjoint::PauliTerm {
            paulis: vec![3u8; 3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let adj_thetas: Vec<f64> = (0..6).map(|i| 0.2 + 0.1 * i as f64).collect();

        let mut adj = CatPlan::new(&dag, Mode::Adj).unwrap();
        let want = adj.execute_gradient(&adj_thetas, &terms).unwrap();

        // Bond 8 is the full 2^3 range for this shape, so the chain is exact.
        let mut mps = CatPlan::new(&dag, Mode::Mps).unwrap();
        assert_eq!(mps.mode(), Mode::Mps);
        let got = mps.execute_gradient_mps(&adj_thetas, &terms, 8).unwrap();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "mps vs adjoint at exact bond: {a} vs {b}"
            );
        }

        // A truncating cap is a different functional, not a crash: finite,
        // deterministic, and stable across repeated calls on the same handle.
        let truncated = mps.execute_gradient_mps(&adj_thetas, &terms, 1).unwrap();
        assert!(truncated.iter().all(|g| g.is_finite()));
        let again = mps.execute_gradient_mps(&adj_thetas, &terms, 1).unwrap();
        assert_eq!(truncated, again, "the plan's dag must not be left bound");

        // A wrong theta length is named by count rather than panicking.
        let err = mps
            .execute_gradient_mps(&adj_thetas[..3], &terms, 8)
            .unwrap_err();
        assert!(err.contains("3") && err.contains("6"), "unexpected: {err}");
    }

    // ── Dynamic-circuit guard (tableau modes) ──

    /// Same two-branch GHZ as `ghz_dag`, but with classical bits so the
    /// measurement tests can write their outcomes somewhere.
    fn ghz_dag_c2(n: usize) -> QuantumDAG {
        let mut dag = QuantumDAG::new(n, 2);
        dag.add_op(OpType::H, &[0]);
        for i in 1..n {
            dag.add_op(OpType::CNOT, &[0, i]);
        }
        dag
    }

    fn rejection(dag: &QuantumDAG) -> String {
        match CatPlan::new(dag, Mode::Stab) {
            Ok(_) => panic!("dynamic circuit must be rejected"),
            Err(msg) => msg,
        }
    }

    #[test]
    fn terminal_measurement_is_accepted() {
        let mut dag = ghz_dag_c2(2);
        dag.add_measure(0, 0);
        dag.add_measure(1, 1);
        let plan = CatPlan::new(&dag, Mode::Stab).expect("terminal measurement is legal");
        let counts = plan.execute_sample(2_000, Some(7)).unwrap();
        assert_eq!(counts.values().sum::<usize>(), 2_000);
        assert_eq!(counts.len(), 2, "GHZ must collapse to two branches");
    }

    #[test]
    fn measure_all_at_the_end_is_accepted() {
        let mut dag = ghz_dag_c2(2);
        dag.add_op(OpType::MeasureAll, &[]);
        let plan = CatPlan::new(&dag, Mode::Stab).expect("measure_all is terminal");
        let total: usize = plan.execute_sample(64, Some(1)).unwrap().values().sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn gate_after_measurement_is_rejected() {
        let mut dag = ghz_dag_c2(2);
        dag.add_measure(0, 0);
        dag.add_op(OpType::X, &[0]);
        let msg = rejection(&dag);
        assert!(msg.contains("measured"), "{msg}");
        assert!(msg.contains("simulate_dynamic"), "{msg}");
    }

    #[test]
    fn conditioned_gate_is_rejected() {
        let mut dag = ghz_dag_c2(2);
        dag.add_measure(0, 0);
        dag.add_op_conditioned(OpType::X, &[1], 0, 1);
        let msg = rejection(&dag);
        assert!(msg.contains("conditioned"), "{msg}");
        assert!(msg.contains("simulate_dynamic"), "{msg}");
    }

    #[test]
    fn reset_after_use_is_rejected() {
        let mut dag = ghz_dag_c2(2);
        dag.add_op(OpType::Reset, &[0]);
        let msg = rejection(&dag);
        assert!(msg.contains("Reset on qubit 0"), "{msg}");
    }

    #[test]
    fn leading_reset_is_accepted() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Reset, &[0]);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        let plan = CatPlan::new(&dag, Mode::Stab).expect("leading reset is a no-op");
        let total: usize = plan.execute_sample(128, Some(3)).unwrap().values().sum();
        assert_eq!(total, 128);
    }

    #[test]
    fn clifford_mode_shares_the_guard() {
        let mut dag = ghz_dag_c2(2);
        dag.add_measure(1, 1);
        dag.add_op(OpType::H, &[1]);
        assert!(CatPlan::new(&dag, Mode::Cliff).is_err());
    }

    /// `ry(theta) ; measure ; rz(phi)` with observable X has `<X> == 0` for
    /// every angle, so both derivatives are exactly 0.  Before the guard the
    /// adjoint lane returned the unitary-only `cos(theta)cos(phi)` (0.7307 at
    /// theta=0.7, phi=0.3) -- a gradient that does not exist.
    #[test]
    fn adjoint_rejects_a_gate_after_measurement() {
        let mut dag = QuantumDAG::new(1, 1);
        dag.add_op(OpType::Ry(var("th")), &[0]);
        dag.add_measure(0, 0);
        dag.add_op(OpType::Rz(var("ph")), &[0]);
        let msg = match CatPlan::new(&dag, Mode::Adj) {
            Ok(_) => panic!("a collapse makes the unitary-only gradient wrong"),
            Err(msg) => msg,
        };
        assert!(msg.contains("after it was measured"), "{msg}");
        assert!(msg.contains("simulate_dynamic"), "{msg}");
    }

    #[test]
    fn adjoint_rejects_feed_forward() {
        let mut dag = QuantumDAG::new(2, 2);
        dag.add_op(OpType::Ry(var("t0")), &[0]);
        dag.add_measure(0, 0);
        dag.add_op_conditioned(OpType::Ry(var("t1")), &[1], 0, 1);
        let msg = match CatPlan::new(&dag, Mode::Adj) {
            Ok(_) => panic!("a conditioned gate has no unitary-only gradient"),
            Err(msg) => msg,
        };
        assert!(msg.contains("conditioned"), "{msg}");
    }

    #[test]
    fn adjoint_accepts_terminal_measurement() {
        // A terminal measurement cannot change a unitary or a Pauli
        // expectation, so the gradient is unchanged: keep it legal.
        let mut dag = QuantumDAG::new(2, 2);
        dag.add_op(OpType::Ry(var("t0")), &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_measure(0, 0);
        dag.add_measure(1, 1);
        let plan = CatPlan::new(&dag, Mode::Adj).expect("terminal measurement is legal");
        assert_eq!(plan.parameter_names(), &["t0".to_string()]);
    }

    /// The adjoint lane's gate-set limit comes from `AdjProgram`, one code path
    /// over from the dynamic guard, so it used to report `cat adj:` while every
    /// neighbouring error said `cat: <lane> plan:`.  A caller comparing lanes
    /// saw two unrelated error styles.
    #[test]
    fn adjoint_gate_set_error_names_its_own_lane() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::ECR, &[0, 1]);
        let msg = match CatPlan::new(&dag, Mode::Adj) {
            Ok(_) => panic!("ECR is outside the adjoint gate set"),
            Err(msg) => msg,
        };
        assert!(msg.starts_with("cat: adjoint plan:"), "{msg}");
        assert!(msg.contains("RY/RZ"), "{msg}");
    }

    fn bell_dag(with_readout: bool) -> QuantumDAG {
        let mut dag = QuantumDAG::new(2, 2);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        if with_readout {
            dag.add_measure(0, 0);
            dag.add_measure(1, 1);
        }
        dag
    }

    /// A trailing readout is dropped, and the returned state must be *exactly*
    /// the state of the same circuit without it -- not merely a state with the
    /// same distribution.
    #[test]
    fn statevector_accepts_a_trailing_readout() {
        let empty = HashMap::new();
        let bare = CatPlan::new(&bell_dag(false), Mode::Sv).unwrap();
        let read = CatPlan::new(&bell_dag(true), Mode::Sv)
            .expect("a trailing readout cannot change the state");
        let a = bare.execute_statevector(&empty).unwrap();
        let b = read.execute_statevector(&empty).unwrap();
        assert_eq!(a, b, "trailing readout must not perturb the statevector");
        assert_eq!(read.parameter_names(), bare.parameter_names());
    }

    /// A readout with a gate after it is *not* droppable: the returned state
    /// would still show the measured qubit as coherent.
    #[test]
    fn statevector_refuses_a_readout_with_a_gate_after_it() {
        let mut dag = QuantumDAG::new(2, 1);
        dag.add_op(OpType::H, &[0]);
        dag.add_measure(0, 0);
        dag.add_op(OpType::X, &[1]);
        let msg = match CatPlan::new(&dag, Mode::Sv) {
            Ok(_) => panic!("a mid-circuit readout cannot be dropped from a state"),
            Err(msg) => msg,
        };
        assert!(msg.contains("runs after a measurement"), "{msg}");
    }

    /// A real collapse on the measured qubit keeps its own, more specific error.
    #[test]
    fn statevector_still_refuses_a_gate_on_a_measured_qubit() {
        let mut dag = QuantumDAG::new(2, 1);
        dag.add_op(OpType::H, &[0]);
        dag.add_measure(0, 0);
        dag.add_op(OpType::H, &[0]);
        let msg = match CatPlan::new(&dag, Mode::Sv) {
            Ok(_) => panic!("collapsing and continuing is not a unitary"),
            Err(msg) => msg,
        };
        assert!(msg.contains("after it was measured"), "{msg}");
    }

    /// A leading reset is `|0> -> |0>`, so the state must be *exactly* the
    /// state of the same circuit without it -- the same rule the adjoint lane
    /// follows, and the same answer Qiskit's `Statevector.from_instruction`
    /// gives for this shape.
    #[test]
    fn statevector_accepts_a_leading_reset() {
        let empty = HashMap::new();
        let bare = CatPlan::new(&bell_dag(false), Mode::Sv).unwrap();

        let mut with_reset = QuantumDAG::new(2, 2);
        with_reset.add_op(OpType::Reset, &[0]);
        with_reset.add_op(OpType::H, &[0]);
        with_reset.add_op(OpType::CNOT, &[0, 1]);
        let reset = CatPlan::new(&with_reset, Mode::Sv)
            .expect("a reset on an untouched qubit cannot change the state");

        let a = bare.execute_statevector(&empty).unwrap();
        let b = reset.execute_statevector(&empty).unwrap();
        assert_eq!(a, b, "leading reset must not perturb the statevector");
        assert_eq!(reset.parameter_names(), bare.parameter_names());
    }

    /// A reset *after* use is collapse-and-reprepare: it produces a mixed state,
    /// which one pure vector cannot hold.  Refusing it is deliberately stricter
    /// than Qiskit, whose pure-state path post-selects and hands back a definite
    /// state instead of the channel's output.
    #[test]
    fn statevector_refuses_a_reset_after_use() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Reset, &[0]);
        let msg = match CatPlan::new(&dag, Mode::Sv) {
            Ok(_) => panic!("a reset after use is not a unitary"),
            Err(msg) => msg,
        };
        assert!(msg.contains("Reset on qubit 0"), "{msg}");
    }
}
