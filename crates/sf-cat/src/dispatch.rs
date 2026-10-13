//! Mode-routed execution seam for [`CatPlan`].
//!
//! `run` is the single entry point the PyO3 handle calls.  It inspects the
//! plan's [`Mode`] and produces an [`Exec`] result.  New milestones (DM,
//! adjoint, stabilizer, Clifford) add a variant to [`Exec`] and a branch here
//! without changing the public `sf.compile(...).execute(...)` surface.

use std::collections::HashMap;

use num_complex::Complex64;
use sf_ir::adjoint::PauliTerm;

use crate::plan::{CatPlan, Mode};

/// Result of a [`run`]/[`run_gradient`] call, tagged by the mode that
/// produced it.
#[derive(Debug)]
pub enum Exec {
    /// Pure statevector amplitudes (length 2^n).
    State(Vec<Complex64>),
    /// Adjoint gradient d⟨O⟩/dθ in `parameter_names()` order (length P).
    Grad(Vec<f64>),
}

/// Execute `plan` with `values`, routing on the plan's mode.
pub fn run(plan: &CatPlan, values: &HashMap<String, f64>) -> Result<Exec, String> {
    match plan.mode() {
        Mode::Sv => plan.execute_statevector(values).map(Exec::State),
        // Adjoint needs an observable, so it has its own entry point; the
        // remaining modes are added in later milestones.  Both are surfaced
        // as errors rather than silently returning an empty/incorrect result.
        Mode::Adj => {
            Err("cat: adjoint plans execute via run_gradient (observable required)".to_string())
        }
        Mode::Dm | Mode::Stab | Mode::Cliff => {
            Err(format!("cat: mode {:?} is not executable yet", plan.mode()))
        }
    }
}

/// Execute an adjoint-mode plan: gradient of ⟨O⟩ for `terms` at `theta`.
pub fn run_gradient(
    plan: &mut CatPlan,
    theta: &[f64],
    terms: &[PauliTerm],
) -> Result<Exec, String> {
    match plan.mode() {
        Mode::Adj => plan.execute_gradient(theta, terms).map(Exec::Grad),
        other => Err(format!("cat: mode {other:?} has no gradient executor")),
    }
}
