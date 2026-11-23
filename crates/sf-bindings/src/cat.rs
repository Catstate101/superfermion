//! PyO3 handle for the `cat` compile-once backend.
//!
//! Mirrors `prepared::PyPreparedCircuit`: build a circuit into a reusable
//! [`CatPlan`] once, then execute it with freshly-bound parameters without
//! rebuilding a DAG or re-parsing gates.  Live modes: `statevector`
//! ([`PyCatCircuit::simulate`]), `adjoint` ([`PyCatCircuit::run_gradient`]),
//! `stabilizer`/`clifford` ([`PyCatCircuit::sample`]), and `mps`
//! ([`PyCatCircuit::simulate`] again, returning a tensor-network `State`
//! instead of a flat buffer).  Density-matrix mode is served by the same
//! [`PyCatCircuit::simulate`] call and returns a density-matrix `State`, so
//! `simulate` reports which lane produced it rather than the caller having to
//! remember.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use num_complex::Complex64;
use pyo3::prelude::*;
use sf_cat::{CatPlan, Mode};
use sf_ir::adjoint::PauliTerm;
use sf_ir::dm::DensityMatrixState;
use sf_ir::state::{DensityMatrixStateWrapper, MPSStateWrapper, StatevectorState};
use sf_ir::MPSState;

use crate::{PyQuantumDAG, PyState};

fn parse_mode(mode: &str) -> PyResult<Mode> {
    match mode.to_ascii_lowercase().as_str() {
        "sv" | "statevector" => Ok(Mode::Sv),
        "dm" | "density_matrix" => Ok(Mode::Dm),
        "adj" | "adjoint" => Ok(Mode::Adj),
        "stab" | "stabilizer" => Ok(Mode::Stab),
        "cliff" | "clifford" => Ok(Mode::Cliff),
        "mps" => Ok(Mode::Mps),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown cat mode '{other}'; use statevector|density_matrix|adjoint|stabilizer|clifford|mps"
        ))),
    }
}

/// Parse a Pauli-string observable ("IXYZ…", character i = qubit i) into
/// the single `PauliTerm` the adjoint engine consumes (coefficient 1.0).
fn parse_pauli_term(n: usize, observable: &str) -> PyResult<PauliTerm> {
    let paulis: Result<Vec<u8>, String> = observable
        .chars()
        .map(|c| match c.to_ascii_uppercase() {
            'I' => Ok(0),
            'X' => Ok(1),
            'Y' => Ok(2),
            'Z' => Ok(3),
            other => Err(format!("invalid Pauli character '{other}' (use I/X/Y/Z)")),
        })
        .collect();
    let paulis = paulis.map_err(pyo3::exceptions::PyValueError::new_err)?;
    if paulis.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "observable length {} does not match n_qubits {n}",
            paulis.len()
        )));
    }
    Ok(PauliTerm {
        paulis,
        coef: Complex64::new(1.0, 0.0),
    })
}

#[pyclass(name = "CatCircuit", frozen)]
pub struct PyCatCircuit {
    // Mutex only because the adjoint and MPS lanes need `&mut` across calls
    // (resident buffers, in-place parameter rebinding); the sv/stab lanes only
    // ever need `&`.  Immutable metadata is mirrored outside the lock so the
    // getters never contend.
    inner: Mutex<CatPlan>,
    n_qubits: usize,
    parameter_names: Vec<String>,
    mode: Mode,
    bond_dim: usize,
}

impl PyCatCircuit {
    fn lock_plan(&self) -> Result<MutexGuard<'_, CatPlan>, String> {
        self.inner
            .lock()
            .map_err(|_| "cat: plan lock poisoned".to_string())
    }
}

#[pymethods]
impl PyCatCircuit {
    /// Compile `dag` once into a reusable handle for `mode` (default
    /// statevector).  `bond_dim` is the MPS Schmidt-rank cap; only the `mps`
    /// lane reads it.
    #[new]
    #[pyo3(signature = (dag, mode="statevector", bond_dim=64))]
    fn new(dag: &PyQuantumDAG, mode: &str, bond_dim: usize) -> PyResult<Self> {
        let m = parse_mode(mode)?;
        if m == Mode::Mps && bond_dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cat: bond_dim must be >= 1 (an MPS with zero bond holds no state)",
            ));
        }
        let plan = CatPlan::new(&dag.inner, m).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let n_qubits = plan.n_qubits();
        let parameter_names = plan.parameter_names().to_vec();
        Ok(Self {
            inner: Mutex::new(plan),
            n_qubits,
            parameter_names,
            mode: m,
            bond_dim,
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The mode this handle was compiled in, as the string `sf.cat` accepted.
    #[getter]
    fn mode(&self) -> &'static str {
        self.mode.as_str()
    }

    /// The MPS Schmidt-rank cap this handle was built with (unused by the
    /// pure lanes, reported so a handle can describe itself).
    #[getter]
    fn bond_dim(&self) -> usize {
        self.bond_dim
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.parameter_names.clone()
    }

    /// Execute the handle and return a `State` (GIL released).
    ///
    /// `statevector` plans return a pure `State`; `density_matrix` plans
    /// return a density-matrix `State` (reset-after-use and non-selective
    /// readouts included); `mps` plans return an `MPSState` holding the tensor
    /// network at this handle's `bond_dim`.  The plan owns the routing so the
    /// lanes cannot drift apart here.
    fn simulate(&self, py: Python<'_>, values: HashMap<String, f64>) -> PyResult<PyState> {
        // MPS answers with a tensor network, not a flat buffer: wrap the
        // `MPSState` directly rather than densifying a 2^n vector (which would
        // throw away the only property that makes the lane worth having).
        if self.mode == Mode::Mps {
            let mps: MPSState = py
                .detach(|| -> Result<MPSState, String> {
                    let mut plan = self.lock_plan()?;
                    plan.execute_mps(&values, self.bond_dim)
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            // Truncation telemetry travels with the state, exactly as it does
            // for `sf.run(method="mps")`: a truncated bond is reported through
            // `truncation_report()`, never silently absorbed.
            let report = crate::MpsTruncationReport {
                discarded_weight: mps.discarded_weight,
                max_discarded_weight: mps.max_discarded_weight,
                truncation_events: mps.truncation_events,
            };
            let mut state = PyState::new(Box::new(MPSStateWrapper::new(mps, "cat")));
            state.mps_report = Some(report);
            return Ok(state);
        }
        let (is_dm, data) = py
            .detach(|| -> Result<(bool, Vec<Complex64>), String> {
                let plan = self.lock_plan()?;
                let is_dm = plan.mode() == Mode::Dm;
                let data = if is_dm {
                    plan.execute_density_matrix(&values)?
                } else {
                    plan.execute_statevector(&values)?
                };
                Ok((is_dm, data))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        if is_dm {
            let dm = DensityMatrixState::from_data(data, self.n_qubits);
            return Ok(PyState::new(Box::new(DensityMatrixStateWrapper::new(
                dm, "cat",
            ))));
        }
        Ok(PyState::new(Box::new(StatevectorState::new(
            data,
            self.n_qubits,
            "cat",
        ))))
    }

    /// Gradient of <observable> at `theta`, aligned with
    /// `parameter_names` (GIL released).  `observable` is a Pauli string
    /// ("IXYZ…", character i = qubit i; coefficient 1.0).
    ///
    /// `adjoint` plans run one forward plus one reverse sweep.  `mps` plans run
    /// the exact shift rule on the truncated chain (`2·P` evolutions, `4·P` for
    /// the gates without an exact rule), so the two lanes answer the same
    /// question with different cost profiles: passes for `adjoint`, memory for
    /// `mps`.  Every other mode still has no gradient here.
    fn run_gradient(
        &self,
        py: Python<'_>,
        observable: &str,
        theta: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let terms = vec![parse_pauli_term(self.n_qubits, observable)?];
        if theta.len() != self.parameter_names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "theta has {} values but the plan has {} parameters",
                theta.len(),
                self.parameter_names.len()
            )));
        }
        py.detach(|| {
            let mut plan = self.lock_plan()?;
            if self.mode == Mode::Mps {
                plan.execute_gradient_mps(&theta, &terms, self.bond_dim)
            } else {
                plan.execute_gradient(&theta, &terms)
            }
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Gradient of a weighted sum of Pauli strings, `⟨O⟩ = Σ coef·⟨P⟩`.
    ///
    /// `terms` is a list of `(pauli_string, coefficient)` pairs.  On an
    /// `adjoint` plan the engine accumulates every term into a single adjoint
    /// state before the reverse sweep, so the cost is one forward plus one
    /// backward pass no matter how many terms the observable carries — a
    /// k-term observable is *not* k separate one-term calls, which is the
    /// difference that matters for Hamiltonian workloads.  An `mps` plan has no
    /// reverse sweep, so its cost scales with the term count too: the shift rule
    /// is run per term and the results are summed.
    #[pyo3(signature = (terms, theta))]
    fn run_gradient_terms(
        &self,
        py: Python<'_>,
        terms: Vec<(String, f64)>,
        theta: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if terms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gradient_terms needs at least one (pauli_string, coefficient) pair",
            ));
        }
        let mut parsed = Vec::with_capacity(terms.len());
        for (observable, coef) in terms {
            let mut term = parse_pauli_term(self.n_qubits, &observable)?;
            term.coef = Complex64::new(coef, 0.0);
            parsed.push(term);
        }
        if theta.len() != self.parameter_names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "theta has {} values but the plan has {} parameters",
                theta.len(),
                self.parameter_names.len()
            )));
        }
        py.detach(|| {
            let mut plan = self.lock_plan()?;
            if self.mode == Mode::Mps {
                plan.execute_gradient_mps(&theta, &parsed, self.bond_dim)
            } else {
                plan.execute_gradient(&theta, &parsed)
            }
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Stabilizer/Clifford-mode sampling: `shots` bulk samples as a counts
    /// dict with little-endian keys (GIL released).
    ///
    /// An MPS plan samples through the state it returns, not through the
    /// handle: `cat(...).simulate(values).sample(shots)` draws from the same
    /// tensor network after binding.  There is no bound version on the handle
    /// itself, so this is refused by name instead of guessing values.
    #[pyo3(signature = (shots, seed=None))]
    fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, usize>> {
        if self.mode == Mode::Mps {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cat: an mps handle samples through its state - call \
                 cat(..., mode=\"mps\").simulate(values).sample(shots, seed) instead \
                 (the handle holds no bound parameter values)",
            ));
        }
        py.detach(|| {
            let plan = self.lock_plan()?;
            plan.execute_sample(shots, seed)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
