//! PyO3 handle for the `cat` compile-once backend.
//!
//! Mirrors `prepared::PyPreparedCircuit`: build a circuit into a reusable
//! [`CatPlan`] once, then execute it with freshly-bound parameters without
//! rebuilding a DAG or re-parsing gates.  Live modes: `statevector`
//! ([`PyCatCircuit::simulate`]), `adjoint` ([`PyCatCircuit::run_gradient`]),
//! and `stabilizer`/`clifford` ([`PyCatCircuit::sample`]).  Density-matrix
//! mode is served by the same [`PyCatCircuit::simulate`] call and returns a
//! density-matrix `State`, so `simulate` reports which lane produced it rather
//! than the caller having to remember.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use num_complex::Complex64;
use pyo3::prelude::*;
use sf_cat::{CatPlan, Mode};
use sf_ir::adjoint::PauliTerm;
use sf_ir::dm::DensityMatrixState;
use sf_ir::state::{DensityMatrixStateWrapper, StatevectorState};

use crate::{PyQuantumDAG, PyState};

fn parse_mode(mode: &str) -> PyResult<Mode> {
    match mode.to_ascii_lowercase().as_str() {
        "sv" | "statevector" => Ok(Mode::Sv),
        "dm" | "density_matrix" => Ok(Mode::Dm),
        "adj" | "adjoint" => Ok(Mode::Adj),
        "stab" | "stabilizer" => Ok(Mode::Stab),
        "cliff" | "clifford" => Ok(Mode::Cliff),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown cat mode '{other}'; use statevector|density_matrix|adjoint|stabilizer|clifford"
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
    // Mutex only because the adjoint lane needs `&mut` across calls (resident
    // buffers); the sv/stab lanes only ever need `&`.  Immutable metadata is
    // mirrored outside the lock so the getters never contend.
    inner: Mutex<CatPlan>,
    n_qubits: usize,
    parameter_names: Vec<String>,
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
    /// statevector).
    #[new]
    #[pyo3(signature = (dag, mode="statevector"))]
    fn new(dag: &PyQuantumDAG, mode: &str) -> PyResult<Self> {
        let m = parse_mode(mode)?;
        let plan = CatPlan::new(&dag.inner, m).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let n_qubits = plan.n_qubits();
        let parameter_names = plan.parameter_names().to_vec();
        Ok(Self {
            inner: Mutex::new(plan),
            n_qubits,
            parameter_names,
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.parameter_names.clone()
    }

    /// Execute the handle and return a `State` (GIL released).
    ///
    /// `statevector` plans return a pure `State`; `density_matrix` plans
    /// return a density-matrix `State` (reset-after-use and non-selective
    /// readouts included).  The plan owns the routing so the two lanes cannot
    /// drift apart here.
    fn simulate(&self, py: Python<'_>, values: HashMap<String, f64>) -> PyResult<PyState> {
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

    /// Adjoint-mode gradient of <observable> at `theta`, aligned with
    /// `parameter_names` (GIL released).  `observable` is a Pauli string
    /// ("IXYZ…", character i = qubit i; coefficient 1.0).
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
            plan.execute_gradient(&theta, &terms)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Stabilizer/Clifford-mode sampling: `shots` bulk samples as a counts
    /// dict with little-endian keys (GIL released).
    #[pyo3(signature = (shots, seed=None))]
    fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, usize>> {
        py.detach(|| {
            let plan = self.lock_plan()?;
            plan.execute_sample(shots, seed)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
