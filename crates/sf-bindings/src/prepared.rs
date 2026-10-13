//! Opt-in prepared CPU/f64 execution. Batch calls are serial across circuits;
//! the existing statevector kernels control within-circuit Rayon parallelism.

use std::collections::HashMap;

use numpy::PyArrayMethods;
use pyo3::prelude::*;
use sf_ir::dag::PreparedStatevector;
use sf_ir::state::StatevectorState;

use crate::{method_error, PyQuantumDAG, PyState};

#[pyclass(name = "DiagonalObservable", frozen)]
pub struct PyDiagonalObservable {
    pub(super) weights: Vec<f64>,
}

#[pymethods]
impl PyDiagonalObservable {
    #[new]
    fn new(weights: Vec<f64>) -> PyResult<Self> {
        if weights.is_empty()
            || !weights.len().is_power_of_two()
            || weights.iter().any(|v| !v.is_finite())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "diagonal weights must be finite and have a nonzero power-of-two length",
            ));
        }
        Ok(Self { weights })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.weights.len()
    }
}

#[pyclass(name = "PreparedCircuit", frozen)]
pub struct PyPreparedCircuit {
    inner: PreparedStatevector,
}

#[pymethods]
impl PyPreparedCircuit {
    #[new]
    fn new(dag: &PyQuantumDAG) -> PyResult<Self> {
        let inner = PreparedStatevector::new(&dag.inner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    fn simulate(&self, py: Python<'_>, values: HashMap<String, f64>) -> PyResult<PyState> {
        let data = py
            .allow_threads(|| self.inner.simulate(&values))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyState::new(Box::new(StatevectorState::new(
            data,
            self.inner.n_qubits,
            "cpu",
        ))))
    }

    /// Return a dense row-major state bank. Each simulation is copied once into
    /// the bank, then NumPy adopts that bank; no State.numpy() clone is needed.
    fn states_batch<'py>(
        &self,
        py: Python<'py>,
        values: Vec<HashMap<String, f64>>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<num_complex::Complex64>>> {
        let rows = values.len();
        let dim = 1usize << self.inner.n_qubits;
        let elements = rows
            .checked_mul(dim)
            .ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("batch dimension overflow"))?;
        let bytes = elements
            .checked_mul(16)
            .ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("batch byte size overflow"))?;
        if bytes > 1usize << 30 {
            return Err(pyo3::exceptions::PyMemoryError::new_err(
                "states_batch is limited to 1 GiB; split the input into smaller batches",
            ));
        }
        let data = py
            .allow_threads(|| -> Result<Vec<num_complex::Complex64>, String> {
                let mut data = Vec::with_capacity(elements);
                for parameters in &values {
                    data.extend(self.inner.simulate(parameters)?);
                }
                Ok(data)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        numpy::PyArray1::from_vec(py, data).reshape([rows, dim])
    }

    fn mean_z_batch<'py>(
        &self,
        py: Python<'py>,
        values: Vec<HashMap<String, f64>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        if self.inner.n_qubits == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mean_z requires at least one qubit",
            ));
        }
        let results = py
            .allow_threads(|| -> Result<Vec<f64>, String> {
                let mut results = Vec::with_capacity(values.len());
                for parameters in &values {
                    let state = StatevectorState::new(
                        self.inner.simulate(parameters)?,
                        self.inner.n_qubits,
                        "cpu",
                    );
                    results.push(state.mean_z().map_err(|e| e.to_string())?);
                }
                Ok(results)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(numpy::PyArray1::from_vec(py, results))
    }

    fn diagonal_batch<'py>(
        &self,
        py: Python<'py>,
        values: Vec<HashMap<String, f64>>,
        observable: &PyDiagonalObservable,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        if observable.weights.len() != 1usize << self.inner.n_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "diagonal dimension does not match circuit",
            ));
        }
        let results = py
            .allow_threads(|| -> Result<Vec<f64>, String> {
                let mut results = Vec::with_capacity(values.len());
                for parameters in &values {
                    let state = StatevectorState::new(
                        self.inner.simulate(parameters)?,
                        self.inner.n_qubits,
                        "cpu",
                    );
                    results.push(
                        state
                            .diagonal_expectation(&observable.weights)
                            .map_err(|e| e.to_string())?,
                    );
                }
                Ok(results)
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(numpy::PyArray1::from_vec(py, results))
    }
}

pub(super) fn statevector<'s>(
    state: &'s PyState,
    py: Python<'_>,
) -> PyResult<&'s StatevectorState> {
    state
        .inner
        .as_any()
        .downcast_ref::<StatevectorState>()
        .ok_or_else(|| method_error(py, "native readout requires a statevector State".into()))
}

pub(super) fn mean_z(state: &PyState, py: Python<'_>) -> PyResult<f64> {
    let state = statevector(state, py)?;
    py.allow_threads(|| state.mean_z())
        .map_err(|e| method_error(py, e.to_string()))
}

pub(super) fn diagonal(
    state: &PyState,
    observable: &PyDiagonalObservable,
    py: Python<'_>,
) -> PyResult<f64> {
    let state = statevector(state, py)?;
    py.allow_threads(|| state.diagonal_expectation(&observable.weights))
        .map_err(|e| method_error(py, e.to_string()))
}
