//! PyO3 bindings for Superfermion's Rust IR, Compiler, Router, Pulse, and QEC.
//!
//! Creates the `_sf_core` Python extension module:
//! - PyQuantumDAG: Python wrapper around QuantumDAG
//! - PyCompiler: Python wrapper around the compilation pipeline
//! - PyRouter: Python wrapper around SABRE routing
//! - PyCouplingMap: Hardware topology
//! - PyPulseSchedule: Pulse-level control
//! - PyQECCode: Quantum error correction codes
//!
//! Usage from Python:
//! ```python
//! from superfermion._sf_core import QuantumDAG, Compiler, Router, CouplingMap
//! ```

#![allow(clippy::redundant_closure)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use sf_ir::gate_list::GateSequence;
use sf_ir::state::{
    DensityMatrixStateWrapper, MPSStateWrapper, QuantumStateImpl, StabilizerStateWrapper,
    StatevectorState,
};
use sf_ir::{MPSState, OpType, Parameter, QuantumDAG, SerializedCircuit};

// ═══════════════════════════════════════════════════════════
// Error mapping — Rust MethodError -> Python MethodError
// ═══════════════════════════════════════════════════════════

/// Map a Rust `MethodError` to the Python `MethodError` class from
/// `superfermion.utils.exceptions` (which subclasses RuntimeError), so
/// unsupported-method calls surface as the documented exception type.
/// Falls back to RuntimeError if the class cannot be imported (e.g. when
/// `_sf_core` is used standalone).
static METHOD_ERROR_TYPE: OnceLock<Py<PyAny>> = OnceLock::new();

fn method_error(py: Python<'_>, msg: String) -> PyErr {
    let cls = METHOD_ERROR_TYPE
        .get_or_init(|| {
            py.import("superfermion.utils.exceptions")
                .and_then(|m| m.getattr("MethodError"))
                .unwrap_or_else(|_| py.get_type::<pyo3::exceptions::PyRuntimeError>().into_any())
                .unbind()
        })
        .bind(py);
    match cls.call1((msg,)) {
        Ok(instance) => PyErr::from_value(instance),
        Err(e) => e,
    }
}

// ═══════════════════════════════════════════════════════════
// MPS State Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around the Rust MPSState.
#[pyclass(name = "MPSState")]
pub struct PyMPSState {
    pub inner: MPSState,
}

#[pymethods]
impl PyMPSState {
    /// Get the MPS tensors as a list of 2D NumPy arrays.
    /// Each tensor has shape (left_bond * 2, right_bond) = (D_L * 2, D_R).
    fn tensors(&self, py: pyo3::Python) -> pyo3::PyResult<Vec<pyo3::PyObject>> {
        use numpy::PyArray1;
        let mut result = Vec::new();
        for tensor in &self.inner.tensors {
            let rows = tensor.nrows();
            let cols = tensor.ncols();
            // Convert nalgebra column-major to numpy array
            let mut data = Vec::with_capacity(rows * cols);
            for i in 0..rows {
                for j in 0..cols {
                    let c = tensor[(i, j)];
                    data.push(num_complex::Complex64::new(c.re, c.im));
                }
            }
            // Create 1D array and reshape to 2D
            let arr1d = PyArray1::from_vec(py, data);
            let arr2d = arr1d.reshape([rows, cols])?;
            result.push(arr2d.as_any().clone().into());
        }
        Ok(result)
    }

    /// Compute <psi|P|psi> for a Pauli string.
    fn pauli_expval(&self, pauli: Vec<u8>) -> (f64, f64) {
        let z = self.inner.pauli_expval(&pauli);
        (z.re, z.im)
    }

    /// Batched Pauli expvals against this state.
    fn pauli_expval_batch(&self, paulis: Vec<Vec<u8>>) -> Vec<(f64, f64)> {
        paulis
            .iter()
            .map(|p| {
                let z = self.inner.pauli_expval(p);
                (z.re, z.im)
            })
            .collect()
    }

    /// Return the virtual-to-physical qubit permutation from lazy-SWAP routing.
    /// perm[virtual_qubit] = physical_site. Identity if no long-range gates.
    fn perm(&self) -> Vec<usize> {
        self.inner.perm.clone()
    }

    /// Accumulated discarded-weight fraction Σε over all 2q-gate QR steps.
    /// 0.0 = exact evolution; > 0 means the `bond_dim` cap removed real weight.
    fn discarded_weight(&self) -> f64 {
        self.inner.discarded_weight
    }

    /// Largest single-step discarded fraction observed.
    fn max_discarded_weight(&self) -> f64 {
        self.inner.max_discarded_weight
    }

    /// Number of 2q-gate steps that discarded non-negligible weight.
    fn truncation_events(&self) -> usize {
        self.inner.truncation_events
    }

    /// Conservative lower bound on the state fidelity: 1 − Σε.
    fn fidelity_lower_bound(&self) -> f64 {
        self.inner.fidelity_lower_bound()
    }

    fn __repr__(&self) -> String {
        format!(
            "MPSState(n_qubits={}, bond_dim={})",
            self.inner.n_qubits, self.inner.bond_dim,
        )
    }
}

// ═══════════════════════════════════════════════════════════
// State — Rust-native quantum state handle (sf.State)
// ═══════════════════════════════════════════════════════════

/// Truncation telemetry captured for method="mps" states.
#[derive(Clone, Copy)]
pub struct MpsTruncationReport {
    pub discarded_weight: f64,
    pub max_discarded_weight: f64,
    pub truncation_events: usize,
}

#[pyclass(name = "State")]
pub struct PyState {
    inner: Box<dyn QuantumStateImpl>,
    /// Set only for method="mps" handles: evolution truncation telemetry.
    mps_report: Option<MpsTruncationReport>,
}

#[pymethods]
impl PyState {
    fn expectation(&self, py: Python<'_>, observable: Vec<(Vec<u8>, f64, f64)>) -> PyResult<f64> {
        let terms = Self::parse_observable(&observable);
        self.inner
            .expectation(&terms)
            .map_err(|e| method_error(py, e.to_string()))
    }

    fn variance(&self, py: Python<'_>, observable: Vec<(Vec<u8>, f64, f64)>) -> PyResult<f64> {
        let terms = Self::parse_observable(&observable);
        self.inner
            .variance(&terms)
            .map_err(|e| method_error(py, e.to_string()))
    }

    #[pyo3(signature = (shots, seed=42))]
    fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: u64,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        self.inner
            .sample(shots, seed)
            .map_err(|e| method_error(py, e.to_string()))
    }

    fn grad(
        &self,
        py: Python<'_>,
        observable: Vec<(Vec<u8>, f64, f64)>,
        dag: &PyQuantumDAG,
        param_values: std::collections::HashMap<String, f64>,
    ) -> PyResult<std::collections::HashMap<String, f64>> {
        let terms = Self::parse_observable(&observable);
        let gradients = self
            .inner
            .grad(&terms, &dag.inner, &param_values)
            .map_err(|e| method_error(py, e.to_string()))?;
        let names = dag.inner.parameter_names();
        let mut result = std::collections::HashMap::new();
        for (name, grad) in names.into_iter().zip(gradients) {
            result.insert(name, grad);
        }
        Ok(result)
    }

    fn entropy(&self, py: Python<'_>) -> PyResult<f64> {
        self.inner
            .entropy()
            .map_err(|e| method_error(py, e.to_string()))
    }

    fn purity(&self, py: Python<'_>) -> PyResult<f64> {
        self.inner
            .purity()
            .map_err(|e| method_error(py, e.to_string()))
    }

    fn fidelity(&self, py: Python<'_>, other: &PyState) -> PyResult<f64> {
        self.inner
            .fidelity(other.inner.as_ref())
            .map_err(|e| method_error(py, e.to_string()))
    }

    fn qfim<'py>(
        &self,
        py: Python<'py>,
        dag: &PyQuantumDAG,
        param_values: std::collections::HashMap<String, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let matrix = self
            .inner
            .qfim(&dag.inner, &param_values)
            .map_err(|e| method_error(py, e.to_string()))?;
        let n = matrix.len();
        if n == 0 {
            return Ok(numpy::PyArray2::from_vec2(py, &[]).unwrap());
        }
        Ok(numpy::PyArray2::from_vec2(py, &matrix).unwrap())
    }

    fn numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let v = self
            .inner
            .to_vec()
            .map_err(|e| method_error(py, e.to_string()))?;
        Ok(numpy::PyArray1::from_vec(py, v))
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let p = self
            .inner
            .probabilities()
            .map_err(|e| method_error(py, e.to_string()))?;
        Ok(numpy::PyArray1::from_vec(py, p))
    }

    /// Truncation telemetry for MPS states (None for exact methods).
    ///
    /// Returns a dict with:
    ///   discarded_weight     — Σε over all 2q-gate QR steps (0.0 = exact)
    ///   max_discarded_weight — largest single-step discard
    ///   truncation_events    — steps that discarded non-negligible weight
    ///   fidelity_lower_bound — 1 − Σε, clamped to [0, 1]; a conservative
    ///                          floor (see MPSState::fidelity_lower_bound)
    fn truncation_report<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let Some(report) = self.mps_report else {
            return Ok(None);
        };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("discarded_weight", report.discarded_weight)?;
        d.set_item("max_discarded_weight", report.max_discarded_weight)?;
        d.set_item("truncation_events", report.truncation_events)?;
        d.set_item(
            "fidelity_lower_bound",
            (1.0 - report.discarded_weight).clamp(0.0, 1.0),
        )?;
        Ok(Some(d))
    }

    fn partial_trace(&self, py: Python<'_>, keep_qubits: Vec<usize>) -> PyResult<Self> {
        let new_state = self
            .inner
            .partial_trace(&keep_qubits)
            .map_err(|e| method_error(py, e.to_string()))?;
        Ok(PyState {
            inner: new_state,
            mps_report: None,
        })
    }

    fn mutual_info(
        &self,
        py: Python<'_>,
        wires_a: Vec<usize>,
        wires_b: Vec<usize>,
    ) -> PyResult<f64> {
        self.inner
            .mutual_info(&wires_a, &wires_b)
            .map_err(|e| method_error(py, e.to_string()))
    }

    #[staticmethod]
    fn from_numpy(
        data: numpy::PyReadonlyArray1<num_complex::Complex64>,
        n_qubits: usize,
    ) -> PyResult<Self> {
        let v: Vec<num_complex::Complex64> = data.as_slice()?.to_vec();
        if v.len() != 1 << n_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array length {} doesn't match 2^{} = {}",
                v.len(),
                n_qubits,
                1usize << n_qubits
            )));
        }
        Ok(PyState {
            inner: Box::new(StatevectorState::new(v, n_qubits, "cpu")),
            mps_report: None,
        })
    }

    /// Build a density-matrix state handle from an explicit rho.
    ///
    /// ``rho`` is the row-major flatten (complex128, length 2^(2n)) of the
    /// density matrix in the little-endian computational basis (qubit q at
    /// bit position q) — the layout returned by ``State.numpy()`` for
    /// ``method='density_matrix'``.  Used to build the handle for noisy
    /// density-matrix runs, where the rho is only available after the
    /// noisy simulation (``simulate_to_state`` has no noise path).
    #[staticmethod]
    fn from_dm(
        rho: numpy::PyReadonlyArray1<num_complex::Complex64>,
        n_qubits: usize,
    ) -> PyResult<Self> {
        // Borrow the input array directly: the permutation below only reads
        // it, and copying it first would keep a second 4^n buffer live.
        let flat = rho.as_slice()?;
        let dim = 1usize << n_qubits;
        if flat.len() != dim * dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array length {} doesn't match 2^{} * 2^{} = {}",
                flat.len(),
                n_qubits,
                n_qubits,
                dim * dim
            )));
        }
        // Wrappers index the density matrix in the interleaved vectorized
        // layout data[ket | (bra << n)] (see DensityMatrixStateWrapper),
        // while the input rho is row-major; permute accordingly.
        let mut data = vec![num_complex::Complex64::new(0.0, 0.0); dim * dim];
        for ket in 0..dim {
            for bra in 0..dim {
                data[ket | (bra << n_qubits)] = flat[ket * dim + bra];
            }
        }
        // Adopt `data` directly (`DensityMatrixState::new` would allocate
        // and zero-fill a throwaway 4^n buffer here; same fix as
        // `simulate_dm_noisy_state`).
        let dm = sf_ir::dm::DensityMatrixState::from_data(data, n_qubits);
        Ok(PyState::new(Box::new(DensityMatrixStateWrapper::new(
            dm, "cpu",
        ))))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn method(&self) -> &str {
        self.inner.method_name()
    }

    #[getter]
    fn device(&self) -> &str {
        self.inner.device_name()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        let n = self.inner.n_qubits();
        match self.inner.method_name() {
            "density_matrix" => vec![1 << n, 1 << n],
            _ => vec![1 << n],
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "State(n_qubits={}, method='{}', device='{}')",
            self.inner.n_qubits(),
            self.inner.method_name(),
            self.inner.device_name(),
        )
    }
}

impl PyState {
    pub fn new(inner: Box<dyn QuantumStateImpl>) -> Self {
        PyState {
            inner,
            mps_report: None,
        }
    }

    fn parse_observable(terms: &[(Vec<u8>, f64, f64)]) -> Vec<sf_ir::PauliTerm> {
        terms
            .iter()
            .map(|(paulis, re, im)| sf_ir::PauliTerm {
                paulis: paulis.clone(),
                coef: num_complex::Complex64::new(*re, *im),
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
// QuantumDAG Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around the Rust QuantumDAG.
#[pyclass(name = "QuantumDAG")]
pub struct PyQuantumDAG {
    pub inner: QuantumDAG,
}

#[pymethods]
impl PyQuantumDAG {
    #[new]
    fn new(n_qubits: usize, n_cbits: usize) -> Self {
        PyQuantumDAG {
            inner: QuantumDAG::new(n_qubits, n_cbits),
        }
    }

    /// Add a gate to the circuit.
    /// Qubit indices are passed through directly to Rust (LSB convention).
    /// The Python RustBackend handles MSB↔LSB conversion via statevector transpose.
    fn add_gate(
        &mut self,
        gate_name: &str,
        qubits: Vec<usize>,
        params: Vec<pyo3::Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let rust_params: Vec<Parameter> = params
            .into_iter()
            .map(|p| {
                if let Ok(f) = p.extract::<f64>() {
                    Parameter::Const(f)
                } else if let Ok(s) = p.extract::<String>() {
                    Parameter::Variable { name: s, id: 0 }
                } else {
                    Parameter::Const(0.0)
                }
            })
            .collect();

        let op = Self::parse_gate(gate_name, &rust_params)?;
        self.inner.add_op(op, &qubits);
        Ok(())
    }

    /// Add a gate that only executes when classical bit `cbit` equals
    /// `value` (classical feed-forward, OpenQASM 3 `if` semantics).
    fn add_gate_cond(
        &mut self,
        gate_name: &str,
        qubits: Vec<usize>,
        params: Vec<pyo3::Bound<'_, pyo3::PyAny>>,
        cbit: usize,
        value: u64,
    ) -> PyResult<()> {
        let rust_params: Vec<Parameter> = params
            .into_iter()
            .map(|p| {
                if let Ok(f) = p.extract::<f64>() {
                    Parameter::Const(f)
                } else if let Ok(s) = p.extract::<String>() {
                    Parameter::Variable { name: s, id: 0 }
                } else {
                    Parameter::Const(0.0)
                }
            })
            .collect();

        let op = Self::parse_gate(gate_name, &rust_params)?;
        self.inner.add_op_conditioned(op, &qubits, cbit, value);
        Ok(())
    }

    /// Add an opaque unitary matrix gate directly to the DAG.
    fn add_unitary(
        &mut self,
        qubits: Vec<usize>,
        matrix: numpy::PyReadonlyArray2<num_complex::Complex64>,
    ) -> PyResult<()> {
        let shape = matrix.shape();
        let rows = shape[0];
        let cols = shape[1];
        if rows != cols || !rows.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unitary matrix must be square with power-of-2 dimension, got {}x{}",
                rows, cols
            )));
        }
        let slice = matrix.as_slice()?;
        let dm = nalgebra::DMatrix::from_row_slice(rows, cols, slice);
        self.inner.add_op(OpType::Unitary(dm), &qubits);
        Ok(())
    }

    /// Add measurement.
    /// Qubit index is passed through directly (LSB convention).
    fn add_measure(&mut self, qubit: usize, cbit: usize) -> PyResult<()> {
        self.inner.add_measure(qubit, cbit);
        Ok(())
    }

    /// Get the circuit depth.
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Get the gate count.
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Count operations of a specific type.
    fn count_ops(&self, op_name: &str) -> usize {
        self.inner.count_ops_of_type(op_name)
    }

    /// Get the number of qubits.
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    /// Get the number of classical bits.
    fn n_cbits(&self) -> usize {
        self.inner.n_cbits
    }

    /// Get the number of free parameters.
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters()
    }

    /// Get parameter names.
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    /// Bind parameters to concrete values.
    fn bind(&self, values: std::collections::HashMap<String, f64>) -> Self {
        PyQuantumDAG {
            inner: self.inner.bind(&values),
        }
    }

    /// In-place update of parameters (fast path for QML).
    fn update_parameters(&mut self, values: std::collections::HashMap<String, f64>) {
        self.inner.update_parameters(&values);
    }

    /// Export to OpenQASM 3.0 string.
    fn to_qasm3(&self) -> String {
        self.inner.to_qasm3()
    }

    /// Serialize to JSON.
    fn to_json(&self) -> PyResult<String> {
        let serialized = SerializedCircuit::from_dag(&self.inner);
        serialized
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Export to full unitary matrix (NumPy).
    fn to_unitary<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<num_complex::Complex64>>> {
        let u = self.inner.to_unitary();
        let (r, c) = u.shape();
        // nalgebra matrices are column-major by default
        let flat: Vec<num_complex::Complex64> = u.as_slice().to_vec();
        let arr = numpy::PyArray1::from_vec(py, flat);
        let arr2 = arr.reshape([c, r])?.call_method0("transpose")?;
        Ok(arr2.downcast_into::<numpy::PyArray2<num_complex::Complex64>>()?)
    }

    /// Run high-performance statevector simulation and return the final statevector.
    ///
    /// With `SF_F32=1` the opt-in f32 lane runs instead and a complex64
    /// array is returned (the qsim-class f32 backend); the default path is
    /// unchanged f64/complex128.
    fn simulate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if sf_ir::f32lane::lane_enabled() {
            let state = self.inner.simulate_f32();
            return Ok(numpy::PyArray1::from_vec(py, state).into_any());
        }
        let state = self.inner.simulate();
        Ok(numpy::PyArray1::from_vec(py, state).into_any())
    }

    /// Explicit f32-lane simulation (complex64 output) regardless of SF_F32.
    fn simulate_f32<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex32>>> {
        let state = self.inner.simulate_f32();
        Ok(numpy::PyArray1::from_vec(py, state))
    }

    /// Run high-performance density-matrix simulation and return the vectorized DM.
    fn simulate_dm<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let state = self.inner.simulate_dm();
        Ok(numpy::PyArray1::from_vec(py, state))
    }

    /// Compute batched Pauli expectation values for statevector.
    fn simulate_pauli_expval_batch(&self, paulis: Vec<Vec<u8>>) -> Vec<f64> {
        self.inner.simulate_pauli_expval_batch(paulis)
    }

    /// Run high-performance MPS simulation and return counts directly.
    fn sample_mps(
        &self,
        bond_dim: usize,
        shots: usize,
        seed: u64,
    ) -> std::collections::HashMap<String, usize> {
        self.inner.sample_mps(bond_dim, shots, seed)
    }

    /// MPS-evolve the circuit using QR-based 2q gate factorization (no
    /// SVD truncation), then return <psi|P|psi> via boundary contraction.
    /// Pauli is encoded one byte per site: 0=I, 1=X, 2=Y, 3=Z. Returns
    /// (real, imag).
    fn simulate_mps_pauli_expval(&self, bond_dim: usize, pauli: Vec<u8>) -> (f64, f64) {
        let z = self.inner.simulate_mps_pauli_expval(bond_dim, &pauli);
        (z.re, z.im)
    }

    /// Batched MPS Pauli expvals: evolve the MPS once (the expensive part),
    /// then contract every Pauli string in `paulis` against the same
    /// evolved state.  Far cheaper than calling
    /// `simulate_mps_pauli_expval` in a loop for multi-term Hamiltonians.
    ///
    /// Each Pauli is one byte per site (0=I, 1=X, 2=Y, 3=Z).  Returns a
    /// list of (real, imag) pairs parallel to the input.
    fn simulate_mps_pauli_expval_batch(
        &self,
        bond_dim: usize,
        paulis: Vec<Vec<u8>>,
    ) -> Vec<(f64, f64)> {
        self.inner
            .simulate_mps_pauli_expval_batch(bond_dim, &paulis)
            .into_iter()
            .map(|z| (z.re, z.im))
            .collect()
    }

    /// Diagnostic: MPS-evolve and report the squared norm via the same
    /// boundary contraction. For a unitary evolution this should be 1.
    fn mps_norm_sq(&self, bond_dim: usize) -> f64 {
        let pauli = vec![0u8; self.inner.n_qubits];
        let z = self.inner.simulate_mps_pauli_expval(bond_dim, &pauli);
        z.re
    }

    /// Evolve the circuit into an MPS state and return it for reuse.
    fn evolve_mps(&self, bond_dim: usize) -> PyMPSState {
        PyMPSState {
            inner: self.inner.evolve_mps(bond_dim),
        }
    }

    /// Simulate and return a State handle (sf.State).
    /// Method can be "statevector", "density_matrix", "mps", or "stabilizer".
    #[pyo3(signature = (method="statevector", device="cpu", bond_dim=64))]
    fn simulate_to_state(&self, method: &str, device: &str, bond_dim: usize) -> PyResult<PyState> {
        match method {
            "statevector" => {
                // Opt-in f32 lane (SF_F32=1): simulate in complex64, then scale
                // up once to the complex128 handle so the public State API
                // (numpy/sample/expectation) stays byte-for-byte unchanged.
                if device != "gpu" && sf_ir::f32lane::lane_enabled() {
                    let sv32 = self.inner.simulate_f32();
                    let sv: Vec<num_complex::Complex64> = sv32
                        .into_iter()
                        .map(|z| num_complex::Complex64::new(z.re as f64, z.im as f64))
                        .collect();
                    return Ok(PyState::new(Box::new(StatevectorState::new(
                        sv,
                        self.inner.n_qubits,
                        device,
                    ))));
                }
                let sv = if device == "gpu" {
                    self.inner
                        .simulate_on("gpu")
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
                } else {
                    self.inner.simulate()
                };
                Ok(PyState::new(Box::new(StatevectorState::new(
                    sv,
                    self.inner.n_qubits,
                    device,
                ))))
            }
            "density_matrix" => {
                // `simulate_dm` returns a fully-populated 4^n buffer; adopt
                // it instead of allocating + zero-filling a throwaway one.
                let dm_vec = self.inner.simulate_dm();
                let dm = sf_ir::dm::DensityMatrixState::from_data(dm_vec, self.inner.n_qubits);
                Ok(PyState::new(Box::new(DensityMatrixStateWrapper::new(
                    dm, device,
                ))))
            }
            "mps" => {
                let mut mps = self.inner.evolve_mps(bond_dim);
                // Canonicalize so boundary contraction (expval) and per-site
                // sampling are numerically stable right after construction.
                mps.canonicalize_right();
                let report = MpsTruncationReport {
                    discarded_weight: mps.discarded_weight,
                    max_discarded_weight: mps.max_discarded_weight,
                    truncation_events: mps.truncation_events,
                };
                let mut state = PyState::new(Box::new(MPSStateWrapper::new(mps, device)));
                state.mps_report = Some(report);
                Ok(state)
            }
            "stabilizer" => {
                let gates = self.inner.to_gate_records();
                let gate_list: Vec<(String, Vec<usize>)> = gates
                    .iter()
                    .filter(|(name, _, _)| {
                        let up = name.to_uppercase();
                        up != "BARRIER" && up != "MEASURE" && up != "RESET"
                    })
                    .map(|(name, qubits, _params)| (name.to_uppercase(), qubits.clone()))
                    .collect();
                let tab = sf_ir::stabilizer::StabilizerTableau::from_gate_list(
                    self.inner.n_qubits,
                    &gate_list,
                )
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
                Ok(PyState::new(Box::new(StabilizerStateWrapper::new(
                    tab, device,
                ))))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown method '{}'. Use 'statevector', 'density_matrix', 'mps', or 'stabilizer'.",
                method
            ))),
        }
    }

    /// Parse an OpenQASM 2.0 string into a QuantumDAG.
    /// Fast Rust-native parser — no regex, no Python overhead.
    #[staticmethod]
    fn from_qasm2(qasm_str: &str) -> PyResult<Self> {
        let dag = sf_ir::qasm::parse_qasm2(qasm_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(PyQuantumDAG { inner: dag })
    }

    /// Export all gate nodes as a list of (name, qubits, params) tuples.
    /// Qubits are in MSB-first order (matching sf.Circuit internal layout).
    /// Skips barrier nodes. Useful for fast DAG→Circuit conversion.
    fn to_gate_records(&self) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        self.inner.to_gate_records()
    }

    /// Apply Pauli twirling to 2Q gates and return twirled (name, qubits, params) list.
    /// All gate data stays in Rust — zero Python-side tuple allocation for 100K+ gates.
    fn pauli_twirl_gates(&self, seed: u64) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        let records = self.inner.to_gate_records();
        sf_ir::stabilizer::pauli_twirl_gate_records(&records, seed)
    }

    /// Deserialize from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let serialized = SerializedCircuit::from_json(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyQuantumDAG {
            inner: serialized.to_dag(),
        })
    }

    /// Adjoint differentiation: compute d<O>/d(theta) for all parameters
    /// in a single forward + backward pass. O(M * 2^n) regardless of N.
    ///
    /// Args:
    ///   observable: list of (pauli_per_qubit: list[int], coef_real: float, coef_imag: float)
    ///   param_values: dict mapping parameter name -> float value
    ///
    /// Returns: dict mapping parameter name -> gradient value.
    fn adjoint_grad(
        &self,
        py: Python<'_>,
        observable: Vec<(Vec<u8>, f64, f64)>,
        param_values: std::collections::HashMap<String, f64>,
    ) -> PyResult<std::collections::HashMap<String, f64>> {
        let terms: Vec<sf_ir::PauliTerm> = observable
            .iter()
            .map(|(paulis, re, im)| sf_ir::PauliTerm {
                paulis: paulis.clone(),
                coef: num_complex::Complex64::new(*re, *im),
            })
            .collect();

        let result = sf_ir::adjoint_grad(&self.inner, &terms, &param_values)
            .map_err(|e| method_error(py, e.to_string()))?;
        let mut out = std::collections::HashMap::new();
        for (name, grad) in result.param_names.iter().zip(result.gradients.iter()) {
            out.insert(name.clone(), *grad);
        }
        Ok(out)
    }

    /// Simulate and sample bitstrings directly in Rust (no statevector return to Python).
    /// Returns a dict of bitstring -> count.
    fn simulate_and_sample(
        &self,
        shots: usize,
        seed: u64,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        let sv = self.inner.simulate();
        let n = self.inner.n_qubits;
        let dim = sv.len();

        let probs: Vec<f64> = sv.iter().map(|c| c.re * c.re + c.im * c.im).collect();

        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        // Build cumulative distribution
        let mut cumulative = vec![0.0f64; dim + 1];
        for i in 0..dim {
            cumulative[i + 1] = cumulative[i] + probs[i];
        }
        let total = cumulative[dim];

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let idx = match cumulative
                .binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(i) => i.min(dim - 1),
                Err(i) => (i - 1).min(dim - 1),
            };
            let bitstring: String = (0..n)
                .rev()
                .map(|q| if (idx >> q) & 1 == 1 { '1' } else { '0' })
                .collect();
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(counts)
    }

    /// Mid-circuit (dynamic-circuit) simulation: per-shot trajectory
    /// replay with state collapse at every Measure op and classical
    /// feed-forward (conditioned ops) support. Returns a bitstring ->
    /// count dict over all qubits (same orientation as
    /// `simulate_and_sample`).
    fn simulate_dynamic(
        &self,
        shots: usize,
        seed: u64,
    ) -> std::collections::HashMap<String, usize> {
        self.inner.simulate_dynamic(shots, seed)
    }

    /// Simulate and apply MSB/LSB endianness transpose in Rust before returning.
    /// Returns statevector in MSB (q0=leftmost) convention directly.
    fn simulate_msb<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let sv = self.inner.simulate();
        let n = self.inner.n_qubits;
        let dim = sv.len();

        let mut msb_sv = vec![num_complex::Complex64::new(0.0, 0.0); dim];
        for i in 0..dim {
            let reversed = reverse_bits(i, n);
            msb_sv[reversed] = sv[i];
        }
        Ok(numpy::PyArray1::from_vec(py, msb_sv))
    }

    /// Simulate on a specified device: "cpu" or "gpu".
    /// Returns statevector in LSB convention (same as simulate()).
    /// Raises RuntimeError if GPU is unavailable or circuit is too large for VRAM.
    fn simulate_on<'py>(
        &self,
        py: Python<'py>,
        device: &str,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let sv = self
            .inner
            .simulate_on(device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(numpy::PyArray1::from_vec(py, sv))
    }

    /// Check if GPU simulation is available at runtime.
    #[staticmethod]
    fn gpu_available() -> bool {
        QuantumDAG::gpu_available()
    }

    /// Apply Kraus operators to the density matrix for noisy simulation.
    /// noise_ops: list of (qubit, kraus_flat) where kraus_flat is a flat list of
    /// f64 values encoding Kraus matrices. Each 2x2 Kraus matrix = 8 floats
    /// (re00, im00, re01, im01, re10, im10, re11, im11).
    /// noise_2q_ops: optional list of 2q channels; each channel is a flat
    /// list of 4x4 Kraus matrices (32 floats each), applied sequentially
    /// after every 2-qubit gate (after the fused gate + 1q-noise sweep).
    /// placement: None/"touch" (default) applies each 1q channel after every
    /// instruction touching its qubit (a 2q gate picks up both qubits' 1q
    /// channels); "gate" applies 1q channels only after 1q gates — the
    /// Qiskit-noise-model placement.  See `placement_per_touch`.
    /// noise_2q_targets: optional per-channel ordered (q0, q1) pair aligned
    /// with `noise_2q_ops`; `None` entries (or a missing/short list) fire on
    /// every 2q gate, a pair only on an instruction emitted in that exact
    /// order (Qiskit `add_quantum_error` semantics).
    #[pyo3(signature = (noise_ops, noise_2q_ops=None, placement=None, noise_2q_targets=None))]
    fn simulate_dm_noisy<'py>(
        &self,
        py: Python<'py>,
        noise_ops: Vec<(usize, Vec<f64>)>,
        noise_2q_ops: Option<Vec<Vec<f64>>>,
        placement: Option<&str>,
        noise_2q_targets: Option<Vec<Option<(usize, usize)>>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let per_touch = placement_per_touch(placement)?;
        let noise_2q_ops = noise_2q_ops.unwrap_or_default();
        let noise_2q_targets = noise_2q_targets.unwrap_or_default();
        let data = dm_noisy_core(
            &self.inner,
            &noise_ops,
            &noise_2q_ops,
            &noise_2q_targets,
            per_touch,
        );
        Ok(numpy::PyArray1::from_vec(py, data))
    }

    /// Noisy density-matrix simulation returning the Rust state handle
    /// directly: the DM data is moved into the handle (zero extra copies),
    /// avoiding the numpy round-trip (vec -> numpy -> vec) that
    /// ``simulate_dm_noisy`` + ``State.from_dm`` would incur on the noisy
    /// path. Also returns the PUBLIC row-major rho (qubit-0-first, the
    /// ``metadata["density_matrix"]`` convention) built by one in-Rust
    /// gather, replacing the Python ``numpy()`` + ``_reverse_qubits_dm``
    /// round-trip.
    // Silence the uninit_vec lint for the deliberately uninitialized public
    // rho buffer inside (pass 1 writes every element before any read).
    /// Accepts the same `noise_2q_targets` ordered-pair list as
    /// `simulate_dm_noisy`.
    #[allow(clippy::uninit_vec)]
    #[pyo3(signature = (noise_ops, noise_2q_ops=None, placement=None, noise_2q_targets=None))]
    fn simulate_dm_noisy_state<'py>(
        &self,
        py: Python<'py>,
        noise_ops: Vec<(usize, Vec<f64>)>,
        noise_2q_ops: Option<Vec<Vec<f64>>>,
        placement: Option<&str>,
        noise_2q_targets: Option<Vec<Option<(usize, usize)>>>,
    ) -> PyResult<(PyState, Bound<'py, numpy::PyArray1<num_complex::Complex64>>)> {
        let trace = std::env::var_os("SF_DM_TRACE").is_some();
        let per_touch = placement_per_touch(placement)?;
        let t_core = std::time::Instant::now();
        let noise_2q_ops = noise_2q_ops.unwrap_or_default();
        let noise_2q_targets = noise_2q_targets.unwrap_or_default();
        let data = dm_noisy_core(
            &self.inner,
            &noise_ops,
            &noise_2q_ops,
            &noise_2q_targets,
            per_touch,
        );
        let t_core = t_core.elapsed();
        let n = self.inner.n_qubits;
        let dim = 1usize << n;
        // Public rho[k][b] = data[rev(k) + rev(b)*dim] (verified against
        // the validated scatter gather, and against an independent numpy
        // oracle at n=10). The storage layout
        // i = ket | (bra << n) means the row-major view of the vec is
        // M[bra][ket], so the public matrix equals the both-axes
        // bit-reversed M, transposed. Applied as two contiguous,
        // cache-friendly passes — blocked transpose (doubles as the copy;
        // it writes every element, so the destination needs no zero-init),
        // then the row swap r <-> rev(r) fused with the in-row column
        // bit-reversal — each written so every element is produced by
        // exactly one task and the regions are cache-line aligned, so they
        // run under Rayon. A direct per-element gather is a scattered
        // permutation (latency-bound, ~50-100 ms at n=10); these passes
        // are memcpy-speed.
        // Send+Sync wrapper for the two buffers used by the parallel perm
        // passes, with method access so the Rayon closures capture the
        // wrapper as a whole (a bare field access would capture the raw
        // pointer field, which is not Sync). Each pass is written so every
        // element is produced by exactly one task and the regions are
        // cache-line aligned.
        #[derive(Clone, Copy)]
        struct DmPermPtrs {
            src: *const num_complex::Complex64,
            dst: *mut num_complex::Complex64,
        }
        unsafe impl Send for DmPermPtrs {}
        unsafe impl Sync for DmPermPtrs {}
        impl DmPermPtrs {
            #[inline(always)]
            unsafe fn get(&self, i: usize) -> num_complex::Complex64 {
                *self.src.add(i)
            }
            #[inline(always)]
            unsafe fn set(&self, i: usize, v: num_complex::Complex64) {
                *self.dst.add(i) = v;
            }
            /// Start of destination row `r`. Method access so Rayon closures
            /// capture the whole wrapper (a bare `p.dst` field access would
            /// capture the raw pointer, which is not Sync).
            #[inline(always)]
            unsafe fn dst_row(&self, r: usize, dim: usize) -> *mut num_complex::Complex64 {
                self.dst.add(r * dim)
            }
        }
        // Uninitialized destination: pass 1 writes every element of the
        // matrix (it is the blocking copy), so the usual zero-fill would
        // be a wasted 16 MB write. Complex64 is Copy (no Drop), so a raw
        // set_len over the reserved buffer is fine as long as nothing
        // reads it before pass 1 completes — which the passes guarantee.
        let mut public: Vec<num_complex::Complex64> = Vec::with_capacity(dim * dim);
        unsafe { public.set_len(dim * dim) };
        let t_p1 = std::time::Instant::now();
        {
            const BS: usize = 32;
            let p = DmPermPtrs {
                src: data.as_ptr(),
                dst: public.as_mut_ptr(),
            };
            if dim >= BS {
                // One task per BS-row source band; band `bi` writes the
                // disjoint destination columns [i0, i0+BS) of every row.
                (0..dim / BS).into_par_iter().for_each(|bi| {
                    let i0 = bi * BS;
                    let mut j0 = 0;
                    while j0 < dim {
                        for b in 0..BS.min(dim - j0) {
                            let dst_base = (j0 + b) * dim + i0;
                            let src_base = i0 * dim + (j0 + b);
                            unsafe {
                                for a in 0..BS {
                                    p.set(dst_base + a, p.get(src_base + a * dim));
                                }
                            }
                        }
                        j0 += BS;
                    }
                });
            } else {
                for i in 0..dim {
                    for j in 0..dim {
                        public[j * dim + i] = data[i * dim + j];
                    }
                }
            }
        }
        let t_p1 = t_p1.elapsed();
        let t_p2 = std::time::Instant::now();
        {
            // Merged row swap r <-> rev(r) + in-row column bit-reversal in
            // ONE pass (previously two sweeps over `public`): each row pair
            // is handled once, by the task owning its smaller index, and
            // the column reversal is applied while both rows are still
            // cache-hot from the swap. Rows fixed by the reversal
            // (r == rev(r)) get the column reversal alone. The row fits in
            // cache, so pairwise column swaps suffice; the pair table is
            // precomputed once (no per-index reverse_bits).
            unsafe fn colrev_row(row: *mut num_complex::Complex64, pairs: &[(usize, usize)]) {
                for &(c, rc) in pairs {
                    std::ptr::swap(row.add(c), row.add(rc));
                }
            }
            let col_pairs: Vec<(usize, usize)> = (0..dim)
                .map(|c| (c, rev_n(c, n)))
                .filter(|&(c, rc)| c < rc)
                .collect();
            let row_pairs: Vec<(usize, usize)> = (0..dim)
                .map(|r| (r, rev_n(r, n)))
                .filter(|&(r, rr)| r < rr)
                .collect();
            let p = DmPermPtrs {
                src: data.as_ptr(),
                dst: public.as_mut_ptr(),
            };
            row_pairs.par_iter().for_each_init(
                || vec![num_complex::Complex64::new(0.0, 0.0); dim],
                |tmp, &(r, rr)| unsafe {
                    let row_r = p.dst_row(r, dim);
                    let row_rr = p.dst_row(rr, dim);
                    // tmp = colrev(old row rr) = new row r
                    std::ptr::copy_nonoverlapping(row_rr, tmp.as_mut_ptr(), dim);
                    colrev_row(tmp.as_mut_ptr(), &col_pairs);
                    // row r = colrev(old row r) = new row rr
                    colrev_row(row_r, &col_pairs);
                    std::ptr::copy_nonoverlapping(row_r, row_rr, dim);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), row_r, dim);
                },
            );
            let fixed: Vec<usize> = (0..dim).filter(|&r| rev_n(r, n) == r).collect();
            fixed.par_iter().for_each(|&r| unsafe {
                colrev_row(p.dst_row(r, dim), &col_pairs);
            });
        }
        let t_p2 = t_p2.elapsed();
        if trace {
            eprintln!(
                "SF_DM_TRACE core={:.2}ms pass1(transpose)={:.2}ms pass23(rowswap+inrev)={:.2}ms",
                t_core.as_secs_f64() * 1e3,
                t_p1.as_secs_f64() * 1e3,
                t_p2.as_secs_f64() * 1e3,
            );
        }
        let arr = numpy::PyArray1::from_vec(py, public);
        // Adopt `data` directly: `DensityMatrixState::new(n)` would allocate
        // and zero-fill a fresh 4^n buffer that is then immediately
        // overwritten — a throwaway 1 GiB allocation + memset at n=13,
        // visible as the third full-size buffer in peak RSS while `data`
        // and `public` are both live.
        let dm = sf_ir::dm::DensityMatrixState::from_data(data, n);
        let state = PyState::new(Box::new(DensityMatrixStateWrapper::new(dm, "cpu")));
        Ok((state, arr))
    }

    /// Noisy density-matrix simulation returning the state handle together
    /// with a **view of its buffer**, instead of a second permuted copy.
    ///
    /// The engine buffer is permuted from the internal layout
    /// `data[ket | (bra << n)]` to the public row-major big-endian layout
    /// (`public[r][c] = rho_LE(rev(r), rev(c))`) in place: on flat indices the
    /// map is `phi(e) = rev(e & (dim-1)) * dim + rev(e >> n)`, an involution,
    /// so one pass swapping the pairs `(f, phi(f))` with `f < phi(f)`
    /// realises it with no destination buffer.  The buffer is then
    /// adopted by the state with the matching layout
    /// (`from_data_public`) and exposed to Python as a read-only 2-D numpy
    /// view whose base object is the state — so the handle and
    /// `metadata["density_matrix"]` share ONE 4^n buffer (peak DM memory
    /// halves versus `simulate_dm_noisy_state`, whose public rho is a second
    /// buffer).
    /// buffer).  Accepts the same `noise_2q_targets` ordered-pair list as
    /// `simulate_dm_noisy`.
    #[pyo3(signature = (noise_ops, noise_2q_ops=None, placement=None, noise_2q_targets=None))]
    fn simulate_dm_noisy_state_view<'py>(
        &self,
        py: Python<'py>,
        noise_ops: Vec<(usize, Vec<f64>)>,
        noise_2q_ops: Option<Vec<Vec<f64>>>,
        placement: Option<&str>,
        noise_2q_targets: Option<Vec<Option<(usize, usize)>>>,
    ) -> PyResult<(
        Bound<'py, PyState>,
        Bound<'py, numpy::PyArray2<num_complex::Complex64>>,
    )> {
        let trace = std::env::var_os("SF_DM_TRACE").is_some();
        let per_touch = placement_per_touch(placement)?;
        let noise_2q_ops = noise_2q_ops.unwrap_or_default();
        let noise_2q_targets = noise_2q_targets.unwrap_or_default();
        let t_core = std::time::Instant::now();
        let mut data = dm_noisy_core(
            &self.inner,
            &noise_ops,
            &noise_2q_ops,
            &noise_2q_targets,
            per_touch,
        );
        let t_core = t_core.elapsed();
        let n = self.inner.n_qubits;
        let dim = 1usize << n;
        debug_assert_eq!(data.len(), dim * dim);

        let t_perm = std::time::Instant::now();
        {
            let rev: Vec<usize> = (0..dim).map(|i| rev_n(i, n)).collect();
            // Raw base address as `usize`: a bare `*mut` is not `Sync`, so the
            // Rayon closure cannot capture the pointer itself (`usize` can).
            // SAFETY: the pairs form a partition of the flat index range into
            // 2-element orbits (phi is an involution), and each orbit is
            // swapped exactly once — by the iteration owning its lower member
            // `f` — so no two threads ever write the same element.
            let base = data.as_mut_ptr() as usize;
            (0..dim).into_par_iter().for_each(|r| {
                let ptr = base as *mut num_complex::Complex64;
                let rr = rev[r];
                for c in 0..dim {
                    let f = r * dim + c;
                    let g = rev[c] * dim + rr;
                    // Both (f, g) and (g, f) are visited; each pair is
                    // swapped exactly once, on the smaller-index side.
                    if f < g {
                        unsafe { std::ptr::swap(ptr.add(f), ptr.add(g)) };
                    }
                }
            });
        }
        let t_perm = t_perm.elapsed();
        if trace {
            eprintln!(
                "SF_DM_TRACE core={:.2}ms perm_inplace={:.2}ms buffers=1",
                t_core.as_secs_f64() * 1e3,
                t_perm.as_secs_f64() * 1e3,
            );
        }

        let ptr = data.as_ptr();
        let dm = sf_ir::dm::DensityMatrixState::from_data_public(data, n);
        let state = Bound::new(
            py,
            PyState::new(Box::new(DensityMatrixStateWrapper::new(dm, "cpu"))),
        )?;
        // SAFETY: `ptr` points into the buffer owned by `state`, which is set
        // as the array's base object (numpy keeps it alive); nothing mutates
        // or reallocates that buffer (no binding evolves an existing handle —
        // the accessors reject it) and the array is made read-only below.
        let view: numpy::ndarray::ArrayView2<'_, num_complex::Complex64> =
            unsafe { numpy::ndarray::ArrayView2::from_shape_ptr((dim, dim), ptr) };
        let arr =
            unsafe { numpy::PyArray2::borrow_from_array(&view, state.clone().into_any()) };
        // Read-only: a writable view could desynchronise the handle's rho.
        arr.readwrite().make_nonwriteable();
        Ok((state, arr))
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumDAG(n_qubits={}, n_cbits={}, depth={}, gates={}, params={})",
            self.inner.n_qubits,
            self.inner.n_cbits,
            self.inner.depth(),
            self.inner.gate_count(),
            self.inner.n_parameters(),
        )
    }
}

impl PyQuantumDAG {
    fn parse_gate(name: &str, params: &[Parameter]) -> PyResult<OpType> {
        match name.to_lowercase().as_str() {
            "h" => Ok(OpType::H),
            "x" => Ok(OpType::X),
            "y" => Ok(OpType::Y),
            "z" => Ok(OpType::Z),
            "s" => Ok(OpType::S),
            "sdg" => Ok(OpType::Sdg),
            "t" => Ok(OpType::T),
            "tdg" => Ok(OpType::Tdg),
            "sx" => Ok(OpType::SX),
            "sxdg" => Ok(OpType::SXdg),
            "id" => Ok(OpType::Id),
            "cx" | "cnot" => Ok(OpType::CNOT),
            "cz" => Ok(OpType::CZ),
            "cy" => Ok(OpType::CY),
            "swap" => Ok(OpType::SWAP),
            "iswap" => Ok(OpType::ISWAP),
            "ecr" => Ok(OpType::ECR),
            "ccx" => Ok(OpType::CCX),
            "cswap" => Ok(OpType::CSWAP),
            "rx" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Rx(theta))
            }
            "ry" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Ry(theta))
            }
            "rz" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Rz(theta))
            }
            "r1" => {
                let phi = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::R1(phi))
            }
            "p" => {
                let lam = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::P(lam))
            }
            "u" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                let phi = params.get(1).cloned().unwrap_or(Parameter::Const(0.0));
                let lam = params.get(2).cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::U(theta, phi, lam))
            }
            "cu" | "cu3" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                let phi = params.get(1).cloned().unwrap_or(Parameter::Const(0.0));
                let lam = params.get(2).cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Cu(theta, phi, lam))
            }
            "rzz" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Rzz(theta))
            }
            "rxx" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Rxx(theta))
            }
            "ryy" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::Ryy(theta))
            }
            "crx" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::CRx(theta))
            }
            "crz" => {
                let theta = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::CRz(theta))
            }
            "cp" | "cr1" | "cphase" => {
                let phi = params.first().cloned().unwrap_or(Parameter::Const(0.0));
                Ok(OpType::CP(phi))
            }
            "measure" => Ok(OpType::Measure),
            "barrier" => Ok(OpType::Barrier),
            "reset" => Ok(OpType::Reset),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown gate: '{}'. Available: H, X, Y, Z, S, T, SX, CX, CZ, SWAP, Rx, Ry, Rz, CP, CRx, CRz, Rzz, Rxx, Ryy, etc.",
                other
            ))),
        }
    }
}

/// Reverse the low `n` bits of `x` (qubit-order reversal of an index).
#[inline]
fn rev_n(x: usize, n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - n)
}

/// Shared core of the noisy density-matrix simulation: runs the gate list
/// with the pre-parsed 1q Kraus channels, fusing each gate with its
/// touching channels into single superoperator sweeps (one memory pass
/// instead of 1 + #touching passes).  Optional 2q channels are applied
/// sequentially after every 2-qubit gate (after the fused sweep); each
/// channel may carry an ordered wire pair via `noise_2q_targets` and then
/// fires only on gates emitted with exactly those qubits in that order.
/// Returns the interleaved DM vector.
/// Parse the optional `placement` argument of the noisy-DM bindings.
///
/// `None`/`"touch"` (default) — the 1q channels registered for a qubit are
/// applied after EVERY instruction touching that qubit, so a 2q gate picks up
/// both qubits' 1q channels (the historical Superfermion placement).
/// `"gate"` — 1q channels are applied only after 1q gates, which is the
/// Qiskit noise-model placement (`depolarizing_error(lam, 1)` on 1q
/// instructions and `(lam, 2)` on 2q instructions).
fn placement_per_touch(spec: Option<&str>) -> PyResult<bool> {
    match spec.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("touch") | Some("touching") | Some("per_touch") => Ok(true),
        Some("gate") | Some("per_gate") | Some("gate_type") => Ok(false),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown placement '{other}' (expected 'touch' or 'gate')"
        ))),
    }
}

fn dm_noisy_core(
    dag: &QuantumDAG,
    noise_ops: &[(usize, Vec<f64>)],
    noise_2q_ops: &[Vec<f64>],
    noise_2q_targets: &[Option<(usize, usize)>],
    per_touch: bool,
) -> Vec<num_complex::Complex64> {
    use sf_ir::dm::DensityMatrixState;
    let trace = std::env::var_os("SF_DM_TRACE").is_some();
    let (mut acc1, mut acc2) = (0.0f64, 0.0f64);
    let mut state = DensityMatrixState::new(dag.n_qubits);
    let instructions = dag.to_instructions();

    // Pre-parse each Kraus channel ONCE (it is re-applied after every
    // instruction that touches the qubit; re-parsing per instruction
    // was pure overhead on top of the DM pass itself).
    let kraus_channels: Vec<(usize, Vec<nalgebra::DMatrix<num_complex::Complex64>>)> = noise_ops
        .iter()
        .map(|(noise_qubit, kraus_flat)| {
            let matrices = kraus_flat
                .chunks(8)
                .map(|flat| {
                    let mut m = nalgebra::DMatrix::<num_complex::Complex64>::zeros(2, 2);
                    m[(0, 0)] = num_complex::Complex64::new(flat[0], flat[1]);
                    m[(0, 1)] = num_complex::Complex64::new(flat[2], flat[3]);
                    m[(1, 0)] = num_complex::Complex64::new(flat[4], flat[5]);
                    m[(1, 1)] = num_complex::Complex64::new(flat[6], flat[7]);
                    m
                })
                .collect();
            (*noise_qubit, matrices)
        })
        .collect();

    // 2q channels: each entry is ONE channel's flat Kraus set (4x4 matrices,
    // 32 floats each, row-major re/im pairs) plus its optional ordered wire
    // pair. Applied in add-order after every 2q gate, after the fused
    // gate+1q sweep.  A `None` target fires on every 2q gate; a pair fires
    // only when the instruction's qubits match it in order (Qiskit
    // `add_quantum_error` parity — an error on (0, 1) is silent on
    // `cx(1, 0)`).
    let kraus_2q: Vec<(Option<(usize, usize)>, Vec<nalgebra::DMatrix<num_complex::Complex64>>)> =
        noise_2q_ops
            .iter()
            .enumerate()
            .map(|(idx, flat)| {
                let target = noise_2q_targets.get(idx).copied().flatten();
                let matrices = flat
                    .chunks(32)
                    .map(|ch| {
                        let mut m = nalgebra::DMatrix::<num_complex::Complex64>::zeros(4, 4);
                        for r in 0..4 {
                            for c in 0..4 {
                                let base = 2 * (r * 4 + c);
                                m[(r, c)] =
                                    num_complex::Complex64::new(ch[base], ch[base + 1]);
                            }
                        }
                        m
                    })
                    .collect();
                (target, matrices)
            })
            .collect();

    // A run of consecutive noisy 1-qubit gates on pairwise-distinct qubits
    // shares one sweep per pair: the two fused 4×4 channel superoperators
    // are applied per joint 16-block, halving the memory traffic of two
    // sequential sweeps with identical per-element arithmetic. The pending
    // batch is flushed before any instruction that touches one of its
    // qubits (or is not a plain 1-qubit gate), so program order and
    // semantics are unchanged.
    type Sup4 = [[num_complex::Complex64; 4]; 4];
    fn flush_pending(state: &mut DensityMatrixState, pending: &mut Vec<(usize, Sup4)>) -> f64 {
        let t = std::time::Instant::now();
        match pending.len() {
            0 => {}
            1 => state.apply_1q_superop(pending[0].0, &pending[0].1),
            _ => state.apply_1q_superop_pair(
                pending[0].0,
                &pending[0].1,
                pending[1].0,
                &pending[1].1,
            ),
        }
        pending.clear();
        t.elapsed().as_secs_f64()
    }
    let mut pending: Vec<(usize, Sup4)> = Vec::with_capacity(2);

    for inst in &instructions {
        let u = inst.op_type.to_matrix();
        // Collect the Kraus channels that touch this instruction's
        // qubits — when any do, apply gate+noise in ONE fused
        // superoperator sweep (one memory pass instead of 1+N).
        //
        // Placement (`per_touch`): the default applies a qubit's 1q channels
        // after every instruction touching that qubit (so a 2q gate picks up
        // both qubits' 1q channels); `placement="gate"` applies them only
        // after 1q gates — a 2q gate then gets its 2q channels and nothing
        // else (Qiskit-noise-model parity).
        let touching: Vec<(usize, &[nalgebra::DMatrix<num_complex::Complex64>])> =
            if per_touch || inst.qubits.len() == 1 {
                kraus_channels
                    .iter()
                    .filter(|(noise_qubit, _)| inst.qubits.contains(noise_qubit))
                    .map(|(noise_qubit, kraus_matrices)| (*noise_qubit, kraus_matrices.as_slice()))
                    .collect()
            } else {
                Vec::new()
            };

        // Plain noisy 1-qubit gate → batch (a second disjoint gate joins it
        // in one sweep; a repeat/other instruction flushes first).
        if inst.qubits.len() == 1 && !touching.is_empty() && !inst.op_type.is_measurement() {
            let q = inst.qubits[0];
            if pending.iter().any(|(pq, _)| *pq == q) {
                let dt = flush_pending(&mut state, &mut pending);
                if trace {
                    acc1 += dt;
                }
            }
            let fused: Vec<nalgebra::DMatrix<num_complex::Complex64>> =
                touching[0].1.iter().map(|k| k * &u).collect();
            pending.push((q, DensityMatrixState::kraus_superop_1q(&fused)));
            if pending.len() == 2 {
                let dt = flush_pending(&mut state, &mut pending);
                if trace {
                    acc1 += dt;
                }
            }
            continue;
        }
        let dt = flush_pending(&mut state, &mut pending);
        if trace {
            acc1 += dt;
        }

        if touching.is_empty() {
            let t = std::time::Instant::now();
            state.apply_unitary(&u, &inst.qubits);
            if trace {
                if inst.qubits.len() == 1 {
                    acc1 += t.elapsed().as_secs_f64();
                } else {
                    acc2 += t.elapsed().as_secs_f64();
                }
            }
        } else {
            let t = std::time::Instant::now();
            state.apply_gate_noise_fused(&u, &inst.qubits, &touching);
            if trace {
                if inst.qubits.len() == 1 {
                    acc1 += t.elapsed().as_secs_f64();
                } else {
                    acc2 += t.elapsed().as_secs_f64();
                }
            }
        }
        // 2q channels: applied after every 2q (non-measurement) instruction,
        // after the fused gate+1q sweep — add-order sequential composition.
        // A targeted channel only fires on its exact ordered pair.
        if !kraus_2q.is_empty() && inst.qubits.len() == 2 && !inst.op_type.is_measurement() {
            let pair = (inst.qubits[0], inst.qubits[1]);
            let t = std::time::Instant::now();
            for (target, ks) in &kraus_2q {
                if let Some(want) = target {
                    if *want != pair {
                        continue;
                    }
                }
                state.apply_kraus_2q(ks, inst.qubits[0], inst.qubits[1]);
            }
            if trace {
                acc2 += t.elapsed().as_secs_f64();
            }
        }
    }
    let dt = flush_pending(&mut state, &mut pending);
    if trace {
        acc1 += dt;
    }
    if trace {
        eprintln!(
            "SF_DM_TRACE sweeps sum_1q={:.2}ms sum_2q={:.2}ms",
            acc1 * 1e3,
            acc2 * 1e3
        );
    }
    state.into_data()
}

// ═══════════════════════════════════════════════════════════
// Compiler Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around the Rust compiler.
#[pyclass(name = "Compiler")]
pub struct PyCompiler {
    backend: sf_compiler::BackendSpec,
}

#[pymethods]
impl PyCompiler {
    #[new]
    #[pyo3(signature = (name, native_gates, n_qubits, connectivity, optimization_level=None))]
    fn new(
        name: &str,
        native_gates: Vec<String>,
        n_qubits: usize,
        connectivity: Vec<(usize, usize)>,
        optimization_level: Option<u8>,
    ) -> Self {
        PyCompiler {
            backend: sf_compiler::BackendSpec {
                name: name.to_string(),
                native_gates,
                connectivity,
                n_qubits,
                optimization_level: optimization_level.unwrap_or(1),
            },
        }
    }

    /// Compile a circuit for the target backend.
    fn compile(&self, dag: &PyQuantumDAG) -> PyResult<PyQuantumDAG> {
        let compiler = sf_compiler::Compiler::new(self.backend.clone());
        let result = compiler
            .compile(&dag.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyQuantumDAG { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "Compiler(backend='{}', n_qubits={}, gates={:?})",
            self.backend.name, self.backend.n_qubits, self.backend.native_gates,
        )
    }
}

// ═══════════════════════════════════════════════════════════
// Router Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around a hardware coupling map.
#[pyclass(name = "CouplingMap")]
pub struct PyCouplingMap {
    inner: sf_router::CouplingMap,
}

#[pymethods]
impl PyCouplingMap {
    /// Create a linear coupling map: 0—1—2—…—(n-1).
    #[staticmethod]
    fn linear(n: usize) -> Self {
        PyCouplingMap {
            inner: sf_router::CouplingMap::linear(n),
        }
    }

    /// Create a grid coupling map (rows × cols).
    #[staticmethod]
    fn grid(rows: usize, cols: usize) -> Self {
        PyCouplingMap {
            inner: sf_router::CouplingMap::grid(rows, cols),
        }
    }

    /// Create an all-to-all coupling map.
    #[staticmethod]
    fn all_to_all(n: usize) -> Self {
        PyCouplingMap {
            inner: sf_router::CouplingMap::all_to_all(n),
        }
    }

    /// Create from edge list.
    #[staticmethod]
    fn from_edges(n_qubits: usize, edges: Vec<(usize, usize)>) -> Self {
        PyCouplingMap {
            inner: sf_router::CouplingMap::from_edges(n_qubits, &edges),
        }
    }

    /// Predefined IBM Eagle topology.
    #[staticmethod]
    fn ibm_eagle() -> Self {
        PyCouplingMap {
            inner: sf_router::HardwareTopology::ibm_eagle(),
        }
    }

    /// Predefined IonQ Forte topology (all-to-all, 36 qubits).
    #[staticmethod]
    fn ionq_forte() -> Self {
        PyCouplingMap {
            inner: sf_router::HardwareTopology::ionq_forte(),
        }
    }

    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    fn is_connected(&self, a: usize, b: usize) -> bool {
        self.inner.is_connected(a, b)
    }

    fn distance(&self, a: usize, b: usize) -> usize {
        self.inner.distance(a, b)
    }

    /// Get all edges as a list of (qubit_a, qubit_b) pairs.
    fn edges(&self) -> Vec<(usize, usize)> {
        self.inner.edges()
    }

    /// Create a heavy-hex coupling map with n qubits.
    #[staticmethod]
    fn heavy_hex(n: usize) -> Self {
        PyCouplingMap {
            inner: sf_router::CouplingMap::heavy_hex(n),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CouplingMap(n_qubits={}, n_edges={})",
            self.inner.n_qubits(),
            self.inner.n_edges(),
        )
    }
}

/// Python wrapper around the SABRE router.
#[pyclass(name = "Router")]
pub struct PyRouter {
    inner: sf_router::Router,
}

#[pymethods]
impl PyRouter {
    #[new]
    fn new(coupling: &PyCouplingMap) -> Self {
        PyRouter {
            inner: sf_router::Router::new(coupling.inner.clone()),
        }
    }

    /// Route a circuit for the hardware topology.
    fn route(&self, dag: &PyQuantumDAG) -> PyResult<PyQuantumDAG> {
        let (routed, _mapping) = self
            .inner
            .route(&dag.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyQuantumDAG { inner: routed })
    }

    fn __repr__(&self) -> String {
        format!(
            "Router(topology=CouplingMap(n_qubits={}))",
            self.inner.topology.n_qubits(),
        )
    }
}

// ═══════════════════════════════════════════════════════════
// Pulse Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around pulse schedules.
#[pyclass(name = "PulseSchedule")]
pub struct PyPulseSchedule {
    inner: sf_pulse::PulseSchedule,
}

#[pymethods]
impl PyPulseSchedule {
    #[new]
    fn new(name: &str) -> Self {
        PyPulseSchedule {
            inner: sf_pulse::PulseSchedule::new(name),
        }
    }

    /// Play a Gaussian pulse on a drive channel.
    fn play_gaussian(&mut self, qubit: usize, duration: usize, sigma: f64, amp: f64) {
        self.inner.play(
            sf_pulse::waveforms::PulseEnvelope::gaussian(duration, sigma, amp),
            sf_pulse::ChannelId::drive(qubit),
            None,
        );
    }

    /// Play a DRAG pulse on a drive channel.
    fn play_drag(&mut self, qubit: usize, duration: usize, sigma: f64, amp: f64, beta: f64) {
        self.inner.play(
            sf_pulse::waveforms::PulseEnvelope::drag(duration, sigma, amp, beta),
            sf_pulse::ChannelId::drive(qubit),
            None,
        );
    }

    /// Play a square pulse.
    fn play_square(&mut self, qubit: usize, duration: usize, amp: f64) {
        self.inner.play(
            sf_pulse::waveforms::PulseEnvelope::square(duration, amp),
            sf_pulse::ChannelId::drive(qubit),
            None,
        );
    }

    /// Barrier all channels.
    fn barrier(&mut self) {
        self.inner.barrier();
    }

    fn duration(&self) -> usize {
        self.inner.duration()
    }

    fn n_instructions(&self) -> usize {
        self.inner.n_instructions()
    }

    fn channels(&self) -> Vec<String> {
        self.inner.channels()
    }

    fn __repr__(&self) -> String {
        format!(
            "PulseSchedule(name='{}', duration={}, instructions={})",
            self.inner.name,
            self.inner.duration(),
            self.inner.n_instructions(),
        )
    }
}

// ═══════════════════════════════════════════════════════════
// QEC Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around QEC codes.
#[pyclass(name = "QECCode")]
pub struct PyQECCode {
    code_type: String,
    n_data: usize,
    n_ancilla: usize,
    distance: usize,
}

#[pymethods]
impl PyQECCode {
    /// Create a repetition code.
    #[staticmethod]
    fn repetition(n: usize) -> Self {
        use sf_qec::codes::StabilizerCode;
        let code = sf_qec::RepetitionCode::new(n);
        PyQECCode {
            code_type: "repetition".to_string(),
            n_data: code.n_data(),
            n_ancilla: code.n_ancilla(),
            distance: code.distance(),
        }
    }

    /// Create a surface code.
    #[staticmethod]
    fn surface(distance: usize) -> Self {
        use sf_qec::codes::StabilizerCode;
        let code = sf_qec::SurfaceCode::new(distance);
        PyQECCode {
            code_type: "surface".to_string(),
            n_data: code.n_data(),
            n_ancilla: code.n_ancilla(),
            distance: code.distance(),
        }
    }

    /// Create a Steane [[7,1,3]] code.
    #[staticmethod]
    fn steane() -> Self {
        use sf_qec::codes::StabilizerCode;
        let code = sf_qec::SteaneCode::new();
        PyQECCode {
            code_type: "steane".to_string(),
            n_data: code.n_data(),
            n_ancilla: code.n_ancilla(),
            distance: code.distance(),
        }
    }

    fn code_type(&self) -> &str {
        &self.code_type
    }

    fn n_data(&self) -> usize {
        self.n_data
    }

    fn n_ancilla(&self) -> usize {
        self.n_ancilla
    }

    fn distance(&self) -> usize {
        self.distance
    }

    fn total_qubits(&self) -> usize {
        self.n_data + self.n_ancilla
    }

    fn __repr__(&self) -> String {
        format!(
            "QECCode(type='{}', n_data={}, n_ancilla={}, distance={})",
            self.code_type, self.n_data, self.n_ancilla, self.distance,
        )
    }
}

// ═══════════════════════════════════════════════════════════
// QEC Decoder Bindings
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "MWPMDecoder")]
pub struct PyMWPMDecoder {
    inner: sf_qec::decoders::MWPMDecoder,
}

#[pymethods]
impl PyMWPMDecoder {
    #[new]
    fn new(n_data: usize, syndrome_qubit_map: Vec<Vec<usize>>) -> Self {
        PyMWPMDecoder {
            inner: sf_qec::decoders::MWPMDecoder::new(n_data, syndrome_qubit_map),
        }
    }

    #[staticmethod]
    fn for_repetition(n: usize) -> Self {
        PyMWPMDecoder {
            inner: sf_qec::decoders::MWPMDecoder::for_repetition(n),
        }
    }

    fn decode(&self, syndrome: Vec<u8>) -> Vec<(usize, String)> {
        let correction = sf_qec::decoders::Decoder::decode(&self.inner, &syndrome);
        correction
            .corrections
            .into_iter()
            .map(|(q, t)| {
                let type_str = match t {
                    sf_qec::decoders::CorrectionType::X => "X",
                    sf_qec::decoders::CorrectionType::Y => "Y",
                    sf_qec::decoders::CorrectionType::Z => "Z",
                };
                (q, type_str.to_string())
            })
            .collect()
    }
}

#[pyclass(name = "UnionFindDecoder")]
pub struct PyUnionFindDecoder {
    inner: sf_qec::decoders::UnionFindDecoder,
}

#[pymethods]
impl PyUnionFindDecoder {
    #[new]
    fn new(n_data: usize, syndrome_qubit_map: Vec<Vec<usize>>) -> Self {
        PyUnionFindDecoder {
            inner: sf_qec::decoders::UnionFindDecoder::new(n_data, syndrome_qubit_map),
        }
    }

    #[staticmethod]
    fn for_repetition(n: usize) -> Self {
        PyUnionFindDecoder {
            inner: sf_qec::decoders::UnionFindDecoder::for_repetition(n),
        }
    }

    fn decode(&self, syndrome: Vec<u8>) -> Vec<(usize, String)> {
        let correction = sf_qec::decoders::Decoder::decode(&self.inner, &syndrome);
        correction
            .corrections
            .into_iter()
            .map(|(q, t)| {
                let type_str = match t {
                    sf_qec::decoders::CorrectionType::X => "X",
                    sf_qec::decoders::CorrectionType::Y => "Y",
                    sf_qec::decoders::CorrectionType::Z => "Z",
                };
                (q, type_str.to_string())
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
// Stabilizer Tableau Bindings
// ═══════════════════════════════════════════════════════════

/// Python wrapper around the Rust StabilizerTableau (Aaronson-Gottesman 2004).
/// Fast Clifford simulation — O(n) per gate, O(n³) sampling.
#[pyclass(name = "StabilizerTableau")]
pub struct PyStabilizerTableau {
    inner: sf_ir::stabilizer::StabilizerTableau,
}

#[pymethods]
impl PyStabilizerTableau {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        Ok(PyStabilizerTableau {
            inner: sf_ir::stabilizer::StabilizerTableau::new(n),
        })
    }

    /// Number of qubits.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Apply a Clifford gate by name (H, S, SDG, X, Y, Z, CX, CNOT, CZ, CY, SWAP).
    fn apply_gate(&mut self, name: &str, qubits: Vec<usize>) -> PyResult<()> {
        self.inner
            .apply_gate(name, &qubits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Sample `shots` bitstrings from the stabilizer state.
    #[pyo3(signature = (shots, seed=None))]
    fn sample(&self, shots: usize, seed: Option<u64>) -> std::collections::HashMap<String, usize> {
        self.inner.sample(shots, seed)
    }

    /// Export tableau data as numpy arrays: (x, z, r) where x/z are (2n, n) uint8.
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<u8>>,
        Bound<'py, numpy::PyArray2<u8>>,
        Bound<'py, numpy::PyArray1<u8>>,
    )> {
        let (x_flat, z_flat, r_flat) = self.inner.to_raw();
        let n = self.inner.n;
        let x_arr = numpy::PyArray1::from_vec(py, x_flat).reshape([2 * n, n])?;
        let z_arr = numpy::PyArray1::from_vec(py, z_flat).reshape([2 * n, n])?;
        let r_arr = numpy::PyArray1::from_vec(py, r_flat);
        Ok((x_arr, z_arr, r_arr))
    }

    /// Compute <psi|P|psi> for a Pauli string encoded as (px, pz) uint8 vectors.
    fn pauli_expval(&self, px: Vec<u8>, pz: Vec<u8>) -> f64 {
        self.inner.pauli_expval(&px, &pz)
    }

    /// Build a tableau by evolving from a gate list: [(name, [qubits]), ...].
    #[staticmethod]
    fn from_gate_list(n: usize, gates: Vec<(String, Vec<usize>)>) -> PyResult<Self> {
        let inner = sf_ir::stabilizer::StabilizerTableau::from_gate_list(n, &gates)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(PyStabilizerTableau { inner })
    }

    fn __repr__(&self) -> String {
        format!("StabilizerTableau(n={})", self.inner.n)
    }
}

/// Apply Pauli twirling to a list of (gate_name, qubits) tuples.
/// Only wraps CX/CNOT/CZ gates in random Pauli sandwiches.
/// Returns a new gate list — does not mutate the input.
#[pyfunction]
fn pauli_twirl(gates: Vec<(String, Vec<usize>)>, seed: u64) -> Vec<(String, Vec<usize>)> {
    sf_ir::stabilizer::pauli_twirl_gate_list(&gates, seed)
}

/// Apply Pauli twirling directly to an SF Circuit, extracting gate data in Rust.
/// Avoids Python-side tuple allocation by accessing GateRecord attributes via PyO3.
/// Returns a list of (name, qubits, params) ready for Circuit construction.
#[pyfunction]
fn pauli_twirl_circuit(
    _py: pyo3::Python<'_>,
    circuit: &pyo3::Bound<'_, pyo3::PyAny>,
    seed: u64,
) -> PyResult<Vec<(String, Vec<usize>, Vec<f64>)>> {
    // Get circuit._gates (list of GateRecord)
    let gates = circuit.getattr("_gates")?;
    let gate_list: &pyo3::Bound<'_, pyo3::types::PyList> = gates.downcast()?;
    let n_gates = gate_list.len();

    // Fast extraction: batch-extract all fields into Rust Vecs
    let mut records: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::with_capacity(n_gates);
    for gate in gate_list.iter() {
        // Extract name (as_str or via __str__)
        let name_attr = gate.getattr("name")?;
        let name: String = if let Ok(s) = name_attr.extract::<String>() {
            s.to_uppercase()
        } else {
            name_attr.str()?.to_string().to_uppercase()
        };

        // Extract qubits list
        let qubits_attr = gate.getattr("qubits")?;
        let qubits_list: &pyo3::Bound<'_, pyo3::types::PyList> = qubits_attr.downcast()?;
        let qubits: Vec<usize> = qubits_list
            .iter()
            .map(|q| q.extract::<usize>().unwrap_or(0))
            .collect();

        // Extract params (optional, may be empty list)
        let params_attr = gate.getattr("params")?;
        let params_list: &pyo3::Bound<'_, pyo3::types::PyList> = params_attr.downcast()?;
        let params: Vec<f64> = params_list
            .iter()
            .map(|p| p.extract::<f64>().unwrap_or(0.0))
            .collect();

        records.push((name, qubits, params));
    }

    // Apply Pauli twirling
    Ok(sf_ir::stabilizer::pauli_twirl_gate_records(&records, seed))
}

// ═══════════════════════════════════════════════════════════
// GateSequence Bindings (lightweight flat gate list)
// ═══════════════════════════════════════════════════════════

/// Python wrapper around the Rust GateSequence — a lightweight flat array
/// gate store with zero per-gate Python heap overhead.  For memory
/// benchmarks this matters: Rust Vecs are invisible to Python tracemalloc.
#[pyclass(name = "GateSequence")]
pub struct PyGateSequence {
    pub inner: GateSequence,
}

#[pymethods]
impl PyGateSequence {
    #[new]
    fn new(n_qubits: usize, n_cbits: usize) -> Self {
        PyGateSequence {
            inner: GateSequence::new(n_qubits, n_cbits),
        }
    }

    /// Pre-allocate for expected gate count (avoids Vec reallocations).
    #[staticmethod]
    fn with_capacity(n_qubits: usize, n_cbits: usize, expected_gates: usize) -> Self {
        PyGateSequence {
            inner: GateSequence::with_capacity(n_qubits, n_cbits, expected_gates),
        }
    }

    /// Build a Quantum Volume circuit entirely in Rust — zero per-gate FFI.
    ///
    /// Python passes numpy arrays for permutations (u64) and angles (f64).
    /// A single Rust call replaces 25,000 `add_gate()` crossings for QV100.
    #[staticmethod]
    fn from_qv_circuit(
        n_qubits: usize,
        n_cbits: usize,
        depth: usize,
        perms: Vec<u64>,
        angles: Vec<f64>,
    ) -> Self {
        PyGateSequence {
            inner: GateSequence::from_qv_batch(n_qubits, n_cbits, depth, &perms, &angles),
        }
    }

    /// Add a single gate.  Gate name is case-insensitive (uppercased).
    fn add_gate(&mut self, name: &str, qubits: Vec<usize>, params: Vec<f64>) {
        self.inner.push(&name.to_uppercase(), &qubits, &params);
    }

    /// Add an opaque unitary matrix gate.
    /// The matrix is stored in Rust without decomposition and emitted as
    /// OpType::Unitary when converting to a QuantumDAG via to_dag().
    fn add_unitary(
        &mut self,
        qubits: Vec<usize>,
        matrix: numpy::PyReadonlyArray2<num_complex::Complex64>,
    ) -> PyResult<()> {
        let shape = matrix.shape();
        let rows = shape[0];
        let cols = shape[1];
        if rows != cols || !rows.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unitary matrix must be square with power-of-2 dimension, got {}x{}",
                rows, cols
            )));
        }
        let slice = matrix.as_slice()?;
        let dm = nalgebra::DMatrix::from_row_slice(rows, cols, slice);
        self.inner.add_unitary(&qubits, dm);
        Ok(())
    }

    /// Batch-extend from a list of (name, qubits, params) tuples.
    /// Each tuple: (str, list[int], list[float]).
    fn extend(&mut self, records: Vec<(String, Vec<usize>, Vec<f64>)>) {
        self.inner.extend(&records);
    }

    /// Number of gates.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of gates (alias).
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Number of qubits.
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    /// Number of classical bits.
    fn n_cbits(&self) -> usize {
        self.inner.n_cbits
    }

    /// Export as Python-compatible list of (name, qubits, params) tuples.
    fn to_gate_records(&self) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        self.inner.to_gate_records()
    }

    /// Convert to a QuantumDAG for compilation.
    fn to_dag(&self) -> PyQuantumDAG {
        PyQuantumDAG {
            inner: self.inner.to_dag(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GateSequence(n_qubits={}, n_gates={})",
            self.inner.n_qubits,
            self.inner.len(),
        )
    }

    /// Pauli-twirl this gate sequence (all in Rust, zero Python heap).
    /// Returns a new GateSequence with Pauli sandwiches around CX/CZ gates.
    fn pauli_twirl(&self, seed: u64) -> Self {
        // Pauli names: 0=I, 1=X, 2=Z, 3=Y
        let pauli = ["I", "X", "Z", "Y"];

        // 14 valid CNOT Pauli pairs: (p1_before, p2_before, p1_after, p2_after)
        let cnot_pairs: [(usize, usize, usize, usize); 14] = [
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (0, 2, 2, 2),
            (0, 3, 2, 3),
            (1, 0, 1, 1),
            (1, 1, 1, 0),
            (1, 3, 3, 2),
            (2, 0, 2, 0),
            (2, 1, 2, 1),
            (2, 2, 0, 2),
            (2, 3, 0, 3),
            (3, 0, 3, 1),
            (3, 1, 3, 0),
            (3, 2, 1, 3),
        ];
        // 14 valid CZ Pauli pairs
        let cz_pairs: [(usize, usize, usize, usize); 14] = [
            (0, 0, 0, 0),
            (0, 1, 2, 1),
            (0, 2, 0, 2),
            (0, 3, 2, 3),
            (1, 0, 1, 2),
            (1, 1, 3, 3),
            (1, 2, 1, 0),
            (2, 0, 2, 0),
            (2, 1, 0, 1),
            (2, 2, 2, 2),
            (2, 3, 0, 3),
            (3, 0, 3, 2),
            (3, 2, 3, 0),
            (3, 3, 1, 1),
        ];

        let n = self.inner.len();
        let mut out = GateSequence::with_capacity(self.inner.n_qubits, self.inner.n_cbits, n * 3);

        // Simple xorshift64 RNG (fast, deterministic from seed)
        let mut state: u64 = seed ^ 0x5DEECE66D;
        if state == 0 {
            state = 1;
        }
        let mut next_u64 = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for (name, qubits, params) in self.inner.iter() {
            if qubits.len() == 2 && (name == "CX" || name == "CNOT" || name == "CZ") {
                let q0 = qubits[0];
                let q1 = qubits[1];
                let pairs = if name == "CZ" { &cz_pairs } else { &cnot_pairs };
                let idx = (next_u64() % 14) as usize;
                let (p1b, p2b, p1a, p2a) = pairs[idx];

                if p1b != 0 {
                    out.push(pauli[p1b], &[q0], &[]);
                }
                if p2b != 0 {
                    out.push(pauli[p2b], &[q1], &[]);
                }

                out.push(name, qubits, params);

                if p1a != 0 {
                    out.push(pauli[p1a], &[q0], &[]);
                }
                if p2a != 0 {
                    out.push(pauli[p2a], &[q1], &[]);
                }
            } else {
                out.push(name, qubits, params);
            }
        }

        PyGateSequence { inner: out }
    }
}

/// Reverse the bit order of an integer with n_bits bits.
/// Used for MSB↔LSB endianness conversion.
fn reverse_bits(val: usize, n_bits: usize) -> usize {
    let mut result = 0usize;
    let mut v = val;
    for _ in 0..n_bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// ═══════════════════════════════════════════════════════════
// Standalone Pauli expval on existing statevector
// ═══════════════════════════════════════════════════════════

/// Compute weighted Pauli expectation value on an existing statevector.
///
/// Takes a little-endian statevector and a list of (paulis_u8, coef_re, coef_im)
/// terms, and returns the real part of sum_k coef_k * <sv|P_k|sv>.
///
/// Pauli encoding per qubit: 0=I, 1=X, 2=Y, 3=Z.
/// The qubit ordering in each Pauli list is q0-first (paulis[q] = qubit q,
/// and qubit q lives at bit position q of the statevector index).
#[pyfunction]
fn hamiltonian_expval(
    sv: numpy::PyReadonlyArray1<num_complex::Complex64>,
    terms: Vec<(Vec<u8>, f64, f64)>,
) -> f64 {
    let sv = sv.as_slice().unwrap();
    let dim = sv.len();
    let mut total = num_complex::Complex64::new(0.0, 0.0);

    for (paulis, coef_re, coef_im) in &terms {
        let coef = num_complex::Complex64::new(*coef_re, *coef_im);
        let mut expval = num_complex::Complex64::new(0.0, 0.0);

        for i in 0..dim {
            let mut phase = num_complex::Complex64::new(1.0, 0.0);
            let mut target_idx = i;

            for (q, &pauli_op) in paulis.iter().enumerate() {
                let bit_pos = q;
                match pauli_op {
                    1 => {
                        // X
                        target_idx ^= 1 << bit_pos;
                    }
                    2 => {
                        // Y
                        target_idx ^= 1 << bit_pos;
                        if (i >> bit_pos) & 1 == 0 {
                            phase *= num_complex::Complex64::new(0.0, -1.0);
                        } else {
                            phase *= num_complex::Complex64::new(0.0, 1.0);
                        }
                    }
                    3 if (i >> bit_pos) & 1 == 1 => {
                        // Z
                        phase *= -1.0;
                    }
                    3 => {}
                    _ => {} // I
                }
            }
            expval += sv[i].conj() * phase * sv[target_idx];
        }
        total += coef * expval;
    }
    total.re
}

// ═══════════════════════════════════════════════════════════
// Module Registration
// ═══════════════════════════════════════════════════════════

/// The Python module exposed to Python.
#[pymodule]
fn _sf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // IR
    m.add_class::<PyQuantumDAG>()?;
    m.add_class::<PyGateSequence>()?;
    m.add_class::<PyMPSState>()?;
    m.add_class::<PyState>()?;

    // Stabilizer (Clifford simulation + Pauli twirl)
    m.add_class::<PyStabilizerTableau>()?;
    m.add_function(wrap_pyfunction!(pauli_twirl, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_twirl_circuit, m)?)?;

    // Compiler
    m.add_class::<PyCompiler>()?;

    // Router
    m.add_class::<PyCouplingMap>()?;
    m.add_class::<PyRouter>()?;

    // Pulse
    m.add_class::<PyPulseSchedule>()?;

    // QEC
    m.add_class::<PyQECCode>()?;
    m.add_class::<PyMWPMDecoder>()?;
    m.add_class::<PyUnionFindDecoder>()?;

    // GPU availability check and diagnostics
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_diagnose, m)?)?;

    // Standalone compute on existing statevectors
    m.add_function(wrap_pyfunction!(hamiltonian_expval, m)?)?;

    // Process memory introspection (memory-efficiency instrumentation)
    m.add_function(wrap_pyfunction!(rss_bytes, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Module-level GPU availability check.
#[pyfunction]
fn gpu_available() -> bool {
    QuantumDAG::gpu_available()
}

/// Diagnostic: returns why GPU init failed (or "ok").
#[pyfunction]
fn gpu_diagnose() -> String {
    #[cfg(feature = "gpu")]
    {
        sf_ir::gpu_diagnose()
    }
    #[cfg(not(feature = "gpu"))]
    {
        "GPU feature not compiled".to_string()
    }
}

/// Resident set size of the current process in bytes (0 if unavailable).
#[pyfunction]
fn rss_bytes() -> usize {
    sf_ir::sysmem::rss_bytes()
}
