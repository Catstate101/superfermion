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

use pyo3::prelude::*;
use numpy::PyArrayMethods;
use sf_ir::{QuantumDAG, OpType, Parameter, SerializedCircuit, MPSState};
use sf_ir::gate_list::GateSequence;

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

    fn __repr__(&self) -> String {
        format!(
            "MPSState(n_qubits={}, bond_dim={})",
            self.inner.n_qubits, self.inner.bond_dim,
        )
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
    fn add_gate(&mut self, gate_name: &str, qubits: Vec<usize>, params: Vec<pyo3::Bound<'_, pyo3::PyAny>>) -> PyResult<()> {
        let rust_params: Vec<Parameter> = params.into_iter().map(|p| {
            if let Ok(f) = p.extract::<f64>() {
                Parameter::Const(f)
            } else if let Ok(s) = p.extract::<String>() {
                Parameter::Variable { name: s, id: 0 }
            } else {
                Parameter::Const(0.0)
            }
        }).collect();
        
        let op = Self::parse_gate(gate_name, &rust_params)?;
        self.inner.add_op(op, &qubits);
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
    fn to_unitary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<num_complex::Complex64>>> {
        let u = self.inner.to_unitary();
        let (r, c) = u.shape();
        // nalgebra matrices are column-major by default
        let flat: Vec<num_complex::Complex64> = u.as_slice().to_vec();
        let arr = numpy::PyArray1::from_vec(py, flat);
        let arr2 = arr.reshape([c, r])?.call_method0("transpose")?;
        Ok(arr2.downcast_into::<numpy::PyArray2<num_complex::Complex64>>()?)
    }

    /// Run high-performance statevector simulation and return the final statevector.
    fn simulate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let state = self.inner.simulate();
        Ok(numpy::PyArray1::from_vec(py, state))
    }

    /// Run high-performance density-matrix simulation and return the vectorized DM.
    fn simulate_dm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let state = self.inner.simulate_dm();
        Ok(numpy::PyArray1::from_vec(py, state))
    }

    /// Compute batched Pauli expectation values for statevector.
    fn simulate_pauli_expval_batch(&self, paulis: Vec<Vec<u8>>) -> Vec<f64> {
        self.inner.simulate_pauli_expval_batch(paulis)
    }

    /// Run high-performance MPS simulation and return counts directly.
    fn sample_mps(&self, bond_dim: usize, shots: usize, seed: u64) -> std::collections::HashMap<String, usize> {
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
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, numpy::PyArray2<u8>>, Bound<'py, numpy::PyArray2<u8>>, Bound<'py, numpy::PyArray1<u8>>)> {
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
fn pauli_twirl_circuit(_py: pyo3::Python<'_>, circuit: &pyo3::Bound<'_, pyo3::PyAny>, seed: u64) -> PyResult<Vec<(String, Vec<usize>, Vec<f64>)>> {
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
        let cnot_pairs: [(usize,usize,usize,usize); 14] = [
            (0,0,0,0),(0,1,0,1),(0,2,2,2),(0,3,2,3),
            (1,0,1,1),(1,1,1,0),(1,3,3,2),
            (2,0,2,0),(2,1,2,1),(2,2,0,2),(2,3,0,3),
            (3,0,3,1),(3,1,3,0),(3,2,1,3),
        ];
        // 14 valid CZ Pauli pairs
        let cz_pairs: [(usize,usize,usize,usize); 14] = [
            (0,0,0,0),(0,1,2,1),(0,2,0,2),(0,3,2,3),
            (1,0,1,2),(1,1,3,3),(1,2,1,0),
            (2,0,2,0),(2,1,0,1),(2,2,2,2),(2,3,0,3),
            (3,0,3,2),(3,2,3,0),(3,3,1,1),
        ];

        let n = self.inner.len();
        let mut out = GateSequence::with_capacity(
            self.inner.n_qubits, self.inner.n_cbits, n * 3,
        );

        // Simple xorshift64 RNG (fast, deterministic from seed)
        let mut state: u64 = seed ^ 0x5DEECE66D;
        if state == 0 { state = 1; }
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

                if p1b != 0 { out.push(pauli[p1b], &[q0], &[]); }
                if p2b != 0 { out.push(pauli[p2b], &[q1], &[]); }

                out.push(name, qubits, params);

                if p1a != 0 { out.push(pauli[p1a], &[q0], &[]); }
                if p2a != 0 { out.push(pauli[p2a], &[q1], &[]); }
            } else {
                out.push(name, qubits, params);
            }
        }

        PyGateSequence { inner: out }
    }
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

    m.add("__version__", "0.1.0")?;
    Ok(())
}
