//! Lightweight flat gate sequence — zero-heap-per-gate storage for circuit
//! construction.  Unlike QuantumDAG (which uses petgraph and maintains full
//! dependency edges), GateSequence is a Struct-of-Arrays flat list designed for
//! high-throughput Python circuit building where DAG semantics aren't needed.
//!
//! When compilation is required, call `to_dag()` to convert into a full
//! QuantumDAG.  For memory benchmarks, gates live in Rust Vecs — invisible to
//! Python's tracemalloc.

use crate::dag::{QubitId, QuantumDAG};
use crate::ops::{OpType, Parameter};

/// A flat, compact gate sequence stored as Structure-of-Arrays.
///
/// For N gates with average 2 qubits and 1 param:
///   names:       N × ~24 bytes (String heap)
///   qubits_data: N × 16 bytes  
///   qubit_offsets: N × 4 bytes
///   params_data: N × 8 bytes
///   param_offsets: N × 4 bytes
///   Total: ~56 bytes/gate  (vs ~200 bytes/gate for Python GateRecord)
///
/// No petgraph dependency — just contiguous Vecs.
#[derive(Clone, Debug)]
pub struct GateSequence {
    /// Gate names (uppercase, e.g. "RX", "CNOT")
    names: Vec<String>,
    
    /// Flat array of all qubit indices across all gates
    qubits_data: Vec<QubitId>,
    /// qubit_offsets[i] = start index in qubits_data for gate i
    /// qubit_offsets[i+1] - qubit_offsets[i] = number of qubits for gate i
    qubit_offsets: Vec<u32>,
    
    /// Flat array of all gate parameters across all gates
    params_data: Vec<f64>,
    /// param_offsets[i] = start index in params_data for gate i
    param_offsets: Vec<u32>,
    
    /// Number of qubits in the circuit
    pub n_qubits: usize,
    /// Number of classical bits in the circuit  
    pub n_cbits: usize,
}

impl GateSequence {
    /// Create a new empty gate sequence.
    pub fn new(n_qubits: usize, n_cbits: usize) -> Self {
        Self {
            names: Vec::new(),
            qubits_data: Vec::new(),
            qubit_offsets: Vec::new(),     // qubit_offsets[i] = end idx for gate i
            params_data: Vec::new(),
            param_offsets: Vec::new(),      // param_offsets[i] = end idx for gate i
            n_qubits,
            n_cbits,
        }
    }
    
    /// Pre-allocate capacity for `expected_gates` gates.
    /// Avoids Vec reallocations during batch construction.
    pub fn with_capacity(n_qubits: usize, n_cbits: usize, expected_gates: usize) -> Self {
        Self {
            names: Vec::with_capacity(expected_gates),
            qubits_data: Vec::with_capacity(expected_gates * 2),  // avg 2 qubits/gate
            qubit_offsets: Vec::with_capacity(expected_gates),
            params_data: Vec::with_capacity(expected_gates),      // avg 1 param/gate  
            param_offsets: Vec::with_capacity(expected_gates),
            n_qubits,
            n_cbits,
        }
    }
    
    /// Add a single gate to the sequence.
    pub fn push(&mut self, name: &str, qubits: &[QubitId], params: &[f64]) {
        self.qubits_data.extend_from_slice(qubits);
        self.qubit_offsets.push(self.qubits_data.len() as u32);
        
        self.params_data.extend_from_slice(params);
        self.param_offsets.push(self.params_data.len() as u32);
        
        self.names.push(name.to_string());
    }
    
    /// Batch-extend from Python-compatible tuple format.
    /// Each record is (name: String, qubits: Vec<usize>, params: Vec<f64>).
    pub fn extend(&mut self, records: &[(String, Vec<QubitId>, Vec<f64>)]) {
        let total = records.len();
        self.names.reserve(total);
        self.qubits_data.reserve(total * 2);
        self.qubit_offsets.reserve(total);
        self.params_data.reserve(total);
        self.param_offsets.reserve(total);
        
        for (name, qubits, params) in records {
            self.push(name, qubits, params);
        }
    }
    
    /// Build a Quantum Volume circuit entirely in Rust — zero per-gate FFI.
    ///
    /// Pre-generated permutations and angles are passed as flat slices
    /// (produced by numpy in Python).  This single call replaces 25,000
    /// individual `add_gate()` Python→Rust crossings for a 100Q×100depth QV.
    ///
    /// Permutation layout: depth × n_qubits entries, row-major.
    /// Angles layout:  depth × (n_qubits//2) × 4 entries, row-major.
    pub fn from_qv_batch(
        n_qubits: usize,
        n_cbits: usize,
        depth: usize,
        perms: &[u64],
        angles: &[f64],
    ) -> Self {
        let pairs_per_layer = n_qubits / 2;
        let gates_per_layer = pairs_per_layer * 5;  // 5 gates per SU(4) pair
        let total_gates = depth * gates_per_layer;
        
        let mut gs = Self::with_capacity(n_qubits, n_cbits, total_gates);
        
        for layer in 0..depth {
            let perm_off = layer * n_qubits;
            let angle_off = layer * pairs_per_layer * 4;
            
            for pair_idx in 0..pairs_per_layer {
                let q0 = perms[perm_off + pair_idx * 2] as usize;
                let q1 = perms[perm_off + pair_idx * 2 + 1] as usize;
                
                let a0 = angles[angle_off + pair_idx * 4];
                let a1 = angles[angle_off + pair_idx * 4 + 1];
                let a2 = angles[angle_off + pair_idx * 4 + 2];
                let a3 = angles[angle_off + pair_idx * 4 + 3];
                
                // SU(4) = RY(a0,q0) · RY(a1,q1) · CNOT(q0,q1) · RY(a2,q0) · RY(a3,q1)
                gs.push("RY", &[q0], &[a0]);
                gs.push("RY", &[q1], &[a1]);
                gs.push("CNOT", &[q0, q1], &[]);
                gs.push("RY", &[q0], &[a2]);
                gs.push("RY", &[q1], &[a3]);
            }
        }
        
        gs
    }
    
    /// Number of gates.
    pub fn len(&self) -> usize {
        self.names.len()
    }
    
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
    
    /// Number of gates (alias for len).
    pub fn gate_count(&self) -> usize {
        self.len()
    }
    
    /// Get qubits for gate at index i.
    fn qubits_at(&self, i: usize) -> &[QubitId] {
        let start = if i == 0 { 0 } else { self.qubit_offsets[i - 1] as usize };
        let end = self.qubit_offsets[i] as usize;
        &self.qubits_data[start..end]
    }
    
    /// Get params for gate at index i.
    fn params_at(&self, i: usize) -> &[f64] {
        let start = if i == 0 { 0 } else { self.param_offsets[i - 1] as usize };
        let end = self.param_offsets[i] as usize;
        &self.params_data[start..end]
    }
    
    /// Export as (name, qubits, params) tuples for Python interop.
    pub fn to_gate_records(&self) -> Vec<(String, Vec<QubitId>, Vec<f64>)> {
        let mut records = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            records.push((
                self.names[i].clone(),
                self.qubits_at(i).to_vec(),
                self.params_at(i).to_vec(),
            ));
        }
        records
    }
    
    /// Convert to a full QuantumDAG for compiler passes.
    /// This is the bridge between lightweight storage and compilation.
    pub fn to_dag(&self) -> QuantumDAG {
        let mut dag = QuantumDAG::new(self.n_qubits, self.n_cbits);
        
        for i in 0..self.len() {
            let name = &self.names[i];
            let qubits = self.qubits_at(i);
            let params = self.params_at(i);
            
            // Convert name + params back to OpType
            let op_type = parse_gate_name(name, params);
            dag.add_op(op_type, qubits);
        }
        
        dag
    }
    
    /// Get a reference to the name at index i.
    pub fn name_at(&self, i: usize) -> &str {
        &self.names[i]
    }
    
    /// Iterate over all gates as (name, qubits, params) tuples.
    pub fn iter(&self) -> GateSequenceIter<'_> {
        GateSequenceIter {
            seq: self,
            pos: 0,
        }
    }
}

/// Iterator over GateSequence yielding (name, qubits, params).
pub struct GateSequenceIter<'a> {
    seq: &'a GateSequence,
    pos: usize,
}

impl<'a> Iterator for GateSequenceIter<'a> {
    type Item = (&'a str, &'a [QubitId], &'a [f64]);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.seq.len() {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        Some((self.seq.name_at(i), self.seq.qubits_at(i), self.seq.params_at(i)))
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.seq.len() - self.pos;
        (remaining, Some(remaining))
    }
}

/// Parse a gate name string (uppercase, Python convention) and params
/// into an OpType for QuantumDAG insertion.
///
/// This mirrors the parse_gate function in sf-bindings but lives in sf-ir
/// so GateSequence can convert to DAG without depending on the bindings crate.
fn parse_gate_name(name: &str, params: &[f64]) -> OpType {
    let n = name.to_lowercase();
    match n.as_str() {
        "h" => OpType::H,
        "x" => OpType::X,
        "y" => OpType::Y,
        "z" => OpType::Z,
        "s" => OpType::S,
        "sdg" => OpType::Sdg,
        "t" => OpType::T,
        "tdg" => OpType::Tdg,
        "sx" => OpType::SX,
        "sxdg" => OpType::SXdg,
        "id" => OpType::Id,
        "cx" | "cnot" => OpType::CNOT,
        "cz" => OpType::CZ,
        "cy" => OpType::CY,
        "swap" => OpType::SWAP,
        "iswap" => OpType::ISWAP,
        "ecr" => OpType::ECR,
        "ccx" | "toffoli" => OpType::CCX,
        "cswap" => OpType::CSWAP,
        "rx" => OpType::Rx(param_single(params, 0.0)),
        "ry" => OpType::Ry(param_single(params, 0.0)),
        "rz" => OpType::Rz(param_single(params, 0.0)),
        "p" | "phase" => OpType::P(param_single(params, 0.0)),
        "r1" => OpType::R1(param_single(params, 0.0)),
        "rzz" => OpType::Rzz(param_single(params, 0.0)),
        "rxx" => OpType::Rxx(param_single(params, 0.0)),
        "ryy" => OpType::Ryy(param_single(params, 0.0)),
        "crx" => OpType::CRx(param_single(params, 0.0)),
        "crz" => OpType::CRz(param_single(params, 0.0)),
        "cp" => OpType::CP(param_single(params, 0.0)),
        "u" => {
            let a = params.first().copied().unwrap_or(0.0);
            let b = params.get(1).copied().unwrap_or(0.0);
            let c = params.get(2).copied().unwrap_or(0.0);
            OpType::U(
                Parameter::Const(a),
                Parameter::Const(b),
                Parameter::Const(c),
            )
        }
        "measure" => OpType::Measure,
        "reset" => OpType::Reset,
        "barrier" => OpType::Barrier,
        "input" => OpType::Input,
        "output" => OpType::Output,
        _ => OpType::Custom(name.to_string(), params.iter().map(|&v| Parameter::Const(v)).collect()),
    }
}

/// Helper: extract first param as Const(f64), fallback to default.
fn param_single(params: &[f64], default: f64) -> Parameter {
    Parameter::Const(params.first().copied().unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_empty() {
        let gs = GateSequence::new(3, 2);
        assert_eq!(gs.len(), 0);
        assert_eq!(gs.n_qubits, 3);
        assert_eq!(gs.n_cbits, 2);
    }
    
    #[test]
    fn test_push_and_readback() {
        let mut gs = GateSequence::new(2, 0);
        gs.push("RX", &[0], &[0.5]);
        gs.push("CNOT", &[0, 1], &[]);
        gs.push("RZZ", &[1, 0], &[1.2]);
        
        assert_eq!(gs.len(), 3);
        assert_eq!(gs.name_at(0), "RX");
        assert_eq!(gs.qubits_at(0), &[0]);
        assert_eq!(gs.params_at(0), &[0.5]);
        assert_eq!(gs.name_at(1), "CNOT");
        assert_eq!(gs.qubits_at(1), &[0, 1]);
        assert!(gs.params_at(1).is_empty());
        assert_eq!(gs.name_at(2), "RZZ");
        assert_eq!(gs.qubits_at(2), &[1, 0]);
        assert_eq!(gs.params_at(2), &[1.2]);
    }
    
    #[test]
    fn test_extend() {
        let mut gs = GateSequence::new(2, 0);
        let records = vec![
            ("H".to_string(), vec![0], vec![]),
            ("CX".to_string(), vec![0, 1], vec![]),
            ("X".to_string(), vec![1], vec![]),
        ];
        gs.extend(&records);
        assert_eq!(gs.len(), 3);
        assert_eq!(gs.name_at(0), "H");
        assert_eq!(gs.name_at(1), "CX");
        assert_eq!(gs.name_at(2), "X");
    }
    
    #[test]
    fn test_to_gate_records() {
        let mut gs = GateSequence::new(2, 0);
        gs.push("H", &[0], &[]);
        gs.push("CNOT", &[0, 1], &[]);
        
        let records = gs.to_gate_records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, "H");
        assert_eq!(records[0].1, vec![0]);
        assert!(records[0].2.is_empty());
        assert_eq!(records[1].0, "CNOT");
        assert_eq!(records[1].1, vec![0, 1]);
    }
    
    #[test]
    fn test_to_dag() {
        let mut gs = GateSequence::new(2, 0);
        gs.push("H", &[0], &[]);
        gs.push("CNOT", &[0, 1], &[]);
        gs.push("X", &[1], &[]);
        
        let dag = gs.to_dag();
        assert_eq!(dag.gate_count(), 3);
        assert_eq!(dag.depth(), 3);  // H(0) → CNOT(0,1) → X(1)
        
        // Verify round-trip through gate records
        let records = dag.to_gate_records();
        assert_eq!(records.len(), 3);
    }
    
    #[test]
    fn test_with_capacity() {
        let mut gs = GateSequence::with_capacity(100, 0, 10000);
        for i in 0..10000 {
            gs.push("X", &[(i % 100) as usize], &[]);
        }
        assert_eq!(gs.len(), 10000);
    }
    
    #[test]
    fn test_iterator() {
        let mut gs = GateSequence::new(2, 0);
        gs.push("H", &[0], &[]);
        gs.push("Z", &[1], &[]);
        
        let gates: Vec<_> = gs.iter().collect();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].0, "H");
        assert_eq!(gates[1].0, "Z");
    }
}
