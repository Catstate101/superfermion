//! Quantum circuit as a Directed Acyclic Graph (DAG).
//!
//! The DAG is the central data structure of Superfermion's IR.
//! Nodes represent quantum operations, edges represent qubit wires
//! flowing through operations. This structure enables:
//! - Topological ordering (execution order)
//! - Dependency analysis (which gates can run in parallel)
//! - Efficient gate insertion/removal (for compiler passes)
//! - Depth computation (critical path length)

use crate::ops::{OpType, Parameter};
use indexmap::IndexMap;
use petgraph::algo::toposort;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Unique identifier for a node in the DAG.
pub type NodeId = NodeIndex;
/// Unique identifier for a qubit.
pub type QubitId = usize;
/// Unique identifier for a classical bit.
pub type ClassicalBitId = usize;

/// A single quantum operation node in the DAG.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumOp {
    /// What gate this is
    pub op_type: OpType,
    pub original_op_type: OpType,

    /// Which qubits it acts on (ordered: control first, target last)
    pub qubits: SmallVec<[QubitId; 3]>,

    /// Classical bits this op writes to (for measurements)
    pub classical_bits: SmallVec<[ClassicalBitId; 1]>,

    /// Classical condition: only execute if register == value
    pub condition: Option<(usize, u64)>,
}

impl QuantumOp {
    /// Create a new operation.
    pub fn new(op_type: OpType, qubits: &[QubitId]) -> Self {
        let op_copy = op_type.clone();
        Self {
            op_type,
            original_op_type: op_copy,
            qubits: SmallVec::from_slice(qubits),
            classical_bits: SmallVec::new(),
            condition: None,
        }
    }

    /// Create an input boundary node for a qubit.
    pub fn input(qubit: QubitId) -> Self {
        Self::new(OpType::Input, &[qubit])
    }

    /// Create an output boundary node for a qubit.
    pub fn output(qubit: QubitId) -> Self {
        Self::new(OpType::Output, &[qubit])
    }
}

/// Type of wire (qubit or classical bit) on a DAG edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireType {
    Qubit(QubitId),
    Classical(ClassicalBitId),
}

/// Optional hints to guide compilation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CircuitMetadata {
    pub name: Option<String>,
    pub target_backend: Option<String>,
    pub optimization_level: u8,
}

/// The quantum circuit as a Directed Acyclic Graph.
///
/// Nodes = quantum operations.
/// Edges = data dependencies (qubit wires flowing through ops).
///
/// Uses petgraph's StableDiGraph:
/// - Stable: node/edge indices don't change on removal
/// - Di: directed (qubit wire direction)
pub struct QuantumDAG {
    /// The graph itself
    graph: StableDiGraph<QuantumOp, WireType>,

    /// Input boundary nodes: one per qubit (source of each wire)
    input_nodes: Vec<NodeId>,

    /// Output boundary nodes: one per qubit (sink of each wire)
    output_nodes: Vec<NodeId>,

    /// Number of qubits
    pub n_qubits: usize,

    /// Number of classical bits
    pub n_cbits: usize,

    /// Free parameters (variable name → unique id)
    pub parameters: IndexMap<String, usize>,

    /// Tracks which nodes contain which symbolic parameters (param_name -> [NodeId])
    pub param_locations: HashMap<String, Vec<NodeId>>,

    /// Optional metadata
    pub metadata: CircuitMetadata,
}

impl QuantumDAG {
    /// Create a new empty circuit with n_qubits qubits and n_cbits classical bits.
    ///
    /// The DAG starts with input→output boundary node pairs for each qubit.
    /// Gates are inserted by rewiring these edges.
    pub fn new(n_qubits: usize, n_cbits: usize) -> Self {
        let mut graph = StableDiGraph::new();
        let mut input_nodes = Vec::with_capacity(n_qubits);
        let mut output_nodes = Vec::with_capacity(n_qubits);

        // Create input and output boundary nodes for each qubit
        for q in 0..n_qubits {
            let input = graph.add_node(QuantumOp::input(q));
            let output = graph.add_node(QuantumOp::output(q));
            graph.add_edge(input, output, WireType::Qubit(q));
            input_nodes.push(input);
            output_nodes.push(output);
        }

        Self {
            graph,
            input_nodes,
            output_nodes,
            n_qubits,
            n_cbits,
            parameters: IndexMap::new(),
            param_locations: HashMap::new(),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Add a gate to the circuit.
    ///
    /// This is the core DAG mutation:
    /// 1. Creates a new node for the gate
    /// 2. For each affected qubit, finds the predecessor of the output node
    /// 3. Rewires: predecessor → new_node → output
    ///
    /// Returns the new node's ID.
    pub fn add_op(&mut self, op_type: OpType, qubits: &[QubitId]) -> NodeId {
        // Validate qubit indices
        for &q in qubits {
            assert!(
                q < self.n_qubits,
                "Qubit {} out of range (circuit has {} qubits)",
                q,
                self.n_qubits
            );
        }
        // Guard against duplicate qubits which would create a self-loop
        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                assert!(
                    qubits[i] != qubits[j],
                    "Duplicate qubit index {} in gate {:?}",
                    qubits[i],
                    op_type
                );
            }
        }

        // Register any new parameters and track their locations
        let node_id_to_be = self.graph.add_node(QuantumOp::new(op_type.clone(), qubits));

        for param in op_type.parameters() {
            if let Parameter::Variable { ref name, id } = param {
                self.parameters.entry(name.clone()).or_insert(*id);
                self.param_locations
                    .entry(name.clone())
                    .or_default()
                    .push(node_id_to_be);
            }
        }

        let new_node = node_id_to_be;

        // For each qubit this gate touches:
        // find predecessor of output node and rewire
        for &qubit in qubits {
            let output_node = self.output_nodes[qubit];

            // Find incoming edge to output node for this qubit wire
            let pred_edge = self
                .graph
                .edges_directed(output_node, Incoming)
                .find(|e| *e.weight() == WireType::Qubit(qubit))
                .expect("Output node must have incoming qubit wire");

            let pred_node = pred_edge.source();
            let edge_id = pred_edge.id();

            // Remove old wire: pred → output
            self.graph.remove_edge(edge_id);

            // Add new wires: pred → new_node → output
            self.graph
                .add_edge(pred_node, new_node, WireType::Qubit(qubit));
            self.graph
                .add_edge(new_node, output_node, WireType::Qubit(qubit));
        }

        new_node
    }

    /// In-place update of symbolic parameters.
    /// This avoids rebuilding the entire DAG and IR for variational loops.
    pub fn update_parameters(&mut self, values: &HashMap<String, f64>) {
        for name in values.keys() {
            if let Some(nodes) = self.param_locations.get(name) {
                for &node_id in nodes {
                    let op = &mut self.graph[node_id];
                    op.op_type = op.original_op_type.bind_params(values);
                }
            }
        }
    }

    /// Add a measurement on a qubit, writing to a classical bit.
    pub fn add_measure(&mut self, qubit: QubitId, cbit: ClassicalBitId) -> NodeId {
        assert!(qubit < self.n_qubits, "Qubit {} out of range", qubit);
        assert!(cbit < self.n_cbits, "Classical bit {} out of range", cbit);

        let mut op = QuantumOp::new(OpType::Measure, &[qubit]);
        op.classical_bits.push(cbit);
        let new_node = self.graph.add_node(op);

        // Wire the qubit
        let output_node = self.output_nodes[qubit];
        let pred_edge = self
            .graph
            .edges_directed(output_node, Incoming)
            .find(|e| *e.weight() == WireType::Qubit(qubit))
            .unwrap();
        let pred_node = pred_edge.source();
        let edge_id = pred_edge.id();
        self.graph.remove_edge(edge_id);
        self.graph
            .add_edge(pred_node, new_node, WireType::Qubit(qubit));
        self.graph
            .add_edge(new_node, output_node, WireType::Qubit(qubit));

        new_node
    }

    /// Get the topological ordering of gate nodes (excluding boundary nodes).
    /// This is the execution order that respects all qubit dependencies.
    pub fn topological_order(&self) -> Vec<NodeId> {
        toposort(&self.graph, None)
            .expect("DAG must be acyclic")
            .into_iter()
            .filter(|&n| !self.is_boundary_node(n))
            .collect()
    }

    /// Export all gate nodes as (name, qubits, params) tuples for Python interop.
    /// Returns gates in topological order, skipping boundary/barrier nodes.
    /// Qubit indices are in MSB-first convention (matching sf.Circuit layout).
    pub fn to_gate_records(&self) -> Vec<(String, Vec<QubitId>, Vec<f64>)> {
        let order = self.topological_order();
        let mut records = Vec::with_capacity(order.len());
        for &node_id in &order {
            let op = &self.graph[node_id];
            if matches!(op.op_type, OpType::Barrier) {
                continue;
            }
            let (name, params) = op_type_to_name_params(&op.op_type);
            records.push((name, op.qubits.to_vec(), params));
        }
        records
    }

    /// Returns the DAG nodes grouped into parallel layers (qubit-disjoint rounds).
    /// All gates in a single layer can be executed in parallel.
    pub fn parallel_layers(&self) -> Vec<Vec<NodeId>> {
        let topo = self.topological_order();
        if topo.is_empty() {
            return vec![];
        }

        let mut dist: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
        let mut max_depth = 0;

        for &node in &topo {
            let pred_max = self
                .graph
                .neighbors_directed(node, petgraph::Incoming)
                .filter(|n| !self.is_boundary_node(*n))
                .map(|n| dist.get(&n).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let d = pred_max + 1;
            dist.insert(node, d);
            if d > max_depth {
                max_depth = d;
            }
        }

        let mut layers = vec![vec![]; max_depth];
        for (node, depth) in dist {
            layers[depth - 1].push(node);
        }
        layers
    }

    /// Compute circuit depth (length of critical path through DAG).
    pub fn depth(&self) -> usize {
        let topo = self.topological_order();
        if topo.is_empty() {
            return 0;
        }

        let mut dist: HashMap<NodeId, usize> = HashMap::new();

        for &node in &topo {
            let pred_max = self
                .graph
                .neighbors_directed(node, Incoming)
                .filter(|n| !self.is_boundary_node(*n))
                .map(|n| dist.get(&n).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            dist.insert(node, pred_max + 1);
        }

        dist.values().copied().max().unwrap_or(0)
    }

    /// Count total number of gates (excluding boundary nodes).
    pub fn gate_count(&self) -> usize {
        self.topological_order().len()
    }

    /// Count gates of a specific type.
    pub fn count_ops_of_type(&self, op_name: &str) -> usize {
        self.topological_order()
            .iter()
            .filter(|&&n| self.graph[n].op_type.name() == op_name)
            .count()
    }

    /// Bind concrete values to variational parameters.
    /// Returns a new DAG with all matching Variable parameters replaced by Const.
    pub fn bind(&self, values: &HashMap<String, f64>) -> Self {
        let mut bound = self.clone_dag();
        for node in bound.graph.node_weights_mut() {
            node.op_type = node.op_type.bind_params(values);
        }
        bound.parameters.retain(|k, _| !values.contains_key(k));
        bound.param_locations.retain(|k, _| !values.contains_key(k));
        bound
    }

    /// Get the number of free (unbound) parameters.
    pub fn n_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Get parameter names in order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }

    /// Convert to a linear instruction list (topological order).
    pub fn simulate_dm(&self) -> Vec<num_complex::Complex64> {
        use crate::dm::DensityMatrixState;
        let mut state = DensityMatrixState::new(self.n_qubits);
        let instructions = self.to_instructions();

        for inst in instructions {
            let u = inst.op_type.to_matrix();
            state.apply_unitary(&u, &inst.qubits);
        }
        state.data
    }

    pub fn simulate_pauli_expval_batch(&self, paulis: Vec<Vec<u8>>) -> Vec<f64> {
        let sv = self.simulate();
        let n = self.n_qubits;
        let dim = 1 << n;

        paulis
            .iter()
            .map(|p| {
                let mut expval = 0.0;
                // P = P0 \otimes P1 ...
                // We can compute this in one pass over the statevector
                for i in 0..dim {
                    let mut phase = num_complex::Complex64::new(1.0, 0.0);
                    let mut target_idx = i;

                    for (q, &pauli_op) in p.iter().enumerate() {
                        match pauli_op {
                            1 => {
                                // X
                                target_idx ^= 1 << q;
                            }
                            2 => {
                                // Y
                                target_idx ^= 1 << q;
                                if (i >> q) & 1 == 0 {
                                    phase *= num_complex::Complex64::i();
                                } else {
                                    phase *= -num_complex::Complex64::i();
                                }
                            }
                            3 if (i >> q) & 1 == 1 => {
                                // Z
                                phase *= -1.0;
                            }
                            3 => {}
                            _ => {} // I
                        }
                    }
                    expval += (sv[i].conj() * phase * sv[target_idx]).re;
                }
                expval
            })
            .collect()
    }

    /// Convert to a linear instruction list (topological order).
    pub fn to_instructions(&self) -> Vec<&QuantumOp> {
        self.topological_order()
            .iter()
            .map(|&n| &self.graph[n])
            .collect()
    }

    /// High-performance MPS simulation and sampling.
    pub fn sample_mps(
        &self,
        bond_dim: usize,
        shots: usize,
        seed: u64,
    ) -> std::collections::HashMap<String, usize> {
        let mut state = crate::mps::MPSState::new(self.n_qubits, bond_dim);
        self._evolve_into(&mut state);
        state.canonicalize_right();

        let is_identity_perm = state.perm_inv.iter().enumerate().all(|(i, &v)| v == i);
        let raw = state.sample(shots, seed);
        if is_identity_perm {
            return raw;
        }
        // Un-permute: sample() returns bitstrings q0-last in physical-site
        // order (char j = physical site n-1-j).  Map each char to its virtual
        // qubit and emit q0-last in virtual-qubit order.
        let mut out = std::collections::HashMap::new();
        for (phys_bits, count) in raw {
            let chars: Vec<u8> = phys_bits.bytes().collect();
            let mut virt = vec![b'0'; self.n_qubits];
            for (j, &ch) in chars.iter().enumerate() {
                let phys = self.n_qubits - 1 - j;
                virt[state.perm_inv[phys]] = ch;
            }
            virt.reverse();
            let key = String::from_utf8(virt).unwrap();
            *out.entry(key).or_insert(0) += count;
        }
        out
    }

    /// Check if a node is a boundary (input/output) node.
    pub fn is_boundary_node(&self, node: NodeId) -> bool {
        self.graph[node].op_type.is_boundary()
    }

    /// Return immediate predecessor gate nodes for a given node (non-boundary).
    pub fn predecessors(&self, node: NodeId) -> Vec<NodeId> {
        self.graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .filter(|&n| !self.is_boundary_node(n))
            .collect()
    }

    /// Return immediate successor gate nodes for a given node (non-boundary).
    pub fn successors(&self, node: NodeId) -> Vec<NodeId> {
        self.graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .filter(|&n| !self.is_boundary_node(n))
            .collect()
    }

    /// Return the input boundary nodes.
    pub fn input_nodes(&self) -> &[NodeId] {
        &self.input_nodes
    }

    /// Return the output boundary nodes.
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    /// Get a reference to the graph (for compiler passes).
    pub fn graph(&self) -> &StableDiGraph<QuantumOp, WireType> {
        &self.graph
    }

    /// Get a mutable reference to the graph.
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<QuantumOp, WireType> {
        &mut self.graph
    }

    /// Clone the DAG (explicit name to avoid confusion with Clone trait).
    pub fn clone_dag(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            input_nodes: self.input_nodes.clone(),
            output_nodes: self.output_nodes.clone(),
            n_qubits: self.n_qubits,
            n_cbits: self.n_cbits,
            parameters: self.parameters.clone(),
            param_locations: self.param_locations.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Export to OpenQASM 3.0 string (basic version).
    pub fn to_qasm3(&self) -> String {
        let mut out = String::from("OPENQASM 3.0;\n");
        out.push_str(&format!("qubit[{}] q;\n", self.n_qubits));
        if self.n_cbits > 0 {
            out.push_str(&format!("bit[{}] c;\n", self.n_cbits));
        }
        out.push('\n');

        for &node_id in &self.topological_order() {
            let op = &self.graph[node_id];
            let qubits_str: Vec<String> = op.qubits.iter().map(|q| format!("q[{q}]")).collect();

            let line = match &op.op_type {
                OpType::H => format!("h {};", qubits_str[0]),
                OpType::X => format!("x {};", qubits_str[0]),
                OpType::Y => format!("y {};", qubits_str[0]),
                OpType::Z => format!("z {};", qubits_str[0]),
                OpType::S => format!("s {};", qubits_str[0]),
                OpType::Sdg => format!("sdg {};", qubits_str[0]),
                OpType::T => format!("t {};", qubits_str[0]),
                OpType::Tdg => format!("tdg {};", qubits_str[0]),
                OpType::SX => format!("sx {};", qubits_str[0]),
                OpType::Rx(p) => format!("rx({}) {};", p.evaluate(), qubits_str[0]),
                OpType::Ry(p) => format!("ry({}) {};", p.evaluate(), qubits_str[0]),
                OpType::Rz(p) => format!("rz({}) {};", p.evaluate(), qubits_str[0]),
                OpType::CNOT => format!("cx {}, {};", qubits_str[0], qubits_str[1]),
                OpType::CZ => format!("cz {}, {};", qubits_str[0], qubits_str[1]),
                OpType::SWAP => format!("swap {}, {};", qubits_str[0], qubits_str[1]),
                OpType::CCX => format!(
                    "ccx {}, {}, {};",
                    qubits_str[0], qubits_str[1], qubits_str[2]
                ),
                OpType::Measure => {
                    if let Some(&cbit) = op.classical_bits.first() {
                        format!("c[{cbit}] = measure {};", qubits_str[0])
                    } else {
                        format!("measure {};", qubits_str[0])
                    }
                }
                OpType::Barrier => "barrier;".to_string(),
                OpType::Reset => format!("reset {};", qubits_str[0]),
                _ => format!("// {:?}", op.op_type),
            };

            out.push_str(&line);
            out.push('\n');
        }

        out
    }

    /// Convert the DAG to a full unitary matrix.
    ///
    /// WARNING: This scales exponentially (2^n x 2^n). Avoid for n > 12.
    /// Convert the DAG to a full unitary matrix.
    ///
    /// WARNING: This scales exponentially (2^n x 2^n). Avoid for n > 12.
    pub fn to_unitary(&self) -> nalgebra::DMatrix<num_complex::Complex64> {
        use nalgebra::DMatrix;
        use num_complex::Complex64;

        let dim = 1 << self.n_qubits;
        let mut total_u = DMatrix::identity(dim, dim);

        let order = self.topological_order();
        for &node_id in &order {
            let op = &self.graph[node_id];
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier {
                continue;
            }

            let n_op = op.qubits.len();
            if n_op == 0 {
                continue;
            }

            let gate_u = op.op_type.to_matrix();

            // Optimization: build full unitary only if necessary
            let mut current_u = DMatrix::identity(dim, dim);
            if n_op == 1 {
                let target = op.qubits[0];
                let mut temp_u = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
                for q in 0..self.n_qubits {
                    if q == target {
                        temp_u = kronecker(&gate_u, &temp_u);
                    } else {
                        temp_u = kronecker(&DMatrix::identity(2, 2), &temp_u);
                    }
                }
                current_u = temp_u;
            } else if n_op == 2 {
                // For 2-qubit gates, build manually to avoid complex kronecker logic
                // (Simplified for now, real implementation would use sparse representations)
                let mut res = vec![Complex64::new(0.0, 0.0); dim];
                for i in 0..dim {
                    let mut vec_in = vec![Complex64::new(0.0, 0.0); dim];
                    vec_in[i] = Complex64::new(1.0, 0.0);
                    self.apply_gate_into(&vec_in, &mut res, op);
                    for j in 0..dim {
                        current_u[(j, i)] = res[j];
                    }
                }
            }

            total_u = current_u * total_u;
        }

        total_u
    }

    /// Parallelism threshold for Rayon dispatch.
    ///
    /// Below this amplitude count, serial execution is faster (Rayon
    /// work-stealing overhead exceeds the parallelism benefit).
    const PARALLEL_THRESHOLD: usize = 1 << 19; // 524288 amplitudes ~ n=19

    /// High-performance in-place statevector simulation.
    ///
    /// ## Memory strategy: single buffer, in-place modification
    ///
    /// Each gate transforms independent amplitude pairs/groups in-place.
    /// A 1-qubit gate on qubit t transforms pairs (state[i], state[i+2^t])
    /// independently. A 2-qubit gate transforms groups of 4. Since each
    /// group is independent, we read and write only the affected amplitudes.
    ///
    /// DRAM traffic per gate: ~64MB at n=24 (read-modify-write one buffer)
    /// vs the old ping-pong approach which moved ~128MB/gate.
    ///
    /// Gate matrices are pre-computed once before the simulation loop to
    /// avoid per-gate DMatrix allocation and trig calls.
    pub fn simulate(&self) -> Vec<num_complex::Complex64> {
        let dim = 1 << self.n_qubits;
        let mut state = vec![num_complex::Complex64::new(0.0, 0.0); dim];
        state[0] = num_complex::Complex64::new(1.0, 0.0);

        let order = self.topological_order();

        // Pre-compute gate matrices and fuse consecutive 1q gates on the same qubit
        let raw_ops: Vec<_> = order
            .iter()
            .filter_map(|&node_id| {
                let op = &self.graph[node_id];
                if op.op_type.is_boundary()
                    || op.op_type == OpType::Barrier
                    || op.op_type.is_measurement()
                {
                    None
                } else {
                    Some((op, op.op_type.to_matrix()))
                }
            })
            .collect();

        let fused = fuse_1q_sequence(&raw_ops, self.n_qubits);

        for (op, gate_u) in &fused {
            Self::apply_gate_inplace(&mut state, op, gate_u);
        }
        state
    }

    /// Apply a gate in-place on the statevector. Each gate transforms
    /// independent amplitude pairs/groups, so no second buffer is needed.
    fn apply_gate_inplace(
        state: &mut [num_complex::Complex64],
        op: &crate::dag::QuantumOp,
        gate_u: &nalgebra::DMatrix<num_complex::Complex64>,
    ) {
        let dim = state.len();
        let use_par = dim >= Self::PARALLEL_THRESHOLD;

        if op.qubits.len() == 1 {
            let t = op.qubits[0];
            let u00 = gate_u[(0, 0)];
            let u01 = gate_u[(0, 1)];
            let u10 = gate_u[(1, 0)];
            let u11 = gate_u[(1, 1)];
            let stride = 1usize << t;

            // Detect diagonal from the ACTUAL matrix (not op_type, which may
            // be stale after 1q gate fusion).
            let is_diag = u01.norm() < 1e-14 && u10.norm() < 1e-14;
            if is_diag {
                let a = u00;
                let b = u11;
                if use_par {
                    let chunk = (dim / 16).max(1024);
                    state
                        .par_chunks_mut(chunk)
                        .enumerate()
                        .for_each(|(c, chunk_s)| {
                            let off = c * chunk;
                            for (k, amp) in chunk_s.iter_mut().enumerate() {
                                let coef = if ((off + k) & stride) == 0 { a } else { b };
                                *amp = coef * *amp;
                            }
                        });
                } else {
                    for i in 0..dim {
                        let coef = if (i & stride) == 0 { a } else { b };
                        state[i] = coef * state[i];
                    }
                }
                return;
            }

            // X gate: swap pairs in-place (only if matrix is actually X)
            let is_x = u00.norm() < 1e-14
                && u11.norm() < 1e-14
                && (u01 - num_complex::Complex64::new(1.0, 0.0)).norm() < 1e-14
                && (u10 - num_complex::Complex64::new(1.0, 0.0)).norm() < 1e-14;
            if is_x {
                inplace_swap_pairs(state, stride, use_par);
                return;
            }

            // General 1q gate: transform pairs (state[i], state[i+stride]) in-place
            let u00r = u00.re;
            let u00i = u00.im;
            let u01r = u01.re;
            let u01i = u01.im;
            let u10r = u10.re;
            let u10i = u10.im;
            let u11r = u11.re;
            let u11i = u11.im;

            inplace_1q_general(
                state, stride, u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i, use_par,
            );
        } else if op.qubits.len() == 2 {
            let q1 = op.qubits[0];
            let q2 = op.qubits[1];
            let mq1 = 1usize << q1;
            let mq2 = 1usize << q2;

            // CNOT: conditional bit-flip in-place (swap pairs where control=1)
            if op.op_type == OpType::CNOT {
                inplace_cnot(state, q1, q2, use_par);
                return;
            }

            // Diagonal 2q: scale each amplitude by its diagonal element
            let is_diag_2q = matches!(op.op_type, OpType::CZ | OpType::Rzz(_));
            if is_diag_2q {
                let g00 = gate_u[(0, 0)];
                let g11 = gate_u[(1, 1)];
                let g22 = gate_u[(2, 2)];
                let g33 = gate_u[(3, 3)];
                if use_par {
                    let chunk = (dim / 16).max(1024);
                    state
                        .par_chunks_mut(chunk)
                        .enumerate()
                        .for_each(|(c, chunk_s)| {
                            let off = c * chunk;
                            for (k, amp) in chunk_s.iter_mut().enumerate() {
                                let i = off + k;
                                let coef = match ((i & mq1) != 0, (i & mq2) != 0) {
                                    (false, false) => g00,
                                    (false, true) => g11,
                                    (true, false) => g22,
                                    (true, true) => g33,
                                };
                                *amp = coef * *amp;
                            }
                        });
                } else {
                    for i in 0..dim {
                        let coef = match ((i & mq1) != 0, (i & mq2) != 0) {
                            (false, false) => g00,
                            (false, true) => g11,
                            (true, false) => g22,
                            (true, true) => g33,
                        };
                        state[i] = coef * state[i];
                    }
                }
                return;
            }

            // General 2q gate: transform groups of 4 amplitudes in-place
            let g00 = gate_u[(0, 0)];
            let g01 = gate_u[(0, 1)];
            let g02 = gate_u[(0, 2)];
            let g03 = gate_u[(0, 3)];
            let g10 = gate_u[(1, 0)];
            let g11 = gate_u[(1, 1)];
            let g12 = gate_u[(1, 2)];
            let g13 = gate_u[(1, 3)];
            let g20 = gate_u[(2, 0)];
            let g21 = gate_u[(2, 1)];
            let g22 = gate_u[(2, 2)];
            let g23 = gate_u[(2, 3)];
            let g30 = gate_u[(3, 0)];
            let g31 = gate_u[(3, 1)];
            let g32 = gate_u[(3, 2)];
            let g33 = gate_u[(3, 3)];

            inplace_2q_general(
                state, q1, q2, mq1, mq2, g00, g01, g02, g03, g10, g11, g12, g13, g20, g21, g22,
                g23, g30, g31, g32, g33, use_par,
            );
        } else if op.qubits.len() == 3 {
            // CCX and CSWAP: swap pairs in-place where condition is met
            match op.op_type {
                OpType::CCX => {
                    let mc1 = 1usize << op.qubits[0];
                    let mc2 = 1usize << op.qubits[1];
                    let mt = 1usize << op.qubits[2];
                    inplace_ccx(state, mc1, mc2, mt, use_par);
                }
                OpType::CSWAP => {
                    let mc = 1usize << op.qubits[0];
                    let mt1 = 1usize << op.qubits[1];
                    let mt2 = 1usize << op.qubits[2];
                    inplace_cswap(state, mc, mt1, mt2, use_par);
                }
                _ => {
                    // General 3q: iterate over independent groups of 8
                    let q1 = op.qubits[0];
                    let q2 = op.qubits[1];
                    let q3 = op.qubits[2];
                    let qs = sorted_3(q1, q2, q3);
                    let m0 = 1usize << qs[0];
                    let m1 = 1usize << qs[1];
                    let m2 = 1usize << qs[2];
                    let n_groups = dim >> 3;
                    for g in 0..n_groups {
                        let base = deposit_bits_3(g, m0, m1, m2);
                        let mut vals = [num_complex::Complex64::new(0.0, 0.0); 8];
                        for col in 0..8usize {
                            let b1 = (col >> 2) & 1;
                            let b2 = (col >> 1) & 1;
                            let b3 = col & 1;
                            vals[col] = state[base | (b1 << q1) | (b2 << q2) | (b3 << q3)];
                        }
                        for row in 0..8usize {
                            let b1 = (row >> 2) & 1;
                            let b2 = (row >> 1) & 1;
                            let b3 = row & 1;
                            let idx = base | (b1 << q1) | (b2 << q2) | (b3 << q3);
                            let mut acc = num_complex::Complex64::new(0.0, 0.0);
                            for col in 0..8usize {
                                acc += gate_u[(row, col)] * vals[col];
                            }
                            state[idx] = acc;
                        }
                    }
                }
            }
        }
    }

    /// Legacy out-of-place gate application (kept for `to_unitary` which needs src/dst).
    fn apply_gate_into(
        &self,
        src: &[num_complex::Complex64],
        dst: &mut [num_complex::Complex64],
        op: &crate::dag::QuantumOp,
    ) {
        let gate_u = op.op_type.to_matrix();
        if op.qubits.len() == 2 {
            let q1 = op.qubits[0];
            let q2 = op.qubits[1];
            let mq1 = 1usize << q1;
            let mq2 = 1usize << q2;
            let dim = src.len();
            for i in 0..dim {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;
                let i00 = i & !mq1 & !mq2;
                let i01 = i00 | mq2;
                let i10 = i00 | mq1;
                let i11 = i00 | mq1 | mq2;
                dst[i] = match bit1 * 2 + bit2 {
                    0 => {
                        gate_u[(0, 0)] * src[i00]
                            + gate_u[(0, 1)] * src[i01]
                            + gate_u[(0, 2)] * src[i10]
                            + gate_u[(0, 3)] * src[i11]
                    }
                    1 => {
                        gate_u[(1, 0)] * src[i00]
                            + gate_u[(1, 1)] * src[i01]
                            + gate_u[(1, 2)] * src[i10]
                            + gate_u[(1, 3)] * src[i11]
                    }
                    2 => {
                        gate_u[(2, 0)] * src[i00]
                            + gate_u[(2, 1)] * src[i01]
                            + gate_u[(2, 2)] * src[i10]
                            + gate_u[(2, 3)] * src[i11]
                    }
                    _ => {
                        gate_u[(3, 0)] * src[i00]
                            + gate_u[(3, 1)] * src[i01]
                            + gate_u[(3, 2)] * src[i10]
                            + gate_u[(3, 3)] * src[i11]
                    }
                };
            }
        } else if op.qubits.len() == 1 {
            let t = op.qubits[0];
            let stride = 1usize << t;
            let u00 = gate_u[(0, 0)];
            let u01 = gate_u[(0, 1)];
            let u10 = gate_u[(1, 0)];
            let u11 = gate_u[(1, 1)];
            for i in 0..src.len() {
                let partner = i ^ stride;
                if (i & stride) == 0 {
                    dst[i] = u00 * src[i] + u01 * src[partner];
                } else {
                    dst[i] = u10 * src[partner] + u11 * src[i];
                }
            }
        } else {
            dst.copy_from_slice(src);
        }
    }

    /// Simulate the circuit on a specific device target.
    ///
    /// `device` is either `"cpu"` or `"gpu"`. On CPU, uses the existing
    /// optimized Rayon+AVX statevector kernel. On GPU, dispatches to the
    /// CUDA kernels in `sf-gpu` (feature-gated).
    pub fn simulate_on(&self, device: &str) -> Result<Vec<num_complex::Complex64>, String> {
        match device {
            "cpu" => Ok(self.simulate()),
            "gpu" => {
                #[cfg(feature = "gpu")]
                {
                    self.simulate_gpu()
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err("GPU support not compiled. Rebuild with --features gpu".to_string())
                }
            }
            other => Err(format!("Unknown device '{}'. Use 'cpu' or 'gpu'.", other)),
        }
    }

    /// GPU statevector simulation (only available with the `gpu` feature).
    #[cfg(feature = "gpu")]
    fn simulate_gpu(&self) -> Result<Vec<num_complex::Complex64>, String> {
        use sf_gpu::{GateOp, GpuError};

        let order = self.topological_order();
        let mut gates: Vec<GateOp> = Vec::new();

        for &node_id in &order {
            let op = &self.graph[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                continue;
            }

            let mat = op.op_type.to_matrix();
            let qubits: Vec<usize> = op.qubits.iter().map(|&q| q).collect();
            let n = mat.nrows();

            let is_diagonal =
                n == 2 && { mat[(0, 1)].norm() < 1e-15 && mat[(1, 0)].norm() < 1e-15 };

            let (matrix_re, matrix_im) = if is_diagonal && n == 2 {
                (
                    vec![mat[(0, 0)].re, mat[(1, 1)].re],
                    vec![mat[(0, 0)].im, mat[(1, 1)].im],
                )
            } else {
                let mut re = Vec::with_capacity(n * n);
                let mut im = Vec::with_capacity(n * n);
                for r in 0..n {
                    for c in 0..n {
                        re.push(mat[(r, c)].re);
                        im.push(mat[(r, c)].im);
                    }
                }
                (re, im)
            };

            gates.push(GateOp {
                name: format!("{:?}", op.op_type),
                qubits,
                matrix_re,
                matrix_im,
                is_diagonal,
            });
        }

        sf_gpu::simulate_statevector(self.n_qubits, &gates).map_err(|e| match e {
            GpuError::NotAvailable => "No CUDA GPU detected. Use device='cpu'.".to_string(),
            GpuError::InsufficientVram {
                n_qubits,
                required_mb,
                available_mb,
            } => format!(
                "Circuit has {} qubits — requires {}MB VRAM but GPU has {}MB. Use device='cpu'.",
                n_qubits, required_mb, available_mb
            ),
            GpuError::Cuda(msg) => format!("CUDA error: {}", msg),
        })
    }

    /// Check if GPU simulation is available at runtime.
    pub fn gpu_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            sf_gpu::is_available()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// In-place gate kernels. Each operates on independent amplitude pairs
// or groups within a single statevector buffer.
// ──────────────────────────────────────────────────────────────────────

/// Wrapper around a raw mutable pointer to allow safe parallel access.
///
/// SAFETY: Callers MUST guarantee that parallel closures access disjoint
/// index ranges. This is upheld because each gate kernel partitions the
/// state into independent amplitude groups (pairs/quads) that don't overlap.
struct SendPtr(*mut num_complex::Complex64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    unsafe fn get(&self, idx: usize) -> num_complex::Complex64 {
        *self.0.add(idx)
    }
    #[inline(always)]
    unsafe fn set(&self, idx: usize, val: num_complex::Complex64) {
        *self.0.add(idx) = val;
    }
    #[inline(always)]
    unsafe fn swap(&self, a: usize, b: usize) {
        let tmp = *self.0.add(a);
        *self.0.add(a) = *self.0.add(b);
        *self.0.add(b) = tmp;
    }
}

/// Out-of-place 2x2 kernel (used by density matrix module).
#[inline(always)]
pub fn apply_2x2_kernel_f64(
    lo_src: &[num_complex::Complex64],
    hi_src: &[num_complex::Complex64],
    lo_dst: &mut [num_complex::Complex64],
    hi_dst: &mut [num_complex::Complex64],
    u00r: f64,
    u00i: f64,
    u01r: f64,
    u01i: f64,
    u10r: f64,
    u10i: f64,
    u11r: f64,
    u11i: f64,
) {
    let n = lo_src.len();
    for i in 0..n {
        let ar = lo_src[i].re;
        let ai = lo_src[i].im;
        let br = hi_src[i].re;
        let bi = hi_src[i].im;
        lo_dst[i] = num_complex::Complex64::new(
            u00r * ar - u00i * ai + u01r * br - u01i * bi,
            u00r * ai + u00i * ar + u01r * bi + u01i * br,
        );
        hi_dst[i] = num_complex::Complex64::new(
            u10r * ar - u10i * ai + u11r * br - u11i * bi,
            u10r * ai + u10i * ar + u11r * bi + u11i * br,
        );
    }
}

/// Swap amplitude pairs at stride `stride` in-place (X gate).
fn inplace_swap_pairs(state: &mut [num_complex::Complex64], stride: usize, use_par: bool) {
    let dim = state.len();
    let block = stride * 2;
    let n_groups = dim / block;

    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| {
            let base = g * block;
            for k in 0..stride {
                unsafe {
                    sp.swap(base + k, base + k + stride);
                }
            }
        });
    } else {
        for g in 0..n_groups {
            let base = g * block;
            for k in 0..stride {
                state.swap(base + k, base + k + stride);
            }
        }
    }
}

/// CNOT in-place: swap (state[i], state[i|mt]) where control=1 and target=0.
fn inplace_cnot(state: &mut [num_complex::Complex64], ctrl: usize, tgt: usize, use_par: bool) {
    let dim = state.len();
    let mc = 1usize << ctrl;
    let mt = 1usize << tgt;

    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| {
            if (i & mc) != 0 && (i & mt) == 0 {
                unsafe {
                    sp.swap(i, i | mt);
                }
            }
        });
    } else {
        for i in 0..dim {
            if (i & mc) != 0 && (i & mt) == 0 {
                state.swap(i, i | mt);
            }
        }
    }
}

/// General 1q gate in-place: transform pairs (state[i], state[i+stride]).
fn inplace_1q_general(
    state: &mut [num_complex::Complex64],
    stride: usize,
    u00r: f64,
    u00i: f64,
    u01r: f64,
    u01i: f64,
    u10r: f64,
    u10i: f64,
    u11r: f64,
    u11i: f64,
    use_par: bool,
) {
    let dim = state.len();
    let block = stride * 2;
    let n_groups = dim / block;

    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| {
            let base = g * block;
            for k in 0..stride {
                let lo_idx = base + k;
                let hi_idx = lo_idx + stride;
                unsafe {
                    let lo = sp.get(lo_idx);
                    let hi = sp.get(hi_idx);
                    let ar = lo.re;
                    let ai = lo.im;
                    let br = hi.re;
                    let bi = hi.im;
                    sp.set(
                        lo_idx,
                        num_complex::Complex64::new(
                            u00r * ar - u00i * ai + u01r * br - u01i * bi,
                            u00r * ai + u00i * ar + u01r * bi + u01i * br,
                        ),
                    );
                    sp.set(
                        hi_idx,
                        num_complex::Complex64::new(
                            u10r * ar - u10i * ai + u11r * br - u11i * bi,
                            u10r * ai + u10i * ar + u11r * bi + u11i * br,
                        ),
                    );
                }
            }
        });
    } else {
        for g in 0..n_groups {
            let base = g * block;
            for k in 0..stride {
                let lo_idx = base + k;
                let hi_idx = lo_idx + stride;
                let lo = state[lo_idx];
                let hi = state[hi_idx];
                let ar = lo.re;
                let ai = lo.im;
                let br = hi.re;
                let bi = hi.im;

                state[lo_idx] = num_complex::Complex64::new(
                    u00r * ar - u00i * ai + u01r * br - u01i * bi,
                    u00r * ai + u00i * ar + u01r * bi + u01i * br,
                );
                state[hi_idx] = num_complex::Complex64::new(
                    u10r * ar - u10i * ai + u11r * br - u11i * bi,
                    u10r * ai + u10i * ar + u11r * bi + u11i * br,
                );
            }
        }
    }
}

/// General 2q gate in-place: transform groups of 4 amplitudes.
#[allow(clippy::too_many_arguments)]
fn inplace_2q_general(
    state: &mut [num_complex::Complex64],
    q1: usize,
    q2: usize,
    mq1: usize,
    mq2: usize,
    g00: num_complex::Complex64,
    g01: num_complex::Complex64,
    g02: num_complex::Complex64,
    g03: num_complex::Complex64,
    g10: num_complex::Complex64,
    g11: num_complex::Complex64,
    g12: num_complex::Complex64,
    g13: num_complex::Complex64,
    g20: num_complex::Complex64,
    g21: num_complex::Complex64,
    g22: num_complex::Complex64,
    g23: num_complex::Complex64,
    g30: num_complex::Complex64,
    g31: num_complex::Complex64,
    g32: num_complex::Complex64,
    g33: num_complex::Complex64,
    use_par: bool,
) {
    let dim = state.len();
    // Iterate over canonical group representatives: indices with both qubit bits = 0
    let (lo_q, hi_q) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
    let m_lo = 1usize << lo_q;
    let m_hi = 1usize << hi_q;
    let n_groups = dim >> 2;

    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| {
            let base = deposit_bits_2(g, m_lo, m_hi);
            let i00 = base;
            let i01 = base | mq2;
            let i10 = base | mq1;
            let i11 = base | mq1 | mq2;
            unsafe {
                let v00 = sp.get(i00);
                let v01 = sp.get(i01);
                let v10 = sp.get(i10);
                let v11 = sp.get(i11);
                sp.set(i00, g00 * v00 + g01 * v01 + g02 * v10 + g03 * v11);
                sp.set(i01, g10 * v00 + g11 * v01 + g12 * v10 + g13 * v11);
                sp.set(i10, g20 * v00 + g21 * v01 + g22 * v10 + g23 * v11);
                sp.set(i11, g30 * v00 + g31 * v01 + g32 * v10 + g33 * v11);
            }
        });
    } else {
        for g in 0..n_groups {
            let base = deposit_bits_2(g, m_lo, m_hi);
            let i00 = base;
            let i01 = base | mq2;
            let i10 = base | mq1;
            let i11 = base | mq1 | mq2;
            let v00 = state[i00];
            let v01 = state[i01];
            let v10 = state[i10];
            let v11 = state[i11];
            state[i00] = g00 * v00 + g01 * v01 + g02 * v10 + g03 * v11;
            state[i01] = g10 * v00 + g11 * v01 + g12 * v10 + g13 * v11;
            state[i10] = g20 * v00 + g21 * v01 + g22 * v10 + g23 * v11;
            state[i11] = g30 * v00 + g31 * v01 + g32 * v10 + g33 * v11;
        }
    }
}

/// CCX (Toffoli) in-place: swap state[i] and state[i|mt] where both controls set and target=0.
fn inplace_ccx(
    state: &mut [num_complex::Complex64],
    mc1: usize,
    mc2: usize,
    mt: usize,
    use_par: bool,
) {
    let dim = state.len();
    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| {
            if (i & mc1) != 0 && (i & mc2) != 0 && (i & mt) == 0 {
                unsafe {
                    sp.swap(i, i | mt);
                }
            }
        });
    } else {
        for i in 0..dim {
            if (i & mc1) != 0 && (i & mc2) != 0 && (i & mt) == 0 {
                state.swap(i, i | mt);
            }
        }
    }
}

/// CSWAP in-place: swap target bits when control is set.
fn inplace_cswap(
    state: &mut [num_complex::Complex64],
    mc: usize,
    mt1: usize,
    mt2: usize,
    use_par: bool,
) {
    let dim = state.len();
    if use_par {
        let sp = SendPtr(state.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| {
            if (i & mc) != 0 && (i & mt1) != 0 && (i & mt2) == 0 {
                unsafe {
                    sp.swap(i, (i ^ mt1) | mt2);
                }
            }
        });
    } else {
        for i in 0..dim {
            if (i & mc) != 0 && (i & mt1) != 0 && (i & mt2) == 0 {
                state.swap(i, (i ^ mt1) | mt2);
            }
        }
    }
}

/// Map a group index g to a state index with the two qubit bits cleared.
/// `m_lo` and `m_hi` are the masks for the lower and higher qubit (m_lo < m_hi).
#[inline(always)]
fn deposit_bits_2(g: usize, m_lo: usize, m_hi: usize) -> usize {
    // Insert a 0-bit at position lo_q and hi_q in the binary representation of g.
    let lo_pos = m_lo.trailing_zeros() as usize;
    let hi_pos = m_hi.trailing_zeros() as usize;
    let lo_mask = m_lo - 1;
    // Insert zero bit at lo_pos
    let x = (g & lo_mask) | ((g & !lo_mask) << 1);
    // Insert zero bit at hi_pos (which shifted up by 1 if above lo_pos)
    let hi_mask = (1usize << (hi_pos)) - 1; // mask below original hi_pos
    let _ = lo_pos; // used above
    (x & hi_mask) | ((x & !hi_mask) << 1)
}

/// Fuse consecutive 1-qubit gates on the same qubit into a single 2x2 matrix.
///
/// Scans the gate list linearly. When a run of 1q gates on the same qubit
/// is interrupted by a multi-qubit gate (or a 1q gate on a different qubit
/// that isn't fusable), the accumulated product is emitted as a single gate.
///
/// This reduces gate count by 30-50% on typical variational circuits
/// (H-Rz-CNOT-Rz patterns) without rebuilding the DAG.
fn fuse_1q_sequence<'a>(
    ops: &[(
        &'a crate::dag::QuantumOp,
        nalgebra::DMatrix<num_complex::Complex64>,
    )],
    n_qubits: usize,
) -> Vec<(
    &'a crate::dag::QuantumOp,
    nalgebra::DMatrix<num_complex::Complex64>,
)> {
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    // Per-qubit accumulator: Option<(first_op_ref, fused_2x2_matrix)>
    let mut accum: Vec<Option<(&'a crate::dag::QuantumOp, DMatrix<Complex64>)>> =
        vec![None; n_qubits];
    let mut result: Vec<(&'a crate::dag::QuantumOp, DMatrix<Complex64>)> =
        Vec::with_capacity(ops.len());

    let is_identity_2x2 = |m: &DMatrix<Complex64>| -> bool {
        let d00 = (m[(0, 0)] - Complex64::new(1.0, 0.0)).norm();
        let d01 = m[(0, 1)].norm();
        let d10 = m[(1, 0)].norm();
        let d11 = (m[(1, 1)] - Complex64::new(1.0, 0.0)).norm();
        d00 + d01 + d10 + d11 < 1e-10
    };

    for (op, mat) in ops {
        if op.qubits.len() == 1 {
            let q = op.qubits[0];
            accum[q] = Some(match accum[q].take() {
                Some((first_op, existing)) => {
                    // mat * existing: new gate applied after existing
                    (first_op, mat * existing)
                }
                None => (*op, mat.clone()),
            });
        } else {
            // Multi-qubit gate: flush accumulators for all involved qubits
            for &q in &op.qubits {
                if let Some((fused_op, fused_mat)) = accum[q].take() {
                    if !is_identity_2x2(&fused_mat) {
                        result.push((fused_op, fused_mat));
                    }
                }
            }
            result.push((*op, mat.clone()));
        }
    }

    // Flush remaining accumulators
    for q in 0..n_qubits {
        if let Some((fused_op, fused_mat)) = accum[q].take() {
            if !is_identity_2x2(&fused_mat) {
                result.push((fused_op, fused_mat));
            }
        }
    }

    result
}

/// Sort 3 qubit indices for iteration.
fn sorted_3(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut arr = [a, b, c];
    arr.sort_unstable();
    arr
}

/// Map group index to base index with 3 qubit bits cleared.
fn deposit_bits_3(g: usize, m0: usize, m1: usize, m2: usize) -> usize {
    let p0 = m0.trailing_zeros() as usize;
    let p1 = m1.trailing_zeros() as usize;
    let p2 = m2.trailing_zeros() as usize;
    let mask0 = (1usize << p0) - 1;
    let x = (g & mask0) | ((g & !mask0) << 1);
    let mask1 = (1usize << p1) - 1;
    let y = (x & mask1) | ((x & !mask1) << 1);
    let mask2 = (1usize << p2) - 1;
    (y & mask2) | ((y & !mask2) << 1)
}

fn kronecker(
    a: &nalgebra::DMatrix<num_complex::Complex64>,
    b: &nalgebra::DMatrix<num_complex::Complex64>,
) -> nalgebra::DMatrix<num_complex::Complex64> {
    let (ra, ca) = a.shape();
    let (rb, cb) = b.shape();
    let mut res =
        nalgebra::DMatrix::from_element(ra * rb, ca * cb, num_complex::Complex64::new(0.0, 0.0));

    for i in 0..ra {
        for j in 0..ca {
            for k in 0..rb {
                for l in 0..cb {
                    res[(i * rb + k, j * cb + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }
    res
}

/// Map an OpType to its QASM2 gate name string and concrete parameter values.
fn op_type_to_name_params(op: &OpType) -> (String, Vec<f64>) {
    match op {
        OpType::H => ("h".into(), vec![]),
        OpType::X => ("x".into(), vec![]),
        OpType::Y => ("y".into(), vec![]),
        OpType::Z => ("z".into(), vec![]),
        OpType::S => ("s".into(), vec![]),
        OpType::Sdg => ("sdg".into(), vec![]),
        OpType::T => ("t".into(), vec![]),
        OpType::Tdg => ("tdg".into(), vec![]),
        OpType::SX => ("sx".into(), vec![]),
        OpType::SXdg => ("sxdg".into(), vec![]),
        OpType::Id => ("id".into(), vec![]),
        OpType::Rx(p) => ("rx".into(), vec![p.evaluate()]),
        OpType::Ry(p) => ("ry".into(), vec![p.evaluate()]),
        OpType::Rz(p) => ("rz".into(), vec![p.evaluate()]),
        OpType::R1(p) => ("r1".into(), vec![p.evaluate()]),
        OpType::P(p) => ("p".into(), vec![p.evaluate()]),
        OpType::U(a, b, c) => ("u".into(), vec![a.evaluate(), b.evaluate(), c.evaluate()]),
        OpType::CNOT => ("cx".into(), vec![]),
        OpType::CZ => ("cz".into(), vec![]),
        OpType::CY => ("cy".into(), vec![]),
        OpType::SWAP => ("swap".into(), vec![]),
        OpType::ISWAP => ("iswap".into(), vec![]),
        OpType::ECR => ("ecr".into(), vec![]),
        OpType::Rzz(p) => ("rzz".into(), vec![p.evaluate()]),
        OpType::Rxx(p) => ("rxx".into(), vec![p.evaluate()]),
        OpType::Ryy(p) => ("ryy".into(), vec![p.evaluate()]),
        OpType::CRx(p) => ("crx".into(), vec![p.evaluate()]),
        OpType::CRz(p) => ("crz".into(), vec![p.evaluate()]),
        OpType::CP(p) => ("cp".into(), vec![p.evaluate()]),
        OpType::CCX => ("ccx".into(), vec![]),
        OpType::CSWAP => ("cswap".into(), vec![]),
        OpType::Measure => ("measure".into(), vec![]),
        OpType::Reset => ("reset".into(), vec![]),
        OpType::Barrier => ("barrier".into(), vec![]),
        _ => ("unknown".into(), vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dag_has_correct_structure() {
        let dag = QuantumDAG::new(3, 2);
        assert_eq!(dag.n_qubits, 3);
        assert_eq!(dag.n_cbits, 2);
        assert_eq!(dag.gate_count(), 0);
        assert_eq!(dag.depth(), 0);
        assert_eq!(dag.input_nodes.len(), 3);
        assert_eq!(dag.output_nodes.len(), 3);
    }

    #[test]
    fn test_add_single_gate() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        assert_eq!(dag.gate_count(), 1);
        assert_eq!(dag.depth(), 1);
    }

    #[test]
    fn test_add_two_gates_same_qubit() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::X, &[0]);
        assert_eq!(dag.gate_count(), 2);
        assert_eq!(dag.depth(), 2);
    }

    #[test]
    fn test_parallel_gates_depth_1() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::X, &[1]);
        assert_eq!(dag.gate_count(), 2);
        assert_eq!(dag.depth(), 1); // parallel — depth is 1
    }

    #[test]
    fn test_bell_state_circuit() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        assert_eq!(dag.gate_count(), 2);
        assert_eq!(dag.depth(), 2); // H then CNOT — sequential
    }

    #[test]
    fn test_topological_order_respects_dependencies() {
        let mut dag = QuantumDAG::new(2, 0);
        let h = dag.add_op(OpType::H, &[0]);
        let cnot = dag.add_op(OpType::CNOT, &[0, 1]);
        let topo = dag.topological_order();
        let h_pos = topo.iter().position(|&n| n == h).unwrap();
        let cnot_pos = topo.iter().position(|&n| n == cnot).unwrap();
        assert!(h_pos < cnot_pos, "H must come before CNOT");
    }

    #[test]
    fn test_parameterized_circuit() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        assert_eq!(dag.n_parameters(), 1);
        assert_eq!(dag.parameter_names(), vec!["theta"]);
    }

    #[test]
    fn test_bind_parameters() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );

        let mut values = HashMap::new();
        values.insert("theta".into(), 1.57);
        let bound = dag.bind(&values);

        assert_eq!(bound.n_parameters(), 0);
        let ops = bound.to_instructions();
        assert_eq!(ops.len(), 1);
        match &ops[0].op_type {
            OpType::Rx(Parameter::Const(v)) => assert!((v - 1.57).abs() < 1e-10),
            other => panic!("Expected Rx(Const), got {:?}", other),
        }
    }

    #[test]
    fn test_qasm3_export() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let qasm = dag.to_qasm3();
        assert!(qasm.contains("OPENQASM 3.0;"));
        assert!(qasm.contains("qubit[2] q;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
    }

    #[test]
    fn test_complex_circuit_depth() {
        // Build: H(0), H(1), CNOT(0,1), Rx(0), Ry(1)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]); // depth 1 on q0
        dag.add_op(OpType::H, &[1]); // depth 1 on q1 (parallel)
        dag.add_op(OpType::CNOT, &[0, 1]); // depth 2 (depends on both H's)
        dag.add_op(OpType::Rx(Parameter::Const(0.5)), &[0]); // depth 3
        dag.add_op(OpType::Ry(Parameter::Const(0.7)), &[1]); // depth 3 (parallel with Rx)

        assert_eq!(dag.gate_count(), 5);
        assert_eq!(dag.depth(), 3);
    }

    #[test]
    fn test_measurement() {
        let mut dag = QuantumDAG::new(2, 2);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_measure(0, 0);
        dag.add_measure(1, 1);

        assert_eq!(dag.gate_count(), 4); // H, CNOT, 2 measures
        let qasm = dag.to_qasm3();
        assert!(qasm.contains("c[0] = measure q[0];"));
        assert!(qasm.contains("c[1] = measure q[1];"));
    }

    #[test]
    fn test_count_ops_of_type() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[2]);

        assert_eq!(dag.count_ops_of_type("H"), 3);
        assert_eq!(dag.count_ops_of_type("CNOT"), 1);
        assert_eq!(dag.count_ops_of_type("Rx"), 0);
    }

    #[test]
    #[should_panic(expected = "Qubit 5 out of range")]
    fn test_invalid_qubit_panics() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[5]); // should panic
    }

    // ── Adaptive threshold & simulation correctness tests ──

    #[test]
    fn test_simulate_small_circuit_bell() {
        // Bell state: H(0), CNOT(0,1) → (|00> + |11>) / sqrt(2)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let sv = dag.simulate();
        let expected_00 = 1.0 / std::f64::consts::SQRT_2;
        assert!((sv[0].re - expected_00).abs() < 1e-10);
        assert!(sv[1].norm() < 1e-10);
        assert!(sv[2].norm() < 1e-10);
        assert!((sv[3].re - expected_00).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_small_circuit_ghz4() {
        // 4-qubit GHZ: H(0), CNOT chain → (|0000> + |1111>) / sqrt(2)
        // This tests the serial path (n=4, dim=16 < PARALLEL_THRESHOLD)
        let mut dag = QuantumDAG::new(4, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 2]);
        dag.add_op(OpType::CNOT, &[2, 3]);

        let sv = dag.simulate();
        let expected = 1.0 / std::f64::consts::SQRT_2;
        assert!((sv[0].re - expected).abs() < 1e-10, "sv[0]={}", sv[0]);
        assert!((sv[15].re - expected).abs() < 1e-10, "sv[15]={}", sv[15]);
        for i in 1..15 {
            assert!(sv[i].norm() < 1e-10, "sv[{}]={} should be 0", i, sv[i]);
        }
    }

    #[test]
    fn test_simulate_diagonal_gates_small() {
        // Rz + CZ on 3 qubits (serial path), verify diagonal fast paths
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(
            OpType::Rz(Parameter::Const(std::f64::consts::PI / 4.0)),
            &[0],
        );
        dag.add_op(OpType::CZ, &[0, 1]);
        dag.add_op(OpType::S, &[2]);

        let sv = dag.simulate();
        // Just verify normalization
        let norm: f64 = sv.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10, "Norm = {}", norm);
    }

    #[test]
    fn test_simulate_x_gate_small() {
        // X on |0> = |1>
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::X, &[0]);

        let sv = dag.simulate();
        assert!(sv[0].norm() < 1e-10);
        assert!((sv[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_ccx_small() {
        // CCX on |110> → |111>
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::X, &[0]);
        dag.add_op(OpType::X, &[1]);
        dag.add_op(OpType::CCX, &[0, 1, 2]);

        let sv = dag.simulate();
        // |110> = index 6 (binary: q0=1, q1=1, q2=0 in LSB order)
        // CCX flips q2, so result is |111> = index 7
        assert!((sv[7].re - 1.0).abs() < 1e-10, "sv[7]={}", sv[7]);
    }

    #[test]
    fn test_simulate_pauli_expval_small() {
        // H|0> in X basis should give <X> = 1
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);

        let results = dag.simulate_pauli_expval_batch(vec![
            vec![1], // X
            vec![3], // Z
        ]);
        assert!((results[0] - 1.0).abs() < 1e-10, "<X>={}", results[0]);
        assert!(results[1].abs() < 1e-10, "<Z>={}", results[1]);
    }

    #[test]
    fn test_simulate_general_2q_gate() {
        // SWAP gate (non-diagonal 2q, tests general 4x4 kernel)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::X, &[0]); // |10>
        dag.add_op(OpType::SWAP, &[0, 1]); // → |01>

        let sv = dag.simulate();
        // |01> = index 2 in LSB convention (q0=0, q1=1)
        assert!((sv[2].re - 1.0).abs() < 1e-10, "sv[2]={}", sv[2]);
    }

    #[test]
    fn test_to_unitary_h_gate() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);

        let u = dag.to_unitary();
        let h = OpType::H.to_matrix();

        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (u[(r, c)] - h[(r, c)]).norm() < 1e-10,
                    "u[({r},{c})]={}, h[({r},{c})]={}",
                    u[(r, c)],
                    h[(r, c)]
                );
            }
        }
    }

    #[test]
    fn test_update_parameters_in_place() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );

        let mut values = HashMap::new();
        values.insert("theta".into(), std::f64::consts::FRAC_PI_2);
        dag.update_parameters(&values);

        let ops = dag.to_instructions();
        assert_eq!(ops.len(), 1);
        match &ops[0].op_type {
            OpType::Rx(Parameter::Const(v)) => {
                assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
            }
            other => panic!("Expected Rx(Const), got {:?}", other),
        }
    }

    #[test]
    fn test_identity_circuit_yields_zero_state() {
        let dag = QuantumDAG::new(3, 0);
        let sv = dag.simulate();

        assert!((sv[0].re - 1.0).abs() < 1e-10);
        for i in 1..sv.len() {
            assert!(sv[i].norm() < 1e-10, "sv[{i}]={}", sv[i]);
        }
    }
}
