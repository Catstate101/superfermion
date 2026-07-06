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
                q, self.n_qubits
            );
        }

        // Register any new parameters and track their locations
        let node_id_to_be = self.graph.add_node(QuantumOp::new(op_type.clone(), qubits));
        
        for param in op_type.parameters() {
            if let Parameter::Variable { ref name, id } = param {
                self.parameters.entry(name.clone()).or_insert(*id);
                self.param_locations
                    .entry(name.clone())
                    .or_insert_with(Vec::new)
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
            self.graph.add_edge(pred_node, new_node, WireType::Qubit(qubit));
            self.graph.add_edge(new_node, output_node, WireType::Qubit(qubit));
        }

        new_node
    }

    /// In-place update of symbolic parameters.
    /// This avoids rebuilding the entire DAG and IR for variational loops.
    pub fn update_parameters(&mut self, values: &HashMap<String, f64>) {
        for (name, _) in values {
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
        self.graph.add_edge(pred_node, new_node, WireType::Qubit(qubit));
        self.graph.add_edge(new_node, output_node, WireType::Qubit(qubit));

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
        if topo.is_empty() { return vec![]; }
        
        let mut dist: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
        let mut max_depth = 0;
        
        for &node in &topo {
            let pred_max = self.graph.neighbors_directed(node, petgraph::Incoming)
                .filter(|n| !self.is_boundary_node(*n))
                .map(|n| dist.get(&n).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let d = pred_max + 1;
            dist.insert(node, d);
            if d > max_depth { max_depth = d; }
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

        paulis.iter().map(|p| {
            let mut expval = 0.0;
            // P = P0 \otimes P1 ...
            // We can compute this in one pass over the statevector
            for i in 0..dim {
                let mut phase = num_complex::Complex64::new(1.0, 0.0);
                let mut target_idx = i;
                
                for (q, &pauli_op) in p.iter().enumerate() {
                    match pauli_op {
                        1 => { // X
                            target_idx ^= 1 << q;
                        }
                        2 => { // Y
                            target_idx ^= 1 << q;
                            if (i >> q) & 1 == 0 {
                                phase *= num_complex::Complex64::i();
                            } else {
                                phase *= -num_complex::Complex64::i();
                            }
                        }
                        3 => { // Z
                            if (i >> q) & 1 == 1 {
                                phase *= -1.0;
                            }
                        }
                        _ => {} // I
                    }
                }
                expval += (sv[i].conj() * phase * sv[target_idx]).re;
            }
            expval
        }).collect()
    }

    /// Convert to a linear instruction list (topological order).
    pub fn to_instructions(&self) -> Vec<&QuantumOp> {
        self.topological_order()
            .iter()
            .map(|&n| &self.graph[n])
            .collect()
    }

    /// High-performance MPS simulation and sampling.
    pub fn sample_mps(&self, bond_dim: usize, shots: usize, seed: u64) -> std::collections::HashMap<String, usize> {
        let mut state = crate::mps::MPSState::new(self.n_qubits, bond_dim);
        let order = self.topological_order();

        for &node_id in &order {
            let op = &self.graph[node_id];
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier {
                continue;
            }
            
            let gate_u = op.op_type.to_matrix();
            if op.qubits.len() == 1 {
                state.apply_1q_gate(op.qubits[0], &gate_u);
            } else if op.qubits.len() == 2 {
                state.apply_2q_gate(op.qubits[0], op.qubits[1], &gate_u);
            }
        }
        // Left-canonicalize before sampling — required for correct
        // bit-by-bit probability computation.
        state.canonicalize_left();
        state.sample(shots, seed)
    }

    /// Check if a node is a boundary (input/output) node.
    pub fn is_boundary_node(&self, node: NodeId) -> bool {
        self.graph[node].op_type.is_boundary()
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
                OpType::CCX => format!("ccx {}, {}, {};", qubits_str[0], qubits_str[1], qubits_str[2]),
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
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier { continue; }
            
            let n_op = op.qubits.len();
            if n_op == 0 { continue; }

            let gate_u = op.op_type.to_matrix();
            
            // Optimization: build full unitary only if necessary
            let mut current_u = DMatrix::identity(dim, dim);
            if n_op == 1 {
                let target = op.qubits[0];
                let mut temp_u = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
                for q in 0..self.n_qubits {
                    if q == target {
                        temp_u = kronecker(&temp_u, &gate_u);
                    } else {
                        temp_u = kronecker(&temp_u, &DMatrix::identity(2, 2));
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

    /// High-performance statevector simulation.
    /// O(G * 2^n) instead of O(G * 2^3n)
    ///
    /// Fix 2026-04-26 (sf.rust dense gap vs Aer): pre-allocate two buffers
    /// once and ping-pong between them. Previous version allocated a fresh
    /// `Vec<Complex64>` of size 2^n inside every gate (`apply_gate_parallel`),
    /// which at n=20 meant 16 MB malloc + zero-fill per gate × 174 gates =
    /// 2.7 GB of allocator traffic per QAOA simulation. Aer C++ ping-pongs;
    /// matching that strategy here brings sf.rust within striking distance
    /// of Aer's statevector path.
    pub fn simulate(&self) -> Vec<num_complex::Complex64> {
        let dim = 1 << self.n_qubits;
        let mut buf_a = vec![num_complex::Complex64::new(0.0, 0.0); dim];
        let mut buf_b = vec![num_complex::Complex64::new(0.0, 0.0); dim];
        buf_a[0] = num_complex::Complex64::new(1.0, 0.0);
        let mut current_is_a = true;

        let order = self.topological_order();
        for &node_id in &order {
            let op = &self.graph[node_id];
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier || op.op_type.is_measurement() {
                continue;
            }
            if current_is_a {
                self.apply_gate_into(&buf_a, &mut buf_b, op);
            } else {
                self.apply_gate_into(&buf_b, &mut buf_a, op);
            }
            current_is_a = !current_is_a;
        }
        if current_is_a { buf_a } else { buf_b }
    }

    /// Apply gate into a pre-allocated destination buffer (no per-gate alloc).
    fn apply_gate_into(
        &self,
        src: &[num_complex::Complex64],
        dst: &mut [num_complex::Complex64],
        op: &crate::dag::QuantumOp,
    ) {
        let gate_u = op.op_type.to_matrix();

        if op.qubits.len() == 1 {
            let t = op.qubits[0];
            let u00 = gate_u[(0, 0)];
            let u01 = gate_u[(0, 1)];
            let u10 = gate_u[(1, 0)];
            let u11 = gate_u[(1, 1)];

            // ── Chunked, branch-free 1-qubit gate kernel ──
            // Goal: amortise Rayon work-stealing overhead and let LLVM
            // autovectorise the inner pair loop. The state splits into
            // blocks of 2^(t+1) consecutive amplitudes; within a block,
            // the first half has bit t = 0 and the second half has bit
            // t = 1. Each pair (lo[i], hi[i]) maps to:
            //
            //   dst_lo[i] = u00 * lo[i] + u01 * hi[i]
            //   dst_hi[i] = u10 * lo[i] + u11 * hi[i]
            //
            // For small t (block tiny) we parallelise over many blocks;
            // for large t (few blocks) we parallelise INSIDE the block
            // by chunking the inner pair loop.
            let half: usize = 1usize << t;
            let block: usize = half * 2;
            let dim: usize = src.len();
            let n_blocks: usize = dim / block;

            // ── DIAGONAL 1q FAST PATH ──
            // RZ, P/R1, S, Sdg, T, Tdg, Z, Id all have the form
            //   diag(a, b) * |s> = a*|0> if bit_t=0 else b*|1>
            // Memory cost: 1 read + 1 write per amplitude (vs 2+2 general).
            // We can do this fully in-place on the destination buffer
            // because src and dst are independent ping-pong slots; we
            // simply scale src into dst.
            let is_diag_1q = matches!(
                op.op_type,
                OpType::Z | OpType::S | OpType::Sdg | OpType::T | OpType::Tdg
                | OpType::Id | OpType::Rz(_) | OpType::R1(_) | OpType::P(_)
            );
            if is_diag_1q {
                let a = u00; // top-left
                let b = u11; // bottom-right
                let chunk: usize = (dim / 16).max(1024);
                dst.par_chunks_mut(chunk).enumerate().for_each(|(c, dst_chunk)| {
                    let off = c * chunk;
                    apply_diag_1q_kernel(
                        &src[off..off + dst_chunk.len()],
                        dst_chunk, t, a, b, off,
                    );
                });
                return;
            }

            // Specialise X — pure permutation.
            if op.op_type == OpType::X {
                if n_blocks >= 4 {
                    dst.par_chunks_mut(block).enumerate().for_each(|(b, dst_chunk)| {
                        let off = b * block;
                        let (lo_dst, hi_dst) = dst_chunk.split_at_mut(half);
                        let (lo_src, hi_src) = src[off..off + block].split_at(half);
                        for i in 0..half {
                            lo_dst[i] = hi_src[i];
                            hi_dst[i] = lo_src[i];
                        }
                    });
                } else {
                    // High-bit gate: parallelise inside the block.
                    for b in 0..n_blocks {
                        let off = b * block;
                        let (lo_dst, hi_dst) = dst[off..off + block].split_at_mut(half);
                        let (lo_src, hi_src) = src[off..off + block].split_at(half);
                        let stripe = (half + 7) / 8; // ~8 stripes for parallelism
                        lo_dst
                            .par_chunks_mut(stripe)
                            .zip(hi_dst.par_chunks_mut(stripe))
                            .enumerate()
                            .for_each(|(s, (ld, hd))| {
                                let s0 = s * stripe;
                                let len = ld.len();
                                let ls = &lo_src[s0..s0 + len];
                                let hs = &hi_src[s0..s0 + len];
                                for i in 0..len {
                                    ld[i] = hs[i];
                                    hd[i] = ls[i];
                                }
                            });
                    }
                }
                return;
            }

            // General 2x2 gate. Inner loop hand-flattened to f64 arithmetic
            // so LLVM can hoist into AVX-2 + FMA when target-cpu=native is
            // enabled (.cargo/config.toml). Per amplitude pair we now do
            // 16 fma-equivalent ops; with AVX-2 (4 f64 lanes) this fits in
            // ~4 fma cycles instead of ~16.
            let u00r = u00.re; let u00i = u00.im;
            let u01r = u01.re; let u01i = u01.im;
            let u10r = u10.re; let u10i = u10.im;
            let u11r = u11.re; let u11i = u11.im;

            if n_blocks >= 4 {
                dst.par_chunks_mut(block).enumerate().for_each(|(b, dst_chunk)| {
                    let off = b * block;
                    let (lo_dst, hi_dst) = dst_chunk.split_at_mut(half);
                    let (lo_src, hi_src) = src[off..off + block].split_at(half);
                    apply_2x2_kernel_f64(
                        lo_src, hi_src, lo_dst, hi_dst,
                        u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i,
                    );
                });
            } else {
                for b in 0..n_blocks {
                    let off = b * block;
                    let (lo_dst, hi_dst) = dst[off..off + block].split_at_mut(half);
                    let (lo_src, hi_src) = src[off..off + block].split_at(half);
                    let stripe = (half + 7) / 8;
                    lo_dst
                        .par_chunks_mut(stripe)
                        .zip(hi_dst.par_chunks_mut(stripe))
                        .enumerate()
                        .for_each(|(s, (ld, hd))| {
                            let s0 = s * stripe;
                            let len = ld.len();
                            let ls = &lo_src[s0..s0 + len];
                            let hs = &hi_src[s0..s0 + len];
                            apply_2x2_kernel_f64(
                                ls, hs, ld, hd,
                                u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i,
                            );
                            let _ = len;
                        });
                }
            }
        } else if op.qubits.len() == 2 {
            // ── Chunked 2-qubit kernel ──
            // Specialise CNOT (90% of 2q traffic in QAOA/Trotter) — pure
            // permutation, no FMAs.
            let q1 = op.qubits[0];
            let q2 = op.qubits[1];
            let dim: usize = src.len();

            if op.op_type == OpType::CNOT {
                // CNOT(c=q1, t=q2) flips bit q2 iff bit q1 = 1.
                // For bit q1 = 0: dst[i] = src[i]; for bit q1 = 1: dst[i] = src[i ^ (1<<q2)].
                // Chunk over independent index ranges; the per-element cost
                // is just a load + a store, so coarse parallelism wins.
                let chunk: usize = (dim / 16).max(1024);
                dst.par_chunks_mut(chunk).enumerate().for_each(|(c, dst_chunk)| {
                    let off = c * chunk;
                    for (k, target) in dst_chunk.iter_mut().enumerate() {
                        let i = off + k;
                        if (i >> q1) & 1 == 1 {
                            *target = src[i ^ (1 << q2)];
                        } else {
                            *target = src[i];
                        }
                    }
                });
                return;
            }

            // ── DIAGONAL 2q FAST PATH ──
            // CZ, RZZ, CP/CR1 — all diagonal in the computational basis.
            // dst[i] = phase[bit_q1(i), bit_q2(i)] * src[i]
            // 1 read + 1 write per amplitude, half the bandwidth.
            let is_diag_2q = matches!(op.op_type, OpType::CZ | OpType::Rzz(_));
            if is_diag_2q {
                let g00 = gate_u[(0, 0)];   // |q1q2=00>
                let g11 = gate_u[(1, 1)];   // |q1q2=01>
                let g22 = gate_u[(2, 2)];   // |q1q2=10>
                let g33 = gate_u[(3, 3)];   // |q1q2=11>
                let mq1 = 1usize << q1;
                let mq2 = 1usize << q2;
                let chunk: usize = (dim / 16).max(1024);
                dst.par_chunks_mut(chunk).enumerate().for_each(|(c, dst_chunk)| {
                    let off = c * chunk;
                    for (k, target) in dst_chunk.iter_mut().enumerate() {
                        let i = off + k;
                        let b1 = (i & mq1) != 0;
                        let b2 = (i & mq2) != 0;
                        let coef = match (b1, b2) {
                            (false, false) => g00,
                            (false, true)  => g11,
                            (true,  false) => g22,
                            (true,  true)  => g33,
                        };
                        *target = coef * src[i];
                    }
                });
                return;
            }

            // General 4x4 gate — 4 random reads per element. Iterate in
            // chunks to amortise rayon overhead; correctness identical to
            // the per-element version.
            let g00 = gate_u[(0, 0)]; let g01 = gate_u[(0, 1)];
            let g02 = gate_u[(0, 2)]; let g03 = gate_u[(0, 3)];
            let g10 = gate_u[(1, 0)]; let g11 = gate_u[(1, 1)];
            let g12 = gate_u[(1, 2)]; let g13 = gate_u[(1, 3)];
            let g20 = gate_u[(2, 0)]; let g21 = gate_u[(2, 1)];
            let g22 = gate_u[(2, 2)]; let g23 = gate_u[(2, 3)];
            let g30 = gate_u[(3, 0)]; let g31 = gate_u[(3, 1)];
            let g32 = gate_u[(3, 2)]; let g33 = gate_u[(3, 3)];
            let mq1: usize = 1 << q1;
            let mq2: usize = 1 << q2;
            let chunk: usize = (dim / 16).max(1024);
            dst.par_chunks_mut(chunk).enumerate().for_each(|(c, dst_chunk)| {
                let off = c * chunk;
                for (k, target) in dst_chunk.iter_mut().enumerate() {
                    let i = off + k;
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;
                    let i00 = i & !mq1 & !mq2;
                    let i01 = i00 | mq2;
                    let i10 = i00 | mq1;
                    let i11 = i00 | mq1 | mq2;
                    let v00 = src[i00];
                    let v01 = src[i01];
                    let v10 = src[i10];
                    let v11 = src[i11];
                    *target = match bit1 * 2 + bit2 {
                        0 => g00 * v00 + g01 * v01 + g02 * v10 + g03 * v11,
                        1 => g10 * v00 + g11 * v01 + g12 * v10 + g13 * v11,
                        2 => g20 * v00 + g21 * v01 + g22 * v10 + g23 * v11,
                        _ => g30 * v00 + g31 * v01 + g32 * v10 + g33 * v11,
                    };
                }
            });
        } else if op.qubits.len() == 3 {
            // CCX and CSWAP are pure permutations — `OpType::to_matrix()`
            // has no implementation for them and would return an
            // all-ones fallback (silent identity). Handle as permutations
            // here directly, no FMAs needed.
            //
            // SF op semantics (matches Python ``GateRecord.to_unitary``):
            //   * CCX(c1, c2, t)   — flip bit t iff bit c1 = 1 AND bit c2 = 1
            //   * CSWAP(c, t1, t2) — swap bits t1, t2 iff bit c = 1
            let chunk: usize = (src.len() / 16).max(1024);
            match op.op_type {
                OpType::CCX => {
                    let c1 = op.qubits[0];
                    let c2 = op.qubits[1];
                    let t  = op.qubits[2];
                    let mc1 = 1usize << c1;
                    let mc2 = 1usize << c2;
                    let mt  = 1usize << t;
                    dst.par_chunks_mut(chunk).enumerate().for_each(|(c, dst_chunk)| {
                        let off = c * chunk;
                        for (k, target) in dst_chunk.iter_mut().enumerate() {
                            let i = off + k;
                            if (i & mc1) != 0 && (i & mc2) != 0 {
                                *target = src[i ^ mt];
                            } else {
                                *target = src[i];
                            }
                        }
                    });
                }
                OpType::CSWAP => {
                    let c  = op.qubits[0];
                    let t1 = op.qubits[1];
                    let t2 = op.qubits[2];
                    let mc  = 1usize << c;
                    let mt1 = 1usize << t1;
                    let mt2 = 1usize << t2;
                    dst.par_chunks_mut(chunk).enumerate().for_each(|(cc, dst_chunk)| {
                        let off = cc * chunk;
                        for (k, target) in dst_chunk.iter_mut().enumerate() {
                            let i = off + k;
                            if (i & mc) != 0 {
                                let b1 = (i & mt1) != 0;
                                let b2 = (i & mt2) != 0;
                                if b1 != b2 {
                                    *target = src[i ^ mt1 ^ mt2];
                                } else {
                                    *target = src[i];
                                }
                            } else {
                                *target = src[i];
                            }
                        }
                    });
                }
                _ => {
                    // General 3q gate via 8x8 unitary — not currently used by
                    // the supported op set, but kept for forward compatibility.
                    let q1 = op.qubits[0];
                    let q2 = op.qubits[1];
                    let q3 = op.qubits[2];
                    dst.par_iter_mut().enumerate().for_each(|(i, target)| {
                        let bit1 = (i >> q1) & 1;
                        let bit2 = (i >> q2) & 1;
                        let bit3 = (i >> q3) & 1;
                        let i_base = i & !(1 << q1) & !(1 << q2) & !(1 << q3);
                        let row = bit1 * 4 + bit2 * 2 + bit3;
                        let mut acc = num_complex::Complex64::new(0.0, 0.0);
                        for col in 0..8usize {
                            let b1 = (col >> 2) & 1;
                            let b2 = (col >> 1) & 1;
                            let b3 = col & 1;
                            let idx = i_base | (b1 << q1) | (b2 << q2) | (b3 << q3);
                            acc += gate_u[(row, col)] * src[idx];
                        }
                        *target = acc;
                    });
                }
            }
        } else {
            // Fallback: copy src to dst; not expected for current op set.
            dst.copy_from_slice(src);
        }
    }
}

/// Helper for kronecker product of DMatrices.
/// Diagonal 1-qubit gate: dst[i] = a * src[i] if bit_t(i)=0 else b * src[i].
/// Memory cost is 1 complex read + 1 complex write per amplitude — half
/// the bandwidth of the general 2x2 gate.  Hot path for RZ, P, S, T, Z.
#[inline(always)]
fn apply_diag_1q_kernel(
    src: &[num_complex::Complex64],
    dst: &mut [num_complex::Complex64],
    t: usize,
    a: num_complex::Complex64,
    b: num_complex::Complex64,
    chunk_offset: usize,
) {
    debug_assert_eq!(src.len(), dst.len());
    let mt = 1usize << t;
    let n = dst.len();
    for k in 0..n {
        let i = chunk_offset + k;
        let coef = if (i & mt) == 0 { a } else { b };
        dst[k] = coef * src[k];
    }
}

/// Single-qubit gate inner kernel.  The hot path uses **explicit AVX-2 +
/// FMA intrinsics** to do 2 complex multiplications per 256-bit register.
/// On x86_64 CPUs from 2013 onwards (Haswell+) this gives ~2x more
/// throughput than the autovectorised scalar code, and ~4x more than
/// LLVM-without-target-cpu builds.
///
/// Per amplitude pair (lo[i], hi[i]) we compute
///     dst_lo[i] = u00 * lo[i] + u01 * hi[i]
///     dst_hi[i] = u10 * lo[i] + u11 * hi[i]
///
/// The complex-multiply trick (a*b for Complex64 in AVX-2):
///     a       = [a_re, a_im, a_re, a_im, ...]     (each lane is one f64)
///     a_swap  = [a_im, a_re, a_im, a_re, ...]     (permute_pd 0b0101)
///     out     = fmaddsub(b_re_dup, a, b_im_dup * a_swap)
///   where fmaddsub does (x*y - z) on even lanes and (x*y + z) on odd
///   lanes, exactly the (re, im) pattern of a complex multiply.
///
/// On non-x86_64 architectures (or builds without AVX-2 + FMA), the path
/// falls back to a scalar f64 loop that LLVM still autovectorises with
/// SSE2.
#[inline(always)]
pub fn apply_2x2_kernel_f64(
    lo_src: &[num_complex::Complex64],
    hi_src: &[num_complex::Complex64],
    lo_dst: &mut [num_complex::Complex64],
    hi_dst: &mut [num_complex::Complex64],
    u00r: f64, u00i: f64,
    u01r: f64, u01i: f64,
    u10r: f64, u10i: f64,
    u11r: f64, u11i: f64,
) {
    debug_assert_eq!(lo_src.len(), hi_src.len());
    debug_assert_eq!(lo_dst.len(), hi_dst.len());
    debug_assert_eq!(lo_src.len(), lo_dst.len());

    #[cfg(all(target_arch = "x86_64",
              target_feature = "avx2",
              target_feature = "fma"))]
    unsafe {
        apply_2x2_avx2(
            lo_src, hi_src, lo_dst, hi_dst,
            u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i,
        );
        return;
    }

    #[allow(unreachable_code)]
    apply_2x2_scalar(
        lo_src, hi_src, lo_dst, hi_dst,
        u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i,
    );
}

#[inline(always)]
fn apply_2x2_scalar(
    lo_src: &[num_complex::Complex64],
    hi_src: &[num_complex::Complex64],
    lo_dst: &mut [num_complex::Complex64],
    hi_dst: &mut [num_complex::Complex64],
    u00r: f64, u00i: f64,
    u01r: f64, u01i: f64,
    u10r: f64, u10i: f64,
    u11r: f64, u11i: f64,
) {
    let n = lo_src.len();
    // SAFETY: Complex64 is repr(C) with two f64 fields. Layout-identical
    // to a slice of 2N f64. dst slices come from split_at_mut so disjoint.
    let ls: &[f64] = unsafe { std::slice::from_raw_parts(lo_src.as_ptr() as *const f64, n*2) };
    let hs: &[f64] = unsafe { std::slice::from_raw_parts(hi_src.as_ptr() as *const f64, n*2) };
    let ld: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(lo_dst.as_mut_ptr() as *mut f64, n*2) };
    let hd: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(hi_dst.as_mut_ptr() as *mut f64, n*2) };
    let mut i = 0;
    while i < n {
        let ar = ls[2*i];     let ai = ls[2*i + 1];
        let br = hs[2*i];     let bi = hs[2*i + 1];
        let lor = u00r * ar - u00i * ai + u01r * br - u01i * bi;
        let loi = u00r * ai + u00i * ar + u01r * bi + u01i * br;
        let hir = u10r * ar - u10i * ai + u11r * br - u11i * bi;
        let hii = u10r * ai + u10i * ar + u11r * bi + u11i * br;
        ld[2*i] = lor; ld[2*i+1] = loi;
        hd[2*i] = hir; hd[2*i+1] = hii;
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64",
          target_feature = "avx2",
          target_feature = "fma"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_2x2_avx2(
    lo_src: &[num_complex::Complex64],
    hi_src: &[num_complex::Complex64],
    lo_dst: &mut [num_complex::Complex64],
    hi_dst: &mut [num_complex::Complex64],
    u00r: f64, u00i: f64,
    u01r: f64, u01i: f64,
    u10r: f64, u10i: f64,
    u11r: f64, u11i: f64,
) {
    use std::arch::x86_64::*;
    let n = lo_src.len();

    // Broadcast each gate matrix element into a full 256-bit register.
    // Each Complex64 in memory is (re, im) pairs of f64.  A 256-bit AVX
    // register holds 4 f64 = 2 complex.  We work two complex amplitudes
    // at a time.
    let u00r_v = _mm256_set1_pd(u00r);
    let u00i_v = _mm256_set1_pd(u00i);
    let u01r_v = _mm256_set1_pd(u01r);
    let u01i_v = _mm256_set1_pd(u01i);
    let u10r_v = _mm256_set1_pd(u10r);
    let u10i_v = _mm256_set1_pd(u10i);
    let u11r_v = _mm256_set1_pd(u11r);
    let u11i_v = _mm256_set1_pd(u11i);

    let lo_src_ptr = lo_src.as_ptr() as *const f64;
    let hi_src_ptr = hi_src.as_ptr() as *const f64;
    let lo_dst_ptr = lo_dst.as_mut_ptr() as *mut f64;
    let hi_dst_ptr = hi_dst.as_mut_ptr() as *mut f64;

    let pairs = n; // number of complex pairs (lo[i], hi[i])
    let chunks = pairs / 2; // process 2 complex pairs per iteration
    for c in 0..chunks {
        // Load 2 complex from lo_src and hi_src: layout [re0,im0,re1,im1].
        let a = _mm256_loadu_pd(lo_src_ptr.add(c * 4));
        let b = _mm256_loadu_pd(hi_src_ptr.add(c * 4));
        // Swap real/imag within each complex: [im0,re0,im1,re1].
        // _mm256_permute_pd(v, 0b0101) swaps the two doubles within each
        // 128-bit lane.
        let a_swap = _mm256_permute_pd::<0b0101>(a);
        let b_swap = _mm256_permute_pd::<0b0101>(b);

        // Complex mul: u * v = fmaddsub(u_re_v, v, u_im_v * v_swap)
        // u00 * a:
        let mul00_partial = _mm256_mul_pd(u00i_v, a_swap);
        let mul00         = _mm256_fmaddsub_pd(u00r_v, a, mul00_partial);
        // u01 * b:
        let mul01_partial = _mm256_mul_pd(u01i_v, b_swap);
        let mul01         = _mm256_fmaddsub_pd(u01r_v, b, mul01_partial);
        let lo_out = _mm256_add_pd(mul00, mul01);
        _mm256_storeu_pd(lo_dst_ptr.add(c * 4), lo_out);

        // u10 * a:
        let mul10_partial = _mm256_mul_pd(u10i_v, a_swap);
        let mul10         = _mm256_fmaddsub_pd(u10r_v, a, mul10_partial);
        // u11 * b:
        let mul11_partial = _mm256_mul_pd(u11i_v, b_swap);
        let mul11         = _mm256_fmaddsub_pd(u11r_v, b, mul11_partial);
        let hi_out = _mm256_add_pd(mul10, mul11);
        _mm256_storeu_pd(hi_dst_ptr.add(c * 4), hi_out);
    }
    // Tail: handle the last odd amplitude pair scalarly.
    let processed = chunks * 2;
    if processed < pairs {
        let i = processed;
        let ls = std::slice::from_raw_parts(lo_src_ptr.add(i * 2), 2);
        let hs = std::slice::from_raw_parts(hi_src_ptr.add(i * 2), 2);
        let ld = std::slice::from_raw_parts_mut(lo_dst_ptr.add(i * 2), 2);
        let hd = std::slice::from_raw_parts_mut(hi_dst_ptr.add(i * 2), 2);
        let ar = ls[0]; let ai = ls[1];
        let br = hs[0]; let bi = hs[1];
        ld[0] = u00r * ar - u00i * ai + u01r * br - u01i * bi;
        ld[1] = u00r * ai + u00i * ar + u01r * bi + u01i * br;
        hd[0] = u10r * ar - u10i * ai + u11r * br - u11i * bi;
        hd[1] = u10r * ai + u10i * ar + u11r * bi + u11i * br;
    }
}


fn kronecker(
    a: &nalgebra::DMatrix<num_complex::Complex64>,
    b: &nalgebra::DMatrix<num_complex::Complex64>,
) -> nalgebra::DMatrix<num_complex::Complex64> {
    let (ra, ca) = a.shape();
    let (rb, cb) = b.shape();
    let mut res = nalgebra::DMatrix::from_element(ra * rb, ca * cb, num_complex::Complex64::new(0.0, 0.0));

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
            OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }),
            &[0],
        );
        assert_eq!(dag.n_parameters(), 1);
        assert_eq!(dag.parameter_names(), vec!["theta"]);
    }

    #[test]
    fn test_bind_parameters() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }),
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
        dag.add_op(OpType::H, &[0]);           // depth 1 on q0
        dag.add_op(OpType::H, &[1]);           // depth 1 on q1 (parallel)
        dag.add_op(OpType::CNOT, &[0, 1]);     // depth 2 (depends on both H's)
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
}
