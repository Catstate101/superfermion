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

    /// Last DAG node that read or wrote each classical bit. Used to draw
    /// `WireType::Classical` dependency edges so topological order always
    /// runs a conditioned gate after the Measure that set its condition
    /// (and before a later op that rewrites the same bit).
    last_cbit_access: Vec<Option<NodeId>>,

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
            last_cbit_access: vec![None; n_cbits],
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

        // Sequence this measure after any prior op that read or wrote the
        // same classical bit, then record it as the bit's latest accessor so
        // later conditioned gates (or re-measurements) are ordered after it.
        if let Some(prev) = self.last_cbit_access[cbit] {
            self.graph
                .add_edge(prev, new_node, WireType::Classical(cbit));
        }
        self.last_cbit_access[cbit] = Some(new_node);

        new_node
    }

    /// Add a gate that only executes when classical bit `cbit` equals
    /// `value` (classical feed-forward, OpenQASM 3 `if (c[cbit] == value)`
    /// semantics). A classical dependency edge is drawn from the last op
    /// that read or wrote `cbit` (typically the Measure that set it).
    pub fn add_op_conditioned(
        &mut self,
        op_type: OpType,
        qubits: &[QubitId],
        cbit: ClassicalBitId,
        value: u64,
    ) -> NodeId {
        assert!(cbit < self.n_cbits, "Classical bit {} out of range", cbit);
        let new_node = self.add_op(op_type, qubits);
        if let Some(prev) = self.last_cbit_access[cbit] {
            self.graph
                .add_edge(prev, new_node, WireType::Classical(cbit));
        }
        self.graph[new_node].condition = Some((cbit, value));
        self.last_cbit_access[cbit] = Some(new_node);
        new_node
    }

    /// Get the topological ordering of gate nodes (excluding boundary nodes).
    /// This is the execution order that respects all qubit dependencies.
    ///
    /// Stable Kahn's algorithm: among all currently-ready nodes the one
    /// created earliest (lowest stable index) is emitted first. For circuits
    /// built sequentially (every op depends only on earlier ops) this yields
    /// exactly the source order — which the instruction planners rely on to
    /// form long contiguous runs of CNOTs and 1q gates (petgraph's DFS
    /// `toposort` interleaves independent regions and shreds those runs).
    pub fn topological_order(&self) -> Vec<NodeId> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashMap};

        let mut indeg: HashMap<NodeId, usize> = HashMap::new();
        for node in self.graph.node_indices() {
            indeg.insert(
                node,
                self.graph
                    .neighbors_directed(node, petgraph::Direction::Incoming)
                    .count(),
            );
        }
        let mut heap: BinaryHeap<Reverse<usize>> = self
            .graph
            .node_indices()
            .filter(|n| indeg[n] == 0)
            .map(|n| Reverse(n.index()))
            .collect();

        let mut out = Vec::with_capacity(self.graph.node_count());
        let mut seen = 0usize;
        while let Some(Reverse(i)) = heap.pop() {
            let node = NodeId::new(i);
            seen += 1;
            if !self.is_boundary_node(node) {
                out.push(node);
            }
            for succ in self
                .graph
                .neighbors_directed(node, petgraph::Direction::Outgoing)
            {
                if let Some(d) = indeg.get_mut(&succ) {
                    *d -= 1;
                    if *d == 0 {
                        heap.push(Reverse(succ.index()));
                    }
                }
            }
        }
        assert!(seen == self.graph.node_count(), "DAG must be acyclic");
        out
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

    /// Density-matrix simulation: ρ → UρU† applied fully in place.
    ///
    /// The vectorized density matrix lives in the (ket, bra) index space
    /// (`i = ket | (bra << n_qubits)`), so a k-qubit unitary is applied as
    /// two index passes — (U ⊗ I) on the ket bits, then (I ⊗ U*) on the bra
    /// bits — each a plain pair/quad transform over disjoint amplitude
    /// groups. No per-gate allocation and no scattered gathers: the previous
    /// implementation allocated a fresh 4^n buffer (twice) per gate and
    /// visited every output element four times.
    pub fn simulate_dm(&self) -> Vec<num_complex::Complex64> {
        let n = self.n_qubits;
        let dim = 1usize << (2 * n);
        let mut data = vec![num_complex::Complex64::new(0.0, 0.0); dim];
        data[0] = num_complex::Complex64::new(1.0, 0.0);

        for &node_id in &self.topological_order() {
            let op = &self.graph[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                continue;
            }
            let u = op.op_type.to_matrix();
            match op.qubits.len() {
                1 => apply_dm_1q(&mut data, op.qubits[0], n, &u),
                2 => apply_dm_2q(&mut data, op.qubits[0], op.qubits[1], n, &u),
                _ => {
                    // Unsupported arity: keep the historical path (and its
                    // behavior) for exotic multi-qubit unitaries.
                    let mut dm =
                        crate::dm::DensityMatrixState::from_data(std::mem::take(&mut data), n);
                    dm.apply_unitary(&u, &op.qubits);
                    data = dm.into_data();
                }
            }
        }
        data
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
            last_cbit_access: self.last_cbit_access.clone(),
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
    /// work-stealing overhead exceeds the parallelism benefit on the
    /// targeted 4-core/8-thread class; calibrated empirically). Parallel
    /// dispatch is additionally skipped when the Rayon pool has a single
    /// thread (RAYON_NUM_THREADS=1 or 1-core hosts), where the chunked
    /// split would only add overhead.
    const PARALLEL_THRESHOLD: usize = 1 << 16; // 65536 amplitudes ~ n=16

    /// Build the shared simulation plan: pre-compute gate matrices (fusing
    /// consecutive 1q gates on one qubit), then run the algebraic rewrite
    /// passes. Shared by the f64 `simulate` and the opt-in f32 lane.
    #[allow(clippy::type_complexity)]
    pub(crate) fn build_sim_plan(
        &self,
    ) -> (
        Vec<(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)>,
        Vec<SimInst>,
    ) {
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

        // Second planning pass: algebraic rewrites replace whole gate groups
        // with single-pass kernels (CX·D·CX → diagonal 2q; long diagonal
        // runs → one Gray-code phase sweep). See `build_sim_instructions`.
        let insts = build_sim_instructions(&fused, self.n_qubits);
        (fused, insts)
    }

    /// f32 (complex64) statevector lane: identical instruction plan, f32
    /// arithmetic (half-size working set, ~1e-6-class accuracy). Opt-in
    /// via `SF_F32=1` at the binding boundary; see `crate::f32lane`.
    pub fn simulate_f32(&self) -> Vec<num_complex::Complex32> {
        crate::f32lane::simulate_f32(self)
    }

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

        let (fused, insts) = self.build_sim_plan();

        // Optional plan census for tuning (SF_PLAN_DUMP=1, stderr).
        if std::env::var_os("SF_PLAN_DUMP").is_some() {
            let (mut cg, mut d1, mut d2, mut pr, mut fu, mut pm) = (0, 0, 0, 0, 0, 0);
            let mut fu_sizes: Vec<usize> = Vec::new();
            let mut pm_sizes: Vec<usize> = Vec::new();
            for inst in &insts {
                match inst {
                    SimInst::Gate { .. } => cg += 1,
                    SimInst::Diag1 { .. } => d1 += 1,
                    SimInst::Diag2 { .. } => d2 += 1,
                    SimInst::PhaseRun { .. } => pr += 1,
                    SimInst::Fused { qs, .. } => {
                        fu += 1;
                        fu_sizes.push(qs.len());
                    }
                    SimInst::PermuteRun { plan } => {
                        pm += 1;
                        pm_sizes.push(plan.touch.len());
                    }
                }
            }
            eprintln!(
                "[sf-plan] n={} total={} gate={} diag1={} diag2={} phase={} fused={} {:?} perm={} {:?}",
                self.n_qubits, insts.len(), cg, d1, d2, pr, fu, fu_sizes, pm, pm_sizes
            );
            let (mut f_sides, mut f_stages) = (0usize, 0usize);
            for inst in &insts {
                if let SimInst::Fused {
                    pre_diag,
                    post_diag,
                    ..
                } = inst
                {
                    if !pre_diag.is_empty() {
                        f_sides += 1;
                        f_stages += pre_diag.len();
                    }
                    if !post_diag.is_empty() {
                        f_sides += 1;
                        f_stages += post_diag.len();
                    }
                }
            }
            if f_stages > 0 {
                eprintln!("[sf-fold] sides={} stages={}", f_sides, f_stages);
            }
            if insts.len() <= 200 {
                let mut line = String::new();
                for inst in &insts {
                    match inst {
                        SimInst::Gate { idx } => {
                            let op = fused[*idx].0;
                            line.push_str(&format!("G({:?},{:?}) ", op.op_type, op.qubits));
                        }
                        SimInst::Diag1 { q, .. } => line.push_str(&format!("D1({}) ", q)),
                        SimInst::Diag2 { q1, q2, .. } => {
                            line.push_str(&format!("D2({},{}) ", q1, q2))
                        }
                        SimInst::PhaseRun { .. } => line.push_str("PR "),
                        SimInst::Fused { qs, .. } => line.push_str(&format!("F{} ", qs.len())),
                        SimInst::PermuteRun { plan } => {
                            line.push_str(&format!("P{} ", plan.touch.len()))
                        }
                    }
                }
                let short: String = line.chars().take(4000).collect();
                eprintln!("[sf-trace] {}", short);
            }
        }

        // One parallelism decision per call: the planned kernels take it as
        // a parameter instead of re-querying the Rayon pool per instruction.
        let use_par = dim >= Self::PARALLEL_THRESHOLD && rayon::current_num_threads() > 1;

        // Lazy scratch for out-of-place permutation sweeps (perm-merged CNOT
        // runs). The buffer is pooled per thread across calls so repeated
        // simulations at the same size never re-allocate or re-zero the
        // 16·4^n-byte destination; only the first call at a given size pays
        // the zero-init.
        let mut scratch: Vec<num_complex::Complex64> = Vec::new();

        for inst in &insts {
            match inst {
                SimInst::Gate { idx } => {
                    Self::apply_gate_inplace(&mut state, fused[*idx].0, &fused[*idx].1, use_par)
                }
                SimInst::Diag1 { q, d0, d1 } => {
                    apply_diag1_inplace(&mut state, *q, *d0, *d1, use_par)
                }
                SimInst::Diag2 { q1, q2, d } => {
                    apply_diag2_inplace(&mut state, *q1, *q2, d, use_par)
                }
                SimInst::PhaseRun { plan } => {
                    apply_phase_run(&mut state, self.n_qubits, plan, use_par)
                }
                SimInst::Fused {
                    qs,
                    u,
                    pre_diag,
                    post_diag,
                } => apply_fused_qubits_inplace(&mut state, qs, u, pre_diag, post_diag, use_par),
                SimInst::PermuteRun { plan } => {
                    if scratch.len() != dim {
                        let mut s = PERM_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
                        if s.len() < dim {
                            s = vec![num_complex::Complex64::new(0.0, 0.0); dim];
                        } else if s.len() > dim {
                            // A larger buffer left by a previous simulation is
                            // reused with its length clamped back to `dim`:
                            // the swap below must keep the state exactly
                            // 2^n elements long.
                            s.truncate(dim);
                        }
                        scratch = s;
                    }
                    apply_perm_run(&state, &mut scratch, plan, use_par);
                    std::mem::swap(&mut state, &mut scratch);
                }
            }
        }

        // Return the (largest) scratch buffer to the thread pool.
        PERM_SCRATCH.with(|c| {
            let mut slot = c.borrow_mut();
            if slot.len() < scratch.len() {
                *slot = std::mem::take(&mut scratch);
            }
        });

        state
    }

    /// Mid-circuit (dynamic-circuit) simulation: per-shot trajectory replay.
    ///
    /// Each shot restarts from |0...0> and walks the DAG in topological
    /// order (which now respects classical-bit dependencies). Measure ops
    /// sample the qubit from the current state, collapse it, and store the
    /// outcome in the classical register; Reset ops collapse the qubit back
    /// to |0> without recording an outcome; ops carrying a classical
    /// condition are skipped when their register test fails. The final
    /// state of every shot is sampled over all qubits.
    ///
    /// This matches PennyLane's one-shot semantics for finite shots with
    /// mid-circuit measurement (qml.measure + qml.cond).
    pub fn simulate_dynamic(
        &self,
        shots: usize,
        seed: u64,
    ) -> std::collections::HashMap<String, usize> {
        use rand::Rng;
        use rand::SeedableRng;

        let n = self.n_qubits;
        let dim = 1usize << n;
        let use_par = dim >= Self::PARALLEL_THRESHOLD && rayon::current_num_threads() > 1;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut creg = vec![0u64; self.n_cbits];
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for _ in 0..shots {
            let mut state = vec![num_complex::Complex64::new(0.0, 0.0); dim];
            state[0] = num_complex::Complex64::new(1.0, 0.0);
            for b in creg.iter_mut() {
                *b = 0;
            }

            for &node_id in &self.topological_order() {
                let op = &self.graph[node_id];
                if op.op_type.is_boundary() || op.op_type == OpType::Barrier {
                    continue;
                }
                // Every op kind (except Measure/MeasureAll below, which are
                // the sources of classical values) honors a classical
                // condition: skip when the register test fails.
                match &op.op_type {
                    OpType::Measure => {
                        if let Some((cbit, value)) = op.condition {
                            if creg[cbit] != value {
                                continue;
                            }
                        }
                        let q = op.qubits[0];
                        let cbit = op.classical_bits.first().copied().unwrap_or(q);
                        let bit = Self::collapse_measure(&mut state, q, &mut rng);
                        creg[cbit] = bit as u64;
                    }
                    OpType::MeasureAll => {
                        for q in 0..n {
                            let bit = Self::collapse_measure(&mut state, q, &mut rng);
                            creg[q] = bit as u64;
                        }
                    }
                    OpType::Reset => {
                        if let Some((cbit, value)) = op.condition {
                            if creg[cbit] != value {
                                continue;
                            }
                        }
                        let q = op.qubits[0];
                        Self::reset_qubit(&mut state, q, &mut rng);
                    }
                    _ => {
                        if let Some((cbit, value)) = op.condition {
                            if creg[cbit] != value {
                                continue;
                            }
                        }
                        let gate_u = op.op_type.to_matrix();
                        Self::apply_gate_inplace(&mut state, op, &gate_u, use_par);
                    }
                }
            }

            // Sample the final state over all qubits (same orientation and
            // cumulative-search scheme as the non-dynamic samplers).
            let mut cumulative = vec![0.0f64; dim + 1];
            for i in 0..dim {
                let a = state[i];
                cumulative[i + 1] = cumulative[i] + a.re * a.re + a.im * a.im;
            }
            let total = cumulative[dim];
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
        counts
    }

    /// Sample qubit `q` from the current state, project onto the sampled
    /// basis state, and renormalize. Returns the sampled bit.
    fn collapse_measure(
        state: &mut [num_complex::Complex64],
        q: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> u8 {
        use rand::Rng;
        let stride = 1usize << q;
        let mut p0 = 0.0f64;
        for (i, a) in state.iter().enumerate() {
            if (i & stride) == 0 {
                p0 += a.re * a.re + a.im * a.im;
            }
        }
        let bit = if rng.gen::<f64>() < p0 { 0 } else { 1 };
        let pb = if bit == 0 { p0 } else { 1.0 - p0 };
        let scale = 1.0 / pb.sqrt();
        for (i, a) in state.iter_mut().enumerate() {
            if ((i >> q) & 1) == bit {
                *a = num_complex::Complex64::new(a.re * scale, a.im * scale);
            } else {
                *a = num_complex::Complex64::new(0.0, 0.0);
            }
        }
        bit as u8
    }

    /// Mid-circuit reset: measure-and-discard the qubit, then prepare |0>.
    /// (Sample the outcome like a measure, collapse onto it, then flip the
    /// qubit back to |0> when the sampled bit was 1. No outcome is stored.)
    fn reset_qubit(state: &mut [num_complex::Complex64], q: usize, rng: &mut rand::rngs::StdRng) {
        use rand::Rng;
        let stride = 1usize << q;
        let mut p0 = 0.0f64;
        for (i, a) in state.iter().enumerate() {
            if (i & stride) == 0 {
                p0 += a.re * a.re + a.im * a.im;
            }
        }
        let bit = if rng.gen::<f64>() < p0 { 0 } else { 1 };
        let pb = if bit == 0 { p0 } else { 1.0 - p0 };
        let scale = 1.0 / pb.sqrt();
        for (i, a) in state.iter_mut().enumerate() {
            if ((i >> q) & 1) == bit {
                *a = num_complex::Complex64::new(a.re * scale, a.im * scale);
            } else {
                *a = num_complex::Complex64::new(0.0, 0.0);
            }
        }
        if bit == 1 {
            // Prepare |0>: flip the (now pure) qubit with X.
            for i in 0..state.len() {
                let partner = i ^ stride;
                if (i & stride) == 0 {
                    state.swap(i, partner);
                }
            }
        }
    }

    /// Apply a gate in-place on the statevector. Each gate transforms
    /// independent amplitude pairs/groups, so no second buffer is needed.
    fn apply_gate_inplace(
        state: &mut [num_complex::Complex64],
        op: &crate::dag::QuantumOp,
        gate_u: &nalgebra::DMatrix<num_complex::Complex64>,
        use_par: bool,
    ) {
        let dim = state.len();

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
                apply_diag1_inplace(state, t, u00, u11, use_par);
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

            // Diagonal 2q: scale each amplitude by its diagonal element.
            // Detected from the ACTUAL matrix (not op_type) so structurally
            // diagonal gates (CP, fused unitaries, ...) take the
            // single-multiply-per-amplitude path as well.
            let is_diag_2q = gate_u[(0, 1)].norm() < 1e-14
                && gate_u[(0, 2)].norm() < 1e-14
                && gate_u[(0, 3)].norm() < 1e-14
                && gate_u[(1, 0)].norm() < 1e-14
                && gate_u[(1, 2)].norm() < 1e-14
                && gate_u[(1, 3)].norm() < 1e-14
                && gate_u[(2, 0)].norm() < 1e-14
                && gate_u[(2, 1)].norm() < 1e-14
                && gate_u[(2, 3)].norm() < 1e-14
                && gate_u[(3, 0)].norm() < 1e-14
                && gate_u[(3, 1)].norm() < 1e-14
                && gate_u[(3, 2)].norm() < 1e-14;
            if is_diag_2q {
                let d = [
                    gate_u[(0, 0)],
                    gate_u[(1, 1)],
                    gate_u[(2, 2)],
                    gate_u[(3, 3)],
                ];
                apply_diag2_inplace(state, q1, q2, &d, use_par);
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

std::thread_local! {
    /// Per-thread pool for the out-of-place permutation destination buffer
    /// (kept at the largest size seen), so repeated simulations reuse one
    /// allocation instead of allocating + zeroing 16·4^n bytes per call.
    #[allow(clippy::missing_const_for_thread_local)] // init already const; clippy 1.93 false positive
    static PERM_SCRATCH: std::cell::RefCell<Vec<num_complex::Complex64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

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
    /// Raw base pointer (kernel/prefetch use; address only).
    #[inline(always)]
    fn raw(&self) -> *mut num_complex::Complex64 {
        self.0
    }
    #[inline(always)]
    unsafe fn swap(&self, a: usize, b: usize) {
        let tmp = *self.0.add(a);
        *self.0.add(a) = *self.0.add(b);
        *self.0.add(b) = tmp;
    }

    /// Swap two disjoint contiguous runs of `count` elements. Runs must not
    /// overlap (callers guarantee this by construction).
    #[inline(always)]
    unsafe fn swap_run(&self, a: usize, b: usize, count: usize) {
        crate::simd::swap_runs(self.0, a, b, count);
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

/// Apply ρ → UρU† for a 1q gate on qubit `q` of an n-qubit density matrix
/// (in place): a pair pass with U on the ket bit, then with conj(U) on the
/// bra bit (n + q). Disjoint pairs make both passes safely in-place.
fn apply_dm_1q(
    data: &mut [num_complex::Complex64],
    q: usize,
    _n: usize,
    u: &nalgebra::DMatrix<num_complex::Complex64>,
) {
    // Fused single-pass kernel: the ket (bit q) and bra (bit n+q) transforms
    // commute, so one sweep over the closed 4-cycles replaces the previous
    // two full-array pair passes (half the memory traffic, same arithmetic
    // order).
    let m = [[u[(0, 0)], u[(0, 1)]], [u[(1, 0)], u[(1, 1)]]];
    crate::simd::dm_1q_fused(data, q, m);
}

/// Apply ρ → UρU† for a 2q gate on qubits (`q0`, `q1`) (in place), with the
/// same row convention as the statevector path (row = bit(q0)*2 + bit(q1)).
fn apply_dm_2q(
    data: &mut [num_complex::Complex64],
    q0: usize,
    q1: usize,
    _n: usize,
    u: &nalgebra::DMatrix<num_complex::Complex64>,
) {
    // Fused single-pass kernel over the closed 16-blocks: ket 4×4 then bra
    // conj 4×4, register-resident (replaces two full-array 4-way passes).
    let mut g = [[num_complex::Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            g[r][c] = u[(r, c)];
        }
    }
    crate::simd::dm_2q_fused(data, q0, q1, &g);
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
            unsafe {
                sp.swap_run(base, base + stride, stride);
            }
        });
    } else {
        let ptr = state.as_mut_ptr();
        for g in 0..n_groups {
            let base = g * block;
            unsafe {
                crate::simd::swap_runs(ptr, base, base + stride, stride);
            }
        }
    }
}

/// CNOT in-place: flip the target bit wherever the control bit is 1.
///
/// Block-structured implementation: with `lo`/`hi` the lower/upper qubit-bit
/// positions, each block of `2^(hi+1)` amplitudes contains `2^(hi-lo-1)`
/// disjoint contiguous-run swaps of length `2^lo`. Both sides of every swap
/// are therefore sequential in memory and exchanged with
/// `swap_nonoverlapping` (vectorizable, branch-free) — the previous
/// elementwise bit-test scan tested two bits per amplitude and swapped
/// across a 2^lo stride, hurting cache and blocking auto-vectorization.
fn inplace_cnot(state: &mut [num_complex::Complex64], ctrl: usize, tgt: usize, use_par: bool) {
    let dim = state.len();
    let (lo, hi) = if ctrl < tgt { (ctrl, tgt) } else { (tgt, ctrl) };
    let run = 1usize << lo; // contiguous run length of each swap side
    let hi_bit = 1usize << hi;
    let block = hi_bit << 1;
    let n_blocks = dim / block;
    let n_runs = hi_bit >> (lo + 1); // runs per block
                                     // Offset of the first (lower-address) run inside a block, and the
                                     // distance to its partner run, for both control/target orderings:
                                     //   ctrl < tgt (ctrl=lo): pairs (run..|bit hi=0|, +hi_bit)
                                     //   ctrl > tgt (ctrl=hi): pairs (hi_bit + even*runs, +run)
    let (a0, delta) = if ctrl < tgt {
        (run, hi_bit)
    } else {
        (hi_bit, run)
    };

    if use_par {
        // Flatten (block, run, tile) into one index space. Tiling bounds the
        // per-task swap length (a single top-qubit run can be megabytes long),
        // so every thread always gets disjoint, reasonably sized work.
        const TILE: usize = 4096;
        let tile = run.min(TILE);
        let n_tiles = run / tile;
        let per_block = n_runs * n_tiles;
        let sp = SendPtr(state.as_mut_ptr());
        (0..n_blocks * per_block).into_par_iter().for_each(|t| {
            let base = (t / per_block) * block;
            let rem = t % per_block;
            let a = base + a0 + ((rem / n_tiles) << 1) * run + (rem % n_tiles) * tile;
            let b = a + delta;
            unsafe {
                sp.swap_run(a, b, tile);
            }
        });
    } else {
        let ptr = state.as_mut_ptr();
        for g in 0..n_blocks {
            let base = g * block;
            for r in 0..n_runs {
                let a = base + a0 + (r << 1) * run;
                let b = a + delta;
                unsafe {
                    crate::simd::swap_runs(ptr, a, b, run);
                }
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
    let m = [
        [
            num_complex::Complex64::new(u00r, u00i),
            num_complex::Complex64::new(u01r, u01i),
        ],
        [
            num_complex::Complex64::new(u10r, u10i),
            num_complex::Complex64::new(u11r, u11i),
        ],
    ];
    let block = stride * 2;
    if use_par {
        // Chunks aligned to the pair-block size so SIMD blocks never straddle
        // a chunk boundary.
        let chunk = (state.len() / 16).max(1024).div_ceil(block) * block;
        state
            .par_chunks_mut(chunk)
            .for_each(|s| crate::simd::pair_pass(s, stride, m));
    } else {
        crate::simd::pair_pass(state, stride, m);
    }
}

/// General 2q gate in-place: transform groups of 4 amplitudes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inplace_2q_general(
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
/// Scans the gate list linearly, accumulating per-qubit 2x2 products. When a
/// multi-qubit gate arrives, ALL pending accumulators are flushed (in qubit
/// order) before it is emitted: deferred 1q gates never split a run of
/// multi-qubit gates, and consecutive 1q gates stay contiguous — both
/// properties are relied upon by the later run-merging passes
/// (`fuse_1q_blocks`, `merge_cnot_runs`).
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
            // Multi-qubit gate: flush ALL pending 1q accumulators (in qubit
            // order) before emitting it, so deferred 1q gates never split a
            // run of multi-qubit gates and consecutive 1q gates stay
            // contiguous for the later run-merging passes.
            for q in 0..n_qubits {
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

// ──────────────────────────────────────────────────────────────────────
// Instruction planning: algebraic rewrites applied before the apply loop.
//
// Three rewrites run on the (already 1q-fused) linear gate list:
//
//   1. CX(a,b) · D(b) · CX(a,b) → one diagonal-2q pass   (D diagonal)
//      CX(a,b) · D(a) · CX(a,b) → one diagonal-1q pass   (CX commutes with
//                                                         control-side ops)
//      CX(a,b) · D(x) · CX(a,b) → D(x) alone              (x outside {a,b}:
//                                                         the CXs cancel)
//   2. Maximal runs of diagonal instructions (≥ the merge break-even for the
//      current lane count) are collapsed into ONE pass multiplying every
//      amplitude by exp(i·Φ(bits)),
//      where Φ is the quadratic pseudo-Boolean form accumulated from the
//      per-gate diagonal entries. Diagonal gates mutually commute, so the
//      run is order-independent and the reconstruction is exact.
//   3. Diagonal-2q instructions a nearby fused 1q block can host are folded
//      into that block's cache-resident group buffer (`pre_diag`/`post_diag`
//      of `SimInst::Fused`): a diagonal commutes through every disjoint-qubit
//      instruction, so its phase applies during the sweep already streaming
//      those amplitudes, at the same per-entry arithmetic as a standalone
//      diagonal pass. Blocks that cannot host every leftover diagonal are
//      re-partitioned when a pair-wide window exists; the rest fall through
//      to rewrite 2.
//
// Any 2π ambiguity in the entries' arguments is harmless: every coefficient
// multiplies an integer bit monomial (b_q or b_q·b_p), so exp(2πi·k·mono)=1.
// ──────────────────────────────────────────────────────────────────────

/// One diagonal 2q phase attached to a `Fused` block by
/// `fold_diags_into_fused`. Entry `s` of the block's group buffer
/// multiplies by `d[(b_i << 1) | b_j]`, where `b_i`/`b_j` are bits `i`/`j`
/// of `s` (positions inside `Fused::qs`) — the exact per-entry multiply a
/// standalone `Diag2 { q1, q2, d }` on those two qubits performs, only
/// cache-resident inside the sweep that is already streaming the block.
#[derive(Clone, Copy)]
pub(crate) struct Diag2Stage {
    pub(crate) i: u8,
    pub(crate) j: u8,
    pub(crate) d: [num_complex::Complex64; 4],
}

/// One planned step for `QuantumDAG::simulate`.
#[derive(Clone)]
pub(crate) enum SimInst {
    /// Apply the precomputed gate matrix `fused[idx].1` to `fused[idx].0.qubits`
    /// via the standard per-op kernel.
    Gate { idx: usize },
    /// Diagonal 1q gate: state[i] *= (bit q == 0 ? d0 : d1)
    Diag1 {
        q: usize,
        d0: num_complex::Complex64,
        d1: num_complex::Complex64,
    },
    /// Diagonal 2q gate: entry indexed by (bit(q1) << 1) | bit(q2)
    Diag2 {
        q1: usize,
        q2: usize,
        d: [num_complex::Complex64; 4],
    },
    /// A whole run of diagonal gates collapsed into one Gray-code phase sweep.
    PhaseRun { plan: PhasePlan },
    /// A fused block of k single-qubit gates on distinct qubits (k = qs.len(),
    /// bit s of the group index ↔ qubit qs[s]), applied as ONE sweep over
    /// 2^k-element groups. `u` holds k row-major 2×2 blocks (4 complex
    /// entries each), gate t = `u[4t..4t+4]` acting on `qs[t]`; because the
    /// qubits are distinct the gates commute and their in-group order is
    /// irrelevant. Built from maximal runs of 1q gates on distinct qubits.
    /// `pre_diag`/`post_diag` are diagonal 2q phases folded into this same
    /// sweep (applied to each group buffer before/after the `u` stages).
    Fused {
        qs: SmallVec<[usize; 8]>,
        u: Vec<num_complex::Complex64>,
        pre_diag: Vec<Diag2Stage>,
        post_diag: Vec<Diag2Stage>,
    },
    /// A composed run of CNOTs applied as ONE out-of-place permutation sweep.
    PermuteRun { plan: PermPlan },
}

/// Φ(bits) = global + Σ_q lin[q]·b_q + Σ_{q<p} quad[q][p]·b_q·b_p
#[derive(Clone)]
pub(crate) struct PhasePlan {
    pub(crate) global: f64,
    pub(crate) lin: Vec<f64>,
    /// Dense symmetric n×n matrix; diagonal entries are never used.
    pub(crate) quad: Vec<f64>,
    pub(crate) n: usize,
}

/// GF(2)-linear index permutation: dest(i) = XOR of `cols[q]` over set bits q.
#[derive(Clone)]
pub(crate) struct PermPlan {
    pub(crate) cols: Vec<usize>,
    /// Every qubit appearing in the composed CNOT run (conservative hazard
    /// set for the diagonal-clustering pass).
    pub(crate) touch: Vec<usize>,
    /// Byte-chunk gather tables for the INVERSE map, laid out as `n/8`
    /// blocks of 256 `usize` entries: `src(d) = XOR of table[(d >> 8c) &
    /// 0xFF]`. Built once per plan by `build_perm_gather_tables` so the
    /// sweep can write `out[d]` sequentially while gathering `state[src(d)]`
    /// without a per-element bit scan. Empty when the tables cannot be
    /// built (the sweep then falls back to the scatter loop).
    pub(crate) inv_tables: Vec<usize>,
}

/// Minimum diagonal-run length for the merged phase sweep to pay off at one
/// lane (the sweep costs one sin/cos per amplitude; individual passes cost one
/// multiply per amplitude each, so short runs are cheaper applied directly).
const MERGE_MIN_RUN: usize = 32;

/// Dense-state size below which every plan rewrite loses: the state stays
/// cache-resident (8 MB L3 here covers n<=18) and the rewrite only adds
/// per-group scalar work plus one sin/cos per amplitude, while the wins all
/// come from trading DRAM re-streams for compute. Measured with the plain
/// parallel gate (2^16): the auto policy was 1.34-2.63x SLOWER at n=16-18
/// while n=20-24 (>=16 MB, DRAM-bound) won 1.5-2.6x — so auto rewriting
/// starts only where the state stops fitting in L3.
const AUTO_REWRITE_MIN_DIM: usize = 1 << 20;

/// Effective merge break-even for the current lane count, overridable via
/// `SF_MERGE_MIN_RUN` (entry counts).
///
/// The sweep's per-amplitude cost is compute-heavy (phase evaluation plus one
/// sin/cos) and splits across worker lanes, while each individual diagonal
/// pass is a streaming multiply that is bandwidth-bound and does not scale,
/// so the break-even run length falls as lanes are added. Measured
/// crossovers on the n=22 zz shape (sweep cost / per-pass cost): ~32 entries
/// on one lane, ~27 at 2, ~7 at 4, ~14 at 8 (a 21-entry run costs -127 ms and
/// -125 ms at 1-2 lanes but saves +171 ms and +95 ms at 4-8 lanes). Cells
/// below `AUTO_REWRITE_MIN_DIM` keep the one-lane break-even: the merged
/// sweep re-reads the whole state just like the per-gate passes, so on a
/// cache-resident state it buys nothing and only costs the phase evaluation.
fn merge_min_run(n_qubits: usize) -> usize {
    if let Ok(v) = std::env::var("SF_MERGE_MIN_RUN") {
        if let Ok(m) = v.trim().parse::<usize>() {
            return m.max(2);
        }
    }
    if (1usize << n_qubits) < AUTO_REWRITE_MIN_DIM {
        return MERGE_MIN_RUN;
    }
    match rayon::current_num_threads() {
        0 | 1 => MERGE_MIN_RUN,
        2 | 3 => 28,
        _ => 15,
    }
}

/// Minimum CNOT-run length for the perm-merged single sweep to pay off
/// (one out-of-place sweep ≈ 2-4 individual CX passes).
const PERM_MIN_RUN: usize = 4;

/// Dense states above this size skip run merging (the scratch buffer would
/// double peak memory; sequential per-CX passes are kept instead).
const PERM_MAX_DIM: usize = 1 << 24;

#[inline(always)]
pub(crate) fn is_diag_1q(mat: &nalgebra::DMatrix<num_complex::Complex64>) -> bool {
    mat.nrows() == 2 && mat[(0, 1)].norm() < 1e-14 && mat[(1, 0)].norm() < 1e-14
}

#[inline(always)]
pub(crate) fn is_diag_2q(mat: &nalgebra::DMatrix<num_complex::Complex64>) -> bool {
    if mat.nrows() != 4 {
        return false;
    }
    let mut ok = true;
    for r in 0..4 {
        for c in 0..4 {
            if r != c && mat[(r, c)].norm() >= 1e-14 {
                ok = false;
            }
        }
    }
    ok
}

#[inline(always)]
fn unit_modulus(z: num_complex::Complex64) -> bool {
    (z.norm() - 1.0).abs() < 1e-9
}

/// Plan the instruction list from the 1q-fused gate list.
fn build_sim_instructions(
    ops: &[(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)],
    n_qubits: usize,
) -> Vec<SimInst> {
    let mut insts: Vec<SimInst> = Vec::with_capacity(ops.len() + 1);
    let mut i = 0;
    while i < ops.len() {
        // ── Rewrite 1: CNOT(a,b) · D · CNOT(a,b) ────────────────────────
        if i + 2 < ops.len() {
            let op0 = ops[i].0;
            let op2 = ops[i + 2].0;
            if op0.op_type == OpType::CNOT
                && op2.op_type == OpType::CNOT
                && op0.qubits == op2.qubits
            {
                let mid_op = ops[i + 1].0;
                let mid_mat = &ops[i + 1].1;
                if mid_op.qubits.len() == 1 && is_diag_1q(mid_mat) {
                    let q = mid_op.qubits[0];
                    let c = op0.qubits[0];
                    let t = op0.qubits[1];
                    let m0 = mid_mat[(0, 0)];
                    let m1 = mid_mat[(1, 1)];
                    if q == t {
                        // CX·D(t)·CX = diag(m0, m1, m1, m0) — the ZZ phase.
                        insts.push(SimInst::Diag2 {
                            q1: c,
                            q2: t,
                            d: [m0, m1, m1, m0],
                        });
                    } else {
                        // D on the control commutes with CX; D on a disjoint
                        // qubit cancels both CXs. Either way the pair drops.
                        insts.push(SimInst::Diag1 { q, d0: m0, d1: m1 });
                    }
                    i += 3;
                    continue;
                }
            }
        }

        // ── Fallback: classify, else keep as a plain gate ───────────────
        let (op, mat) = (ops[i].0, &ops[i].1);
        if op.qubits.len() == 1 && is_diag_1q(mat) {
            insts.push(SimInst::Diag1 {
                q: op.qubits[0],
                d0: mat[(0, 0)],
                d1: mat[(1, 1)],
            });
        } else if op.qubits.len() == 2 && is_diag_2q(mat) {
            insts.push(SimInst::Diag2 {
                q1: op.qubits[0],
                q2: op.qubits[1],
                d: [mat[(0, 0)], mat[(1, 1)], mat[(2, 2)], mat[(3, 3)]],
            });
        } else {
            insts.push(SimInst::Gate { idx: i });
        }
        i += 1;
    }

    // Third planning pass: consolidate instruction runs into fewer sweeps —
    // (a) maximal runs of commuting 1q gates on distinct qubits become ONE
    //     k-qubit Fused kernel call;
    // (b) maximal runs of consecutive CNOTs become ONE permutation sweep.
    let insts = fuse_1q_blocks(insts, ops, n_qubits);
    let insts = merge_cnot_runs(insts, ops, 1usize << n_qubits);
    let insts = fold_diags_into_fused(insts);

    cluster_commuting_diags(insts, ops, n_qubits)
}

/// Cluster diagonal instructions across commuting (disjoint-qubit) gates.
///
/// A diagonal instruction can be deferred past any gate whose qubit set is
/// disjoint from it (disjoint-qubit unitaries commute), so diagonals that
/// are interleaved with such gates are collected into ONE merged phase
/// sweep. Whenever an incoming gate conflicts with a deferred diagonal, the
/// whole buffer is flushed (as a merged `PhaseRun` when long enough, else as
/// individual instructions) before that gate is applied.
fn cluster_commuting_diags(
    insts: Vec<SimInst>,
    ops: &[(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)],
    n_qubits: usize,
) -> Vec<SimInst> {
    fn inst_qubits(
        inst: &SimInst,
        ops: &[(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)],
    ) -> Vec<usize> {
        match inst {
            SimInst::Gate { idx, .. } => ops[*idx].0.qubits.iter().copied().collect(),
            SimInst::Diag1 { q, .. } => vec![*q],
            SimInst::Diag2 { q1, q2, .. } => vec![*q1, *q2],
            SimInst::Fused { qs, .. } => qs.to_vec(),
            SimInst::PermuteRun { plan } => plan.touch.clone(),
            // A PhaseRun (not produced by the planner) touches every qubit.
            SimInst::PhaseRun { plan } => (0..plan.n).collect(),
        }
    }

    let mut out: Vec<SimInst> = Vec::with_capacity(insts.len() + 1);
    let mut pending: Vec<SimInst> = Vec::new();
    let mut used = vec![false; n_qubits];
    let min_run = merge_min_run(n_qubits);

    let flush = |out: &mut Vec<SimInst>, pending: &mut Vec<SimInst>, used: &mut [bool]| {
        if pending.is_empty() {
            return;
        }
        if pending.len() >= min_run {
            if let Some(plan) = build_phase_plan(pending, n_qubits) {
                out.push(SimInst::PhaseRun { plan });
                pending.clear();
                for f in used.iter_mut() {
                    *f = false;
                }
                return;
            }
        }
        for inst in pending.drain(..) {
            out.push(inst);
        }
        for f in used.iter_mut() {
            *f = false;
        }
    };

    for inst in insts {
        let is_diag = matches!(inst, SimInst::Diag1 { .. } | SimInst::Diag2 { .. });
        if is_diag {
            for q in inst_qubits(&inst, ops) {
                used[q] = true;
            }
            pending.push(inst);
            continue;
        }
        let qs = inst_qubits(&inst, ops);
        if qs.iter().any(|&q| used[q]) {
            flush(&mut out, &mut pending, &mut used);
        }
        out.push(inst);
    }
    flush(&mut out, &mut pending, &mut used);
    out
}

/// Largest supported fused-block size (the group buffer holds 2^k entries).
const FUSE_K_MAX: usize = 10;

/// Max number of qubits fused into one k-qubit pass (2^k values per group).
/// Overridable at runtime via `SF_FUSE_K` (1 disables the rewrite).
///
/// The fused kernel trades DRAM traffic for per-group work, so its
/// break-even is the lane count: measured on 20-24q cells, fusion pays from 4
/// lanes up (1.28-1.81x at 8) while at 1-2 lanes every K loses (0.22-0.98x vs
/// one history-faithful pair_pass sweep per gate).
///
/// The measured K-optimum (same-window lane-8 probes at 20/22/23/24 q,
/// 1 warm + 3 reps p50): K=8 wins across 2^20-2^22 (1.09-1.13x over K=6 at
/// 22q; K=10 loses 1.36-1.38x at 22q and 1.22x on zz_20q, ties on vqc_20q),
/// but the 2^K gather window blows past L3 at 23-24q: K=8 collapses there
/// (vqc_24q 1603 vs 839 ms at K=6; zz_24q 1857 vs 841; vqc_23q 546 vs 401;
/// zz_23q 583 vs 418; K=7 and K=5 also lose to 6; K=10 1.16-1.31x worse).
/// Hence: 8 below 23 qubits, 6 at/above. K=9/10 stay reachable via SF_FUSE_K
/// for experiments only. The former "rayon threads >= 4" condition is gone:
/// the fused plan also wins on 1 thread (lane1 micro 2026-09-17 vs the unfused
/// gate path: zz_20 1.19x, ang_20 1.47x, zz_22 1.26x, ang_22 1.58x at K=8), and
/// qsim/qiskit lane1 baselines sit near their lane8 numbers, so the old gate
/// only cost the lane1 comparisons.
/// Cells below `AUTO_REWRITE_MIN_DIM` keep the faithful pair-pass plan: their
/// state is cache-resident, so there is no DRAM traffic to trade (measured
/// 1.34-2.63x slower at n=16-18 under a plain parallel gate).
fn fuse_1q_max_k(n_qubits: usize) -> usize {
    if let Ok(v) = std::env::var("SF_FUSE_K") {
        if let Ok(k) = v.trim().parse::<usize>() {
            return k.clamp(1, FUSE_K_MAX);
        }
    }
    if (1usize << n_qubits) >= AUTO_REWRITE_MIN_DIM {
        if n_qubits >= 23 {
            6
        } else {
            8
        }
    } else {
        1
    }
}

/// Scatter the low `masks.len()` bits of `s` onto the bits of `masks`.
#[inline(always)]
fn spread_bits(s: usize, masks: &[usize]) -> usize {
    let mut out = 0usize;
    for (b, &m) in masks.iter().enumerate() {
        if (s >> b) & 1 == 1 {
            out |= m;
        }
    }
    out
}

/// Insert zero bits at the (ascending) positions of `masks` in `g` — maps a
/// group index to the base amplitude index with all masked qubit bits cleared.
#[inline(always)]
fn deposit_bits_masked(g: usize, masks: &[usize]) -> usize {
    let mut x = g;
    for &m in masks {
        let p = m.trailing_zeros() as usize;
        let low = (1usize << p) - 1;
        x = (x & low) | ((x & !low) << 1);
    }
    x
}

/// Fuse maximal runs of consecutive 1q instructions on distinct qubits into
/// ONE `Fused` sweep (k ≤ SF_FUSE_K). The per-qubit 2×2 matrices are already
/// circuit-ordered by `fuse_1q_sequence`; disjoint qubits commute, so the
/// kernel simply applies them one after another to each 2^k-element group —
/// same arithmetic as k separate passes, 1/k of the memory traffic.
fn fuse_1q_blocks(
    insts: Vec<SimInst>,
    ops: &[(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)],
    n_qubits: usize,
) -> Vec<SimInst> {
    let kmax = fuse_1q_max_k(n_qubits);
    if kmax < 2 {
        return insts;
    }

    let one_q = |inst: &SimInst| -> Option<(usize, [[num_complex::Complex64; 2]; 2])> {
        match inst {
            SimInst::Gate { idx } => {
                let op = ops[*idx].0;
                if op.qubits.len() != 1 {
                    return None;
                }
                let mat = &ops[*idx].1;
                Some((
                    op.qubits[0],
                    [[mat[(0, 0)], mat[(0, 1)]], [mat[(1, 0)], mat[(1, 1)]]],
                ))
            }
            SimInst::Diag1 { q, d0, d1 } => Some((
                *q,
                [
                    [*d0, num_complex::Complex64::new(0.0, 0.0)],
                    [num_complex::Complex64::new(0.0, 0.0), *d1],
                ],
            )),
            _ => None,
        }
    };

    let mut out = Vec::with_capacity(insts.len());
    let mut i = 0;
    while i < insts.len() {
        let mut qs: Vec<usize> = Vec::new();
        let mut mats: Vec<[[num_complex::Complex64; 2]; 2]> = Vec::new();
        let mut j = i;
        while j < insts.len() && qs.len() < kmax {
            match one_q(&insts[j]) {
                Some((q, m)) if !qs.contains(&q) => {
                    qs.push(q);
                    mats.push(m);
                    j += 1;
                }
                _ => break,
            }
        }
        if qs.len() >= 2 {
            // Order factors by ascending qubit; bit s of the group index is
            // qubit qs[s] (matches `deposit_bits_masked`). Flatten the k 2×2
            // blocks as [u00, u01, u10, u11] each.
            let mut order: Vec<usize> = (0..qs.len()).collect();
            order.sort_by_key(|&t| qs[t]);
            let mut sv: SmallVec<[usize; 8]> = SmallVec::new();
            let mut u: Vec<num_complex::Complex64> = Vec::with_capacity(4 * qs.len());
            for &t in &order {
                sv.push(qs[t]);
                u.push(mats[t][0][0]);
                u.push(mats[t][0][1]);
                u.push(mats[t][1][0]);
                u.push(mats[t][1][1]);
            }
            out.push(SimInst::Fused {
                qs: sv,
                u,
                pre_diag: Vec::new(),
                post_diag: Vec::new(),
            });
            i = j;
        } else {
            out.push(insts[i].clone());
            i += 1;
        }
    }
    out
}

/// Qubit list of a fused block (empty for every other instruction).
fn fused_qs(inst: &SimInst) -> &[usize] {
    match inst {
        SimInst::Fused { qs, .. } => qs,
        _ => &[],
    }
}

/// Fold diagonal 2q instructions into the adjacent fused 1q blocks.
///
/// A `Diag2` on (q1,q2) commutes with every instruction whose qubit set is
/// disjoint from it, so it can drift through consecutive `Fused` blocks until
/// one of their windows contains both qubits and then rides along inside that
/// block's cache-resident group buffer — `post_diag` when it attaches to the
/// block preceding it (D ∘ U), `pre_diag` when it attaches to a block that
/// follows (U ∘ D). One whole-state sweep per diagonal is traded for one
/// multiply per entry inside the host block's gather/scatter. Diagonals no
/// run can host stay
/// standalone and take the existing merge/phase path. `SF_FOLD=0` disables
/// the pass (A/B attribution).
///
/// The zz benchmark shape folds completely: the H+RZ head and the H tail are
/// two fused runs with the RZZ chain between them; the backward run absorbs
/// every pair inside its windows, and the forward run is re-partitioned
/// (union-find over the spanning pairs, then a greedy window fill) so the two
/// boundary-spanning pairs fit as well — the 7-sweep plan (6 fused + 1 merged
/// phase) becomes 6 sweeps with no phase pass.
fn fold_diags_into_fused(insts: Vec<SimInst>) -> Vec<SimInst> {
    if std::env::var("SF_FOLD").map(|v| v == "0").unwrap_or(false) {
        return insts;
    }

    fn holds(qs: &[usize], q1: usize, q2: usize) -> bool {
        qs.contains(&q1) && qs.contains(&q2)
    }
    fn touches(qs: &[usize], q1: usize, q2: usize) -> bool {
        qs.contains(&q1) || qs.contains(&q2)
    }
    fn local_pos(qs: &[usize], q: usize) -> u8 {
        qs.iter().position(|&x| x == q).unwrap() as u8
    }

    enum Slot {
        /// Post-attach stage on the fused block at this `out` index.
        Back(usize),
        /// Pre-attach stage on this block of the forward run.
        Fwd(usize),
        /// Hostable once the forward run is re-partitioned.
        Repair,
        /// Stays a standalone diagonal.
        Left,
    }

    let mut out: Vec<SimInst> = Vec::with_capacity(insts.len());
    let mut i = 0;
    while i < insts.len() {
        if !matches!(insts[i], SimInst::Diag1 { .. } | SimInst::Diag2 { .. }) {
            out.push(insts[i].clone());
            i += 1;
            continue;
        }
        // Maximal run of diagonal instructions at the cursor.
        let mut j = i;
        while j < insts.len() && matches!(insts[j], SimInst::Diag1 { .. } | SimInst::Diag2 { .. }) {
            j += 1;
        }
        // Backward host run: trailing fused blocks already emitted.
        let back_len = out
            .iter()
            .rev()
            .take_while(|x| matches!(x, SimInst::Fused { .. }))
            .count();
        // Forward host run: fused blocks right after the diagonal run.
        let fwd_start = j;
        let mut fwd_len = 0;
        while fwd_start + fwd_len < insts.len()
            && matches!(insts[fwd_start + fwd_len], SimInst::Fused { .. })
        {
            fwd_len += 1;
        }

        // ── Decide where every pending Diag2 goes ────────────────────────
        let pairs: Vec<Option<(usize, usize)>> = (i..j)
            .map(|k| match &insts[k] {
                SimInst::Diag2 { q1, q2, .. } => Some((*q1, *q2)),
                _ => None,
            })
            .collect();
        let mut slots: Vec<Option<Slot>> = Vec::with_capacity(pairs.len());
        for p in &pairs {
            let (q1, q2) = match *p {
                Some(pr) => pr,
                None => {
                    slots.push(None);
                    continue;
                }
            };
            let mut slot = None;
            // Nearest host first: drift left through disjoint blocks.
            for b in (out.len() - back_len..out.len()).rev() {
                let qs = fused_qs(&out[b]);
                if holds(qs, q1, q2) {
                    slot = Some(Slot::Back(b));
                    break;
                }
                if touches(qs, q1, q2) {
                    break; // a later block gates q1/q2 — cannot move past it
                }
            }
            if slot.is_none() {
                for f in 0..fwd_len {
                    let qs = fused_qs(&insts[fwd_start + f]);
                    if holds(qs, q1, q2) {
                        slot = Some(Slot::Fwd(f));
                        break;
                    }
                    if touches(qs, q1, q2) {
                        slot = Some(Slot::Repair);
                        break;
                    }
                }
            }
            slots.push(Some(slot.unwrap_or(Slot::Left)));
        }

        // ── Repair: re-partition the forward run so spanning pairs fit ───
        let mut fwd_blocks: Vec<SimInst> = insts[fwd_start..fwd_start + fwd_len].to_vec();
        let any_repair = slots.iter().any(|s| matches!(s, Some(Slot::Repair)));
        if any_repair {
            let wanted: Vec<(usize, usize)> = slots
                .iter()
                .zip(&pairs)
                .filter_map(|(s, p)| match (s, p) {
                    (Some(Slot::Fwd(_)) | Some(Slot::Repair), Some(pr)) => Some(*pr),
                    _ => None,
                })
                .collect();
            match refold_forward_run(&fwd_blocks, &wanted) {
                Some(rebuilt) => {
                    fwd_blocks = rebuilt;
                    // Re-point every forward slot at the new windows.
                    for (s, p) in slots.iter_mut().zip(&pairs) {
                        if !matches!(s, Some(Slot::Fwd(_)) | Some(Slot::Repair)) {
                            continue;
                        }
                        let (q1, q2) = (*p).expect("forward slot without pair");
                        let mut found = None;
                        for (f, block) in fwd_blocks.iter().enumerate() {
                            if holds(fused_qs(block), q1, q2) {
                                found = Some(Slot::Fwd(f));
                                break;
                            }
                        }
                        *s = Some(found.unwrap_or(Slot::Left));
                    }
                }
                None => {
                    for s in slots.iter_mut() {
                        if matches!(s, Some(Slot::Repair)) {
                            *s = Some(Slot::Left);
                        }
                    }
                }
            }
        }

        // ── Apply the decisions ──────────────────────────────────────────
        for (k, slot) in (i..j).zip(&slots) {
            match slot {
                Some(Slot::Back(b)) => {
                    if let (SimInst::Diag2 { q1, q2, d }, SimInst::Fused { qs, post_diag, .. }) =
                        (&insts[k], &mut out[*b])
                    {
                        post_diag.push(Diag2Stage {
                            i: local_pos(qs, *q1),
                            j: local_pos(qs, *q2),
                            d: *d,
                        });
                    }
                }
                Some(Slot::Fwd(f)) => {
                    if let (SimInst::Diag2 { q1, q2, d }, SimInst::Fused { qs, pre_diag, .. }) =
                        (&insts[k], &mut fwd_blocks[*f])
                    {
                        pre_diag.push(Diag2Stage {
                            i: local_pos(qs, *q1),
                            j: local_pos(qs, *q2),
                            d: *d,
                        });
                    }
                }
                _ => {}
            }
        }
        // Standalone leftovers (and Diag1s) keep their order before the run.
        for (k, slot) in (i..j).zip(&slots) {
            if !matches!(slot, Some(Slot::Back(_)) | Some(Slot::Fwd(_))) {
                out.push(insts[k].clone());
            }
        }
        for block in fwd_blocks.drain(..) {
            out.push(block);
        }
        i = fwd_start + fwd_len;
    }
    out
}

/// `a·b` for two row-major 2×2 matrices ([m00, m01, m10, m11] layout).
fn compose_2x2(
    a: &[num_complex::Complex64; 4],
    b: &[num_complex::Complex64; 4],
) -> [num_complex::Complex64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Try to re-partition a fused run so that every (q1,q2) pair in `pairs`
/// lands inside one window. The pairs merge into union-find components
/// (co-located, or not foldable at all), the qubits are placed in ascending
/// order greedily (a component joins a window whole or starts the next one),
/// and trailing singleton windows are fixed by donating the largest qubit of
/// the previous window when that qubit is unconstrained. Returns None when no
/// valid partition exists; the caller then keeps the original run.
fn refold_forward_run(blocks: &[SimInst], pairs: &[(usize, usize)]) -> Option<Vec<SimInst>> {
    // Local bit indices of already-attached stages would change here.
    if blocks.iter().any(|b| match b {
        SimInst::Fused {
            pre_diag,
            post_diag,
            ..
        } => !pre_diag.is_empty() || !post_diag.is_empty(),
        _ => false,
    }) {
        return None;
    }
    let mut uni: Vec<usize> = Vec::new();
    for b in blocks {
        for &q in fused_qs(b) {
            if !uni.contains(&q) {
                uni.push(q);
            }
        }
    }
    if uni.is_empty() {
        return None;
    }
    uni.sort_unstable();
    let kmax = blocks.iter().map(|b| fused_qs(b).len()).max().unwrap_or(0);
    let idx_of = |q: usize| uni.binary_search(&q).ok();
    let mut mats: Vec<Option<[num_complex::Complex64; 4]>> = vec![None; uni.len()];
    for b in blocks {
        let SimInst::Fused { qs, u, .. } = b else {
            return None;
        };
        for (t, &q) in qs.iter().enumerate() {
            let m = [u[4 * t], u[4 * t + 1], u[4 * t + 2], u[4 * t + 3]];
            let i = idx_of(q).expect("qubit outside run universe");
            mats[i] = Some(match mats[i] {
                Some(prev) => compose_2x2(&m, &prev),
                None => m,
            });
        }
    }

    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    // Pairs whose qubit is missing from the run can never fold here.
    let mut ok: Vec<bool> = pairs
        .iter()
        .map(|&(q1, q2)| idx_of(q1).is_some() && idx_of(q2).is_some())
        .collect();
    if !ok.iter().any(|&o| o) {
        return None;
    }
    // Iteratively enforce the kmax cap on components: pairs whose whole
    // component cannot fit a window fall back to standalone diagonals.
    let mut parent: Vec<usize>;
    let mut comp_of: Vec<usize>;
    let mut comp_members: Vec<Vec<usize>>;
    loop {
        parent = (0..uni.len()).collect();
        for (pi, &(q1, q2)) in pairs.iter().enumerate() {
            if !ok[pi] {
                continue;
            }
            let a = idx_of(q1).unwrap();
            let b = idx_of(q2).unwrap();
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        comp_of = vec![usize::MAX; uni.len()];
        comp_members = Vec::new();
        let mut root_comp: Vec<Option<usize>> = vec![None; uni.len()];
        for u in 0..uni.len() {
            let r = root(&mut parent, u);
            let c = match root_comp[r] {
                Some(c) => c,
                None => {
                    let c = comp_members.len();
                    comp_members.push(Vec::new());
                    root_comp[r] = Some(c);
                    c
                }
            };
            comp_of[u] = c;
            comp_members[c].push(u);
        }
        let mut dropped = false;
        for (pi, &(q1, _)) in pairs.iter().enumerate() {
            if !ok[pi] {
                continue;
            }
            let a = idx_of(q1).unwrap();
            if comp_members[comp_of[a]].len() > kmax {
                ok[pi] = false;
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
        if !ok.iter().any(|&o| o) {
            return None;
        }
    }

    // Greedy ascending placement; components stay whole.
    let mut placed = vec![false; uni.len()];
    let mut windows: Vec<Vec<usize>> = Vec::new();
    for u in 0..uni.len() {
        if placed[u] {
            continue;
        }
        let members = &comp_members[comp_of[u]];
        if members.len() > kmax {
            return None; // defensive: oversized component slipped through
        }
        let mut ms: Vec<usize> = members.iter().map(|&m| uni[m]).collect();
        ms.sort_unstable();
        for &m in members {
            placed[m] = true;
        }
        let cur_len = windows.last().map_or(0, Vec::len);
        if cur_len == 0 || cur_len + ms.len() > kmax {
            windows.push(ms);
        } else {
            windows.last_mut().unwrap().extend(ms);
        }
    }
    // The kernel needs k >= 2: a singleton window donates from its left
    // neighbour (valid only when the donated qubit is unconstrained).
    for wi in 0..windows.len() {
        if windows[wi].len() >= 2 {
            continue;
        }
        if wi == 0 || windows[wi - 1].len() < 3 {
            return None;
        }
        let q_prev = *windows[wi - 1].iter().max().unwrap();
        let uidx = idx_of(q_prev).unwrap();
        if comp_members[comp_of[uidx]].len() != 1 {
            return None;
        }
        windows[wi - 1].retain(|&x| x != q_prev);
        windows[wi].push(q_prev);
        windows[wi].sort_unstable();
    }

    let mut rebuilt: Vec<SimInst> = Vec::with_capacity(windows.len());
    for w in &windows {
        let mut qs: Vec<usize> = w.clone();
        qs.sort_unstable();
        let mut sv: SmallVec<[usize; 8]> = SmallVec::new();
        let mut u: Vec<num_complex::Complex64> = Vec::with_capacity(4 * qs.len());
        for &q in &qs {
            sv.push(q);
            let m = mats[idx_of(q).unwrap()].expect("every run qubit has a matrix");
            u.extend_from_slice(&m);
        }
        rebuilt.push(SimInst::Fused {
            qs: sv,
            u,
            pre_diag: Vec::new(),
            post_diag: Vec::new(),
        });
    }
    Some(rebuilt)
}

/// Compose maximal runs of consecutive CNOT instructions (≥ PERM_MIN_RUN)
/// into ONE out-of-place permutation sweep. CNOTs act as GF(2)-linear maps
/// on the basis index, so the whole run is a single index permutation
/// `dest(i) = XOR of cols[q] over the set bits q of i`; applying it costs one
/// read + one write sweep instead of one sweep per CNOT.
fn merge_cnot_runs(
    insts: Vec<SimInst>,
    ops: &[(&QuantumOp, nalgebra::DMatrix<num_complex::Complex64>)],
    dim: usize,
) -> Vec<SimInst> {
    let perm_off = std::env::var("SF_PERM").map(|v| v == "0").unwrap_or(false);
    if perm_off || dim > PERM_MAX_DIM {
        return insts;
    }
    let cnot = |inst: &SimInst| -> Option<(usize, usize)> {
        if let SimInst::Gate { idx } = inst {
            let op = ops[*idx].0;
            if op.op_type == OpType::CNOT && op.qubits.len() == 2 {
                return Some((op.qubits[0], op.qubits[1]));
            }
        }
        None
    };
    let n = dim.trailing_zeros() as usize;
    let mut out = Vec::with_capacity(insts.len());
    let mut i = 0;
    while i < insts.len() {
        if cnot(&insts[i]).is_some() {
            let mut j = i;
            while j < insts.len() && cnot(&insts[j]).is_some() {
                j += 1;
            }
            if j - i >= PERM_MIN_RUN {
                let mut cols: Vec<usize> = (0..n).map(|q| 1usize << q).collect();
                let mut touch = vec![false; n];
                for k in i..j {
                    let (c, t) = cnot(&insts[k]).unwrap();
                    touch[c] = true;
                    touch[t] = true;
                    for q in 0..n {
                        cols[q] ^= ((cols[q] >> c) & 1) << t;
                    }
                }
                // Identity permutation: the run cancels out; drop it.
                if !cols.iter().enumerate().all(|(q, &c)| c == 1usize << q) {
                    let touch_q: Vec<usize> = (0..n).filter(|&q| touch[q]).collect();
                    let inv_tables = build_perm_gather_tables(&cols);
                    out.push(SimInst::PermuteRun {
                        plan: PermPlan {
                            cols,
                            touch: touch_q,
                            inv_tables,
                        },
                    });
                }
                i = j;
                continue;
            }
        }
        out.push(insts[i].clone());
        i += 1;
    }
    out
}

/// Goal index of amplitude `i` under the composed permutation.
#[inline(always)]
fn perm_dest(cols: &[usize], i: usize) -> usize {
    let mut d = 0usize;
    let mut x = i;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        d ^= cols[q];
        x &= x - 1;
    }
    d
}

/// Build byte-chunk gather tables for the inverse of the plan map
/// `dest(i) = XOR of cols[q] over the set bits q of i`.
///
/// Treats the columns as the n×n GF(2) matrix M (dest = M·i) and
/// Gauss-Jordan-reduces it to the identity while applying the same row
/// operations to a tracked identity, which yields row masks `w[r]` with
/// `i_r = parity(d & w[r])`. Transposing those into per-destination-bit
/// masks `u[k]` and folding each byte of `d` into a 256-entry chunk turns
/// `src(d)` into a handful of independent L1-resident lookups (no
/// per-element bit scan), which is what lets the gather sweep issue its
/// scattered reads early. Returns an empty Vec (scatter fallback) if the
/// matrix is not invertible.
fn build_perm_gather_tables(cols: &[usize]) -> Vec<usize> {
    let n = cols.len();
    // Row r of `m` = mask over q of bit r of column q.
    let mut m: Vec<usize> = (0..n)
        .map(|r| {
            let mut row = 0usize;
            for (q, &c) in cols.iter().enumerate() {
                if (c >> r) & 1 == 1 {
                    row |= 1 << q;
                }
            }
            row
        })
        .collect();
    let mut w: Vec<usize> = (0..n).map(|r| 1usize << r).collect();
    for r in 0..n {
        // The composed CNOT run is invertible, so a pivot must exist.
        let p = match (r..n).find(|&p| (m[p] >> r) & 1 == 1) {
            Some(p) => p,
            None => return Vec::new(),
        };
        m.swap(r, p);
        w.swap(r, p);
        for r2 in 0..n {
            if r2 != r && (m[r2] >> r) & 1 == 1 {
                m[r2] ^= m[r];
                w[r2] ^= w[r];
            }
        }
    }
    // u[k] = mask over i-bits selected by destination bit k.
    let mut u = vec![0usize; n];
    for r in 0..n {
        let mut wr = w[r];
        while wr != 0 {
            let k = wr.trailing_zeros() as usize;
            u[k] |= 1 << r;
            wr &= wr - 1;
        }
    }
    // Fold byte chunks of the destination index: entry b holds the XOR of
    // the single-bit masks for the set bits of b (DP on the lowest set
    // bit). Bits ≥ n never occur in a valid `d`, so they read as 0.
    let n_chunks = n.div_ceil(8);
    let mut tables = vec![0usize; n_chunks * 256];
    for c in 0..n_chunks {
        for b in 1usize..256 {
            let j = b.trailing_zeros() as usize;
            let bit = c * 8 + j;
            let extra = if bit < n { u[bit] } else { 0 };
            tables[c * 256 + b] = tables[c * 256 + (b & (b - 1))] ^ extra;
        }
    }
    tables
}

/// Source index `src(d)` from the byte-chunk gather tables (NCH = n/8).
#[inline(always)]
fn perm_src<const NCH: usize>(tables: &[usize], d: usize) -> usize {
    let mut s = tables[d & 0xFF];
    if NCH > 1 {
        s ^= tables[0x100 + ((d >> 8) & 0xFF)];
    }
    if NCH > 2 {
        s ^= tables[0x200 + ((d >> 16) & 0xFF)];
    }
    s
}

/// Prefetch `p` into L1 (no-op on non-x86_64 targets).
#[inline(always)]
pub(crate) unsafe fn prefetch_t0<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// One contiguous destination range of a gather sweep, generic over the
/// element type (the f64 and f32 lanes share it).
///
/// `dst[d] = src[src(d)]` for `d in start..end`, with the scattered source
/// reads prefetched `pf` elements ahead so they overlap instead of stalling
/// one miss at a time; destination writes stay sequential.
///
/// SAFETY: `src`/`dst` must be valid for `end` elements and callers running
/// in parallel must pass disjoint `dst` ranges.
#[inline(always)]
pub(crate) unsafe fn perm_gather_range<const NCH: usize, T: Copy>(
    src: *const T,
    dst: *mut T,
    tables: &[usize],
    start: usize,
    end: usize,
    pf: usize,
) {
    let head = end.saturating_sub(pf).max(start);
    for d in start..head {
        let s = perm_src::<NCH>(tables, d);
        let sn = perm_src::<NCH>(tables, d + pf);
        prefetch_t0(src.add(sn));
        *dst.add(d) = *src.add(s);
    }
    for d in head..end {
        let s = perm_src::<NCH>(tables, d);
        *dst.add(d) = *src.add(s);
    }
}

/// Dispatch a gather range on the table width (1-3 chunks for n ≤ 24).
///
/// SAFETY: same contract as `perm_gather_range`; `tables` must hold at
/// least one 256-entry chunk.
#[inline(always)]
pub(crate) unsafe fn perm_gather_dispatch<T: Copy>(
    src: *const T,
    dst: *mut T,
    tables: &[usize],
    start: usize,
    end: usize,
    pf: usize,
) {
    match tables.len() / 256 {
        1 => perm_gather_range::<1, T>(src, dst, tables, start, end, pf),
        2 => perm_gather_range::<2, T>(src, dst, tables, start, end, pf),
        _ => perm_gather_range::<3, T>(src, dst, tables, start, end, pf),
    }
}

/// Software prefetch distance (elements ahead) of the gather sweep;
/// overridable via `SF_PERM_PF`.
const PERM_PREFETCH: usize = 32;

/// Gather-sweep decision for a plan: `Some((prefetch_elements, chunks))`
/// when the sweep should use the gather form, `None` for the historical
/// scatter loop. `SF_PERM_SWEEP=0` forces the scatter form.
pub(crate) fn perm_gather_config(plan: &PermPlan) -> Option<(usize, usize)> {
    if plan.inv_tables.is_empty()
        || std::env::var("SF_PERM_SWEEP")
            .map(|v| v == "0")
            .unwrap_or(false)
    {
        return None;
    }
    let n_chunks = plan.inv_tables.len() / 256;
    if !(1..=3).contains(&n_chunks) {
        return None;
    }
    let pf = std::env::var("SF_PERM_PF")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(PERM_PREFETCH)
        .max(1);
    Some((pf, n_chunks))
}

/// Historical scatter sweep: `out[dest(i)] = state[i]` with a per-element
/// bit scan for the destination. Kept as the `SF_PERM_SWEEP=0` escape hatch
/// (A/B) and for the sizes the gather tables do not cover.
fn apply_perm_scatter(
    state: &[num_complex::Complex64],
    out: &mut [num_complex::Complex64],
    cols: &[usize],
    use_par: bool,
) {
    let dim = state.len();
    if use_par {
        let src = SendPtr(state.as_ptr() as *mut num_complex::Complex64);
        let dst = SendPtr(out.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| unsafe {
            let d = perm_dest(cols, i);
            dst.set(d, src.get(i));
        });
    } else {
        for i in 0..dim {
            out[perm_dest(cols, i)] = state[i];
        }
    }
}

/// Out-of-place permutation sweep: `out[dest(i)] = state[i]`.
///
/// Gather form (default): `out[d] = state[src(d)]` with `src` evaluated
/// from the plan's byte-chunk tables — sequential destination writes with
/// prefetched scattered reads. The scatter form issued one write-allocate
/// miss per element on top of a serial per-element bit scan and measured
/// ≈15 individual CX passes at n=24; the gather sweep overlaps the
/// (unavoidable) scattered read misses behind the prefetch distance and
/// streams the writes. Parallel workers split the destination index space
/// into cache-line-aligned contiguous ranges (disjoint writes, shared
/// read side).
fn apply_perm_run(
    state: &[num_complex::Complex64],
    out: &mut [num_complex::Complex64],
    plan: &PermPlan,
    use_par: bool,
) {
    let (pf, _) = match perm_gather_config(plan) {
        Some(cfg) => cfg,
        None => return apply_perm_scatter(state, out, &plan.cols, use_par),
    };
    let dim = state.len();
    let src = SendPtr(state.as_ptr() as *mut num_complex::Complex64);
    let dst = SendPtr(out.as_mut_ptr());
    if use_par {
        let lanes = rayon::current_num_threads().max(1);
        // Contiguous destination ranges, whole cache lines so no line is
        // written by two workers.
        let chunk = ((dim / (lanes * 8)).max(1 << 13) + 7) & !7usize;
        let n_ranges = dim.div_ceil(chunk);
        (0..n_ranges).into_par_iter().for_each(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(dim);
            unsafe { perm_gather_dispatch(src.raw(), dst.raw(), &plan.inv_tables, start, end, pf) };
        });
    } else {
        unsafe { perm_gather_dispatch(src.raw(), dst.raw(), &plan.inv_tables, 0, dim, pf) };
    }
}

/// Apply a fused block of k 1q gates on distinct qubits (tensor-product
/// structure) in ONE sweep. `u` holds k row-major 2×2 blocks (gate t =
/// `u[4t..4t+4]`) acting on `qs[t]` (bit t of the group index).
/// `pre_diag`/`post_diag` hold diagonal 2q phases folded into the same
/// sweep (see `fold_diags_into_fused`), applied to each group buffer
/// before/after the `u` stages. Dispatches to a const-generic instantiation
/// so the inner loops fully unroll for every supported k. Same arithmetic
/// as k separate passes, 1/k memory traffic.
fn apply_fused_qubits_inplace(
    state: &mut [num_complex::Complex64],
    qs: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
) {
    let k = qs.len();
    let masks: Vec<usize> = qs.iter().map(|&q| 1usize << q).collect();
    match k {
        2 => fused_dispatch::<2>(state, &masks, u, pre_diag, post_diag, use_par),
        3 => fused_dispatch::<3>(state, &masks, u, pre_diag, post_diag, use_par),
        4 => fused_dispatch::<4>(state, &masks, u, pre_diag, post_diag, use_par),
        5 => fused_dispatch::<5>(state, &masks, u, pre_diag, post_diag, use_par),
        6 => fused_dispatch::<6>(state, &masks, u, pre_diag, post_diag, use_par),
        7 => fused_dispatch::<7>(state, &masks, u, pre_diag, post_diag, use_par),
        8 => fused_dispatch::<8>(state, &masks, u, pre_diag, post_diag, use_par),
        9 => fused_dispatch::<9>(state, &masks, u, pre_diag, post_diag, use_par),
        10 => fused_dispatch::<10>(state, &masks, u, pre_diag, post_diag, use_par),
        _ => unreachable!("fused block size out of range"),
    }
}

/// Const-generic worker for `apply_fused_qubits_inplace` (K = block size).
/// The 2^K-element group buffer skips initialization entirely (every slot is
/// written by the gather before it is read) — a plain `[Complex64; N]` init
/// would memset up to 16 KiB per group and dominate the small-K kernels. Each
/// stage transforms disjoint pairs at distance 2^t with the shared SIMD pair
/// kernel (same per-pair arithmetic as the historical scalar loop). Folded
/// diagonal stages multiply the resident buffer by one precomputed per-entry
/// factor table per side (built once per call, not per group) through the
/// dispatch-free vector kernel, since every side is a full pass over the
/// L1-resident buffer.
/// Kept as the reference worker for A/B micro-benchmarks; the dispatcher
/// now uses `apply_fused_tensor_banded` (same arithmetic, bit-identical).
#[allow(dead_code)]
#[inline(always)]
fn apply_fused_tensor<const K: usize>(
    state: &mut [num_complex::Complex64],
    masks: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
) {
    use std::mem::MaybeUninit;
    debug_assert_eq!(masks.len(), K);
    debug_assert_eq!(u.len(), 4 * K);
    let sub = 1usize << K;
    let dim = state.len();
    let n_groups = dim >> K;
    let offs: Vec<usize> = (0..sub).map(|s| spread_bits(s, masks)).collect();
    let fac_pre = build_stage_factors(sub, pre_diag);
    let fac_post = build_stage_factors(sub, post_diag);

    let body = |sp: &SendPtr, g: usize| {
        let base = deposit_bits_masked(g, masks);
        // SAFETY: [MaybeUninit<T>; N] has the same layout as MaybeUninit<[T; N]>.
        let mut val: [MaybeUninit<num_complex::Complex64>; 1024] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for s in 0..sub {
            val[s].write(unsafe { sp.get(base | offs[s]) });
        }
        // SAFETY: the gather above initialized all `sub` slots, and `val` is
        // not touched directly again before the scatter below.
        let buf: &mut [num_complex::Complex64] = unsafe {
            std::slice::from_raw_parts_mut(val.as_mut_ptr() as *mut num_complex::Complex64, sub)
        };
        // Folded diagonal stages: one table multiply per side, before/after
        // the pair stages (a diagonal does not commute with the block's 2×2
        // gates, so the side matters). The buffers are L1-resident and the
        // parent callers are already Rayon-chunked, so the small-size gate
        // in `pattern_scale_periodic` does not apply.
        if !fac_pre.is_empty() {
            crate::simd::pattern_scale_chunk(buf, 0, &fac_pre);
        }
        // Apply gate t to every pair (s, s | 1<<t) — tensor structure, no
        // dense 2^k matvec. K vector passes over the resident buffer; the
        // buffer is at most 256 complex and the parent callers are already
        // Rayon-chunked, so take the dispatch-free kernel (the length gate
        // in `pair_pass` would force these stages scalar).
        for t in 0..K {
            let m = [[u[4 * t], u[4 * t + 1]], [u[4 * t + 2], u[4 * t + 3]]];
            crate::simd::pair_pass_chunk(buf, 1usize << t, m);
        }
        if !fac_post.is_empty() {
            crate::simd::pattern_scale_chunk(buf, 0, &fac_post);
        }
        for s in 0..sub {
            unsafe {
                sp.set(base | offs[s], buf[s]);
            }
        }
    };

    let sp = SendPtr(state.as_mut_ptr());
    if use_par {
        (0..n_groups).into_par_iter().for_each(|g| body(&sp, g));
    } else {
        for g in 0..n_groups {
            body(&sp, g);
        }
    }
}

/// Worker-selection knobs. Defaults are the measured winners
/// (`tests::fused_banded_microbench`): per-group below bit 12, banded at
/// 8 groups above (8 groups × 2^K complex is 8 KiB at K=6 — L1-friendly —
/// and 128 KiB at K=10, L2-resident; both keep the band's page window
/// resident for its whole gather/stage/scatter walk). Read per call so
/// probes can interleave configs in one process — the two env reads are
/// negligible against a 2^24-element sweep. The overrides exist for
/// production A/B diagnostics: `SF_BANDED_MIN_BIT=64` forces the
/// per-group worker.
fn banded_knobs() -> (u32, usize) {
    let min_bit = std::env::var("SF_BANDED_MIN_BIT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(12);
    let band = std::env::var("SF_BAND_GROUPS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(8usize);
    (min_bit, band.max(1))
}

/// Tile-worker selection for `fused_dispatch`. Returns the band when the
/// block's qubits form a contiguous run `{lo..lo+K-1}` (a band's gather
/// addresses then collapse to `2^K` contiguous runs — see
/// `apply_fused_tensor_tiled_with`) and `None` to keep the historical
/// per-group/banded workers. Default ON: the tiled worker is bit-identical
/// and won the 20/22/24q probe on every cell (`_nb4f_tile_probe.py`);
/// `SF_FUSED_TILE=0` restores the old selection. `SF_TILE_GROUPS`
/// (default 64) overrides the band, normalized to a power of two and
/// capped by 1 MiB of scratch and by `2^lo`. `lo > 0` blocks carrying
/// folded diagonals stay banded: the s-major scratch has no per-row scale
/// kernel.
fn tile_band_for(
    masks: &[usize],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    dim: usize,
) -> Option<usize> {
    if std::env::var("SF_FUSED_TILE")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    if masks.is_empty() {
        return None;
    }
    let lo = masks[0].trailing_zeros() as usize;
    if !masks
        .iter()
        .enumerate()
        .all(|(i, &m)| m == 1usize << (lo + i))
    {
        return None;
    }
    if lo > 0 && (!pre_diag.is_empty() || !post_diag.is_empty()) {
        return None;
    }
    let sub = 1usize << masks.len();
    let n_groups = dim >> masks.len();
    let raw = std::env::var("SF_TILE_GROUPS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(64usize);
    // ≤ 1 MiB of scratch keeps one band L2-resident.
    let mem_cap = ((1usize << 20) / (sub * 16)).max(1);
    let cap = if lo == 0 {
        n_groups.min(mem_cap)
    } else {
        (1usize << lo).min(n_groups).min(mem_cap)
    };
    let band = raw.max(1).next_power_of_two().min(cap);
    if band < 2 {
        return None;
    }
    Some(band)
}

/// Fused-block worker selection. The per-group worker keeps its 2^K buffer
/// L1-resident and wins when the block's qubits are low; once the largest
/// mask reaches the banded threshold the per-group walk of 2^K pages
/// several KB apart starts thrashing the TLB, and banded processing is
/// faster (see `tests::fused_banded_microbench`). Both workers are
/// bit-identical. Contiguous-run blocks default to the tiled worker
/// (64-group bands measured fastest across the 20/22/24q probe cells);
/// `SF_FUSED_TILE=0` keeps the historical selection.
#[inline(always)]
fn fused_dispatch<const K: usize>(
    state: &mut [num_complex::Complex64],
    masks: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
) {
    let (min_bit, band) = banded_knobs();
    if let Some(tband) = tile_band_for(masks, pre_diag, post_diag, state.len()) {
        return apply_fused_tensor_tiled_with::<K>(
            state, masks, u, pre_diag, post_diag, use_par, tband,
        );
    }
    let hi = masks.iter().map(|m| m.trailing_zeros()).max().unwrap_or(0);
    if hi >= min_bit {
        apply_fused_tensor_banded_with::<K>(state, masks, u, pre_diag, post_diag, use_par, band)
    } else {
        apply_fused_tensor::<K>(state, masks, u, pre_diag, post_diag, use_par)
    }
}

/// Banded worker for `apply_fused_qubits_inplace` (K = block size).
///
/// Same arithmetic and same per-element stage order as `apply_fused_tensor`
/// (bit-identical results), but the groups are processed in bands: a whole
/// band of groups is gathered into one contiguous scratch, the K stage
/// passes run over the scratch with long loops, and the band is scattered
/// back. Two effects: (1) each band's page window is walked in address
/// order instead of re-touching 2^K scattered pages per group (the
/// high-qubit windows were DTLB-bound), and (2) the per-stage setup cost
/// is paid once per band, not once per group.
///
/// The explicit band size exists for the micro-benchmark sweep;
/// production callers go through `fused_dispatch`.
fn apply_fused_tensor_banded_with<const K: usize>(
    state: &mut [num_complex::Complex64],
    masks: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
    band: usize,
) {
    debug_assert_eq!(masks.len(), K);
    debug_assert_eq!(u.len(), 4 * K);
    let sub = 1usize << K;
    let dim = state.len();
    let n_groups = dim >> K;
    let band = band.max(1);
    let n_bands = n_groups.div_ceil(band);
    let offs: Vec<usize> = (0..sub).map(|s| spread_bits(s, masks)).collect();
    let fac_pre = build_stage_factors(sub, pre_diag);
    let fac_post = build_stage_factors(sub, post_diag);

    // Group `g` occupies scratch rows [row, row + sub); every stage pair
    // stays inside one row (sub is a multiple of 2*stride), so concatenated
    // rows can be staged with the same chunk kernels as a single group.
    let run_band = |scratch: &mut Vec<num_complex::Complex64>, b: usize, sp: &SendPtr| {
        let g0 = b * band;
        let g1 = (g0 + band).min(n_groups);
        let len = (g1 - g0) * sub;
        if scratch.len() != len {
            scratch.resize(len, num_complex::Complex64::new(0.0, 0.0));
        }
        for g in g0..g1 {
            let base = deposit_bits_masked(g, masks);
            let row = (g - g0) * sub;
            for s in 0..sub {
                scratch[row + s] = unsafe { sp.get(base | offs[s]) };
            }
        }
        // Folded diagonal stages: one table multiply per side over the
        // whole band (the table period is `sub`, so a phase-0 call carries
        // across rows exactly like the per-group calls did).
        if !fac_pre.is_empty() {
            crate::simd::pattern_scale_chunk(scratch, 0, &fac_pre);
        }
        for t in 0..K {
            let m = [[u[4 * t], u[4 * t + 1]], [u[4 * t + 2], u[4 * t + 3]]];
            crate::simd::pair_pass_chunk(scratch, 1usize << t, m);
        }
        if !fac_post.is_empty() {
            crate::simd::pattern_scale_chunk(scratch, 0, &fac_post);
        }
        for g in g0..g1 {
            let base = deposit_bits_masked(g, masks);
            let row = (g - g0) * sub;
            for s in 0..sub {
                unsafe {
                    sp.set(base | offs[s], scratch[row + s]);
                }
            }
        }
    };

    let sp = SendPtr(state.as_mut_ptr());
    if use_par {
        (0..n_bands)
            .into_par_iter()
            .for_each_init(Vec::new, |scratch, b| run_band(scratch, b, &sp));
    } else {
        let mut scratch: Vec<num_complex::Complex64> = Vec::new();
        for b in 0..n_bands {
            run_band(&mut scratch, b, &sp);
        }
    }
}

/// Tiled worker for `apply_fused_qubits_inplace` (K = block size), used for
/// contiguous-run blocks (`fused_dispatch` picks the band).
///
/// For a block on qubits `{lo..lo+K-1}` the gather addresses of one band are
/// `base(g0+j) | offs[s] == base(g0) + offs[s] + j` for `j < band <= 2^lo`:
/// every group-local index `s` is ONE contiguous run of `band` complex, so
/// gather/scatter become `2^K` contiguous copies per band (16 KiB at band
/// 1024) instead of `2^K` strided 16-byte touches per group. `lo == 0` is
/// fully flat — the 2^K windows tile the state, so one copy of `band * 2^K`
/// in the same g-major layout as the banded worker, and folded diagonals
/// reuse `pattern_scale_chunk` unchanged. For `lo > 0` the scratch is
/// s-major (`scratch[s*band + j]`) and the K stage passes run at distance
/// `2^t * band`. The layout is the only difference to the other workers —
/// every stage applies the same 2x2 kernels to the same disjoint index
/// pairs (checked by `tests::fused_tiled_parity`), so results are
/// bit-identical. Folded diagonals with `lo > 0` fall back to the banded
/// worker (no per-row scale kernel for the s-major layout).
fn apply_fused_tensor_tiled_with<const K: usize>(
    state: &mut [num_complex::Complex64],
    masks: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
    band: usize,
) {
    debug_assert_eq!(masks.len(), K);
    debug_assert_eq!(u.len(), 4 * K);
    let sub = 1usize << K;
    let dim = state.len();
    let n_groups = dim >> K;
    if n_groups == 0 {
        return;
    }
    let lo = masks[0].trailing_zeros() as usize;
    debug_assert!(masks
        .iter()
        .enumerate()
        .all(|(i, &m)| m == 1usize << (lo + i)));
    if lo > 0 && (!pre_diag.is_empty() || !post_diag.is_empty()) {
        return apply_fused_tensor_banded_with::<K>(
            state, masks, u, pre_diag, post_diag, use_par, band,
        );
    }
    // A power-of-two band dividing `n_groups` keeps every band full; for
    // `lo > 0` the `2^lo` cap keeps one row's run inside a base stretch.
    let cap = if lo == 0 {
        n_groups
    } else {
        (1usize << lo).min(n_groups)
    };
    let band = band.max(1).next_power_of_two().min(cap);
    let n_bands = n_groups / band;
    let offs: Vec<usize> = (0..sub).map(|s| spread_bits(s, masks)).collect();
    let fac_pre = build_stage_factors(sub, pre_diag);
    let fac_post = build_stage_factors(sub, post_diag);
    // Stage distance in the scratch: flat g-major at `lo == 0`, one row per
    // group-local index (s-major) otherwise.
    let stride0 = if lo == 0 { 1usize } else { band };

    let run_band = |scratch: &mut Vec<num_complex::Complex64>, b: usize, sp: &SendPtr| {
        let len = band * sub;
        if scratch.len() != len {
            scratch.resize(len, num_complex::Complex64::new(0.0, 0.0));
        }
        let base0 = deposit_bits_masked(b * band, masks);
        let p = sp.raw();
        // SAFETY: the `sub` runs per band, `band` long each, cover exactly
        // the `band * sub` addresses owned by this band's groups (rows are
        // `2^lo >= band` apart, bands advance by `band` inside each linear
        // stretch), and bands are disjoint — parallel callers never touch
        // the same element.
        unsafe {
            if lo == 0 {
                std::ptr::copy_nonoverlapping(p.add(base0), scratch.as_mut_ptr(), len);
            } else {
                for s in 0..sub {
                    std::ptr::copy_nonoverlapping(
                        p.add(base0 + offs[s]),
                        scratch.as_mut_ptr().add(s * band),
                        band,
                    );
                }
            }
            if !fac_pre.is_empty() {
                crate::simd::pattern_scale_chunk(scratch, 0, &fac_pre);
            }
            for t in 0..K {
                let m = [[u[4 * t], u[4 * t + 1]], [u[4 * t + 2], u[4 * t + 3]]];
                crate::simd::pair_pass_chunk(scratch, (1usize << t) * stride0, m);
            }
            if !fac_post.is_empty() {
                crate::simd::pattern_scale_chunk(scratch, 0, &fac_post);
            }
            if lo == 0 {
                std::ptr::copy_nonoverlapping(scratch.as_ptr(), p.add(base0), len);
            } else {
                for s in 0..sub {
                    std::ptr::copy_nonoverlapping(
                        scratch.as_ptr().add(s * band),
                        p.add(base0 + offs[s]),
                        band,
                    );
                }
            }
        }
    };

    let sp = SendPtr(state.as_mut_ptr());
    if use_par {
        (0..n_bands)
            .into_par_iter()
            .for_each_init(Vec::new, |scratch, b| run_band(scratch, b, &sp));
    } else {
        let mut scratch: Vec<num_complex::Complex64> = Vec::new();
        for b in 0..n_bands {
            run_band(&mut scratch, b, &sp);
        }
    }
}

/// Combined per-entry factor table for a list of folded diagonal stages:
/// entry `s` of the group buffer gets the product of every stage's
/// `d[(b_i << 1) | b_j]` (zero-quadrant excluded, stays 1.0). Empty when no
/// stage is present, so the caller skips the buffer multiply entirely.
/// Built once per instruction application; the groups then reuse it. Two
/// wrap-around slack entries follow the period for the vector kernel's
/// 2-complex windows (same convention as `simd::make_table`).
fn build_stage_factors(sub: usize, stages: &[Diag2Stage]) -> Vec<num_complex::Complex64> {
    if stages.is_empty() {
        return Vec::new();
    }
    let mut fac = vec![num_complex::Complex64::new(1.0, 0.0); sub];
    for st in stages {
        let mi = 1usize << (st.i as usize);
        let mj = 1usize << (st.j as usize);
        for (s, f) in fac.iter_mut().enumerate() {
            let idx = (((s & mi) != 0) as usize) << 1 | (((s & mj) != 0) as usize);
            *f *= st.d[idx];
        }
    }
    let slack = [fac[0], fac[1]];
    fac.extend_from_slice(&slack);
    fac
}

/// Try to express a diagonal run as one quadratic phase form.
/// Returns None when any entry deviates from unit modulus.
fn build_phase_plan(run: &[SimInst], n: usize) -> Option<PhasePlan> {
    let mut plan = PhasePlan {
        global: 0.0,
        lin: vec![0.0; n],
        quad: vec![0.0; n * n],
        n,
    };
    for inst in run {
        match inst {
            SimInst::Diag1 { q, d0, d1 } => {
                if !unit_modulus(*d0) || !unit_modulus(*d1) {
                    return None;
                }
                let p0 = d0.arg();
                let p1 = d1.arg();
                plan.global += p0;
                plan.lin[*q] += p1 - p0;
            }
            SimInst::Diag2 { q1, q2, d } => {
                if d.iter().any(|z| !unit_modulus(*z)) {
                    return None;
                }
                let p00 = d[0].arg();
                let p01 = d[1].arg();
                let p10 = d[2].arg();
                let p11 = d[3].arg();
                plan.global += p00;
                plan.lin[*q1] += p10 - p00;
                plan.lin[*q2] += p01 - p00;
                let a12 = p11 - p10 - p01 + p00;
                plan.quad[q1 * n + q2] += a12;
                plan.quad[q2 * n + q1] += a12;
            }
            SimInst::Gate { .. }
            | SimInst::PhaseRun { .. }
            | SimInst::Fused { .. }
            | SimInst::PermuteRun { .. } => return None,
        }
    }
    Some(plan)
}

/// Direct evaluation of the quadratic phase form at `idx`.
pub(crate) fn phase_eval(plan: &PhasePlan, idx: usize) -> f64 {
    let n = plan.n;
    let mut phi = plan.global;
    for q in 0..n {
        if (idx >> q) & 1 == 1 {
            phi += plan.lin[q];
            let row = &plan.quad[q * n..q * n + n];
            for p in (q + 1)..n {
                if (idx >> p) & 1 == 1 {
                    phi += row[p];
                }
            }
        }
    }
    phi
}

/// Apply a whole diagonal run in ONE pass: multiply each amplitude by
/// exp(i·Φ(bits)) while walking the state in Gray-code order, so Φ updates
/// in O(n) per amplitude instead of O(#terms). The complex product is
/// error-free-transformed (rounding residuals folded into the next step)
/// and the phase is re-seeded by a direct evaluation every 256 steps, so
/// multiplicative drift stays at the sin_cos-rounding floor (~1e-15).
fn apply_phase_run(
    state: &mut [num_complex::Complex64],
    n: usize,
    plan: &PhasePlan,
    use_par: bool,
) {
    let dim = state.len();
    debug_assert_eq!(dim, 1usize << n);

    // Table-driven fast path for banded quadratic forms (see `PhaseFast`);
    // `SF_PHASE_TABLE=0` forces the reference loop for A/B attribution.
    let table_ok = std::env::var("SF_PHASE_TABLE")
        .map(|v| v != "0")
        .unwrap_or(true);
    if table_ok {
        if let Some(fast) = build_phase_fast(plan) {
            apply_phase_run_fast(state, plan, &fast, use_par);
            return;
        }
    }

    let process = |s: &mut [num_complex::Complex64], off: usize| {
        let len = s.len();
        let (sin0, cos0) = phase_eval(plan, off).sin_cos();
        let mut ph = num_complex::Complex64::new(cos0, sin0);
        // Compensation residuals of the complex product.
        let mut c_re: f64 = 0.0;
        let mut c_im: f64 = 0.0;
        s[0] *= ph;
        for k in 1..len {
            let idx = off + (k ^ (k >> 1));
            let q = (k as u64).trailing_zeros() as usize;
            let bq = (idx >> q) & 1;
            let mut inner = plan.lin[q];
            let row = &plan.quad[q * plan.n..q * plan.n + plan.n];
            for p in 0..plan.n {
                if p != q {
                    inner += row[p] * (((idx >> p) & 1) as f64);
                }
            }
            let delta = if bq == 1 { inner } else { -inner };
            let (sin_d, cos_d) = delta.sin_cos();
            // Compensated complex product (error-free transform): each
            // multiply's rounding residual is folded into the next, so the
            // phase-angle random walk is driven only by sin_cos rounding.
            let a = ph.re + c_re;
            let b = ph.im + c_im;
            let p1 = a * cos_d;
            let e1 = a.mul_add(cos_d, -p1);
            let q1 = b * sin_d;
            let e2 = b.mul_add(sin_d, -q1);
            let s1 = p1 - q1;
            let e3 = (p1 - s1) - q1;
            let p2 = a * sin_d;
            let e4 = a.mul_add(sin_d, -p2);
            let q2 = b * cos_d;
            let e5 = b.mul_add(cos_d, -q2);
            let s2 = p2 + q2;
            let e6 = (p2 - s2) + q2;
            ph = num_complex::Complex64::new(s1, s2);
            c_re = (e1 - e2) + e3;
            c_im = (e4 + e5) + e6;
            if k & 0x00FF == 0 {
                let (sin_r, cos_r) = phase_eval(plan, idx).sin_cos();
                ph = num_complex::Complex64::new(cos_r, sin_r);
                c_re = 0.0;
                c_im = 0.0;
            }
            s[k ^ (k >> 1)] *= ph;
        }
    };

    if use_par {
        let chunk = (dim / 16).max(1024);
        state.par_chunks_mut(chunk).enumerate().for_each(|(c, s)| {
            process(s, c * chunk);
        });
    } else {
        process(state, 0);
    }
}

/// Table-driven fast path for the merged phase sweep.
///
/// `apply_phase_run` re-evaluates the O(n) quadratic form and calls sin_cos
/// once per amplitude. Whenever every row of `quad` has at most
/// `FAST_MAX_TERMS` non-zero entries — true for the banded forms that
/// nearest-neighbour ZZ chains produce — the phase increment of a step
/// depends only on the few bits that row touches, so all 2^z possible deltas
/// (and both parities of the flipped bit) are pre-computed once per call and
/// looked up per step.  Entries are built with the same ascending-p
/// summation order as the reference loop, so the (sin_d, cos_d) values fed
/// into the error-free-transformed product are bit-identical to the scalar
/// path (reads that contribute exact zeros are skipped; only the sign of a
/// zero delta could differ, which no value comparison can observe).
struct PhaseFast {
    /// Non-zero `quad` columns per qubit (ascending, `!= q`).
    z: Vec<usize>,
    col: Vec<[usize; FAST_MAX_TERMS]>,
    /// `n * FAST_TAB` entries; index = `q * FAST_TAB + mask + (bq << z)`.
    sin: Vec<f64>,
    cos: Vec<f64>,
}

/// Max non-zero `quad` entries per row for the table path (2^z deltas).
const FAST_MAX_TERMS: usize = 4;
/// Table stride per qubit: 2^FAST_MAX_TERMS masks x 2 flipped-bit parities.
const FAST_TAB: usize = 1 << (FAST_MAX_TERMS + 1);

fn build_phase_fast(plan: &PhasePlan) -> Option<PhaseFast> {
    let n = plan.n;
    if n > 63 {
        return None; // step bookkeeping indexes bits with u64 arithmetic
    }
    let mut z = vec![0usize; n];
    let mut col = vec![[0usize; FAST_MAX_TERMS]; n];
    let mut sin = vec![0.0f64; n * FAST_TAB];
    let mut cos = vec![0.0f64; n * FAST_TAB];
    for q in 0..n {
        let row = &plan.quad[q * n..q * n + n];
        let mut zq = 0usize;
        for p in 0..n {
            if p == q || row[p] == 0.0 {
                continue;
            }
            if zq == FAST_MAX_TERMS {
                return None;
            }
            col[q][zq] = p;
            zq += 1;
        }
        z[q] = zq;
        let base = q * FAST_TAB;
        for mask in 0..(1usize << zq) {
            let mut inner = plan.lin[q];
            for t in 0..zq {
                inner += row[col[q][t]] * ((mask >> t) & 1) as f64;
            }
            for bq in 0..2usize {
                let delta = if bq == 1 { inner } else { -inner };
                let (s, c) = delta.sin_cos();
                sin[base + mask + (bq << zq)] = s;
                cos[base + mask + (bq << zq)] = c;
            }
        }
    }
    Some(PhaseFast { z, col, sin, cos })
}

/// `apply_phase_run` with per-step table lookups (see `PhaseFast`).  The
/// Gray walk, reseed cadence, compensation algebra and chunking are the
/// reference implementation's; only the delta lookup changes.
fn apply_phase_run_fast(
    state: &mut [num_complex::Complex64],
    plan: &PhasePlan,
    fast: &PhaseFast,
    use_par: bool,
) {
    let dim = state.len();
    debug_assert_eq!(dim, 1usize << plan.n);

    let process = |s: &mut [num_complex::Complex64], off: usize| {
        let len = s.len();
        let (sin0, cos0) = phase_eval(plan, off).sin_cos();
        let mut ph = num_complex::Complex64::new(cos0, sin0);
        // Compensation residuals of the complex product.
        let mut c_re: f64 = 0.0;
        let mut c_im: f64 = 0.0;
        s[0] *= ph;
        for k in 1..len {
            let idx = off + (k ^ (k >> 1));
            let q = (k as u64).trailing_zeros() as usize;
            let bq = (idx >> q) & 1;
            let zq = fast.z[q];
            let cols = &fast.col[q];
            let mut mask = 0usize;
            for t in 0..zq {
                mask |= ((idx >> cols[t]) & 1) << t;
            }
            let e = q * FAST_TAB + mask + (bq << zq);
            let sin_d = fast.sin[e];
            let cos_d = fast.cos[e];
            // Compensated complex product (error-free transform): each
            // multiply's rounding residual is folded into the next, so the
            // phase-angle random walk is driven only by sin_cos rounding.
            let a = ph.re + c_re;
            let b = ph.im + c_im;
            let p1 = a * cos_d;
            let e1 = a.mul_add(cos_d, -p1);
            let q1 = b * sin_d;
            let e2 = b.mul_add(sin_d, -q1);
            let s1 = p1 - q1;
            let e3 = (p1 - s1) - q1;
            let p2 = a * sin_d;
            let e4 = a.mul_add(sin_d, -p2);
            let q2 = b * cos_d;
            let e5 = b.mul_add(cos_d, -q2);
            let s2 = p2 + q2;
            let e6 = (p2 - s2) + q2;
            ph = num_complex::Complex64::new(s1, s2);
            c_re = (e1 - e2) + e3;
            c_im = (e4 + e5) + e6;
            if k & 0x00FF == 0 {
                let (sin_r, cos_r) = phase_eval(plan, idx).sin_cos();
                ph = num_complex::Complex64::new(cos_r, sin_r);
                c_re = 0.0;
                c_im = 0.0;
            }
            s[k ^ (k >> 1)] *= ph;
        }
    };

    if use_par {
        let chunk = (dim / 16).max(1024);
        state.par_chunks_mut(chunk).enumerate().for_each(|(c, s)| {
            process(s, c * chunk);
        });
    } else {
        process(state, 0);
    }
}

/// Diagonal 1q gate in-place: one complex multiply per amplitude, delegated
/// to the SIMD diagonal kernel (scalar fallback is formula-identical).
fn apply_diag1_inplace(
    state: &mut [num_complex::Complex64],
    q: usize,
    d0: num_complex::Complex64,
    d1: num_complex::Complex64,
    use_par: bool,
) {
    if use_par {
        let chunk = (state.len() / 16).max(1024);
        state.par_chunks_mut(chunk).enumerate().for_each(|(c, s)| {
            crate::simd::diag1_pass(s, c * chunk, q, d0, d1);
        });
    } else {
        crate::simd::diag1_pass(state, 0, q, d0, d1);
    }
}

/// Diagonal 2q gate in-place: entry (bit q1, bit q2) → d[(b1<<1)|b2],
/// delegated to the SIMD diagonal kernel (scalar fallback is same-formula).
fn apply_diag2_inplace(
    state: &mut [num_complex::Complex64],
    q1: usize,
    q2: usize,
    d: &[num_complex::Complex64; 4],
    use_par: bool,
) {
    if use_par {
        let chunk = (state.len() / 16).max(1024);
        state.par_chunks_mut(chunk).enumerate().for_each(|(c, s)| {
            crate::simd::diag2_pass(s, c * chunk, q1, q2, d);
        });
    } else {
        crate::simd::diag2_pass(state, 0, q1, q2, d);
    }
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
        OpType::Cu(a, b, c) => ("cu".into(), vec![a.evaluate(), b.evaluate(), c.evaluate()]),
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

    // ── Engine rewrite tests (CX·D·CX fusion, diagonal-run phase sweep) ──

    #[test]
    fn test_zz_triple_matches_rzz() {
        // CX(0,1)·Rz(θ,1)·CX(0,1) must equal Rzz(θ) on (0,1).
        let theta = 0.734;
        let mut dag_a = QuantumDAG::new(3, 0);
        dag_a.add_op(OpType::H, &[0]);
        dag_a.add_op(OpType::CNOT, &[0, 1]);
        dag_a.add_op(OpType::Rz(Parameter::Const(theta)), &[1]);
        dag_a.add_op(OpType::CNOT, &[0, 1]);
        dag_a.add_op(OpType::Ry(Parameter::Const(0.3)), &[2]);

        let mut dag_b = QuantumDAG::new(3, 0);
        dag_b.add_op(OpType::H, &[0]);
        dag_b.add_op(OpType::Rzz(Parameter::Const(theta)), &[0, 1]);
        dag_b.add_op(OpType::Ry(Parameter::Const(0.3)), &[2]);

        let a = dag_a.simulate();
        let b = dag_b.simulate();
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).norm() < 1e-12,
                "idx {i}: CX·Rz·CX {} != Rzz {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_cx_sandwich_control_and_disjoint() {
        // Control-side: CX(0,1)·Rz(0)·CX(0,1) = Rz(0)  (CX commutes with it).
        // Disjoint:     CX(0,1)·Rz(2)·CX(0,1) = Rz(2).
        let theta = 0.62;
        let mut a = QuantumDAG::new(3, 0);
        a.add_op(OpType::H, &[0]);
        a.add_op(OpType::CNOT, &[0, 1]);
        a.add_op(OpType::Rz(Parameter::Const(theta)), &[2]);
        a.add_op(OpType::CNOT, &[0, 1]);

        let mut b = QuantumDAG::new(3, 0);
        b.add_op(OpType::H, &[0]);
        b.add_op(OpType::Rz(Parameter::Const(theta)), &[2]);

        let mut c = QuantumDAG::new(3, 0);
        c.add_op(OpType::H, &[0]);
        c.add_op(OpType::CNOT, &[0, 1]);
        c.add_op(OpType::Rz(Parameter::Const(theta)), &[0]);
        c.add_op(OpType::CNOT, &[0, 1]);

        let mut d = QuantumDAG::new(3, 0);
        d.add_op(OpType::H, &[0]);
        d.add_op(OpType::Rz(Parameter::Const(theta)), &[0]);

        for (x, y) in [(a.simulate(), b.simulate()), (c.simulate(), d.simulate())] {
            for i in 0..x.len() {
                assert!((x[i] - y[i]).norm() < 1e-12, "idx {i}");
            }
        }
    }

    #[test]
    fn test_diag_run_merge_matches_unitary() {
        // 44-row diagonal run (mixed Rz/CZ/CP/Rzz on overlapping qubits) must
        // reproduce the exact unitary action. The circuit is built on
        // reflected qubit indices so `to_unitary` (MSB-first) and `simulate`
        // (LSB-first) describe the same bit order; `to_unitary` uses the
        // legacy independent kernel path, making it a valid reference.
        use std::f64::consts::PI;
        let n = 4usize;
        let rq = |q: usize| n - 1 - q;
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::H, &[rq(0)]);
        dag.add_op(OpType::Ry(Parameter::Const(0.41)), &[rq(1)]);
        for k in 0..44 {
            let q = k % n;
            match k % 4 {
                0 => {
                    dag.add_op(
                        OpType::Rz(Parameter::Const(0.11 * (k as f64) - PI)),
                        &[rq(q)],
                    );
                }
                1 => {
                    dag.add_op(OpType::CZ, &[rq(q), rq((q + 1) % n)]);
                }
                2 => {
                    dag.add_op(
                        OpType::CP(Parameter::Const(0.07 * (k as f64) + 0.3)),
                        &[rq(q), rq((q + 2) % n)],
                    );
                }
                _ => {
                    dag.add_op(
                        OpType::Rzz(Parameter::Const(0.05 * (k as f64))),
                        &[rq((q + 1) % n), rq((q + 3) % n)],
                    );
                }
            }
        }
        dag.add_op(OpType::CNOT, &[rq(0), rq(1)]);
        dag.add_op(OpType::Rz(Parameter::Const(0.9)), &[rq(1)]);
        dag.add_op(OpType::CNOT, &[rq(0), rq(1)]);
        dag.add_op(OpType::X, &[rq(3)]);

        let sv = dag.simulate();
        let u = dag.to_unitary();
        for row in 0..sv.len() {
            let expected = u[(row, 0)];
            assert!(
                (sv[row] - expected).norm() < 1e-9,
                "row {row}: {} vs {}",
                sv[row],
                expected
            );
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

    // ── Perm-merge gather sweep tests ──

    /// Random CNOT (control, target) pairs for permutation tests.
    fn random_cnot_pairs(n: usize, len: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut s = seed;
        let mut next = |m: usize| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as usize) % m
        };
        let mut pairs = Vec::with_capacity(len);
        for _ in 0..len {
            let c = next(n);
            let mut t = next(n);
            if t == c {
                t = (t + 1) % n;
            }
            pairs.push((c, t));
        }
        pairs
    }

    /// Column composition of a CNOT run (same update as `merge_cnot_runs`).
    fn compose_cols(n: usize, pairs: &[(usize, usize)]) -> Vec<usize> {
        let mut cols: Vec<usize> = (0..n).map(|q| 1usize << q).collect();
        for &(c, t) in pairs {
            for q in 0..n {
                cols[q] ^= ((cols[q] >> c) & 1) << t;
            }
        }
        cols
    }

    #[test]
    fn test_perm_gather_tables_are_exact_inverse() {
        // `out[d] = state[src(d)]` must invert the scatter map exactly:
        // src(dest(i)) == i for every basis index (values only move).
        fn src_from_tables(tables: &[usize], d: usize) -> usize {
            let n_chunks = tables.len() / 256;
            let mut s = 0usize;
            for c in 0..n_chunks {
                s ^= tables[c * 256 + ((d >> (8 * c)) & 0xFF)];
            }
            s
        }
        for (n, seed) in [(8usize, 1u64), (13, 2), (20, 3)] {
            let cols = compose_cols(n, &random_cnot_pairs(n, 40, seed));
            let tables = build_perm_gather_tables(&cols);
            assert_eq!(tables.len(), ((n + 7) / 8) * 256, "n={n} table size");
            for i in 0..(1usize << n) {
                let d = perm_dest(&cols, i);
                assert_eq!(src_from_tables(&tables, d), i, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn test_perm_sweep_gather_matches_scatter() {
        // Function-level sweep equivalence: the serial gather, the
        // Rayon-split gather and the historical scatter loop must produce
        // bit-identical output buffers for the same permutation.
        #[allow(clippy::type_complexity)]
        fn run_three(
            data: &[num_complex::Complex64],
            cols: &[usize],
            tables: &[usize],
            pf: usize,
        ) -> (
            Vec<num_complex::Complex64>,
            Vec<num_complex::Complex64>,
            Vec<num_complex::Complex64>,
        ) {
            let dim = data.len();
            let zero = num_complex::Complex64::new(0.0, 0.0);
            let mut scatter = vec![zero; dim];
            let mut serial = vec![zero; dim];
            let mut par = vec![zero; dim];
            apply_perm_scatter(data, &mut scatter, cols, false);
            let src = SendPtr(data.as_ptr() as *mut num_complex::Complex64);
            let s = SendPtr(serial.as_mut_ptr());
            let p = SendPtr(par.as_mut_ptr());
            unsafe {
                perm_gather_dispatch::<num_complex::Complex64>(
                    src.raw(),
                    s.raw(),
                    tables,
                    0,
                    dim,
                    pf,
                );
            }
            // Rayon-split path: two ranges so the parallel branch runs even
            // on a single-threaded test pool.
            let chunk = ((dim / 2) + 7) & !7usize;
            let n_ranges = (dim + chunk - 1) / chunk;
            (0..n_ranges).into_par_iter().for_each(|c| {
                let start = c * chunk;
                let end = (start + chunk).min(dim);
                unsafe {
                    perm_gather_dispatch::<num_complex::Complex64>(
                        src.raw(),
                        p.raw(),
                        tables,
                        start,
                        end,
                        pf,
                    );
                }
            });
            (scatter, serial, par)
        }

        for (n, seed) in [(12usize, 5u64), (16, 6)] {
            let dim = 1usize << n;
            let cols = compose_cols(n, &random_cnot_pairs(n, 36, seed));
            let tables = build_perm_gather_tables(&cols);
            let data: Vec<num_complex::Complex64> = (0..dim)
                .map(|i| num_complex::Complex64::new(i as f64, 0.25 * i as f64))
                .collect();
            let (scatter, serial, par) = run_three(&data, &cols, &tables, 16);
            for i in 0..dim {
                assert_eq!(
                    (scatter[i].re, scatter[i].im),
                    (serial[i].re, serial[i].im),
                    "gather vs scatter n={n} i={i}"
                );
                assert_eq!(
                    (scatter[i].re, scatter[i].im),
                    (par[i].re, par[i].im),
                    "par gather vs scatter n={n} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_cnot_run_perm_matches_unitary() {
        // A CNOT run long enough to merge (>= PERM_MIN_RUN) must reproduce
        // the exact unitary action through the gather sweep; `to_unitary`
        // walks the legacy per-gate kernel path and is the reference.
        let n = 6usize;
        let rq = |q: usize| n - 1 - q;
        let pairs = random_cnot_pairs(n, 9, 11);
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::H, &[rq(0)]);
        dag.add_op(OpType::Ry(Parameter::Const(0.37)), &[rq(3)]);
        for &(c, t) in &pairs {
            dag.add_op(OpType::CNOT, &[rq(c), rq(t)]);
        }
        dag.add_op(OpType::Rz(Parameter::Const(0.8)), &[rq(1)]);
        let sv = dag.simulate();
        let u = dag.to_unitary();
        for row in 0..sv.len() {
            assert!(
                (sv[row] - u[(row, 0)]).norm() < 1e-9,
                "row {row}: {} vs {}",
                sv[row],
                u[(row, 0)]
            );
        }
    }

    /// Manual micro-benchmark (not run by default): per-window cost of one
    /// fused k=6 block, old per-group worker vs the banded worker across
    /// band sizes, plus the un-fused K-streaming-pass reference. Verifies
    /// bit-exact equality of every banded variant first. Run with:
    /// `cargo test -p sf-ir --release fused_banded_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn fused_banded_microbench() {
        use std::time::Instant;
        let dim = 1usize << 24;
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut nxt = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let state: Vec<num_complex::Complex64> = (0..dim)
            .map(|_| num_complex::Complex64::new(nxt(), nxt()))
            .collect();
        let windows: [(&str, [usize; 6]); 4] = [
            ("low_0_5", [0, 1, 2, 3, 4, 5]),
            ("mid1_6_11", [6, 7, 8, 9, 10, 11]),
            ("mid2_12_17", [12, 13, 14, 15, 16, 17]),
            ("high_18_23", [18, 19, 20, 21, 22, 23]),
        ];
        let bands = [8usize, 16, 32, 64, 128, 256, 512, 1024];
        for (name, qs) in windows {
            let masks: Vec<usize> = qs.iter().map(|&q| 1usize << q).collect();
            let mut u = Vec::with_capacity(24);
            for t in 0..6 {
                let c = (t as f64 + 0.5) * 0.37;
                u.push(num_complex::Complex64::new(c.cos() * 0.9, c.sin() * 0.1));
                u.push(num_complex::Complex64::new(-c.sin() * 0.2, c.cos()));
                u.push(num_complex::Complex64::new(c.sin(), c.cos() * 0.3));
                u.push(num_complex::Complex64::new(c.cos(), -c.sin() * 0.7));
            }
            let mut a = state.clone();
            let mut b = state.clone();
            let t0 = Instant::now();
            apply_fused_tensor::<6>(&mut a, &masks, &u, &[], &[], false);
            let v0 = t0.elapsed().as_secs_f64() * 1e3;
            for &band in &bands {
                b.copy_from_slice(&state);
                apply_fused_tensor_banded_with::<6>(&mut b, &masks, &u, &[], &[], false, band);
                assert!(
                    a.iter().zip(b.iter()).all(|(v, w)| v == w),
                    "{name}: banded band={band} not bit-exact"
                );
            }
            let mut line = format!("{name:10} v0 {v0:8.1} ms |");
            for &band in &bands {
                b.copy_from_slice(&state);
                let t1 = Instant::now();
                apply_fused_tensor_banded_with::<6>(&mut b, &masks, &u, &[], &[], false, band);
                let ms = t1.elapsed().as_secs_f64() * 1e3;
                line.push_str(&format!("  b{band} {ms:7.1}"));
            }
            // Un-fused reference: one streaming pass per gate (strides =
            // the window's actual masks).
            b.copy_from_slice(&state);
            let t2 = Instant::now();
            for t in 0..6 {
                let m = [[u[4 * t], u[4 * t + 1]], [u[4 * t + 2], u[4 * t + 3]]];
                crate::simd::pair_pass_chunk(&mut b, masks[t], m);
            }
            line.push_str(&format!(" | 6pass {:.1}", t2.elapsed().as_secs_f64() * 1e3));
            println!("{line}");
            // Tiled worker (contiguous-run gather): flat copy at `lo == 0`,
            // one contiguous run per group-local index otherwise.
            let mut tl = format!("{name:10} tiled    |");
            for &band in &[64usize, 256, 1024] {
                b.copy_from_slice(&state);
                apply_fused_tensor_tiled_with::<6>(&mut b, &masks, &u, &[], &[], false, band);
                assert!(
                    a.iter().zip(b.iter()).all(|(v, w)| v == w),
                    "{name}: tiled band={band} not bit-exact"
                );
                b.copy_from_slice(&state);
                let t1 = Instant::now();
                apply_fused_tensor_tiled_with::<6>(&mut b, &masks, &u, &[], &[], false, band);
                let ms = t1.elapsed().as_secs_f64() * 1e3;
                tl.push_str(&format!("  t{band} {ms:7.1}"));
            }
            println!("{tl}");
            // Interleaved production check (same process/window, min of 5
            // alternating rounds): the dispatcher must pick the faster
            // worker — this answers whether the banded win survives on the
            // real entry point under current machine load.
            let mut tv = f64::MAX;
            let mut td = f64::MAX;
            for _ in 0..5 {
                a.copy_from_slice(&state);
                let t0 = Instant::now();
                apply_fused_tensor::<6>(&mut a, &masks, &u, &[], &[], false);
                tv = tv.min(t0.elapsed().as_secs_f64() * 1e3);
                b.copy_from_slice(&state);
                let t1 = Instant::now();
                fused_dispatch::<6>(&mut b, &masks, &u, &[], &[], false);
                td = td.min(t1.elapsed().as_secs_f64() * 1e3);
            }
            assert!(
                a.iter().zip(b.iter()).all(|(v, w)| v == w),
                "{name}: dispatch not bit-exact vs v0"
            );
            println!("{name:10} [interleaved min-of-5] v0 {tv:8.1} ms | dispatch {td:8.1} ms");
        }
    }

    /// The tiled worker must be bit-exact against the per-group reference on
    /// contiguous-run blocks: the flat `lo == 0` layout, a mid window whose
    /// runs are `2^lo` wide, and a high window, across band sizes — plus the
    /// flat path with folded diagonal stages.
    #[test]
    fn fused_tiled_parity() {
        let n = 16usize;
        let dim = 1usize << n;
        let mut x: u64 = 0xD1CE_B00C_5EED_1234;
        let mut nxt = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let state: Vec<num_complex::Complex64> = (0..dim)
            .map(|_| num_complex::Complex64::new(nxt(), nxt()))
            .collect();
        let windows: [(&str, [usize; 6]); 3] = [
            ("low_0_5", [0, 1, 2, 3, 4, 5]),
            ("mid_4_9", [4, 5, 6, 7, 8, 9]),
            ("high_10_15", [10, 11, 12, 13, 14, 15]),
        ];
        let bands = [2usize, 4, 8, 16];
        for (name, qs) in windows {
            let masks: Vec<usize> = qs.iter().map(|&q| 1usize << q).collect();
            let mut u = Vec::with_capacity(24);
            for t in 0..6 {
                let c = (t as f64 + 0.5) * 0.41;
                u.push(num_complex::Complex64::new(c.cos(), c.sin() * 0.3));
                u.push(num_complex::Complex64::new(-c.sin(), c.cos() * 0.8));
                u.push(num_complex::Complex64::new(c.sin() * 0.5, c.cos()));
                u.push(num_complex::Complex64::new(c.cos() * 0.7, -c.sin()));
            }
            let mut a = state.clone();
            apply_fused_tensor::<6>(&mut a, &masks, &u, &[], &[], false);
            for &band in &bands {
                let mut b = state.clone();
                apply_fused_tensor_tiled_with::<6>(&mut b, &masks, &u, &[], &[], false, band);
                assert!(
                    a.iter().zip(b.iter()).all(|(v, w)| v == w),
                    "{name}: tiled band={band} not bit-exact"
                );
            }
        }
        // Folded diagonals ride the flat (`lo == 0`) layout only.
        let masks: Vec<usize> = (0..6).map(|q| 1usize << q).collect();
        let stage = Diag2Stage {
            i: 0,
            j: 1,
            d: [
                num_complex::Complex64::new(1.0, 0.0),
                num_complex::Complex64::new(0.3, -0.7),
                num_complex::Complex64::new(-0.2, 0.9),
                num_complex::Complex64::new(0.5, 0.5),
            ],
        };
        let u: Vec<num_complex::Complex64> = (0..6)
            .flat_map(|t| {
                let c = t as f64 * 0.7;
                [
                    num_complex::Complex64::new(c.cos(), 0.0),
                    num_complex::Complex64::new(-c.sin(), 0.0),
                    num_complex::Complex64::new(c.sin(), 0.0),
                    num_complex::Complex64::new(c.cos(), 0.0),
                ]
            })
            .collect();
        let mut a = state.clone();
        apply_fused_tensor::<6>(&mut a, &masks, &u, &[stage], &[stage], false);
        for &band in &bands {
            let mut b = state.clone();
            apply_fused_tensor_tiled_with::<6>(&mut b, &masks, &u, &[stage], &[stage], false, band);
            assert!(
                a.iter().zip(b.iter()).all(|(v, w)| v == w),
                "folded flat tiled band={band} not bit-exact"
            );
        }
    }

    /// Bit-exactness of the tiled worker for every dispatched K (2..=10),
    /// across flat (`lo == 0`) and shifted (`lo == 6`) windows.
    #[test]
    fn fused_tiled_parity_k2_10() {
        let dim = 1usize << 16;
        let state: Vec<num_complex::Complex64> = (0..dim)
            .map(|i| {
                let a = i as f64 * 0.0017;
                num_complex::Complex64::new(a.sin(), (a * 0.61).cos())
            })
            .collect();
        macro_rules! k_check {
            ($($k:literal),*) => {
                $(
                    {
                        let k: usize = $k;
                        for lo in [0usize, 6] {
                            let masks: Vec<usize> =
                                (0..k).map(|i| 1usize << (lo + i)).collect();
                            let u: Vec<num_complex::Complex64> = (0..k * 4)
                                .map(|t| {
                                    let c = t as f64 * 0.23;
                                    num_complex::Complex64::new(c.cos(), c.sin() * 0.5)
                                })
                                .collect();
                            let mut a = state.clone();
                            apply_fused_tensor::<$k>(&mut a, &masks, &u, &[], &[], false);
                            for band in [2usize, 64] {
                                let mut b = state.clone();
                                apply_fused_tensor_tiled_with::<$k>(
                                    &mut b, &masks, &u, &[], &[], false, band,
                                );
                                assert!(
                                    a.iter().zip(b.iter()).all(|(v, w)| v == w),
                                    "K={k} lo={lo} band={band} not bit-exact"
                                );
                            }
                        }
                    }
                )*
            };
        }
        k_check!(2, 3, 4, 5, 6, 7, 8, 9, 10);
    }
}
