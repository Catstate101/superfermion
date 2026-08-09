//! Hardware Topology — coupling map representation.
//!
//! Defines the physical connectivity of quantum hardware.
//! Used by the SABRE router to determine which qubit pairs
//! can interact directly vs needing SWAP insertions.

use std::collections::{BTreeSet, VecDeque};

/// Coupling map — the hardware connectivity graph.
///
/// Each edge (i, j) means physical qubits i and j can perform a 2-qubit gate directly.
/// Edges are treated as undirected (if CX(0,1) is available, CX(1,0) is too via H conjugation).
#[derive(Clone, Debug)]
pub struct CouplingMap {
    n_qubits: usize,
    /// Adjacency list: qubit → set of neighbors
    adjacency: Vec<BTreeSet<usize>>,
    /// Distance matrix (lazily computed): dist[i][j] = shortest path length
    distances: Option<Vec<Vec<usize>>>,
}

impl CouplingMap {
    /// Create from a list of edges.
    pub fn from_edges(n_qubits: usize, edges: &[(usize, usize)]) -> Self {
        let mut adjacency = vec![BTreeSet::new(); n_qubits];
        for &(a, b) in edges {
            adjacency[a].insert(b);
            adjacency[b].insert(a);
        }
        let mut map = Self {
            n_qubits,
            adjacency,
            distances: None,
        };
        map.compute_distances();
        map
    }

    /// Linear chain: 0—1—2—…—(n-1)
    pub fn linear(n: usize) -> Self {
        let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Self::from_edges(n, &edges)
    }

    /// Grid topology: rows × cols
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if c + 1 < cols {
                    edges.push((idx, idx + 1));
                }
                if r + 1 < rows {
                    edges.push((idx, idx + cols));
                }
            }
        }
        Self::from_edges(n, &edges)
    }

    /// Heavy-hex topology (used by IBM Eagle/Heron processors).
    /// Simplified version: creates a heavy-hex-like structure.
    pub fn heavy_hex(n_qubits: usize) -> Self {
        // For a basic heavy-hex, create a grid with extra connector qubits
        // This is a simplified approximation
        let cols = (n_qubits as f64).sqrt().ceil() as usize;
        let rows = n_qubits.div_ceil(cols);
        let _actual_n = rows * cols;
        let mut edges = Vec::new();

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if idx >= n_qubits {
                    continue;
                }
                // Horizontal
                if c + 1 < cols && idx + 1 < n_qubits {
                    edges.push((idx, idx + 1));
                }
                // Vertical (every other column)
                if r + 1 < rows && c % 2 == 0 {
                    let below = (r + 1) * cols + c;
                    if below < n_qubits {
                        edges.push((idx, below));
                    }
                }
            }
        }

        Self::from_edges(n_qubits, &edges)
    }

    /// All-to-all connectivity (for simulators).
    pub fn all_to_all(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j));
            }
        }
        Self::from_edges(n, &edges)
    }

    /// Number of physical qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Check if two qubits are directly connected.
    pub fn is_connected(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        a < self.n_qubits && b < self.n_qubits && self.adjacency[a].contains(&b)
    }

    /// Get neighbors of a qubit.
    pub fn neighbors(&self, qubit: usize) -> &BTreeSet<usize> {
        &self.adjacency[qubit]
    }

    /// Get the shortest distance between two qubits.
    pub fn distance(&self, a: usize, b: usize) -> usize {
        if a == b {
            return 0;
        }
        match &self.distances {
            Some(d) => d[a][b],
            None => self.bfs_distance(a, b),
        }
    }

    /// Compute all-pairs shortest paths via BFS.
    fn compute_distances(&mut self) {
        let n = self.n_qubits;
        let mut dist = vec![vec![usize::MAX; n]; n];

        for source in 0..n {
            dist[source][source] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(source);

            while let Some(current) = queue.pop_front() {
                for &neighbor in &self.adjacency[current] {
                    if dist[source][neighbor] == usize::MAX {
                        dist[source][neighbor] = dist[source][current] + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        self.distances = Some(dist);
    }

    fn bfs_distance(&self, a: usize, b: usize) -> usize {
        let mut visited = vec![false; self.n_qubits];
        let mut queue = VecDeque::new();
        visited[a] = true;
        queue.push_back((a, 0));

        while let Some((node, dist)) = queue.pop_front() {
            if node == b {
                return dist;
            }
            for &neighbor in &self.adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        usize::MAX // Disconnected
    }

    /// Get all edges as a list of pairs.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for (a, neighbors) in self.adjacency.iter().enumerate() {
            for &b in neighbors {
                if a < b {
                    edges.push((a, b));
                }
            }
        }
        edges
    }

    /// Number of edges (coupling links).
    pub fn n_edges(&self) -> usize {
        self.edges().len()
    }
}

/// Predefined hardware topologies for major QPU families.
pub struct HardwareTopology;

impl HardwareTopology {
    /// IBM Eagle r3 (127 qubits, heavy-hex)
    pub fn ibm_eagle() -> CouplingMap {
        CouplingMap::heavy_hex(127)
    }

    /// IBM Heron (133 qubits, heavy-hex)
    pub fn ibm_heron() -> CouplingMap {
        CouplingMap::heavy_hex(133)
    }

    /// Rigetti Ankaa (84 qubits, grid-like)
    pub fn rigetti_ankaa() -> CouplingMap {
        CouplingMap::grid(7, 12)
    }

    /// IonQ Forte (36 qubits, all-to-all trapped ion)
    pub fn ionq_forte() -> CouplingMap {
        CouplingMap::all_to_all(36)
    }

    /// IQM Garnet (20 qubits, grid)
    pub fn iqm_garnet() -> CouplingMap {
        CouplingMap::grid(4, 5)
    }

    /// Generic simulator (no connectivity constraints)
    pub fn simulator(n_qubits: usize) -> CouplingMap {
        CouplingMap::all_to_all(n_qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology() {
        let topo = CouplingMap::linear(5);
        assert_eq!(topo.n_qubits(), 5);
        assert!(topo.is_connected(0, 1));
        assert!(topo.is_connected(1, 2));
        assert!(!topo.is_connected(0, 2)); // Not directly connected
        assert_eq!(topo.distance(0, 2), 2);
        assert_eq!(topo.distance(0, 4), 4);
    }

    #[test]
    fn test_grid_topology() {
        let topo = CouplingMap::grid(3, 3);
        assert_eq!(topo.n_qubits(), 9);
        // (0,0)-(0,1) connected
        assert!(topo.is_connected(0, 1));
        // (0,0)-(1,0) connected
        assert!(topo.is_connected(0, 3));
        // (0,0)-(1,1) NOT connected
        assert!(!topo.is_connected(0, 4));
        assert_eq!(topo.distance(0, 8), 4); // corner to corner
    }

    #[test]
    fn test_all_to_all() {
        let topo = CouplingMap::all_to_all(5);
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(topo.is_connected(i, j));
                    assert_eq!(topo.distance(i, j), 1);
                }
            }
        }
    }

    #[test]
    fn test_hardware_topologies() {
        let eagle = HardwareTopology::ibm_eagle();
        assert_eq!(eagle.n_qubits(), 127);

        let ionq = HardwareTopology::ionq_forte();
        assert_eq!(ionq.n_qubits(), 36);
        // IonQ all-to-all: any pair is distance 1
        assert_eq!(ionq.distance(0, 35), 1);

        let iqm = HardwareTopology::iqm_garnet();
        assert_eq!(iqm.n_qubits(), 20);
    }

    #[test]
    fn test_neighbors() {
        let topo = CouplingMap::linear(4);
        assert_eq!(topo.neighbors(0).len(), 1); // only qubit 1
        assert_eq!(topo.neighbors(1).len(), 2); // qubits 0, 2
        assert_eq!(topo.neighbors(3).len(), 1); // only qubit 2
    }
}
