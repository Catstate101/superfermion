//! Decoders — syndrome-to-correction mapping.
//!
//! Three decoders:
//! 1. **MWPM** (Minimum Weight Perfect Matching) — optimal for surface codes
//! 2. **Union-Find** — near-linear time, suitable for real-time decoding
//! 3. **Lookup Table** — precomputed corrections for small codes

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Correction to apply after decoding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Correction {
    /// Qubit index → Pauli correction to apply
    pub corrections: Vec<(usize, CorrectionType)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionType {
    X, Z, Y,
}

/// Trait for QEC decoders.
pub trait Decoder {
    fn name(&self) -> &str;
    /// Decode a syndrome bit-string into a correction.
    fn decode(&self, syndrome: &[u8]) -> Correction;
}

// ═══════════════════════════════════════════════════════════
// MWPM Decoder
// ═══════════════════════════════════════════════════════════

/// Minimum Weight Perfect Matching decoder.
///
/// Constructs a complete graph where vertices are defects (non-trivial
/// syndrome bits) and edges are weighted by graph distance. The minimum
/// weight perfect matching identifies the most likely error.
pub struct MWPMDecoder {
    /// Syndrome bit → qubit associations
    syndrome_qubit_map: Vec<Vec<usize>>,
    _n_data: usize,
}

impl MWPMDecoder {
    pub fn new(n_data: usize, syndrome_qubit_map: Vec<Vec<usize>>) -> Self {
        Self {
            syndrome_qubit_map,
            _n_data: n_data,
        }
    }

    /// Create for a repetition code.
    pub fn for_repetition(n: usize) -> Self {
        let map: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        Self::new(n, map)
    }

    /// Greedy MWPM approximation: pair defects by nearest distance.
    fn greedy_matching(&self, defects: &[usize]) -> Vec<(usize, usize)> {
        let mut remaining: Vec<usize> = defects.to_vec();
        let mut pairs = Vec::new();

        while remaining.len() >= 2 {
            // Find the closest pair
            let (mut best_i, mut best_j, mut best_dist) = (0, 1, usize::MAX);

            for i in 0..remaining.len() {
                for j in i + 1..remaining.len() {
                    let d = self.defect_distance(remaining[i], remaining[j]);
                    if d < best_dist {
                        best_i = i;
                        best_j = j;
                        best_dist = d;
                    }
                }
            }

            pairs.push((remaining[best_i], remaining[best_j]));
            remaining.remove(best_j);
            remaining.remove(best_i);
        }

        // Odd number of defects: pair last one with boundary
        if remaining.len() == 1 {
            pairs.push((remaining[0], usize::MAX)); // boundary
        }

        pairs
    }

    fn defect_distance(&self, a: usize, b: usize) -> usize {
        if a > b { a - b } else { b - a }
    }
}

impl Decoder for MWPMDecoder {
    fn name(&self) -> &str {
        "MWPM"
    }

    fn decode(&self, syndrome: &[u8]) -> Correction {
        // Find defects (non-trivial syndrome bits)
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Correction { corrections: vec![] };
        }

        let pairs = self.greedy_matching(&defects);

        // For each pair, correct the qubit between them
        let mut corrections = Vec::new();
        for (d1, d2) in pairs {
            if d2 == usize::MAX {
                // Boundary defect
                if d1 < self.syndrome_qubit_map.len() {
                    corrections.push((self.syndrome_qubit_map[d1][0], CorrectionType::X));
                }
            } else {
                // Correct the chain between d1 and d2
                for q in d1..d2 {
                    if q < self.syndrome_qubit_map.len() && !self.syndrome_qubit_map[q].is_empty() {
                        corrections.push((self.syndrome_qubit_map[q][0], CorrectionType::X));
                    }
                }
            }
        }

        Correction { corrections }
    }
}

// ═══════════════════════════════════════════════════════════
// Union-Find Decoder
// ═══════════════════════════════════════════════════════════

/// Union-Find decoder — near-linear time complexity.
///
/// Uses a disjoint-set (union-find) data structure to cluster defects.
/// Connected clusters that have odd parity need correction.
pub struct UnionFindDecoder {
    syndrome_qubit_map: Vec<Vec<usize>>,
    _n_data: usize,
}

impl UnionFindDecoder {
    pub fn new(n_data: usize, syndrome_qubit_map: Vec<Vec<usize>>) -> Self {
        Self {
            syndrome_qubit_map,
            _n_data: n_data,
        }
    }

    pub fn for_repetition(n: usize) -> Self {
        let map: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        Self::new(n, map)
    }
}

impl Decoder for UnionFindDecoder {
    fn name(&self) -> &str {
        "UnionFind"
    }

    fn decode(&self, syndrome: &[u8]) -> Correction {
        let n = syndrome.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0usize; n];

        // Find with path compression
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        // Union by rank
        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx == ry {
                return;
            }
            if rank[rx] < rank[ry] {
                parent[rx] = ry;
            } else if rank[rx] > rank[ry] {
                parent[ry] = rx;
            } else {
                parent[ry] = rx;
                rank[rx] += 1;
            }
        }

        // Find defects
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Correction { corrections: vec![] };
        }

        // Grow clusters: union adjacent defects
        for i in 0..defects.len() {
            for j in i + 1..defects.len() {
                if defects[j] - defects[i] <= 1 {
                    union(&mut parent, &mut rank, defects[i], defects[j]);
                }
            }
        }

        // For each cluster with odd parity, generate corrections
        let mut cluster_sizes: HashMap<usize, Vec<usize>> = HashMap::new();
        for &d in &defects {
            let root = find(&mut parent, d);
            cluster_sizes.entry(root).or_default().push(d);
        }

        let mut corrections = Vec::new();
        for (_, members) in &cluster_sizes {
            if members.len() % 2 == 1 {
                // Odd parity cluster — correct first qubit
                let d = members[0];
                if d < self.syndrome_qubit_map.len() && !self.syndrome_qubit_map[d].is_empty() {
                    corrections.push((self.syndrome_qubit_map[d][0], CorrectionType::X));
                }
            } else {
                // Even parity — correct chain between first and last
                let first = *members.first().unwrap();
                let last = *members.last().unwrap();
                for q in first..last {
                    if q < self.syndrome_qubit_map.len() && !self.syndrome_qubit_map[q].is_empty() {
                        corrections.push((self.syndrome_qubit_map[q][0], CorrectionType::X));
                    }
                }
            }
        }

        Correction { corrections }
    }
}

// ═══════════════════════════════════════════════════════════
// Lookup Table Decoder
// ═══════════════════════════════════════════════════════════

/// Lookup table decoder — precomputed syndrome → correction map.
///
/// Fast (O(1) decode time) but only practical for small codes.
pub struct LookupDecoder {
    table: HashMap<Vec<u8>, Correction>,
}

impl LookupDecoder {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    /// Build lookup table for a repetition code of distance n.
    pub fn for_repetition(n: usize) -> Self {
        let mut table = HashMap::new();

        // No error
        table.insert(vec![0; n - 1], Correction { corrections: vec![] });

        // Single-qubit X errors
        for q in 0..n {
            let mut syndrome = vec![0u8; n - 1];
            if q > 0 {
                syndrome[q - 1] = 1;
            }
            if q < n - 1 {
                syndrome[q] = 1;
            }
            table.insert(
                syndrome,
                Correction {
                    corrections: vec![(q, CorrectionType::X)],
                },
            );
        }

        Self { table }
    }

    pub fn add_entry(&mut self, syndrome: Vec<u8>, correction: Correction) {
        self.table.insert(syndrome, correction);
    }

    pub fn n_entries(&self) -> usize {
        self.table.len()
    }
}

impl Decoder for LookupDecoder {
    fn name(&self) -> &str {
        "LookupTable"
    }

    fn decode(&self, syndrome: &[u8]) -> Correction {
        self.table
            .get(syndrome)
            .cloned()
            .unwrap_or(Correction { corrections: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mwpm_no_error() {
        let decoder = MWPMDecoder::for_repetition(5);
        let correction = decoder.decode(&[0, 0, 0, 0]);
        assert!(correction.corrections.is_empty());
    }

    #[test]
    fn test_mwpm_single_error() {
        let decoder = MWPMDecoder::for_repetition(5);
        // Error on qubit 2: syndrome bits 1 and 2 light up
        let correction = decoder.decode(&[0, 1, 1, 0]);
        assert!(!correction.corrections.is_empty());
    }

    #[test]
    fn test_union_find_no_error() {
        let decoder = UnionFindDecoder::for_repetition(5);
        let correction = decoder.decode(&[0, 0, 0, 0]);
        assert!(correction.corrections.is_empty());
    }

    #[test]
    fn test_union_find_single_error() {
        let decoder = UnionFindDecoder::for_repetition(5);
        let correction = decoder.decode(&[0, 1, 1, 0]);
        assert!(!correction.corrections.is_empty());
    }

    #[test]
    fn test_lookup_decoder() {
        let decoder = LookupDecoder::for_repetition(3);
        assert!(decoder.n_entries() > 0);

        // No error
        let c = decoder.decode(&[0, 0]);
        assert!(c.corrections.is_empty());

        // Error on qubit 0: syndrome = [1, 0]
        let c = decoder.decode(&[1, 0]);
        assert_eq!(c.corrections.len(), 1);
        assert_eq!(c.corrections[0].0, 0);

        // Error on qubit 1: syndrome = [1, 1]
        let c = decoder.decode(&[1, 1]);
        assert_eq!(c.corrections.len(), 1);
        assert_eq!(c.corrections[0].0, 1);
    }
}
