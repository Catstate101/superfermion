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
    X,
    Z,
    Y,
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
        a.abs_diff(b)
    }
}

/// Unique data qubit shared by checks `a` and `b` (`None` if not exactly one).
fn shared_qubit(map: &[Vec<usize>], a: usize, b: usize) -> Option<usize> {
    match (map.get(a), map.get(b)) {
        (Some(x), Some(y)) => {
            let mut it = x.iter().filter(|q| y.contains(q));
            match (it.next(), it.next()) {
                (Some(&q), None) => Some(q),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Exact minimum-weight decoding for 1D chain maps.
///
/// A clean chain map has `n_data` qubits, `n_data - 1` checks, every check
/// touching exactly 2 qubits, and consecutive checks sharing exactly one
/// qubit (like `[[0,1],[1,2],...]` up to relabeling). Then each syndrome bit
/// is the parity of two adjacent data bits, so fixing `e[0]` determines the
/// whole error vector — exactly two candidates exist; the lighter wins.
/// This is ML-optimal (not an approximation) for such maps.
///
/// Returns `None` unless the map is a clean chain (caller falls back to
/// greedy + validation, so non-chain codes are unaffected).
fn chain_exact_decode(
    map: &[Vec<usize>],
    n_data: usize,
    syndrome: &[u8],
) -> Option<Vec<(usize, CorrectionType)>> {
    let m = map.len();
    if m + 1 != n_data || syndrome.len() != m || n_data < 2 {
        return None;
    }
    for qubits in map.iter() {
        if qubits.len() != 2 {
            return None;
        }
    }
    // Walk the chain: order[0] is the boundary qubit of check 0 (absent
    // from check 1); each subsequent order[k+1] is the other member of
    // check[k], which must also sit in check[k+1] (except the last).
    // Any deviation means "not a clean chain" -> None (caller falls back).
    let (a0, b0) = (map[0][0], map[0][1]);
    let start = if m == 1 || !map[1].contains(&a0) {
        a0
    } else if !map[1].contains(&b0) {
        b0
    } else {
        return None;
    };
    let mut order: Vec<usize> = vec![start, if start == a0 { b0 } else { a0 }];
    for k in 1..m {
        if !map[k].contains(&order[k]) {
            return None;
        }
        let mut others = map[k].iter().filter(|q| **q != order[k]);
        match (others.next(), others.next()) {
            (Some(&q), None) => {
                if order.contains(&q) {
                    return None;
                }
                if k + 1 < m && !map[k + 1].contains(&q) {
                    return None;
                }
                order.push(q);
            }
            _ => return None,
        }
    }
    if order.len() != n_data {
        return None;
    }
    // Two candidates: e[0] = 0 or 1, rest forced by adjacent parities.
    let mut best: Vec<(usize, CorrectionType)> = Vec::new();
    let mut best_w = usize::MAX;
    for first_bit in [0u8, 1u8] {
        let mut e = vec![0u8; n_data];
        e[0] = first_bit;
        for k in 0..m {
            let s = if k < syndrome.len() {
                syndrome[k]
            } else {
                return None;
            };
            // check k constrains order[k] (+) order[k+1]
            e[k + 1] = e[k] ^ (s & 1);
        }
        let w: usize = e.iter().map(|&b| b as usize).sum();
        if w < best_w {
            best_w = w;
            best = e
                .iter()
                .enumerate()
                .filter(|(_, &b)| b == 1)
                .map(|(i, _)| (order[i], CorrectionType::X))
                .collect();
        }
    }
    Some(best)
}

/// Corrections pairing a single defect `k` to its nearest boundary.
/// Exact at chain ends (left: data 0..=k; right: data k+1..); interior
/// single defects have no unambiguous side — returns empty so that
/// H*e==s validation upstream rejects it and the fallback handles it.
fn boundary_correction(
    map: &[Vec<usize>],
    n_data: usize,
    k: usize,
) -> Vec<(usize, CorrectionType)> {
    let m = map.len();
    if k == 0 {
        (0..=k).map(|q| (q, CorrectionType::X)).collect()
    } else if m > 0 && k + 1 == m {
        ((k + 1)..n_data).map(|q| (q, CorrectionType::X)).collect()
    } else {
        Vec::new()
    }
}

/// Corrections for defect pair (d1, d2): the lighter of the connecting
/// chain (shared-qubit walk, exact for chain-like maps such as the
/// repetition code) vs pairing both defects to their boundaries (always
/// H-consistent for chain maps). Non-chain dead ends fall through to
/// upstream H*e==s validation + fallback (correctness preserved).
fn pair_corrections(
    map: &[Vec<usize>],
    n_data: usize,
    d1: usize,
    d2: usize,
) -> Vec<(usize, CorrectionType)> {
    let (lo, hi) = (d1.min(d2), d1.max(d2));
    let mut chain: Vec<(usize, CorrectionType)> = Vec::new();
    let mut chain_ok = true;
    for k in lo..hi {
        match shared_qubit(map, k, k + 1) {
            Some(q) => chain.push((q, CorrectionType::X)),
            None => {
                chain_ok = false;
                break;
            }
        }
    }
    let mut boundary: Vec<(usize, CorrectionType)> =
        (0..=lo).map(|q| (q, CorrectionType::X)).collect();
    if hi + 1 < n_data {
        boundary.extend(((hi + 1)..n_data).map(|q| (q, CorrectionType::X)));
    }
    if chain_ok && chain.len() <= boundary.len() {
        chain
    } else {
        boundary
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
            return Correction {
                corrections: vec![],
            };
        }

        // Exact path for chain maps (repetition-like): ML-optimal.
        // Wrong outputs are impossible here beyond validation — the
        // Python wrapper re-checks H*e==s and falls back to BP+OSD.
        if let Some(exact) = chain_exact_decode(&self.syndrome_qubit_map, self._n_data, syndrome) {
            return Correction { corrections: exact };
        }

        let pairs = self.greedy_matching(&defects);

        // Translate each pair via the shared chain/boundary helper.
        let mut corrections = Vec::new();
        for (d1, d2) in pairs {
            if d2 == usize::MAX {
                // Unpaired boundary defect.
                corrections.extend(boundary_correction(
                    &self.syndrome_qubit_map,
                    self._n_data,
                    d1,
                ));
            } else {
                corrections.extend(pair_corrections(
                    &self.syndrome_qubit_map,
                    self._n_data,
                    d1,
                    d2,
                ));
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
            return Correction {
                corrections: vec![],
            };
        }

        // Exact path for chain maps (repetition-like): ML-optimal.
        // Python-side H*e==s validation still guards the result.
        if let Some(exact) = chain_exact_decode(&self.syndrome_qubit_map, self._n_data, syndrome) {
            return Correction { corrections: exact };
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
        for members in cluster_sizes.values() {
            if members.len() % 2 == 1 {
                // Odd cluster: pair consecutive members, boundary-pair the leftover.
                let mut prev: Option<usize> = None;
                for &d in members {
                    if let Some(p) = prev {
                        corrections.extend(pair_corrections(
                            &self.syndrome_qubit_map,
                            self._n_data,
                            p,
                            d,
                        ));
                        prev = None;
                    } else {
                        prev = Some(d);
                    }
                }
                if let Some(last) = prev {
                    corrections.extend(boundary_correction(
                        &self.syndrome_qubit_map,
                        self._n_data,
                        last,
                    ));
                }
            } else {
                // Even cluster — pair first and last defects.
                let first = *members.first().unwrap();
                let last = *members.last().unwrap();
                corrections.extend(pair_corrections(
                    &self.syndrome_qubit_map,
                    self._n_data,
                    first,
                    last,
                ));
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
        table.insert(
            vec![0; n - 1],
            Correction {
                corrections: vec![],
            },
        );

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
        self.table.get(syndrome).cloned().unwrap_or(Correction {
            corrections: vec![],
        })
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

    fn applies_rep(c: &Correction, syn: &[u8], n: usize) -> bool {
        // H*e == s over the repetition chain map [[k,k+1]].
        let mut e = vec![0u8; n];
        for (q, _) in &c.corrections {
            if *q >= n {
                return false;
            }
            e[*q] ^= 1;
        }
        for k in 0..n - 1 {
            if e[k] ^ e[k + 1] != syn[k] {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_mwpm_all_single_errors_exact() {
        // Every single data error must decode to itself (was: check-index
        // confusion returned map[q][0], failing H*e==s).
        for n in [3usize, 5, 9] {
            let decoder = MWPMDecoder::for_repetition(n);
            for err in 0..n {
                let mut syn = vec![0u8; n - 1];
                if err > 0 {
                    syn[err - 1] ^= 1;
                }
                if err + 1 < n {
                    syn[err] ^= 1;
                }
                let c = decoder.decode(&syn);
                assert!(applies_rep(&c, &syn, n), "n={n} err={err}");
                assert_eq!(c.corrections.len(), 1);
                assert_eq!(c.corrections[0].0, err);
            }
        }
    }

    #[test]
    fn test_mwpm_boundary_pair_picks_light_side() {
        // Defects {0,3} on d=5: chain {1,2,3} (w3) vs boundaries {0},{4}
        // (w2) — must pick the boundary option.
        let decoder = MWPMDecoder::for_repetition(5);
        let c = decoder.decode(&[1, 0, 0, 1]);
        assert!(applies_rep(&c, &[1, 0, 0, 1], 5));
        assert_eq!(c.corrections.len(), 2);
    }

    #[test]
    fn test_union_find_all_single_errors_exact() {
        for n in [3usize, 5, 9] {
            let decoder = UnionFindDecoder::for_repetition(n);
            for err in 0..n {
                let mut syn = vec![0u8; n - 1];
                if err > 0 {
                    syn[err - 1] ^= 1;
                }
                if err + 1 < n {
                    syn[err] ^= 1;
                }
                let c = decoder.decode(&syn);
                assert!(applies_rep(&c, &syn, n), "n={n} err={err}");
            }
        }
    }

    fn brute_optimal_weight(map: &[Vec<usize>], n: usize, syn: &[u8]) -> Option<usize> {
        // Minimum error weight satisfying H*e == s, by exhaustive search.
        let m = map.len();
        let mut best: Option<usize> = None;
        for mask in 0..(1usize << n) {
            let mut ok = true;
            for (a, row) in map.iter().enumerate() {
                let mut p = 0u8;
                for &q in row {
                    p ^= ((mask >> q) & 1) as u8;
                }
                if p != syn[a] {
                    ok = false;
                    break;
                }
            }
            if ok {
                let w = mask.count_ones() as usize;
                best = Some(best.map_or(w, |b: usize| b.min(w)));
            }
        }
        best
    }

    #[test]
    fn test_chain_exact_matches_brute_force_optimum() {
        // On chain maps, chain_exact_decode must attain the true minimum
        // weight (ML-optimal), not just a valid correction.
        for n in [3usize, 5, 7] {
            let map: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
            for mask in 0..(1usize << (n - 1)) {
                let syn: Vec<u8> = (0..n - 1).map(|k| ((mask >> k) & 1) as u8).collect();
                let got = chain_exact_decode(&map, n, &syn);
                let opt = brute_optimal_weight(&map, n, &syn);
                match (got, opt) {
                    (Some(c), Some(w)) => {
                        assert_eq!(c.len(), w, "n={n} syn={syn:?}");
                    }
                    (None, None) => {}
                    (g, o) => panic!("n={n} syn={syn:?} got={g:?} opt={o:?}"),
                }
            }
        }
    }

    #[test]
    fn test_chain_exact_rejects_non_chain() {
        // Surface-code-style map (checks share 0 or 2+ qubits): must be None.
        let map = vec![vec![0, 1, 2, 3], vec![1, 2, 4, 5]];
        assert!(chain_exact_decode(&map, 6, &[1, 0]).is_none());
        // Wrong syndrome length.
        let rep: Vec<Vec<usize>> = vec![vec![0, 1], vec![1, 2]];
        assert!(chain_exact_decode(&rep, 3, &[1, 0, 0]).is_none());
    }
}
