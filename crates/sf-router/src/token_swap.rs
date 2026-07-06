//! Token Swapping — SWAP sequence to realize a qubit permutation.
//!
//! Given a coupling map and a target permutation, finds a sequence
//! of SWAPs that transforms the identity into the target permutation.
//!
//! Uses iterative displacement reduction: each round, process all
//! edges and swap if it reduces total displacement.

use crate::topology::CouplingMap;
use crate::RouterError;

/// Find a sequence of SWAPs to realize the given permutation.
///
/// `perm[i] = j` means position i should end up holding token j.
///
/// Returns a list of (qubit_a, qubit_b) SWAPs to apply.
pub fn token_swap_route(
    coupling: &CouplingMap,
    target_perm: &[usize],
) -> Result<Vec<(usize, usize)>, RouterError> {
    let n = target_perm.len();
    if n != coupling.n_qubits() {
        return Err(RouterError::TokenSwapError(format!(
            "Permutation size {} != coupling map size {}", n, coupling.n_qubits()
        )));
    }

    let mut current: Vec<usize> = (0..n).collect();
    let mut swaps = Vec::new();
    let edges = coupling.edges();

    let max_rounds = n * n;

    for _ in 0..max_rounds {
        if current == target_perm {
            return Ok(swaps);
        }

        let mut made_progress = false;

        // For each edge, swap if it reduces total displacement
        for &(a, b) in &edges {
            if current == target_perm {
                return Ok(swaps);
            }

            let d_before = displacement(coupling, current[a], target_perm[a])
                + displacement(coupling, current[b], target_perm[b]);
            let d_after = displacement(coupling, current[b], target_perm[a])
                + displacement(coupling, current[a], target_perm[b]);

            if d_after < d_before {
                current.swap(a, b);
                swaps.push((a, b));
                made_progress = true;
            }
        }

        if !made_progress {
            break;
        }
    }

    if current == target_perm {
        Ok(swaps)
    } else {
        Err(RouterError::TokenSwapError(
            "Failed to converge to target permutation".to_string(),
        ))
    }
}

/// Distance from current token value to target value at a position.
fn displacement(coupling: &CouplingMap, token: usize, target: usize) -> usize {
    if token == target {
        0
    } else {
        coupling.distance(token, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_permutation() {
        let coupling = CouplingMap::linear(4);
        let perm = vec![0, 1, 2, 3];
        let swaps = token_swap_route(&coupling, &perm).unwrap();
        assert!(swaps.is_empty());
    }

    #[test]
    fn test_adjacent_swap() {
        let coupling = CouplingMap::linear(4);
        let perm = vec![1, 0, 2, 3]; // swap positions 0 and 1
        let swaps = token_swap_route(&coupling, &perm).unwrap();
        // Verify result
        let mut test = vec![0, 1, 2, 3];
        for &(a, b) in &swaps {
            test.swap(a, b);
        }
        assert_eq!(test, perm);
    }

    #[test]
    fn test_reverse_permutation_grid() {
        let coupling = CouplingMap::grid(2, 2); // 4 qubits, richer connectivity
        let perm = vec![3, 2, 1, 0];
        let swaps = token_swap_route(&coupling, &perm).unwrap();
        let mut test = vec![0, 1, 2, 3];
        for &(a, b) in &swaps {
            test.swap(a, b);
        }
        assert_eq!(test, perm);
    }

    #[test]
    fn test_all_to_all_any_permutation() {
        let coupling = CouplingMap::all_to_all(5);
        let perm = vec![4, 3, 2, 1, 0]; // full reverse — trivial on all-to-all
        let swaps = token_swap_route(&coupling, &perm).unwrap();
        let mut test: Vec<usize> = (0..5).collect();
        for &(a, b) in &swaps {
            test.swap(a, b);
        }
        assert_eq!(test, perm);
    }
}
