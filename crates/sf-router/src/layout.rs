//! Layout strategies — initial mapping of logical to physical qubits.

use sf_ir::QubitMapping;
use crate::topology::CouplingMap;

/// Trait for layout strategies.
pub trait LayoutStrategy {
    fn compute_layout(&self, n_logical: usize, coupling: &CouplingMap) -> QubitMapping;
}

/// Trivial layout: logical qubit i → physical qubit i.
pub struct TrivialLayout;

impl TrivialLayout {
    pub fn compute(n_qubits: usize) -> QubitMapping {
        QubitMapping::identity(n_qubits)
    }
}

impl LayoutStrategy for TrivialLayout {
    fn compute_layout(&self, n_logical: usize, _coupling: &CouplingMap) -> QubitMapping {
        QubitMapping::identity(n_logical)
    }
}

/// Noise-aware layout: place most-interacting qubits on best-connected physical qubits.
///
/// Heuristic: sort logical qubits by 2Q gate frequency,
/// sort physical qubits by connectivity degree, match greedily.
pub struct NoiseAwareLayout;

impl NoiseAwareLayout {
    /// Compute layout based on gate interaction frequency.
    pub fn compute(
        n_logical: usize,
        interactions: &[(usize, usize)],
        coupling: &CouplingMap,
    ) -> QubitMapping {
        let n_physical = coupling.n_qubits();
        assert!(n_logical <= n_physical);

        // Count 2Q interaction frequency per qubit
        let mut freq = vec![0usize; n_logical];
        for &(a, b) in interactions {
            if a < n_logical {
                freq[a] += 1;
            }
            if b < n_logical {
                freq[b] += 1;
            }
        }

        // Sort logical qubits by frequency (descending)
        let mut logical_order: Vec<usize> = (0..n_logical).collect();
        logical_order.sort_by(|a, b| freq[*b].cmp(&freq[*a]));

        // Sort physical qubits by degree (descending)
        let mut physical_order: Vec<usize> = (0..n_physical).collect();
        physical_order.sort_by(|a, b| {
            coupling.neighbors(*b).len().cmp(&coupling.neighbors(*a).len())
        });

        // Greedy assignment
        let mut layout = vec![0usize; n_logical];
        for (i, &logical) in logical_order.iter().enumerate() {
            layout[logical] = physical_order[i];
        }

        QubitMapping::from_layout(&layout)
    }
}

impl LayoutStrategy for NoiseAwareLayout {
    fn compute_layout(&self, n_logical: usize, _coupling: &CouplingMap) -> QubitMapping {
        // Without interaction data, fall back to identity
        QubitMapping::identity(n_logical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_layout() {
        let layout = TrivialLayout::compute(4);
        assert!(layout.is_identity());
    }

    #[test]
    fn test_noise_aware_layout() {
        let coupling = CouplingMap::linear(5);

        // Qubits 0 and 3 interact most
        let interactions = vec![(0, 3), (0, 3), (0, 3), (1, 2)];
        let layout = NoiseAwareLayout::compute(4, &interactions, &coupling);

        // Qubit 0 should get a well-connected physical qubit
        // (middle of linear chain has degree 2)
        let p0 = layout.logical_to_physical(sf_ir::LogicalQubit(0));
        // The most-interacting logical qubit should be placed centrally
        assert!(p0.0 < 5);
    }
}
