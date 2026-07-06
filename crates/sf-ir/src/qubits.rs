//! Qubit management — physical/logical qubit mapping and allocation.
//!
//! Provides types for tracking qubit identity across compilation
//! and routing stages where logical-to-physical mappings change.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A logical qubit — the user's abstract qubit identity.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct LogicalQubit(pub usize);

/// A physical qubit — the hardware device's qubit index.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct PhysicalQubit(pub usize);

impl fmt::Display for LogicalQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}", self.0)
    }
}

impl fmt::Display for PhysicalQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// Bidirectional mapping between logical and physical qubits.
///
/// Maintained by the router; updated every time a SWAP is inserted.
/// At the start of routing, logical == physical (identity map).
#[derive(Clone, Debug)]
pub struct QubitMapping {
    /// logical → physical
    l2p: Vec<usize>,
    /// physical → logical
    p2l: Vec<usize>,
    n_qubits: usize,
}

impl QubitMapping {
    /// Create an identity mapping for `n` qubits.
    pub fn identity(n: usize) -> Self {
        Self {
            l2p: (0..n).collect(),
            p2l: (0..n).collect(),
            n_qubits: n,
        }
    }

    /// Create from an explicit logical→physical map.
    pub fn from_layout(layout: &[usize]) -> Self {
        let n = layout.len();
        let mut p2l = vec![0usize; n];
        for (l, &p) in layout.iter().enumerate() {
            p2l[p] = l;
        }
        Self {
            l2p: layout.to_vec(),
            p2l,
            n_qubits: n,
        }
    }

    /// Logical → Physical lookup.
    pub fn logical_to_physical(&self, logical: LogicalQubit) -> PhysicalQubit {
        PhysicalQubit(self.l2p[logical.0])
    }

    /// Physical → Logical lookup.
    pub fn physical_to_logical(&self, physical: PhysicalQubit) -> LogicalQubit {
        LogicalQubit(self.p2l[physical.0])
    }

    /// Swap two physical qubits in the mapping (called when router inserts a SWAP).
    pub fn swap_physical(&mut self, p0: PhysicalQubit, p1: PhysicalQubit) {
        let l0 = self.p2l[p0.0];
        let l1 = self.p2l[p1.0];
        self.p2l[p0.0] = l1;
        self.p2l[p1.0] = l0;
        self.l2p[l0] = p1.0;
        self.l2p[l1] = p0.0;
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get the full logical-to-physical map as a slice.
    pub fn as_l2p(&self) -> &[usize] {
        &self.l2p
    }

    /// Get the full physical-to-logical map as a slice.
    pub fn as_p2l(&self) -> &[usize] {
        &self.p2l
    }

    /// Check if this is the identity mapping.
    pub fn is_identity(&self) -> bool {
        self.l2p.iter().enumerate().all(|(i, &p)| i == p)
    }
}

/// Qubit allocator — tracks which physical qubits are available.
#[derive(Clone, Debug)]
pub struct QubitAllocator {
    n_total: usize,
    allocated: Vec<bool>,
}

impl QubitAllocator {
    pub fn new(n_total: usize) -> Self {
        Self {
            n_total,
            allocated: vec![false; n_total],
        }
    }

    /// Allocate the next available physical qubit.
    pub fn allocate(&mut self) -> Option<PhysicalQubit> {
        for (i, used) in self.allocated.iter_mut().enumerate() {
            if !*used {
                *used = true;
                return Some(PhysicalQubit(i));
            }
        }
        None
    }

    /// Allocate a specific physical qubit.
    pub fn allocate_specific(&mut self, qubit: PhysicalQubit) -> bool {
        if qubit.0 < self.n_total && !self.allocated[qubit.0] {
            self.allocated[qubit.0] = true;
            true
        } else {
            false
        }
    }

    /// Release a physical qubit.
    pub fn release(&mut self, qubit: PhysicalQubit) {
        if qubit.0 < self.n_total {
            self.allocated[qubit.0] = false;
        }
    }

    /// Number of available qubits.
    pub fn available(&self) -> usize {
        self.allocated.iter().filter(|&&a| !a).count()
    }

    pub fn n_total(&self) -> usize {
        self.n_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mapping() {
        let m = QubitMapping::identity(4);
        assert!(m.is_identity());
        assert_eq!(m.logical_to_physical(LogicalQubit(2)), PhysicalQubit(2));
        assert_eq!(m.physical_to_logical(PhysicalQubit(3)), LogicalQubit(3));
    }

    #[test]
    fn test_swap_physical() {
        let mut m = QubitMapping::identity(4);
        m.swap_physical(PhysicalQubit(0), PhysicalQubit(2));
        assert_eq!(m.logical_to_physical(LogicalQubit(0)), PhysicalQubit(2));
        assert_eq!(m.logical_to_physical(LogicalQubit(2)), PhysicalQubit(0));
        assert_eq!(m.physical_to_logical(PhysicalQubit(0)), LogicalQubit(2));
        assert!(!m.is_identity());
    }

    #[test]
    fn test_from_layout() {
        // Logical 0→Physical 2, Logical 1→Physical 0, Logical 2→Physical 1
        let m = QubitMapping::from_layout(&[2, 0, 1]);
        assert_eq!(m.logical_to_physical(LogicalQubit(0)), PhysicalQubit(2));
        assert_eq!(m.physical_to_logical(PhysicalQubit(2)), LogicalQubit(0));
    }

    #[test]
    fn test_allocator() {
        let mut a = QubitAllocator::new(4);
        assert_eq!(a.available(), 4);
        let q = a.allocate().unwrap();
        assert_eq!(q, PhysicalQubit(0));
        assert_eq!(a.available(), 3);
        a.release(q);
        assert_eq!(a.available(), 4);
    }
}
