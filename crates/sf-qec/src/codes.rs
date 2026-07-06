//! Stabilizer code definitions — Surface, Steane, Repetition.

use serde::{Deserialize, Serialize};

/// Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pauli {
    I, X, Y, Z,
}

/// A stabilizer generator: product of Paulis on data qubits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stabilizer {
    pub paulis: Vec<(usize, Pauli)>, // (qubit_index, pauli)
    pub ancilla: usize,               // ancilla qubit used for measurement
}

/// Trait for stabilizer codes.
pub trait StabilizerCode {
    fn name(&self) -> &str;
    fn n_data(&self) -> usize;
    fn n_ancilla(&self) -> usize;
    fn n_logical(&self) -> usize;
    fn distance(&self) -> usize;
    fn stabilizers(&self) -> &[Stabilizer];
    fn x_stabilizers(&self) -> Vec<&Stabilizer>;
    fn z_stabilizers(&self) -> Vec<&Stabilizer>;
}

/// Repetition code [[n, 1, n]] — simplest QEC code, corrects bit-flip only.
#[derive(Clone, Debug)]
pub struct RepetitionCode {
    n: usize,
    stabilizers: Vec<Stabilizer>,
}

impl RepetitionCode {
    pub fn new(n: usize) -> Self {
        assert!(n >= 3, "Repetition code needs n >= 3");
        let stabilizers = (0..n - 1)
            .map(|i| Stabilizer {
                paulis: vec![(i, Pauli::Z), (i + 1, Pauli::Z)],
                ancilla: n + i,
            })
            .collect();
        Self { n, stabilizers }
    }
}

impl StabilizerCode for RepetitionCode {
    fn name(&self) -> &str { "RepetitionCode" }
    fn n_data(&self) -> usize { self.n }
    fn n_ancilla(&self) -> usize { self.n - 1 }
    fn n_logical(&self) -> usize { 1 }
    fn distance(&self) -> usize { self.n }
    fn stabilizers(&self) -> &[Stabilizer] { &self.stabilizers }
    fn x_stabilizers(&self) -> Vec<&Stabilizer> { vec![] }
    fn z_stabilizers(&self) -> Vec<&Stabilizer> { self.stabilizers.iter().collect() }
}

/// Steane code [[7, 1, 3]] — smallest CSS code.
#[derive(Clone, Debug)]
pub struct SteaneCode {
    stabilizers: Vec<Stabilizer>,
}

impl SteaneCode {
    pub fn new() -> Self {
        // [[7,1,3]] stabilizers
        // X stabilizers: X on qubits {0,1,2,3}, {0,1,4,5}, {0,2,4,6}
        // Z stabilizers: Z on same sets
        let x_sets = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4, 5],
            vec![0, 2, 4, 6],
        ];
        let z_sets = x_sets.clone();

        let mut stabilizers = Vec::new();
        for (i, qubits) in x_sets.iter().enumerate() {
            stabilizers.push(Stabilizer {
                paulis: qubits.iter().map(|&q| (q, Pauli::X)).collect(),
                ancilla: 7 + i,
            });
        }
        for (i, qubits) in z_sets.iter().enumerate() {
            stabilizers.push(Stabilizer {
                paulis: qubits.iter().map(|&q| (q, Pauli::Z)).collect(),
                ancilla: 7 + 3 + i,
            });
        }

        Self { stabilizers }
    }
}

impl StabilizerCode for SteaneCode {
    fn name(&self) -> &str { "SteaneCode" }
    fn n_data(&self) -> usize { 7 }
    fn n_ancilla(&self) -> usize { 6 }
    fn n_logical(&self) -> usize { 1 }
    fn distance(&self) -> usize { 3 }
    fn stabilizers(&self) -> &[Stabilizer] { &self.stabilizers }
    fn x_stabilizers(&self) -> Vec<&Stabilizer> {
        self.stabilizers[..3].iter().collect()
    }
    fn z_stabilizers(&self) -> Vec<&Stabilizer> {
        self.stabilizers[3..].iter().collect()
    }
}

/// Surface code [[d², d²-1, d]] — the most practical QEC code.
///
/// On a d × d grid, data qubits sit on vertices,
/// X-stabilizers on faces, Z-stabilizers on edges.
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    distance: usize,
    n_data: usize,
    n_ancilla: usize,
    stabilizers: Vec<Stabilizer>,
}

impl SurfaceCode {
    pub fn new(distance: usize) -> Self {
        assert!(distance >= 3 && distance % 2 == 1, "Distance must be odd >= 3");

        let d = distance;
        let n_data = d * d;
        // For rotated surface code: (d²-1)/2 X-checks + (d²-1)/2 Z-checks
        let _n_ancilla = d * d - 1;

        let mut stabilizers = Vec::new();
        let mut ancilla_idx = n_data;

        // X-stabilizers (faces)
        for r in 0..d - 1 {
            for c in 0..d - 1 {
                if (r + c) % 2 == 0 {
                    // X-plaquette
                    let qubits = vec![
                        r * d + c,
                        r * d + c + 1,
                        (r + 1) * d + c,
                        (r + 1) * d + c + 1,
                    ];
                    stabilizers.push(Stabilizer {
                        paulis: qubits.into_iter().map(|q| (q, Pauli::X)).collect(),
                        ancilla: ancilla_idx,
                    });
                    ancilla_idx += 1;
                }
            }
        }

        // Z-stabilizers (edges)
        for r in 0..d - 1 {
            for c in 0..d - 1 {
                if (r + c) % 2 == 1 {
                    let qubits = vec![
                        r * d + c,
                        r * d + c + 1,
                        (r + 1) * d + c,
                        (r + 1) * d + c + 1,
                    ];
                    stabilizers.push(Stabilizer {
                        paulis: qubits.into_iter().map(|q| (q, Pauli::Z)).collect(),
                        ancilla: ancilla_idx,
                    });
                    ancilla_idx += 1;
                }
            }
        }

        // Boundary stabilizers (weight-2)
        // Top boundary
        for c in (0..d - 1).step_by(2) {
            stabilizers.push(Stabilizer {
                paulis: vec![(c, Pauli::Z), (c + 1, Pauli::Z)],
                ancilla: ancilla_idx,
            });
            ancilla_idx += 1;
        }
        // Bottom boundary
        for c in (1..d - 1).step_by(2) {
            stabilizers.push(Stabilizer {
                paulis: vec![
                    ((d - 1) * d + c, Pauli::Z),
                    ((d - 1) * d + c + 1, Pauli::Z),
                ],
                ancilla: ancilla_idx,
            });
            ancilla_idx += 1;
        }

        Self {
            distance: d,
            n_data,
            n_ancilla: ancilla_idx - n_data,
            stabilizers,
        }
    }
}

impl StabilizerCode for SurfaceCode {
    fn name(&self) -> &str { "SurfaceCode" }
    fn n_data(&self) -> usize { self.n_data }
    fn n_ancilla(&self) -> usize { self.n_ancilla }
    fn n_logical(&self) -> usize { 1 }
    fn distance(&self) -> usize { self.distance }
    fn stabilizers(&self) -> &[Stabilizer] { &self.stabilizers }
    fn x_stabilizers(&self) -> Vec<&Stabilizer> {
        self.stabilizers
            .iter()
            .filter(|s| s.paulis.iter().any(|(_, p)| *p == Pauli::X))
            .collect()
    }
    fn z_stabilizers(&self) -> Vec<&Stabilizer> {
        self.stabilizers
            .iter()
            .filter(|s| s.paulis.iter().all(|(_, p)| *p == Pauli::Z))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_code() {
        let code = RepetitionCode::new(5);
        assert_eq!(code.n_data(), 5);
        assert_eq!(code.n_ancilla(), 4);
        assert_eq!(code.distance(), 5);
        assert_eq!(code.stabilizers().len(), 4);
    }

    #[test]
    fn test_steane_code() {
        let code = SteaneCode::new();
        assert_eq!(code.n_data(), 7);
        assert_eq!(code.n_ancilla(), 6);
        assert_eq!(code.distance(), 3);
        assert_eq!(code.stabilizers().len(), 6);
        assert_eq!(code.x_stabilizers().len(), 3);
        assert_eq!(code.z_stabilizers().len(), 3);
    }

    #[test]
    fn test_surface_code() {
        let code = SurfaceCode::new(3);
        assert_eq!(code.n_data(), 9);
        assert_eq!(code.distance(), 3);
        assert!(code.stabilizers().len() > 0);
    }

    #[test]
    fn test_surface_code_d5() {
        let code = SurfaceCode::new(5);
        assert_eq!(code.n_data(), 25);
        assert_eq!(code.distance(), 5);
    }
}
pub mod ldpc; pub mod bivariate;
