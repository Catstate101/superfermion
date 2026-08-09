//! Superfermion Quantum Error Correction (sf-qec)
//!
//! Rust-native QEC library providing:
//! - Stabilizer code representations (Surface, Steane, Repetition)
//! - Syndrome extraction circuits
//! - Decoders (MWPM, Union-Find, Lookup Table)
//! - Logical error rate estimation

#![allow(clippy::new_without_default)]

pub mod codes;
pub mod decoders;
pub mod syndrome;

pub use codes::{RepetitionCode, StabilizerCode, SteaneCode, SurfaceCode};
pub use decoders::{Decoder, LookupDecoder, MWPMDecoder, UnionFindDecoder};
pub use syndrome::SyndromeExtractor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        let code = RepetitionCode::new(3);
        assert_eq!(code.n_data(), 3);
        assert_eq!(code.n_ancilla(), 2);
    }
}
pub mod logical_ops;
