//! Superfermion Quantum Error Correction (sf-qec)
//!
//! Rust-native QEC library providing:
//! - Stabilizer code representations (Surface, Steane, Repetition)
//! - Syndrome extraction circuits
//! - Decoders (MWPM, Union-Find, Lookup Table)
//! - Logical error rate estimation

pub mod codes;
pub mod syndrome;
pub mod decoders;

pub use codes::{StabilizerCode, SurfaceCode, RepetitionCode, SteaneCode};
pub use syndrome::SyndromeExtractor;
pub use decoders::{Decoder, MWPMDecoder, UnionFindDecoder, LookupDecoder};

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
