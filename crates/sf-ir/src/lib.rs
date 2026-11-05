//! Superfermion Quantum Intermediate Representation (sf-ir)
//!
//! This is the foundation of Superfermion. The IR is:
//! - **Hardware-agnostic**: represents abstract quantum circuits
//! - **DAG-based**: enables compiler optimizations via graph transformations
//! - **Parameterized**: supports variational/trainable quantum circuits
//! - **Serializable**: circuits can be saved, loaded, and transmitted
//!
//! # Architecture
//! ```text
//! User API (Python)
//!     ↓
//! sf.Circuit → builds → QuantumDAG
//!     ↓
//! sf-compiler → transforms → QuantumDAG (hardware-specific)
//!     ↓
//! sf-router → routes → QuantumDAG (connectivity-mapped)
//!     ↓
//! Backend → executes → Results
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::match_like_matches_macro)]

pub mod adjoint;
pub mod classical;
pub mod commutation;
pub mod dag;
pub mod dm;
pub mod f32lane;
pub mod gate_list;
pub mod mps;
pub mod ops;
pub mod qasm;
pub mod qubits;
pub mod serialize;
pub mod simd;
pub mod stabilizer;
pub mod state;
pub mod sysmem;

// Re-export core types at crate root for convenience
pub use adjoint::{adjoint_grad, AdjointGradResult, PauliTerm};
pub use classical::{ClassicalRegFile, ClassicalRegister};
pub use dag::{CircuitMetadata, NodeId, QuantumDAG, QuantumOp, QubitId, WireType};
pub use mps::MPSState;
pub use ops::{OpType, Parameter, ParameterExpr};
pub use qubits::{LogicalQubit, PhysicalQubit, QubitAllocator, QubitMapping};
pub use serialize::SerializedCircuit;
pub use state::{
    DensityMatrixStateWrapper, MPSStateWrapper, MethodError, QuantumStateImpl,
    StabilizerStateWrapper, StatevectorState,
};

/// Derive the seed of the `index`-th independent random sub-stream of `seed`.
///
/// Every sampler that draws several independent streams from one user seed
/// (one stream per shot, or one per block of shots) must go through this
/// function instead of `seed.wrapping_add(index)`.
///
/// `wrapping_add` is not a decorrelation: the stream for seed `s` at block
/// `b` is the stream for seed `s + 1` at block `b - 1`, so runs that differ
/// only in the seed re-draw ~62 of the 63 blocks of a 4000-shot run and
/// return near-identical counts. That collapsed the spread across seeds to
/// 5.3 on a GHZ-4 stabilizer run whose binomial spread is 31.6, which makes
/// error bars built from repeated seeds far too narrow. Mixing both the seed
/// and the index through the splitmix64 finalizer gives streams that share
/// no state, while staying fully deterministic and reproducible.
#[inline]
pub fn mix_seed(seed: u64, index: u64) -> u64 {
    let mut z = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Diagnostic: returns a human-readable reason why GPU init failed (or "ok").
#[cfg(feature = "gpu")]
pub fn gpu_diagnose() -> String {
    sf_gpu::diagnose()
}

/// Diagnostic: GPU not compiled.
#[cfg(not(feature = "gpu"))]
pub fn gpu_diagnose() -> String {
    "GPU feature not compiled".to_string()
}

#[cfg(test)]
mod mix_seed_tests {
    use super::mix_seed;

    /// Adjacent seeds must not share sub-streams: `mix_seed(s, i)` and
    /// `mix_seed(s + 1, i)` have to differ for every index (with
    /// `wrapping_add` they were adjacent by construction).
    #[test]
    fn adjacent_seeds_do_not_collide() {
        for s in 0..64u64 {
            for i in 0..64u64 {
                assert_ne!(mix_seed(s, i), mix_seed(s, i + 1));
                assert_ne!(mix_seed(s, i), mix_seed(s + 1, i));
            }
        }
    }

    /// No two of the first 1024 `(seed, index)` pairs share a stream.
    ///
    /// This is a window, not a proof.  The construction is injective in each
    /// argument on its own -- `index` enters as a product with an odd constant
    /// and the splitmix64 finalizer is a bijection -- but not over the pair:
    /// `mix_seed(0, 0) == mix_seed(0x9E37_79B9_7F4A_7C15, 1) == 0`, because
    /// both feed the finalizer a zero.  That is harmless for sampling: within
    /// one run only the index varies (so streams never alias), and two runs
    /// collide only when `seed1 ^ seed2` happens to be the exact odd multiple
    /// of the mixing constant that the two indices differ by.
    #[test]
    fn a_thousand_pairs_do_not_alias() {
        let mut seen = std::collections::HashSet::new();
        for s in 0..32u64 {
            for i in 0..32u64 {
                assert!(seen.insert(mix_seed(s, i)), "collision at ({s}, {i})");
            }
        }
    }

    /// Determinism: the same inputs always give the same stream seed.
    #[test]
    fn deterministic() {
        assert_eq!(mix_seed(12345, 678), mix_seed(12345, 678));
        assert_ne!(mix_seed(12345, 678), mix_seed(678, 12345));
    }
}
