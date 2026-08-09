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
pub mod gate_list;
pub mod mps;
pub mod ops;
pub mod qasm;
pub mod qubits;
pub mod serialize;
pub mod stabilizer;
pub mod state;

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
