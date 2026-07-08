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

pub mod dag;
pub mod ops;
pub mod qubits;
pub mod classical;
pub mod serialize;
pub mod mps;
pub mod dm;
pub mod qasm;
pub mod stabilizer;
pub mod gate_list;
pub mod adjoint;
pub mod commutation;
pub mod state;

// Re-export core types at crate root for convenience
pub use dag::{QuantumDAG, QuantumOp, WireType, CircuitMetadata, NodeId, QubitId};
pub use ops::{OpType, Parameter, ParameterExpr};
pub use qubits::{LogicalQubit, PhysicalQubit, QubitMapping, QubitAllocator};
pub use classical::{ClassicalRegister, ClassicalRegFile};
pub use serialize::SerializedCircuit;
pub use mps::MPSState;
pub use adjoint::{adjoint_grad, AdjointGradResult, PauliTerm};
pub use state::{QuantumStateImpl, MethodError, StatevectorState, DensityMatrixStateWrapper, MPSStateWrapper, StabilizerStateWrapper};

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
