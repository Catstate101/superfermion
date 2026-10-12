//! SuperFermion "cat" backend.
//!
//! `cat` is an opt-in execution core that sits above the vetted `sf-ir`
//! kernels.  Its value is a *compile-once* handle: a circuit is lowered to a
//! [`CatPlan`] a single time (topological ordering, constant gate matrices and
//! fusion metadata cached), then re-bound and executed per call without ever
//! rebuilding a DAG or re-parsing gates.  This removes the fixed per-call cost
//! that dominates small-qubit statevector and adjoint workloads.
//!
//! The crate reuses the AVX2/Rayon kernels already in `sf-ir` rather than
//! reimplementing SIMD primitives; the new logic is the scheduling / IR /
//! handle layer.  Modes are added incrementally (statevector first).

#![allow(clippy::needless_range_loop)]

pub mod adj;
pub mod dispatch;
pub mod kernels;
pub mod paulframe;
pub mod plan;
pub mod router;

pub use adj::AdjProgram;
pub use dispatch::{run, run_gradient, Exec};
pub use paulframe::{FrameTableau, PEntry};
pub use plan::{CatPlan, Mode};
pub use router::{classify, select, CircuitProfile, Representation};
