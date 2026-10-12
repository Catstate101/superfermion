//! Optional C++ kernel lane (additive Lever-1 experiment).
//!
//! When `build.rs` finds a local MinGW-w64 g++, it compiles
//! `csrc/transpose.cpp` into `libsf_cat_cxx.a` and sets `sf_cat_cxx`; this
//! module then exposes the two C-ABI entry points to the pack path.
//!
//! Kernel choice is runtime-switched with `SF_CAT_KERNEL`:
//!   * unset / `ds` — C++ delta-swap transpose (the default when built):
//!     same-window ABCCBA A/B measured 4.4-4.5x faster pack on the brick
//!     ladder (2.6x median over the rung set, counts improve 1.2-1.5x);
//!     on already-sub-ms GHZ pack it is ~0.06 ms slower absolute.
//!   * `rust`    — the Rust scalar loop (still the fallback everywhere)
//!   * `literal` — C++ port of the same loop (codegen comparison; lost to
//!     the Rust loop on brick, kept for reproducibility)
//!
//! Every variant is pinned bit-exact against the Rust scalar transpose by
//! `paulframe::tests::cpp_pack_kernels_match_scalar_transpose`, and the Rust
//! loop remains the always-available fallback (also on machines without a
//! C++ toolchain, where the FFI items do not exist at all).

use std::sync::OnceLock;

/// Which pack transpose runs in `sample_packed`'s fallback lane.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PackKernel {
    Rust,
    #[cfg(sf_cat_cxx)]
    CppLiteral,
    #[cfg(sf_cat_cxx)]
    CppDs,
}

/// Name of the active pack kernel, for reporting from examples/benches.
pub fn current_kernel() -> &'static str {
    match selected() {
        PackKernel::Rust => "rust",
        #[cfg(sf_cat_cxx)]
        PackKernel::CppLiteral => "literal",
        #[cfg(sf_cat_cxx)]
        PackKernel::CppDs => "ds",
    }
}

/// True when the C++ archive was linked in at build time.
pub fn cxx_built() -> bool {
    cfg!(sf_cat_cxx)
}

#[cfg(sf_cat_cxx)]
extern "C" {
    fn sf_cat_transpose_literal(
        lanes: *const u64,
        packed: *mut u64,
        n: usize,
        words: usize,
        cnt: usize,
    );
    fn sf_cat_transpose_ds(lanes: *const u64, packed: *mut u64, n: usize, words: usize, cnt: usize);
}

/// Resolve the runtime kernel choice once (`SF_CAT_KERNEL`).
pub(crate) fn selected() -> PackKernel {
    static SEL: OnceLock<PackKernel> = OnceLock::new();
    *SEL.get_or_init(|| {
        match std::env::var("SF_CAT_KERNEL").unwrap_or_default().as_str() {
            "rust" => PackKernel::Rust,
            #[cfg(sf_cat_cxx)]
            "literal" => PackKernel::CppLiteral,
            // `ds` and the unset default: the delta-swap lane when built in;
            // machines without a C++ toolchain fall through to `_` = Rust.
            #[cfg(sf_cat_cxx)]
            "ds" | "" => PackKernel::CppDs,
            _ => PackKernel::Rust,
        }
    })
}

/// Run the selected non-Rust transpose if one is active.  Returns `false`
/// when the caller should run its own (Rust) scalar loop instead.
#[cfg_attr(not(sf_cat_cxx), allow(unused_variables))]
pub(crate) fn transpose(lanes: &[u64], packed: &mut [u64], words: usize, cnt: usize) -> bool {
    match selected() {
        PackKernel::Rust => false,
        #[cfg(sf_cat_cxx)]
        PackKernel::CppLiteral => {
            debug_assert!(packed.len() >= cnt * words, "block slice sized cnt*words");
            // Safety: `lanes` is the block's n outcome lanes and `packed` is
            // the block's `cnt * words` key slice (checked above); the C
            // entries only touch those ranges.
            unsafe {
                sf_cat_transpose_literal(
                    lanes.as_ptr(),
                    packed.as_mut_ptr(),
                    lanes.len(),
                    words,
                    cnt,
                );
            }
            true
        }
        #[cfg(sf_cat_cxx)]
        PackKernel::CppDs => {
            debug_assert!(packed.len() >= cnt * words, "block slice sized cnt*words");
            // Safety: same contract as the literal kernel above.
            unsafe {
                sf_cat_transpose_ds(lanes.as_ptr(), packed.as_mut_ptr(), lanes.len(), words, cnt);
            }
            true
        }
    }
}

/// Direct entry for tests/calibration: run one named C++ kernel without
/// touching the process-wide `SF_CAT_KERNEL` selection.
#[cfg(sf_cat_cxx)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn transpose_with(
    kernel: &str,
    lanes: &[u64],
    packed: &mut [u64],
    words: usize,
    cnt: usize,
) {
    debug_assert!(packed.len() >= cnt * words, "block slice sized cnt*words");
    match kernel {
        // Safety: same contract as `transpose` above.
        "literal" => unsafe {
            sf_cat_transpose_literal(lanes.as_ptr(), packed.as_mut_ptr(), lanes.len(), words, cnt);
        },
        "ds" => unsafe {
            sf_cat_transpose_ds(lanes.as_ptr(), packed.as_mut_ptr(), lanes.len(), words, cnt);
        },
        other => panic!("unknown C++ kernel {other:?}"),
    }
}
