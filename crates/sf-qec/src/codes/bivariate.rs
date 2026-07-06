//! Bivariate Bicycle codes.
use super::ldpc::LDPCCode;

// TODO: BivariateBicycleCode currently wraps an empty LDPCCode.
// Plan: construct the full n=2*l*m BB code from circulant matrices A, B
// as described in https://arxiv.org/abs/2308.07915.
pub struct BivariateBicycleCode {
    pub inner: LDPCCode,
}

impl BivariateBicycleCode {
    pub fn new(l: usize, m: usize) -> Self {
        Self {
            inner: LDPCCode::new(2 * l * m, 2, 4), // Placeholder
        }
    }
}
