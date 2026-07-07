// Single-qubit gate kernel: applies a 2x2 unitary to qubit `target` across 2^n amplitudes.
// Each thread handles one pair of amplitudes separated by stride = 2^target.
extern "C" __global__ void apply_gate_1q(
    double *state_re, double *state_im,
    int n_qubits, int target,
    double m00_re, double m00_im,
    double m01_re, double m01_im,
    double m10_re, double m10_im,
    double m11_re, double m11_im
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = 1ULL << target;
    unsigned long long num_pairs = 1ULL << (n_qubits - 1);

    if (idx >= num_pairs) return;

    // Compute the two indices: i0 has bit `target` = 0, i1 has bit `target` = 1
    unsigned long long block = idx >> target;
    unsigned long long local = idx & (stride - 1);
    unsigned long long i0 = (block << (target + 1)) | local;
    unsigned long long i1 = i0 | stride;

    double a_re = state_re[i0], a_im = state_im[i0];
    double b_re = state_re[i1], b_im = state_im[i1];

    // new_a = m00 * a + m01 * b
    state_re[i0] = (m00_re * a_re - m00_im * a_im) + (m01_re * b_re - m01_im * b_im);
    state_im[i0] = (m00_re * a_im + m00_im * a_re) + (m01_re * b_im + m01_im * b_re);

    // new_b = m10 * a + m11 * b
    state_re[i1] = (m10_re * a_re - m10_im * a_im) + (m11_re * b_re - m11_im * b_im);
    state_im[i1] = (m10_re * a_im + m10_im * a_re) + (m11_re * b_im + m11_im * b_re);
}

// Diagonal single-qubit gate (RZ, S, T, P): only multiplies phases, no mixing.
// More memory-efficient — each thread touches only one amplitude.
extern "C" __global__ void apply_diagonal_1q(
    double *state_re, double *state_im,
    int n_qubits, int target,
    double d0_re, double d0_im,
    double d1_re, double d1_im
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long dim = 1ULL << n_qubits;

    if (idx >= dim) return;

    int bit = (idx >> target) & 1;
    double d_re = bit ? d1_re : d0_re;
    double d_im = bit ? d1_im : d0_im;

    double a_re = state_re[idx];
    double a_im = state_im[idx];

    state_re[idx] = d_re * a_re - d_im * a_im;
    state_im[idx] = d_re * a_im + d_im * a_re;
}

// Two-qubit gate kernel: applies a 4x4 unitary to qubits (ctrl, target).
// Matrix stored in row-major as 16 complex numbers (32 doubles).
extern "C" __global__ void apply_gate_2q(
    double *state_re, double *state_im,
    int n_qubits, int q0, int q1,
    const double *mat_re, const double *mat_im
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_quads = 1ULL << (n_qubits - 2);

    if (idx >= num_quads) return;

    int hi = (q0 > q1) ? q0 : q1;
    int lo = (q0 > q1) ? q1 : q0;

    // Remove bits at positions hi and lo to get base index
    unsigned long long mask_lo = (1ULL << lo) - 1;
    unsigned long long mask_mid = ((1ULL << (hi - 1)) - 1) ^ mask_lo;
    unsigned long long mask_hi_bits = ~((1ULL << hi) - 1) & ((1ULL << n_qubits) - 1);

    unsigned long long a = idx & mask_lo;
    unsigned long long b = (idx >> lo) & (mask_mid >> lo);
    unsigned long long c = (idx >> (hi - 1));

    unsigned long long base = a | (b << (lo + 1)) | (c << (hi + 1));

    unsigned long long indices[4];
    indices[0] = base;
    indices[1] = base | (1ULL << lo);
    indices[2] = base | (1ULL << hi);
    indices[3] = base | (1ULL << lo) | (1ULL << hi);

    // The matrix is indexed as (q0_bit*2 + q1_bit), but indices[] are ordered
    // by (bit_lo, bit_hi). When q0 < q1, lo=q0 and hi=q1, so
    // indices[1]=(q0=1,q1=0) and indices[2]=(q0=0,q1=1) need swapping
    // to match matrix row ordering (q0*2+q1): row1=(0,1), row2=(1,0).
    if (q0 < q1) {
        unsigned long long tmp = indices[1];
        indices[1] = indices[2];
        indices[2] = tmp;
    }

    double amp_re[4], amp_im[4];
    for (int i = 0; i < 4; i++) {
        amp_re[i] = state_re[indices[i]];
        amp_im[i] = state_im[indices[i]];
    }

    for (int i = 0; i < 4; i++) {
        double new_re = 0.0, new_im = 0.0;
        for (int j = 0; j < 4; j++) {
            double mr = mat_re[i * 4 + j];
            double mi = mat_im[i * 4 + j];
            new_re += mr * amp_re[j] - mi * amp_im[j];
            new_im += mr * amp_im[j] + mi * amp_re[j];
        }
        state_re[indices[i]] = new_re;
        state_im[indices[i]] = new_im;
    }
}

// Compute probabilities: |amplitude|^2 for each basis state.
extern "C" __global__ void compute_probabilities(
    const double *state_re, const double *state_im,
    double *probs, unsigned long long dim
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    double re = state_re[idx];
    double im = state_im[idx];
    probs[idx] = re * re + im * im;
}
