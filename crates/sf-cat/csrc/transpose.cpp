// sf-cat C++ kernel lane — additive Lever-1 experiment (pack path).
//
// Two entry points, both producing exactly the same output as
// `sample_packed`'s Rust fallback loop, i.e. for shot i and qubit-word
// wq:  packed[i * words + wq] |=  bit_i(lane_q) << (q % 64).
//
//   sf_cat_transpose_literal : literal port of the Rust scalar loop
//                              (codegen-comparison baseline).
//   sf_cat_transpose_ds      : 64x64 bit-matrix transpose by a 6-stage
//                              delta-swap network, replacing the
//                              O(n * 64) per-bit scatter with register-
//                              local work per qubit word.
//
// Delta-swap orientation note (verified numerically before writing this
// file, see sim4_bench_suite/_sf_cat_dscheck.py): with masks
// "s ones then s zeros" and stages s = 32..1, the network yields the
// transpose with BOTH indices reversed, so the output row `63 - i` is
// bit-reversed before being stored at row i.
//
// Compiled by build.rs with the local MinGW-w64 g++ when available; the
// Rust side falls back to its own loop when this object is absent.
// Plain C-style code only: no STL, no exceptions/RTTI, so it links into
// the windows-gnu Rust binary without a libstdc++ runtime dependency.

#include <stddef.h>
#include <stdint.h>

namespace {

inline uint64_t stage_mask(int s) {
    uint64_t m = 0;
    for (int k = 0; k < 64; k += 2 * s) {
        m |= ((1ull << s) - 1ull) << k;
    }
    return m;
}

inline uint64_t reverse_bits64(uint64_t v) {
    v = ((v & 0x5555555555555555ull) << 1) | ((v >> 1) & 0x5555555555555555ull);
    v = ((v & 0x3333333333333333ull) << 2) | ((v >> 2) & 0x3333333333333333ull);
    v = ((v & 0x0F0F0F0F0F0F0F0Full) << 4) | ((v >> 4) & 0x0F0F0F0F0F0F0F0Full);
    return __builtin_bswap64(v);
}

}  // namespace

extern "C" void sf_cat_transpose_literal(const uint64_t* lanes, uint64_t* packed,
                                         size_t n, size_t words, size_t cnt) {
    for (size_t q = 0; q < n; ++q) {
        const uint64_t lane = lanes[q];
        const unsigned sh = (unsigned)(q & 63);
        const size_t wq = q >> 6;
        for (size_t i = 0; i < cnt; ++i) {
            packed[i * words + wq] |= ((lane >> i) & 1ull) << sh;
        }
    }
}

extern "C" void sf_cat_transpose_ds(const uint64_t* lanes, uint64_t* packed,
                                    size_t n, size_t words, size_t cnt) {
    const size_t nwords = (n + 63) >> 6;
    for (size_t wq = 0; wq < nwords; ++wq) {
        uint64_t t[64];
        const size_t base = wq << 6;
        for (size_t q = 0; q < 64; ++q) {
            t[q] = (base + q < n) ? lanes[base + q] : 0;
        }
        for (int s = 32; s > 0; s >>= 1) {
            const uint64_t m = stage_mask(s);
            for (int i = 0; i < 64; ++i) {
                if ((i & s) == 0) {
                    const int j = i + s;
                    const uint64_t a = t[i];
                    const uint64_t b = t[j];
                    const uint64_t x = (a ^ (b >> s)) & m;
                    t[i] = a ^ x;
                    t[j] = b ^ (x << s);
                }
            }
        }
        // Row `63 - i` holds shot i's qubit word, bit-reversed.
        for (size_t i = 0; i < cnt; ++i) {
            packed[i * words + wq] |= reverse_bits64(t[63 - i]);
        }
    }
}
