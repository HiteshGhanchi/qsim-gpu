// Shared definitions for every kernel file.
//
// The backend prepends one line before compiling, either
//     typedef float  real_t;   (complex64 state)
// or  typedef double real_t;   (complex128 state)
// so the same source builds both precisions.

#include <cupy/complex.cuh>

typedef complex<real_t> cplx;
typedef long long idx_t;  // 64-bit: at 30 qubits, byte offsets pass 2^32

// Insert a 0 bit at every position set in `mask`, lowest position first.
// Example: idx = 0b11, mask = 0b010  ->  0b101.
// This turns a compact thread number into a state index whose "fixed" bits are 0.
__device__ __forceinline__ idx_t insert_zero_bits(idx_t idx, idx_t mask) {
    while (mask) {
        const int p = __ffsll(mask) - 1;      // lowest set bit
        mask &= mask - 1;                     // clear it
        const idx_t low = idx & ((1LL << p) - 1);
        idx = ((idx - low) << 1) | low;       // shift the high part up by one
    }
    return idx;
}
