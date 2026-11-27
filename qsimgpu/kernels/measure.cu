// Sampling without a second full-size buffer (two passes).
//
// Pass 1, chunk_prob_sums: split the state into chunks of 2^chunk_log2 amplitudes and write
//   the total probability of each chunk (accumulated in double). Extra memory: one double
//   per chunk, i.e. 2^(n - chunk_log2) values instead of 2^n.
// Host: cumulative sum over chunks, draw uniform numbers, pick a chunk per shot.
// Pass 2, locate_in_chunk: one thread per shot walks its chunk until the running sum passes
//   the shot's residual.

#define REDUCE_BLOCK 256

__device__ __forceinline__ double prob(const cplx a) {
    const double re = (double)a.real(), im = (double)a.imag();
    return re * re + im * im;
}

extern "C" __global__ void chunk_prob_sums(const cplx* __restrict__ psi, const int chunk_log2,
                                           const idx_t n_chunks, double* __restrict__ out) {
    __shared__ double sh[REDUCE_BLOCK];
    const idx_t chunk = 1LL << chunk_log2;
    for (idx_t c = blockIdx.x; c < n_chunks; c += gridDim.x) {
        double s = 0.0;
        for (idx_t i = threadIdx.x; i < chunk; i += blockDim.x) s += prob(psi[c * chunk + i]);
        sh[threadIdx.x] = s;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[c] = sh[0];
        __syncthreads();
    }
}

extern "C" __global__ void locate_in_chunk(const cplx* __restrict__ psi, const int chunk_log2,
                                           const idx_t* __restrict__ chunk_ids,
                                           const double* __restrict__ residual,
                                           idx_t* __restrict__ out, const idx_t n_shots) {
    const idx_t chunk = 1LL << chunk_log2;
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t s = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; s < n_shots; s += stride) {
        const idx_t base = chunk_ids[s] << chunk_log2;
        const double r = residual[s];
        double acc = 0.0;
        idx_t pick = -1, last_nonzero = base;
        for (idx_t i = 0; i < chunk; ++i) {
            const double p = prob(psi[base + i]);
            if (p > 0.0) last_nonzero = base + i;
            acc += p;
            if (acc > r) { pick = base + i; break; }
        }
        out[s] = (pick >= 0) ? pick : last_nonzero;  // rounding fallback: last non-zero entry
    }
}
