// Exact Pauli-string expectation value <psi|P|psi> in one pass.
//
// For x_mask (bits with X or Y), z_mask (bits with Z or Y) and n_y Y's:
//   <psi|P|psi> = i^n_y * sum_i conj(psi[i ^ x_mask]) * (-1)^popcount(i & z_mask) * psi[i]
// This kernel computes the sum (without the i^n_y factor, applied on the host). Each block
// writes one partial sum; the host adds the partials. Accumulation is in double even for a
// complex64 state, so summing 2^n terms does not lose precision.
// With x_mask = z_mask = 0 the result is <psi|psi> = norm^2.

#define REDUCE_BLOCK 256

extern "C" __global__ void pauli_partial(const cplx* __restrict__ psi, const idx_t n_amp,
                                         const idx_t x_mask, const idx_t z_mask,
                                         double* __restrict__ out_re,
                                         double* __restrict__ out_im) {
    __shared__ double sre[REDUCE_BLOCK];
    __shared__ double sim[REDUCE_BLOCK];
    double re = 0.0, im = 0.0;
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t i = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_amp; i += stride) {
        const cplx a = psi[i];
        const cplx b = psi[i ^ x_mask];
        const double ar = a.real(), ai = a.imag(), br = b.real(), bi = -(double)b.imag();
        const double sign = (__popcll(i & z_mask) & 1) ? -1.0 : 1.0;
        re += sign * (br * ar - bi * ai);
        im += sign * (br * ai + bi * ar);
    }
    sre[threadIdx.x] = re;
    sim[threadIdx.x] = im;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            sre[threadIdx.x] += sre[threadIdx.x + w];
            sim[threadIdx.x] += sim[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_re[blockIdx.x] = sre[0];
        out_im[blockIdx.x] = sim[0];
    }
}
