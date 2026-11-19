// Gate kernels. Every kernel uses a grid-stride loop, so any launch size covers all work.

// Plain copy: dst[i] = src[i]. Used as the memory-bandwidth baseline (benchmark B0)
// and as the first "does the GPU toolchain work" kernel.
extern "C" __global__ void copy_state(const cplx* __restrict__ src, cplx* __restrict__ dst,
                                      const idx_t n) {
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t i = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// General single-qubit gate [[m00, m01], [m10, m11]] on `target`, applied only where every
// control bit is 1.
//
// Each thread handles one amplitude pair (i0, i1) that differs only in the target bit.
// fixed_mask = target bit | control bits, ctrl_mask = control bits.
// Work items: 2^(n - 1 - n_controls). Thread number t becomes i0 by inserting a 0 at every
// fixed position and then switching the control bits on. With no controls this is the
// textbook formula i0 = ((t >> k) << (k + 1)) | (t & ((1 << k) - 1)).
extern "C" __global__ void apply_1q(cplx* __restrict__ psi, const idx_t n_work, const int target,
                                    const idx_t fixed_mask, const idx_t ctrl_mask,
                                    const cplx m00, const cplx m01,
                                    const cplx m10, const cplx m11) {
    const idx_t tbit = 1LL << target;
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t i0 = insert_zero_bits(t, fixed_mask) | ctrl_mask;
        const idx_t i1 = i0 | tbit;
        const cplx a = psi[i0];
        const cplx b = psi[i1];
        psi[i0] = m00 * a + m01 * b;
        psi[i1] = m10 * a + m11 * b;
    }
}
