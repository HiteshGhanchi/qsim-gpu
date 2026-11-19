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
