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

// General k-qubit matrix (2^k x 2^k, row-major in `mat`) on k targets, applied where every
// control bit is 1. Used for SWAP and other multi-target gates in the general path, and for
// the fused "unitary" gates produced by the fusion pass.
//
// target_pack holds the k target positions, 6 bits each (targets[0] in the lowest bits).
// Bit i of a local index j selects the value of targets[i] (little-endian, like the matrix).
// Work items: 2^(n - k - n_controls); each thread gathers 2^k amplitudes, multiplies, scatters.
//
// K is a compile-time constant, one kernel per size (apply_k1 .. apply_k5). With a fixed K
// the loops unroll and the 2^K amplitudes stay in registers instead of spilling to local
// memory. The matrix is copied into shared memory once per block.
template <int K>
__device__ __forceinline__ void apply_kq_impl(cplx* __restrict__ psi, const idx_t n_work,
                                              const idx_t target_pack, const idx_t fixed_mask,
                                              const idx_t ctrl_mask,
                                              const cplx* __restrict__ mat) {
    constexpr int DIM = 1 << K;
    __shared__ cplx smat[DIM * DIM];
    for (int i = threadIdx.x; i < DIM * DIM; i += blockDim.x) smat[i] = mat[i];
    __syncthreads();
    idx_t tbit[K];
#pragma unroll
    for (int i = 0; i < K; ++i) tbit[i] = 1LL << ((target_pack >> (6 * i)) & 63);
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t base = insert_zero_bits(t, fixed_mask) | ctrl_mask;
        cplx amp[DIM];
#pragma unroll
        for (int j = 0; j < DIM; ++j) {
            idx_t off = 0;
#pragma unroll
            for (int i = 0; i < K; ++i) if ((j >> i) & 1) off |= tbit[i];
            amp[j] = psi[base | off];
        }
#pragma unroll
        for (int r = 0; r < DIM; ++r) {
            cplx acc = cplx(0, 0);
#pragma unroll
            for (int c = 0; c < DIM; ++c) acc += smat[r * DIM + c] * amp[c];
            idx_t off = 0;
#pragma unroll
            for (int i = 0; i < K; ++i) if ((r >> i) & 1) off |= tbit[i];
            psi[base | off] = acc;
        }
    }
}

#define DEFINE_APPLY_K(K)                                                                   \
    extern "C" __global__ void apply_k##K(cplx* __restrict__ psi, const idx_t n_work,       \
                                          const idx_t target_pack, const idx_t fixed_mask,  \
                                          const idx_t ctrl_mask,                            \
                                          const cplx* __restrict__ mat) {                   \
        apply_kq_impl<K>(psi, n_work, target_pack, fixed_mask, ctrl_mask, mat);             \
    }
DEFINE_APPLY_K(1)
DEFINE_APPLY_K(2)
DEFINE_APPLY_K(3)
DEFINE_APPLY_K(4)
DEFINE_APPLY_K(5)

// ---------------------------------------------------------------------------------------
// Specialised kernels (one per gate family). They do the same job as apply_1q/apply_kq for
// their gates, with less arithmetic and, for phases and SWAP, less memory traffic.
// ---------------------------------------------------------------------------------------

// Phase on one subspace: psi[i] *= phase for every i whose `fixed_mask` bits all equal
// `set_mask`. Covers Z, S, T, P, CZ, CP and multi-controlled Z: their matrices are
// diag(1, phase), so amplitudes with target bit 0 are left alone and never even read.
// Work items: 2^(n - popcount(fixed_mask)).
extern "C" __global__ void apply_phase(cplx* __restrict__ psi, const idx_t n_work,
                                       const idx_t fixed_mask, const idx_t set_mask,
                                       const cplx phase) {
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t i = insert_zero_bits(t, fixed_mask) | set_mask;
        psi[i] *= phase;
    }
}

// General diagonal diag(d0, d1) on `target` (e.g. Rz), where controls are 1: one thread per
// amplitude, no pairing. fixed_mask = control bits, ctrl_mask = control bits.
extern "C" __global__ void apply_diag(cplx* __restrict__ psi, const idx_t n_work,
                                      const int target, const idx_t fixed_mask,
                                      const idx_t ctrl_mask, const cplx d0, const cplx d1) {
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t i = insert_zero_bits(t, fixed_mask) | ctrl_mask;
        psi[i] *= ((i >> target) & 1) ? d1 : d0;
    }
}

// X / CX / CCX / multi-controlled X: swap each amplitude pair, no multiplications.
extern "C" __global__ void apply_x(cplx* __restrict__ psi, const idx_t n_work, const int target,
                                   const idx_t fixed_mask, const idx_t ctrl_mask) {
    const idx_t tbit = 1LL << target;
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t i0 = insert_zero_bits(t, fixed_mask) | ctrl_mask;
        const cplx a = psi[i0];
        psi[i0] = psi[i0 | tbit];
        psi[i0 | tbit] = a;
    }
}

// SWAP of qubits a and b (optionally controlled): only the |01> and |10> amplitudes of each
// group move, so half of the state is never touched.
extern "C" __global__ void apply_swap(cplx* __restrict__ psi, const idx_t n_work, const int a,
                                      const int b, const idx_t fixed_mask,
                                      const idx_t ctrl_mask) {
    const idx_t abit = 1LL << a, bbit = 1LL << b;
    const idx_t stride = (idx_t)blockDim.x * gridDim.x;
    for (idx_t t = (idx_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_work; t += stride) {
        const idx_t base = insert_zero_bits(t, fixed_mask) | ctrl_mask;
        const cplx v = psi[base | abit];
        psi[base | abit] = psi[base | bbit];
        psi[base | bbit] = v;
    }
}
