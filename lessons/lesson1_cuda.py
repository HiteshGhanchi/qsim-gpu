# Lesson 1: the same pair trick as lesson0.py, but running on the GPU.
#
# Run it with the project's environment (needs an NVIDIA GPU):
#     .venv/bin/python lessons/lesson1_cuda.py

import math

import cupy as cp  # NumPy-like arrays that live in GPU memory
import numpy as np

# --- Part 1: CPU memory vs GPU memory ---------------------------------------------------
host = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)  # lives in normal RAM
device = cp.asarray(host)  # copied into GPU memory
print("on the GPU: ", device, "| type:", type(device).__name__)
print("back on CPU:", device.get())  # .get() copies GPU -> CPU

# --- Part 2: a CUDA kernel, written as a string of C code -------------------------------
# A kernel is one small function that thousands of GPU threads run at the same time.
# Each thread gets its own number t and handles ONE pair of amplitudes (i0, i1).
SOURCE = r"""
#include <cupy/complex.cuh>
extern "C" __global__
void apply_gate(complex<float>* psi, long long n_pairs, int k,
                complex<float> g00, complex<float> g01,
                complex<float> g10, complex<float> g11) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;  // my thread number
    if (t >= n_pairs) return;                    // extra threads do nothing
    long long low  = t & ((1LL << k) - 1);       // bits below k
    long long i0   = ((t >> k) << (k + 1)) | low; // insert a 0 at bit k
    long long i1   = i0 | (1LL << k);             // same index with bit k = 1
    complex<float> a = psi[i0], b = psi[i1];
    psi[i0] = g00 * a + g01 * b;                 // exactly lesson0's apply_on()
    psi[i1] = g10 * a + g11 * b;
}
"""
kernel = cp.RawKernel(SOURCE, "apply_gate", options=("-std=c++17",))  # compiled on first use

S = np.complex64(1 / math.sqrt(2))
H = [S, S, S, -S]  # g00, g01, g10, g11


def apply_on_gpu(psi: cp.ndarray, k: int, g: list) -> None:
    n_pairs = psi.size // 2
    threads = 256  # threads per block
    blocks = (n_pairs + threads - 1) // threads  # enough blocks to cover every pair
    kernel((blocks,), (threads,), (psi, np.int64(n_pairs), np.int32(k), *g))


# --- Part 3: the Bell state again, on the GPU --------------------------------------------
psi = cp.asarray(np.array([1, 0, 0, 0], dtype=np.complex64))  # |00>
apply_on_gpu(psi, 0, H)  # H on qubit 0
# CNOT(control 0 -> target 1) as an array operation: swap the scores of 01 and 11
psi[[1, 3]] = psi[[3, 1]]
print("Bell state from the GPU:", np.round(psi.get(), 3))

# --- Part 4: a big state, where the GPU earns its keep -----------------------------------
n = 24  # 2^24 = 16.7 million amplitudes = 128 MB
big = cp.zeros(1 << n, dtype=cp.complex64)
big[0] = 1
start, stop = cp.cuda.Event(), cp.cuda.Event()  # GPU-side stopwatch
start.record()
for k in range(n):  # H on every qubit -> equal superposition
    apply_on_gpu(big, k, H)
stop.record()
stop.synchronize()  # wait for the GPU to finish
ms = cp.cuda.get_elapsed_time(start, stop)
print(f"{n} H gates on a {n}-qubit state: {ms:.1f} ms total")
print("every amplitude should be 1/sqrt(2^n):", float(abs(big[12345])), "vs",
      1 / math.sqrt(2**n))

# --- Part 5: the same thing through the real project --------------------------------------
from qsim.backends.cuda_backend import CudaBackend  # noqa: E402
from qsim.circuit import Circuit  # noqa: E402

sim = CudaBackend(2)
sim.run(Circuit(2).h(0).cx(0, 1))
print("qsim Bell state:", np.round(sim.statevector(), 3), "samples:", sim.sample(10, seed=1))
