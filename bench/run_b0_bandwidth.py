"""B0: achievable memory bandwidth, the baseline for every efficiency number.

Copies a state-sized buffer with (a) our copy kernel and (b) CuPy's own copy, for complex64
buffers of 2^n amplitudes. Bytes moved per copy = 2 * 8 * 2^n (one read + one write).

    python -m bench.run_b0_bandwidth --max-qubits 27
"""

from __future__ import annotations

import argparse
from typing import Any

import cupy as cp
import numpy as np

from bench.run_env import save_result, time_gpu
from qsimgpu.backends.cuda_backend import BLOCK, MAX_BLOCKS, _module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-qubits", type=int, default=16)
    ap.add_argument("--max-qubits", type=int, default=27)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    copy = _module("gates.cu", "complex64").get_function("copy_state")
    rows = []
    for n in range(args.min_qubits, args.max_qubits + 1):
        size = 1 << n
        src = cp.ones(size, dtype=cp.complex64)
        dst = cp.empty_like(src)
        nbytes = 2 * src.nbytes
        grid = (max(1, min(MAX_BLOCKS, (size + BLOCK - 1) // BLOCK)),)
        def run_kernel(src: cp.ndarray = src, dst: cp.ndarray = dst, size: int = size,
                       grid: tuple[int] = grid) -> None:
            copy(grid, (BLOCK,), (src, dst, np.int64(size)))

        def run_cupy(src: cp.ndarray = src, dst: cp.ndarray = dst) -> None:
            cp.copyto(dst, src)

        t_kernel = time_gpu(run_kernel, repeats=args.repeats)
        t_cupy = time_gpu(run_cupy, repeats=args.repeats)
        row = {"n": n, "bytes_moved": nbytes,
               "copy_kernel": t_kernel | {"GBps": nbytes / t_kernel["median_ms"] / 1e6},
               "cupy_copyto": t_cupy | {"GBps": nbytes / t_cupy["median_ms"] / 1e6}}
        rows.append(row)
        print(f"n={n:2d}  kernel {row['copy_kernel']['GBps']:7.1f} GB/s   "
              f"cupy {row['cupy_copyto']['GBps']:7.1f} GB/s")
        del src, dst
        cp.get_default_memory_pool().free_all_blocks()
    def best_of(rs: list[dict[str, Any]]) -> float:
        return max(max(r["copy_kernel"]["GBps"], r["cupy_copyto"]["GBps"]) for r in rs)

    # Small buffers fit in the L2 cache and would overstate DRAM bandwidth, so the baseline
    # uses only buffers of at least 2^23 amplitudes (64 MiB), far larger than any L2.
    dram = [r for r in rows if r["n"] >= 23] or rows
    path = save_result("b0_bandwidth", {"dtype": "complex64", "rows": rows,
                                        "achievable_GBps": best_of(dram),
                                        "achievable_definition": "best copy, buffers >= 2^23",
                                        "small_buffer_peak_GBps": best_of(rows)})
    print(f"achievable DRAM bandwidth = {best_of(dram):.1f} GB/s "
          f"(small-buffer peak {best_of(rows):.1f}); saved {path}")


if __name__ == "__main__":
    main()
