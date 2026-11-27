"""B2: how does the time of one gate grow with the number of qubits?

H on the middle qubit with the general kernel, complex64, n = 10..max. Once the GPU is
saturated the time should double per extra qubit (twice the memory to stream).

    python -m bench.run_b2_scaling --max-qubits 28
"""

from __future__ import annotations

import argparse

import numpy as np

from bench.run_env import save_result, time_gpu
from qsimgpu import gates
from qsimgpu.backends.cuda_backend import CudaBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-qubits", type=int, default=10)
    ap.add_argument("--max-qubits", type=int, default=28)
    ap.add_argument("--per-timing", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    args = ap.parse_args()

    import cupy as cp

    h = gates.base_matrix("h")
    itemsize = np.dtype(args.dtype).itemsize
    rows = []
    for n in range(args.min_qubits, args.max_qubits + 1):
        b = CudaBackend(n, np.dtype(args.dtype))
        k = n // 2
        t = time_gpu(lambda b=b, k=k: [b.apply_1q_matrix(h, k) for _ in range(args.per_timing)],
                     repeats=args.repeats)
        per_gate = t["median_ms"] / args.per_timing
        nbytes = 2 * itemsize * (1 << n)
        rows.append({"n": n, "target": k, "ms_per_gate": per_gate,
                     "GBps": nbytes / per_gate / 1e6, "timing": t})
        prev = rows[-2]["ms_per_gate"] if len(rows) > 1 else None
        ratio = f"x{per_gate / prev:4.2f}" if prev else ""
        print(f"n={n:2d}  {per_gate:9.4f} ms/gate  {nbytes / per_gate / 1e6:7.1f} GB/s  {ratio}")
        del b
        cp.get_default_memory_pool().free_all_blocks()
    path = save_result(f"b2_scaling_{args.dtype}", {"dtype": args.dtype, "gate": "h",
                                                    "kernel": "apply_1q (general)", "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
