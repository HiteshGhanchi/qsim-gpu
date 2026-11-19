"""B1: does the target qubit change the time of a single-qubit gate?

Applies H to target k = 0..n-1 on an n-qubit complex64 state with the general 2x2 kernel.
Each timing covers `--per-timing` back-to-back gates, so launch gaps don't count much.
Effective bandwidth = 16 bytes * 2^n per gate (read + write every amplitude once).

    python -m bench.run_b1_target --qubits 26
"""

from __future__ import annotations

import argparse

import numpy as np

from bench.run_env import save_result, time_gpu
from qsim import gates
from qsim.backends.cuda_backend import CudaBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=26)
    ap.add_argument("--per-timing", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    n = args.qubits
    b = CudaBackend(n, np.complex64)
    h = gates.base_matrix("h")
    nbytes = 16 * (1 << n)
    rows = []
    for k in range(n):
        t = time_gpu(lambda k=k: [b.apply_1q_matrix(h, k) for _ in range(args.per_timing)],
                     repeats=args.repeats)
        per_gate = t["median_ms"] / args.per_timing
        rows.append({"target": k, "ms_per_gate": per_gate, "GBps": nbytes / per_gate / 1e6,
                     "timing": t})
        print(f"k={k:2d}  {per_gate:8.3f} ms/gate  {nbytes / per_gate / 1e6:7.1f} GB/s")
    path = save_result(f"b1_target_{args.tag}", {"n": n, "dtype": "complex64", "gate": "h",
                                                 "kernel": "apply_1q (general)",
                                                 "gates_per_timing": args.per_timing, "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
