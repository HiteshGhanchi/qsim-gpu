"""B3: how close does a gate get to the copy-kernel bandwidth?

Measures the copy kernel and gate kernels back to back on the same state size, in the same
process, so both see the same GPU clocks. Efficiency = gate GB/s / copy GB/s, where both move
16 bytes per complex64 amplitude (read + write).

    python -m bench.run_b3_efficiency --qubits 26 --tag baseline
"""

from __future__ import annotations

import argparse
from typing import Any

import cupy as cp
import numpy as np

from bench.run_env import save_result, time_gpu
from qsim.backends.cuda_backend import BLOCK, MAX_BLOCKS, CudaBackend, _module
from qsim.circuit import Gate


def cases(n: int) -> list[tuple[str, Gate]]:
    hi, lo = n - 1, 0
    return [
        ("h_high_target", Gate("h", (hi,))),
        ("h_low_target", Gate("h", (lo,))),
        ("rz_high_target", Gate("rz", (hi,), params=(0.3,))),
        ("x_high_target", Gate("x", (hi,))),
        ("cz", Gate("cz", (hi,), (hi - 1,))),
        ("cx", Gate("cx", (hi,), (hi - 1,))),
        ("swap", Gate("swap", (hi, hi - 1))),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=26)
    ap.add_argument("--per-timing", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--generic", action="store_true",
                    help="force the general kernels (only meaningful once specialised ones exist)")
    args = ap.parse_args()

    n = args.qubits
    size = 1 << n
    nbytes = 16 * size
    b = CudaBackend(n, np.complex64)
    if args.generic and hasattr(b, "specialized"):
        b.specialized = False
    copy = _module("gates.cu", "complex64").get_function("copy_state")
    dst = cp.empty_like(b.state)
    grid = (max(1, min(MAX_BLOCKS, (size + BLOCK - 1) // BLOCK)),)

    def copy_once() -> None:
        copy(grid, (BLOCK,), (b.state, dst, np.int64(size)))

    t_copy = time_gpu(lambda: [copy_once() for _ in range(args.per_timing)], repeats=args.repeats)
    copy_ms = t_copy["median_ms"] / args.per_timing
    copy_gbps = nbytes / copy_ms / 1e6
    print(f"copy: {copy_ms:.3f} ms  {copy_gbps:.1f} GB/s")
    rows: list[dict[str, Any]] = []
    for name, g in cases(n):
        t = time_gpu(lambda g=g: [b.apply(g) for _ in range(args.per_timing)],
                     repeats=args.repeats)
        ms = t["median_ms"] / args.per_timing
        gbps = nbytes / ms / 1e6  # "effective": as if every amplitude were read and written
        rows.append({"case": name, "gate": g.name, "ms_per_gate": ms, "effective_GBps": gbps,
                     "fraction_of_copy": gbps / copy_gbps, "timing": t})
        print(f"{name:15s} {ms:9.3f} ms  {gbps:7.1f} GB/s  {100 * gbps / copy_gbps:6.1f}% of copy")
    path = save_result(f"b3_efficiency_{args.tag}", {
        "n": n, "dtype": "complex64", "copy": {"ms": copy_ms, "GBps": copy_gbps, "timing": t_copy},
        "roofline_ms_per_gate": copy_ms, "rows": rows})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
